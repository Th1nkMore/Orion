"""Fail-closed frozen-ORION chronological actual-target runner primitives.

The command-line entry point for this module is deliberately preflight-only:
it does not construct ORION, touch CUDA, submit Slurm, run CARLA, or train.  A
real integration must inject audited occupancy/support/target hooks and then
call :func:`run_chronological_actual_target_replay` from the ORION environment.

This module keeps the deployed ``orion_stage3_agent.py`` unchanged.  It loads
that Python config into memory, deep-copies it, and makes a dedicated
actual-target test-pipeline mutation with traffic-light annotations enabled.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from uq_estimator.bev_target_rasterizer import (
    GT_RASTERIZER_ID,
    PAIRWISE_BEV_IOU_POLICY_ID,
    SELECTED_MODE_RASTERIZER_ID,
    pairwise_bev_iou_v1,
    rasterize_planningmetric_gt_v1,
    selected_mode_occupancy_callback_v1,
)
from uq_estimator.decoded_actual_target_export import (
    ActualTargetBranchBundleV1,
    BRANCH_BUNDLE_SCHEMA_VERSION,
    FailureEventPolicyV1,
    PairedActualTargetBundleV1,
    bridge_actual_target_bundle_to_v2_record,
    pair_actual_target_branches,
)
from uq_estimator.orion_decode_adapter import (
    AdaptedORIONBatchV1,
    ORIONDecodeAdapterConfigV1,
    adapt_orion_head_outputs_v1,
)
from uq_estimator.orion_replay_smoke import (
    REPLAY_PLAN_SCHEMA_VERSION,
    RUNTIME_ATTESTATION_SCHEMA_VERSION,
)
from uq_estimator.projected_visible_support import (
    ORION_CAMERA_ORDER,
    VISIBLE_SUPPORT_PROJECTION_VERSION,
    project_boxes_to_visible_patch_support,
)


RUNNER_PREFLIGHT_SCHEMA_VERSION = "orion-actual-target-runner-preflight/v1"
RUNNER_FRAME_AUDIT_SCHEMA_VERSION = "orion-actual-target-frame-audit/v1"
RUNNER_INTEGRATION_SCHEMA_VERSION = "orion-actual-target-runtime-hooks/v1"
RUNNER_EXECUTION_SCHEMA_VERSION = "orion-actual-target-runner-result/v1"
ACTUAL_TARGET_PIPELINE_ID = "orion-stage3-agent-actual-target-pipeline/v1"
TRAFFIC_SEMANTICS_ID = "orion-v1-affects-ego-light-state-only/v1"
PATCH_FEATURE_HOOK_ID = "orion-evavit-position-level-6x40x40-row-major/v1"
GT_BOX_Z_ORIGIN = "bottom"
DECODED_BOX_Z_ORIGIN = "center"
BOX_Z_ORIGIN_POLICY_ID = "b2d-gt-bottom-orion-decoded-center/v1"
PILOT_CALIBRATION_POLICY_ID = (
    "preregistered-pilot-thresholds-0p50-support-0p01/v1"
)
PILOT_MATCH_POLICY_ID = "class-score-distance-gated-one-to-one/v1"
PILOT_MINIMUM_PREDICTION_SCORE = 0.50
PILOT_MAXIMUM_CENTER_DISTANCE_M = 4.0
PILOT_SENSITIVITY_CENTER_DISTANCE_M = 2.0

CANONICAL_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
CAMERA_DIRECTORY_BY_NAME = {
    "CAM_FRONT": "rgb_front",
    "CAM_FRONT_LEFT": "rgb_front_left",
    "CAM_FRONT_RIGHT": "rgb_front_right",
    "CAM_BACK": "rgb_back",
    "CAM_BACK_LEFT": "rgb_back_left",
    "CAM_BACK_RIGHT": "rgb_back_right",
}
HEAD_OUTPUT_KEYS = (
    "all_cls_scores",
    "all_bbox_preds",
    "all_traj_preds",
    "all_traj_cls_scores",
    "all_traffic_states",
)
MEMORY_FIELDS = (
    "memory_embedding",
    "memory_reference_point",
    "memory_timestamp",
    "memory_egopose",
    "memory_velo",
    "sample_time",
    "memory_canbus",
    "memory_scene_tokens",
    "his_memory_canbus_len",
    "memory_scene_query",
    "scene_memory_timestamp",
)
REQUIRED_REAL_HOOKS = (
    "occupancy_rasterizer",
    "branch_target_builder",
    "patch_feature_extractor",
    "corruption_transform",
    "record_sink",
    "decoder_parity_check",
    "selected_motion_mode_check",
    "projection_overlay_check",
    "gt_axis_alignment_check",
    "gt_occupancy_rasterizer",
    "pairwise_bev_iou",
    "project_visible_support",
)
EXTERNAL_PRODUCTION_HOOKS = (
    "corruption_transform",
    "record_sink",
    "decoder_parity_check",
    "selected_motion_mode_check",
    "projection_overlay_check",
    "gt_axis_alignment_check",
)
PRODUCTION_BRANCH_BUILDER_TYPE = (
    "uq_estimator.orion_actual_target_builder",
    "ProductionActualTargetBranchBuilderV1",
)
SAFETY_ACTOR_CLASS_IDS = (0, 1, 2, 3, 7)
TRAFFIC_LIGHT_CLASS_ID = 6


class OrionActualTargetRunnerError(RuntimeError):
    """Raised whenever execution would require an unaudited assumption."""


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OrionActualTargetRunnerError("%s must be a non-empty string" % name)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stage3_agent_config(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute the repository's plain Python config and retain only config data."""

    path = Path(path).resolve()
    if not path.is_file():
        raise OrionActualTargetRunnerError("stage3 agent config does not exist")
    namespace = runpy.run_path(str(path))
    config = {
        key: value
        for key, value in namespace.items()
        if not key.startswith("__")
    }
    if not isinstance(config.get("test_pipeline"), list):
        raise OrionActualTargetRunnerError("config lacks test_pipeline")
    data = config.get("data")
    if not isinstance(data, Mapping) or not isinstance(data.get("test"), Mapping):
        raise OrionActualTargetRunnerError("config lacks data.test")
    return config, {
        "path": str(path),
        "sha256": _sha256(path),
        "loader": "python-runpy-local-read-only",
    }


def _pipeline_nodes(pipeline: Sequence[Mapping[str, Any]]) -> Iterable[Dict[str, Any]]:
    for raw in pipeline:
        if not isinstance(raw, dict):
            raise OrionActualTargetRunnerError("every pipeline node must be a dict")
        yield raw
        transforms = raw.get("transforms")
        if transforms is not None:
            if not isinstance(transforms, list):
                raise OrionActualTargetRunnerError("pipeline transforms must be a list")
            for nested in _pipeline_nodes(transforms):
                yield nested


def mutate_stage3_agent_config_for_actual_targets(
    source: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create an in-memory, exporter-only pipeline without changing deployment."""

    mutated = copy.deepcopy(dict(source))
    root_pipeline = mutated.get("test_pipeline")
    data = mutated.get("data")
    if not isinstance(root_pipeline, list) or not isinstance(data, dict):
        raise OrionActualTargetRunnerError("stage3 config structure is incomplete")
    test = data.get("test")
    if not isinstance(test, dict) or not isinstance(test.get("pipeline"), list):
        raise OrionActualTargetRunnerError("stage3 config lacks data.test.pipeline")
    if root_pipeline != test["pipeline"]:
        raise OrionActualTargetRunnerError(
            "root test_pipeline and data.test.pipeline disagree before mutation"
        )

    load_count = 0
    collect_count = 0
    for node in _pipeline_nodes(root_pipeline):
        node_type = node.get("type")
        if node_type == "LoadAnnotations3D":
            node["with_light_state"] = True
            node["with_actor_ids"] = True
            load_count += 1
        if node_type == "CustomCollect3D":
            keys = node.get("keys")
            if not isinstance(keys, list):
                raise OrionActualTargetRunnerError("CustomCollect3D keys must be a list")
            for required in ("traffic_state", "traffic_state_mask", "gt_actor_ids"):
                if required not in keys:
                    keys.append(required)
            collect_count += 1
    if load_count != 1:
        raise OrionActualTargetRunnerError(
            "actual-target pipeline requires exactly one LoadAnnotations3D"
        )
    if collect_count != 1:
        raise OrionActualTargetRunnerError(
            "actual-target pipeline requires exactly one CustomCollect3D"
        )

    # Keep the resolved test pipeline and its data.test reference identical,
    # but deliberately leave inference_only_pipeline and the source file alone.
    test["pipeline"] = root_pipeline
    data["samples_per_gpu"] = 1
    test["test_mode"] = True
    mutated["actual_target_export"] = {
        "pipeline_id": ACTUAL_TARGET_PIPELINE_ID,
        "source_deployment_config_mutated": False,
        "batch_size": 1,
        "with_light_state": True,
        "with_actor_ids": True,
        "target_axis_collect_keys": [
            "traffic_state",
            "traffic_state_mask",
            "gt_actor_ids",
        ],
        "traffic_semantics_id": TRAFFIC_SEMANTICS_ID,
    }

    nodes = list(_pipeline_nodes(root_pipeline))
    loaders = [node for node in nodes if node.get("type") == "LoadAnnotations3D"]
    collectors = [node for node in nodes if node.get("type") == "CustomCollect3D"]
    passed = (
        len(loaders) == 1
        and loaders[0].get("with_light_state") is True
        and loaders[0].get("with_actor_ids") is True
        and len(collectors) == 1
        and all(
            key in collectors[0].get("keys", [])
            for key in ("traffic_state", "traffic_state_mask", "gt_actor_ids")
        )
        and data.get("samples_per_gpu") == 1
    )
    return mutated, {
        "schema_version": ACTUAL_TARGET_PIPELINE_ID,
        "passed": bool(passed),
        "load_annotations_mutation_count": load_count,
        "custom_collect_mutation_count": collect_count,
        "with_light_state_enabled": True,
        "with_actor_ids_enabled": True,
        "collected_keys": ["traffic_state", "traffic_state_mask", "gt_actor_ids"],
        "samples_per_gpu": data.get("samples_per_gpu"),
        "source_deployment_config_mutated": False,
    }


def verify_local_traffic_formatter_fix(repo_root: Path) -> Dict[str, Any]:
    """Statically and functionally verify the object-aligned formatter repair."""

    repo_root = Path(repo_root).resolve()
    formatter = repo_root / "mmcv" / "datasets" / "pipelines" / "formating.py"
    helper = (
        repo_root
        / "mmcv"
        / "datasets"
        / "pipelines"
        / "traffic_state_alignment.py"
    )
    actor_helper = (
        repo_root / "mmcv" / "datasets" / "pipelines" / "actor_id_alignment.py"
    )
    loading = repo_root / "mmcv" / "datasets" / "pipelines" / "loading.py"
    transforms = (
        repo_root / "mmcv" / "datasets" / "pipelines" / "transforms_3d.py"
    )
    if not all(path.is_file() for path in (formatter, helper, actor_helper, loading, transforms)):
        raise OrionActualTargetRunnerError("traffic formatter/helper files are missing")
    text = formatter.read_text(encoding="utf-8")
    loading_text = loading.read_text(encoding="utf-8")
    transforms_text = transforms.read_text(encoding="utf-8")
    required_fragments = (
        "from .traffic_state_alignment import filter_traffic_state_by_box_mask",
        "filter_traffic_state_by_box_mask(",
        "results['traffic_state']",
        "results['traffic_state_mask']",
        "gt_bboxes_3d_mask",
        "filter_actor_ids_by_box_mask(",
        "results['gt_actor_ids']",
    )
    static_pass = all(fragment in text for fragment in required_fragments)
    actor_static_pass = (
        "with_actor_ids=False" in loading_text
        and "results['gt_actor_ids']" in loading_text
        and transforms_text.count("filter_actor_ids_by_box_mask(") >= 2
        and "from .actor_id_alignment import filter_actor_ids_by_box_mask" in text
        and "results['gt_actor_ids'] = filter_actor_ids_by_box_mask(" in text
    )

    spec = importlib.util.spec_from_file_location(
        "_orion_traffic_state_alignment_preflight", str(helper)
    )
    if spec is None or spec.loader is None:
        raise OrionActualTargetRunnerError("cannot load traffic alignment helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        import numpy as np

        state = np.asarray([[0, 1], [2, 0], [1, 1]], dtype=np.int64)
        loader_mask = np.asarray([True, False, True], dtype=np.bool_)
        box_mask = np.asarray([True, False, True], dtype=np.bool_)
        filtered_state, filtered_mask = module.filter_traffic_state_by_box_mask(
            state, loader_mask, box_mask
        )
        functional_pass = (
            filtered_state.tolist() == [[0, 1], [1, 1]]
            and filtered_mask.tolist() == [True, True]
        )
        actor_spec = importlib.util.spec_from_file_location(
            "_orion_actor_id_alignment_preflight", str(actor_helper)
        )
        if actor_spec is None or actor_spec.loader is None:
            raise OrionActualTargetRunnerError("cannot load actor-ID alignment helper")
        actor_module = importlib.util.module_from_spec(actor_spec)
        actor_spec.loader.exec_module(actor_module)
        actor_result = actor_module.filter_actor_ids_by_box_mask(
            np.asarray([101, 102, 103], dtype=np.int64), box_mask
        )
        actor_functional_pass = actor_result.tolist() == [101, 103]
    except Exception as exc:  # pragma: no cover - defensive real preflight.
        raise OrionActualTargetRunnerError(
            "traffic formatter functional verification failed: %s" % exc
        ) from exc
    return {
        "passed": bool(
            static_pass
            and functional_pass
            and actor_static_pass
            and actor_functional_pass
        ),
        "static_callsite_verified": bool(static_pass),
        "functional_alignment_fixture_verified": bool(functional_pass),
        "actor_id_all_filter_stages_verified": bool(actor_static_pass),
        "actor_id_functional_alignment_verified": bool(actor_functional_pass),
        "formatter_path": str(formatter),
        "formatter_sha256": _sha256(formatter),
        "helper_path": str(helper),
        "helper_sha256": _sha256(helper),
        "actor_helper_path": str(actor_helper),
        "actor_helper_sha256": _sha256(actor_helper),
        "loading_path": str(loading),
        "loading_sha256": _sha256(loading),
        "transforms_path": str(transforms),
        "transforms_sha256": _sha256(transforms),
    }


def verify_box_z_origin_lineage(repo_root: Path) -> Dict[str, Any]:
    """Verify the distinct GT-bottom and pre-wrapper decoded-center evidence."""

    repo_root = Path(repo_root).resolve()
    dataset = repo_root / "mmcv" / "datasets" / "b2d_orion_dataset.py"
    base_boxes = repo_root / "mmcv" / "core" / "bbox" / "structures" / "base_box3d.py"
    head = repo_root / "mmcv" / "models" / "dense_heads" / "orion_head.py"
    for path in (dataset, base_boxes, head):
        if not path.is_file():
            raise OrionActualTargetRunnerError("z-origin evidence file is missing")
    dataset_text = dataset.read_text(encoding="utf-8")
    base_text = base_boxes.read_text(encoding="utf-8")
    head_text = head.read_text(encoding="utf-8")
    checks = {
        "dataset_constructs_center_origin_wrapper": (
            "origin=(0.5, 0.5, 0.5)" in dataset_text
        ),
        "base_wrapper_converts_tensor_to_bottom_origin": (
            "dst = self.tensor.new_tensor((0.5, 0.5, 0))" in base_text
            and "self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)" in base_text
        ),
        "orion_targets_use_gravity_center": (
            "gt_bboxes.gravity_center" in head_text
        ),
        "repository_wrapper_subtracts_half_height_only_after_decode": (
            "bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5" in head_text
        ),
    }
    return {
        "policy_id": BOX_Z_ORIGIN_POLICY_ID,
        "passed": all(checks.values()),
        "checks": checks,
        "gt_box_z_origin": GT_BOX_Z_ORIGIN,
        "decoded_adapter_box_z_origin": DECODED_BOX_Z_ORIGIN,
        "claim": (
            "canonical gt_bboxes_3d.tensor is bottom-origin; decoded adapter "
            "boxes remain center-origin before repository box wrapping"
        ),
        "source_lineage": [
            {"path": str(dataset), "sha256": _sha256(dataset)},
            {"path": str(base_boxes), "sha256": _sha256(base_boxes)},
            {"path": str(head), "sha256": _sha256(head)},
        ],
    }


def pilot_failure_event_policy() -> FailureEventPolicyV1:
    """Return the explicit preregistered pilot policy, never a mock default."""

    return FailureEventPolicyV1(
        component_thresholds=(0.50, 0.50, 0.50, 0.50, 0.50, 0.50),
        minimum_patch_support=0.01,
        calibration_policy_id=PILOT_CALIBRATION_POLICY_ID,
    )


@dataclass(frozen=True)
class TrafficTargetV1:
    state_labels: torch.Tensor
    state_valid_affects_ego: torch.Tensor
    loader_valid: torch.Tensor
    affects_ego: torch.Tensor
    semantics_id: str = TRAFFIC_SEMANTICS_ID


@dataclass(frozen=True)
class GTEligibilityResultV1:
    """Synchronously filtered v1 GT axes and their actor-ID audit."""

    axes: Dict[str, Any]
    eligibility_mask: torch.Tensor
    selected_actor_ids: Tuple[str, ...]
    audit: Dict[str, Any]


@dataclass(frozen=True)
class BuiltBranchTargetV1:
    """Actual-target bundle plus the exact GT/support-axis filter audit."""

    bundle: ActualTargetBranchBundleV1
    eligibility_audit: Mapping[str, Any]


def derive_v1_traffic_targets(
    traffic_state: torch.Tensor,
    traffic_state_mask: torch.Tensor,
) -> TrafficTargetV1:
    """Use state column 0 and gate it by loader validity AND affects-ego.

    ORION's prediction has four independent focal-loss logits: columns 0:3
    represent light state and column 3 audits ``affects_ego``.  This function
    handles only privileged GT.  It never softmaxes four logits together and
    never mixes the predicted affects-ego logit into the v1 state error.
    """

    if not isinstance(traffic_state, torch.Tensor) or traffic_state.ndim != 2:
        raise OrionActualTargetRunnerError("traffic_state must have shape [N,2]")
    if traffic_state.shape[1] != 2 or traffic_state.dtype == torch.bool:
        raise OrionActualTargetRunnerError("traffic_state must be integer [N,2]")
    if traffic_state.is_floating_point() or traffic_state.is_complex():
        raise OrionActualTargetRunnerError("traffic_state must be integer [N,2]")
    if (
        not isinstance(traffic_state_mask, torch.Tensor)
        or traffic_state_mask.dtype != torch.bool
        or traffic_state_mask.shape != (traffic_state.shape[0],)
    ):
        raise OrionActualTargetRunnerError(
            "traffic_state_mask must be boolean [N] aligned to traffic_state"
        )
    if traffic_state.device != traffic_state_mask.device:
        raise OrionActualTargetRunnerError("traffic tensors must share a device")
    loader_valid = traffic_state_mask.clone()
    if bool(loader_valid.any()):
        valid_rows = traffic_state[loader_valid]
        if torch.any(valid_rows[:, 0] < 0) or torch.any(valid_rows[:, 0] > 2):
            raise OrionActualTargetRunnerError(
                "loader-valid traffic light state must lie in [0,2]"
            )
        if torch.any((valid_rows[:, 1] != 0) & (valid_rows[:, 1] != 1)):
            raise OrionActualTargetRunnerError(
                "loader-valid affects_ego must be binary"
            )
    affects_ego = torch.zeros_like(loader_valid)
    affects_ego[loader_valid] = traffic_state[loader_valid, 1].to(torch.bool)
    state_valid = loader_valid & affects_ego
    labels = traffic_state[:, 0].to(torch.long).clone()
    labels[~state_valid] = -1
    return TrafficTargetV1(
        state_labels=labels,
        state_valid_affects_ego=state_valid,
        loader_valid=loader_valid,
        affects_ego=affects_ego,
    )


def _actor_id_digest(actor_ids: Sequence[Any]) -> str:
    payload = json.dumps([_normalize_actor_id(value) for value in actor_ids], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_actor_id(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise OrionActualTargetRunnerError("each actor ID tensor must be scalar")
        value = value.detach().cpu().item()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    result = str(value).strip()
    if not result:
        raise OrionActualTargetRunnerError("actor IDs must be non-empty")
    return result


def filter_v1_gt_target_eligibility(
    *,
    boxes: Any,
    classes: torch.Tensor,
    gt_attr: torch.Tensor,
    traffic_state: torch.Tensor,
    traffic_state_mask: torch.Tensor,
    actor_ids: Sequence[Any],
    projected_support: torch.Tensor,
    support_actor_ids: Sequence[Any],
) -> GTEligibilityResultV1:
    """Filter every GT/support axis with one frozen safety-actor mask.

    Main v1 targets contain car/van/truck/bicycle/pedestrian, plus only
    loader-valid traffic lights that affect ego.  Traffic signs, cones,
    ``others``, and non-affecting lights are excluded.  Actor IDs are required
    on both the GT and support axes so an order mismatch cannot be hidden by
    equal tensor lengths.
    """

    if (
        not isinstance(classes, torch.Tensor)
        or classes.ndim != 1
        or classes.dtype == torch.bool
        or classes.is_floating_point()
        or classes.is_complex()
    ):
        raise OrionActualTargetRunnerError("classes must be an integer [N] tensor")
    count = int(classes.shape[0])
    if not hasattr(boxes, "__len__") or len(boxes) != count:
        raise OrionActualTargetRunnerError("boxes must have object axis length N")
    if not isinstance(gt_attr, torch.Tensor) or gt_attr.shape[0] != count:
        raise OrionActualTargetRunnerError("gt_attr must have leading object axis N")
    if (
        gt_attr.ndim != 2
        or gt_attr.shape[1] < 34
        or not gt_attr.is_floating_point()
        or not torch.isfinite(gt_attr).all()
    ):
        raise OrionActualTargetRunnerError(
            "gt_attr must be finite floating PlanningMetric [N,D>=34]"
        )
    if len(actor_ids) != count or len(support_actor_ids) != count:
        raise OrionActualTargetRunnerError(
            "GT and support actor-ID axes must both have length N"
        )
    normalized_actor_ids = tuple(_normalize_actor_id(value) for value in actor_ids)
    normalized_support_ids = tuple(
        _normalize_actor_id(value) for value in support_actor_ids
    )
    if len(set(normalized_actor_ids)) != count:
        raise OrionActualTargetRunnerError("GT actor IDs must be unique per frame")
    if normalized_actor_ids != normalized_support_ids:
        raise OrionActualTargetRunnerError(
            "projected-support actor IDs are not exactly aligned to GT actor IDs"
        )
    if (
        not isinstance(projected_support, torch.Tensor)
        or projected_support.ndim != 3
        or projected_support.shape[-1] != count
    ):
        raise OrionActualTargetRunnerError(
            "projected_support must have shape [V,P,N] aligned to actor IDs"
        )
    if projected_support.device != classes.device:
        raise OrionActualTargetRunnerError("support/classes must share a device")
    traffic = derive_v1_traffic_targets(traffic_state, traffic_state_mask)
    if traffic.state_labels.shape[0] != count or traffic.state_labels.device != classes.device:
        raise OrionActualTargetRunnerError("traffic axis must align to classes")
    if gt_attr.device != classes.device:
        raise OrionActualTargetRunnerError("gt_attr/classes must share a device")
    if not torch.equal(gt_attr[:, 27].to(torch.long), classes.to(torch.long)) or not torch.allclose(
        gt_attr[:, 27], classes.to(dtype=gt_attr.dtype)
    ):
        raise OrionActualTargetRunnerError(
            "PlanningMetric gt_attr[:,27] category must exactly equal gt_labels_3d"
        )

    safety = torch.zeros(count, dtype=torch.bool, device=classes.device)
    for class_id in SAFETY_ACTOR_CLASS_IDS:
        safety |= classes == class_id
    affecting_light = (
        (classes == TRAFFIC_LIGHT_CLASS_ID)
        & traffic.state_valid_affects_ego
    )
    eligibility = safety | affecting_light
    selected_indices = torch.nonzero(eligibility, as_tuple=False).flatten()
    selected_ids = tuple(normalized_actor_ids[index] for index in selected_indices.tolist())
    try:
        filtered_boxes = boxes[eligibility]
    except Exception as exc:
        raise OrionActualTargetRunnerError(
            "boxes do not support the shared boolean eligibility mask"
        ) from exc
    axes = {
        "boxes": filtered_boxes,
        "classes": classes[eligibility],
        "gt_attr": gt_attr[eligibility],
        "traffic_state": traffic_state[eligibility],
        "traffic_state_mask": traffic_state_mask[eligibility],
        "traffic_state_labels": traffic.state_labels[eligibility],
        "traffic_state_valid": traffic.state_valid_affects_ego[eligibility],
        "projected_support": projected_support[..., eligibility],
        "actor_ids": selected_ids,
        "support_actor_ids": selected_ids,
    }
    post_count = len(selected_ids)
    if any(
        (
            value.shape[-1] if key == "projected_support" else value.shape[0]
        )
        != post_count
        for key, value in axes.items()
        if isinstance(value, torch.Tensor)
    ):
        raise OrionActualTargetRunnerError("eligibility filtering lost axis alignment")
    return GTEligibilityResultV1(
        axes=axes,
        eligibility_mask=eligibility,
        selected_actor_ids=selected_ids,
        audit={
            "policy": "safety-actors-plus-affecting-traffic-light/v1",
            "allowed_safety_actor_class_ids": list(SAFETY_ACTOR_CLASS_IDS),
            "conditional_traffic_light_class_id": TRAFFIC_LIGHT_CLASS_ID,
            "pre_filter_count": count,
            "post_filter_count": post_count,
            "excluded_count": count - post_count,
            "safety_actor_count": int(safety.sum().item()),
            "affecting_traffic_light_count": int(affecting_light.sum().item()),
            "raw_loader_valid_light_count": int(traffic.loader_valid.sum().item()),
            "affects_ego_valid_count": int(
                traffic.state_valid_affects_ego.sum().item()
            ),
            "selected_indices": selected_indices.detach().cpu().tolist(),
            "pre_filter_actor_ids_sha256": _actor_id_digest(normalized_actor_ids),
            "post_filter_actor_ids_sha256": _actor_id_digest(selected_ids),
            "gt_and_support_actor_ids_exactly_equal": True,
            "all_axes_filtered_by_one_boolean_mask": True,
            "excluded_classes_never_become_negative_targets": True,
            "planningmetric_category_axis_matches_classes": True,
            "gt_rasterizer_attr_shape": [1, post_count, int(gt_attr.shape[1])],
            "gt_rasterizer_attr_contract": (
                "pass eligible gt_attr.unsqueeze(0) to rasterize_planningmetric_gt_v1"
            ),
        },
    )


def _unwrap_data_container(value: Any) -> Any:
    """Unwrap MMCV DataContainer-like objects without importing MMCV."""

    seen = set()
    while hasattr(value, "data") and not isinstance(value, torch.Tensor):
        identifier = id(value)
        if identifier in seen:
            raise OrionActualTargetRunnerError("cyclic DataContainer wrapper")
        seen.add(identifier)
        value = value.data
    return value


def _unwrap_singleton(value: Any, name: str) -> Any:
    value = _unwrap_data_container(value)
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = _unwrap_data_container(value[0])
    if isinstance(value, (list, tuple)) and name != "img_metas":
        raise OrionActualTargetRunnerError(
            "%s retains ambiguous non-singleton test augmentation nesting" % name
        )
    return value


@dataclass(frozen=True)
class CanonicalORIONBatchV1:
    data: Dict[str, Any]
    img_metas: Tuple[Mapping[str, Any], ...]
    traffic: TrafficTargetV1
    frame_idx: int
    scene_token: str
    camera_order: Tuple[str, ...]
    processed_image_hw: Tuple[int, int]


def canonicalize_orion_test_batch(batch: Mapping[str, Any]) -> CanonicalORIONBatchV1:
    """Convert a batch-size-one MMCV test batch to the head's direct inputs."""

    if not isinstance(batch, Mapping):
        raise OrionActualTargetRunnerError("ORION batch must be a mapping")
    required = (
        "img",
        "img_metas",
        "traffic_state",
        "traffic_state_mask",
        "lidar2img",
        "cam_intrinsic",
        "timestamp",
        "ego_pose",
        "ego_pose_inv",
        "command",
        "can_bus",
        "gt_bboxes_3d",
        "gt_labels_3d",
        "gt_attr_labels",
        "gt_actor_ids",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise OrionActualTargetRunnerError(
            "actual-target batch is missing: " + ", ".join(missing)
        )
    data = {key: _unwrap_singleton(value, key) for key, value in batch.items()}

    metas = data.pop("img_metas")
    if isinstance(metas, Mapping):
        metas = [metas]
    if not isinstance(metas, (list, tuple)) or len(metas) != 1:
        raise OrionActualTargetRunnerError("img_metas must resolve to batch size one")
    meta = _unwrap_data_container(metas[0])
    if not isinstance(meta, Mapping):
        raise OrionActualTargetRunnerError("img_metas[0] must be a mapping")

    img = data["img"]
    if not isinstance(img, torch.Tensor):
        raise OrionActualTargetRunnerError("img must resolve to a tensor")
    if img.ndim == 4:
        if img.shape[0] != len(CANONICAL_CAMERAS):
            raise OrionActualTargetRunnerError("img must contain exactly six cameras")
        img = img.unsqueeze(0)
    if img.ndim != 5 or img.shape[:2] != (1, len(CANONICAL_CAMERAS)):
        raise OrionActualTargetRunnerError("img must have shape [1,6,C,H,W]")
    data["img"] = img

    lidar2img = data["lidar2img"]
    if not isinstance(lidar2img, torch.Tensor):
        lidar2img = torch.as_tensor(lidar2img)
    if lidar2img.ndim == 3:
        lidar2img = lidar2img.unsqueeze(0)
    if lidar2img.shape != (1, 6, 4, 4):
        raise OrionActualTargetRunnerError(
            "post-augmentation lidar2img must have shape [1,6,4,4]"
        )
    data["lidar2img"] = lidar2img

    explicit_order = meta.get("camera_order")
    if explicit_order is not None:
        camera_order = tuple(explicit_order)
    else:
        filenames = meta.get("filename")
        if not isinstance(filenames, (list, tuple)) or len(filenames) != 6:
            raise OrionActualTargetRunnerError(
                "camera order requires explicit camera_order or six filenames"
            )
        inferred = []
        directory_to_camera = {
            directory: camera for camera, directory in CAMERA_DIRECTORY_BY_NAME.items()
        }
        for filename in filenames:
            parts = Path(str(filename)).parts
            matches = [directory_to_camera[part] for part in parts if part in directory_to_camera]
            if len(matches) != 1:
                raise OrionActualTargetRunnerError(
                    "cannot infer exactly one camera name from filename"
                )
            inferred.append(matches[0])
        camera_order = tuple(inferred)
    if camera_order != CANONICAL_CAMERAS:
        raise OrionActualTargetRunnerError("camera order is not canonical ORION order")
    pad_shape = meta.get("pad_shape")
    if not isinstance(pad_shape, (list, tuple)) or len(pad_shape) != 6:
        raise OrionActualTargetRunnerError("img_metas.pad_shape must contain six views")
    first_shape = tuple(pad_shape[0])
    if len(first_shape) < 2 or any(tuple(shape)[:2] != first_shape[:2] for shape in pad_shape):
        raise OrionActualTargetRunnerError("all processed views must share image shape")

    traffic_state = data["traffic_state"]
    traffic_mask = data["traffic_state_mask"]
    if not isinstance(traffic_state, torch.Tensor):
        traffic_state = torch.as_tensor(traffic_state)
    if not isinstance(traffic_mask, torch.Tensor):
        traffic_mask = torch.as_tensor(traffic_mask)
    traffic_mask = traffic_mask.to(torch.bool)
    traffic = derive_v1_traffic_targets(traffic_state, traffic_mask)
    data["traffic_state"] = traffic_state
    data["traffic_state_mask"] = traffic_mask

    gt_classes = data["gt_labels_3d"]
    if not isinstance(gt_classes, torch.Tensor):
        gt_classes = torch.as_tensor(gt_classes)
    if gt_classes.ndim != 1 or gt_classes.dtype == torch.bool or gt_classes.is_floating_point():
        raise OrionActualTargetRunnerError("gt_labels_3d must be integer [N]")
    data["gt_labels_3d"] = gt_classes
    gt_attr = data["gt_attr_labels"]
    if not isinstance(gt_attr, torch.Tensor):
        gt_attr = torch.as_tensor(gt_attr)
    if gt_attr.shape[0] != gt_classes.shape[0]:
        raise OrionActualTargetRunnerError("gt_attr_labels must align to GT classes")
    data["gt_attr_labels"] = gt_attr
    if len(data["gt_bboxes_3d"]) != gt_classes.shape[0]:
        raise OrionActualTargetRunnerError("gt_bboxes_3d must align to GT classes")
    if traffic_state.shape[0] != gt_classes.shape[0]:
        raise OrionActualTargetRunnerError("traffic tensors must align to GT classes")
    actor_ids = data["gt_actor_ids"]
    if not isinstance(actor_ids, torch.Tensor):
        actor_ids = torch.as_tensor(actor_ids)
    if (
        actor_ids.ndim != 1
        or actor_ids.shape[0] != gt_classes.shape[0]
        or actor_ids.dtype == torch.bool
        or actor_ids.is_floating_point()
        or actor_ids.is_complex()
    ):
        raise OrionActualTargetRunnerError(
            "gt_actor_ids must be integer [N] aligned after every GT filter"
        )
    if actor_ids.unique().numel() != actor_ids.numel():
        raise OrionActualTargetRunnerError("gt_actor_ids must be unique per frame")
    data["gt_actor_ids"] = actor_ids

    frame_idx = meta.get("frame_idx")
    if isinstance(frame_idx, bool) or not isinstance(frame_idx, int) or frame_idx < 0:
        raise OrionActualTargetRunnerError("img_metas.frame_idx must be non-negative int")
    scene_token = _nonempty(meta.get("scene_token"), "img_metas.scene_token")
    return CanonicalORIONBatchV1(
        data=data,
        img_metas=(meta,),
        traffic=traffic,
        frame_idx=frame_idx,
        scene_token=scene_token,
        camera_order=camera_order,
        processed_image_hw=(int(first_shape[0]), int(first_shape[1])),
    )


def _base_model(model: Any) -> Any:
    return getattr(model, "module", model)


def reset_and_assert_orion_memory(model: Any) -> Dict[str, Any]:
    """Reset every participating temporal head and assert known fields are empty."""

    core = _base_model(model)
    head = getattr(core, "pts_bbox_head", None)
    if head is None or not callable(getattr(head, "reset_memory", None)):
        raise OrionActualTargetRunnerError("model.pts_bbox_head.reset_memory is required")
    head.reset_memory()
    reset_heads = ["pts_bbox_head"]
    map_head = getattr(core, "map_head", None)
    map_participates = bool(getattr(core, "with_map_head", False))
    if map_participates:
        if map_head is None or not callable(getattr(map_head, "reset_memory", None)):
            raise OrionActualTargetRunnerError(
                "participating model.map_head.reset_memory is required"
            )
        map_head.reset_memory()
        reset_heads.append("map_head")
    nonempty_fields: List[str] = []
    audited_fields: List[str] = []
    for head_name, temporal_head in (("pts_bbox_head", head), ("map_head", map_head)):
        if temporal_head is None or (head_name == "map_head" and not map_participates):
            continue
        for field in MEMORY_FIELDS:
            if hasattr(temporal_head, field):
                audited_fields.append("%s.%s" % (head_name, field))
                if getattr(temporal_head, field) is not None:
                    nonempty_fields.append("%s.%s" % (head_name, field))
    if nonempty_fields:
        raise OrionActualTargetRunnerError(
            "memory reset left non-empty fields: " + ", ".join(nonempty_fields)
        )
    if not audited_fields:
        raise OrionActualTargetRunnerError("no known ORION memory fields were audited")
    return {
        "reset_heads": reset_heads,
        "audited_fields": audited_fields,
        "all_audited_fields_are_none": True,
    }


@dataclass(frozen=True)
class PerceptionFrameV1:
    canonical_batch: CanonicalORIONBatchV1
    raw_head_outputs: Mapping[str, torch.Tensor]
    adapted: AdaptedORIONBatchV1
    image_features: torch.Tensor


class FrozenORIONPerceptionHookV1:
    """Direct frozen perception path that excludes LLM/VAE/planning decode."""

    def __init__(
        self,
        decode_config: ORIONDecodeAdapterConfigV1,
        occupancy_rasterizer: Callable[[Any], torch.Tensor],
    ) -> None:
        if not callable(occupancy_rasterizer):
            raise OrionActualTargetRunnerError("real occupancy rasterizer is required")
        self.decode_config = decode_config
        self.occupancy_rasterizer = occupancy_rasterizer

    def __call__(self, model: Any, batch: Mapping[str, Any]) -> PerceptionFrameV1:
        core = _base_model(model)
        for method in ("extract_feat", "prepare_location", "position_embeding"):
            if not callable(getattr(core, method, None)):
                raise OrionActualTargetRunnerError(
                    "ORION perception hook requires model.%s" % method
                )
        head = getattr(core, "pts_bbox_head", None)
        if not callable(head):
            raise OrionActualTargetRunnerError("model.pts_bbox_head must be callable")
        core.eval()
        canonical = canonicalize_orion_test_batch(batch)
        data = dict(canonical.data)
        with torch.inference_mode():
            image_features = core.extract_feat(data["img"])
            if not isinstance(image_features, torch.Tensor) or image_features.ndim != 5:
                raise OrionActualTargetRunnerError(
                    "extract_feat must return ORION tensor [B,V,C,H,W]"
                )
            if image_features.shape[:2] != (1, 6):
                raise OrionActualTargetRunnerError(
                    "ORION image features must preserve batch-one six-view order"
                )
            data["img_feats"] = image_features
            location = core.prepare_location(list(canonical.img_metas), **data)
            position = core.position_embeding(
                data, location, list(canonical.img_metas)
            )
            head_result = head(list(canonical.img_metas), position, **data)
        if not isinstance(head_result, (tuple, list)) or len(head_result) < 1:
            raise OrionActualTargetRunnerError(
                "pts_bbox_head must return a tuple whose first item is raw outs"
            )
        raw = head_result[0]
        if not isinstance(raw, Mapping):
            raise OrionActualTargetRunnerError("raw ORION head outs must be a mapping")
        missing = [key for key in HEAD_OUTPUT_KEYS if key not in raw]
        if missing:
            raise OrionActualTargetRunnerError(
                "raw ORION head outs missing: " + ", ".join(missing)
            )
        adapted = adapt_orion_head_outputs_v1(
            raw,
            config=self.decode_config,
            occupancy_rasterizer=self.occupancy_rasterizer,
        )
        if len(adapted.frames) != 1 or len(adapted.audits) != 1:
            raise OrionActualTargetRunnerError(
                "actual-target runner requires exactly one decoded frame"
            )
        if adapted.frames[0].traffic_state_logits.shape[-1] != 4:
            raise OrionActualTargetRunnerError(
                "ORION v1 must retain four traffic logits for audit"
            )
        return PerceptionFrameV1(
            canonical_batch=canonical,
            raw_head_outputs=raw,
            adapted=adapted,
            image_features=image_features,
        )


@dataclass(frozen=True)
class ActualTargetRuntimeHooksV1:
    """Caller-owned integrations that the repository cannot safely guess."""

    perception_hook: FrozenORIONPerceptionHookV1
    branch_target_builder: Callable[[Any, CanonicalORIONBatchV1, Mapping[str, Any]], BuiltBranchTargetV1]
    patch_feature_extractor: Callable[[Any, PerceptionFrameV1, Mapping[str, Any]], torch.Tensor]
    corruption_transform: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
    record_sink: Callable[[Any, PairedActualTargetBundleV1, Mapping[str, Any]], None]
    decoder_parity_check: Callable[[], bool]
    selected_motion_mode_check: Callable[[], bool]
    projection_overlay_check: Callable[[], bool]
    gt_axis_alignment_check: Callable[[], bool]
    gt_occupancy_rasterizer: Callable[..., Any]
    pairwise_bev_iou: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    project_visible_support: Callable[..., Any]
    occupancy_rasterizer_id: str
    support_projector_id: str
    gt_occupancy_rasterizer_id: str
    pairwise_bev_iou_policy_id: str
    patch_feature_hook_id: str
    gt_box_z_origin: str
    decoded_box_z_origin: str
    integration_kind: str = "unconnected"
    schema_version: str = RUNNER_INTEGRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUNNER_INTEGRATION_SCHEMA_VERSION:
            raise OrionActualTargetRunnerError("unsupported runtime hook schema")
        for name in (
            "occupancy_rasterizer_id",
            "support_projector_id",
            "gt_occupancy_rasterizer_id",
            "pairwise_bev_iou_policy_id",
            "patch_feature_hook_id",
        ):
            _nonempty(getattr(self, name), name)
        if self.integration_kind not in ("real_orion", "cpu_mock", "unconnected"):
            raise OrionActualTargetRunnerError("unknown integration_kind")
        if (
            self.gt_box_z_origin != GT_BOX_Z_ORIGIN
            or self.decoded_box_z_origin != DECODED_BOX_Z_ORIGIN
        ):
            raise OrionActualTargetRunnerError(
                "B2D GT requires bottom-origin while decoded ORION boxes require center-origin"
            )


def extract_evavit_patch_features_v1(
    model: Any,
    frame: PerceptionFrameV1,
    context: Mapping[str, Any],
) -> torch.Tensor:
    """Convert real position-level EVAViT features to ``[6,1600,C]``."""

    del model, context
    features = frame.image_features
    if not isinstance(features, torch.Tensor) or features.ndim != 5:
        raise OrionActualTargetRunnerError("EVAViT features must be [B,V,C,H,W]")
    if features.shape[0] != 1 or features.shape[1] != len(ORION_CAMERA_ORDER):
        raise OrionActualTargetRunnerError(
            "EVAViT feature batch must be batch-one in canonical six-view order"
        )
    if frame.canonical_batch.camera_order != ORION_CAMERA_ORDER:
        raise OrionActualTargetRunnerError("EVAViT camera order is not canonical")
    if tuple(features.shape[-2:]) != (40, 40):
        raise OrionActualTargetRunnerError(
            "production patch feature hook requires exact 40x40 feature grid"
        )
    return features[0].permute(0, 2, 3, 1).reshape(
        len(ORION_CAMERA_ORDER), 1600, features.shape[2]
    )


def _production_visible_support(
    boxes_lidar: torch.Tensor,
    post_augmentation_lidar2img: torch.Tensor,
    processed_image_hw: Sequence[Sequence[int]],
    *,
    box_source: str,
    image_transform_id: str,
    **kwargs: Any,
) -> Any:
    """Production projection with distinct audited GT/predicted z origins."""

    if box_source == "privileged_gt":
        box_z_origin = GT_BOX_Z_ORIGIN
    elif box_source == "decoded_orion":
        box_z_origin = DECODED_BOX_Z_ORIGIN
    else:
        raise OrionActualTargetRunnerError(
            "box_source must be 'privileged_gt' or 'decoded_orion'"
        )
    return project_boxes_to_visible_patch_support(
        boxes_lidar.detach().cpu(),
        post_augmentation_lidar2img.detach().cpu(),
        processed_image_hw,
        camera_order=ORION_CAMERA_ORDER,
        matrix_camera_order=ORION_CAMERA_ORDER,
        image_shape_camera_order=ORION_CAMERA_ORDER,
        image_transform_id=image_transform_id,
        box_z_origin=box_z_origin,
        patch_hw=(40, 40),
        expected_camera_order=ORION_CAMERA_ORDER,
        expected_patch_hw=(40, 40),
        **kwargs,
    )


def build_production_runtime_hooks_v1(
    *,
    decode_config: ORIONDecodeAdapterConfigV1,
    branch_target_builder: Callable[[Any, CanonicalORIONBatchV1, Mapping[str, Any]], BuiltBranchTargetV1],
    corruption_transform: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
    record_sink: Callable[[Any, PairedActualTargetBundleV1, Mapping[str, Any]], None],
    decoder_parity_check: Callable[[], bool],
    selected_motion_mode_check: Callable[[], bool],
    projection_overlay_check: Callable[[], bool],
    gt_axis_alignment_check: Callable[[], bool],
    gt_box_z_origin: str,
    decoded_box_z_origin: str,
) -> ActualTargetRuntimeHooksV1:
    """Build production hooks with repository BEV/projection primitives fixed."""

    if (
        gt_box_z_origin != GT_BOX_Z_ORIGIN
        or decoded_box_z_origin != DECODED_BOX_Z_ORIGIN
    ):
        raise OrionActualTargetRunnerError(
            "explicit production origins must be GT='bottom', decoded='center'"
        )
    if decode_config.occupancy_rasterizer_id != SELECTED_MODE_RASTERIZER_ID:
        raise OrionActualTargetRunnerError(
            "decode_config must use the production selected-mode rasterizer ID"
        )
    return ActualTargetRuntimeHooksV1(
        perception_hook=FrozenORIONPerceptionHookV1(
            decode_config, selected_mode_occupancy_callback_v1
        ),
        branch_target_builder=branch_target_builder,
        patch_feature_extractor=extract_evavit_patch_features_v1,
        corruption_transform=corruption_transform,
        record_sink=record_sink,
        decoder_parity_check=decoder_parity_check,
        selected_motion_mode_check=selected_motion_mode_check,
        projection_overlay_check=projection_overlay_check,
        gt_axis_alignment_check=gt_axis_alignment_check,
        gt_occupancy_rasterizer=rasterize_planningmetric_gt_v1,
        pairwise_bev_iou=pairwise_bev_iou_v1,
        project_visible_support=_production_visible_support,
        occupancy_rasterizer_id=SELECTED_MODE_RASTERIZER_ID,
        support_projector_id=VISIBLE_SUPPORT_PROJECTION_VERSION,
        gt_occupancy_rasterizer_id=GT_RASTERIZER_ID,
        pairwise_bev_iou_policy_id=PAIRWISE_BEV_IOU_POLICY_ID,
        patch_feature_hook_id=PATCH_FEATURE_HOOK_ID,
        gt_box_z_origin=gt_box_z_origin,
        decoded_box_z_origin=decoded_box_z_origin,
        integration_kind="real_orion",
    )


def runtime_hook_readiness(hooks: Optional[ActualTargetRuntimeHooksV1]) -> Dict[str, Any]:
    if hooks is None:
        connected = {name: False for name in REQUIRED_REAL_HOOKS}
        identifiers: Dict[str, Optional[str]] = {
            "occupancy_rasterizer_id": None,
            "support_projector_id": None,
            "gt_occupancy_rasterizer_id": None,
            "pairwise_bev_iou_policy_id": None,
            "patch_feature_hook_id": None,
            "gt_box_z_origin": None,
            "decoded_box_z_origin": None,
        }
        audit_results: Dict[str, bool] = {}
        external_hook_ids: Dict[str, Optional[str]] = {
            name: None for name in EXTERNAL_PRODUCTION_HOOKS
        }
        kind = "unconnected"
    else:
        connected = {
            "occupancy_rasterizer": callable(hooks.perception_hook.occupancy_rasterizer),
            "branch_target_builder": callable(hooks.branch_target_builder),
            "patch_feature_extractor": callable(hooks.patch_feature_extractor),
            "corruption_transform": callable(hooks.corruption_transform),
            "record_sink": callable(hooks.record_sink),
            "decoder_parity_check": callable(hooks.decoder_parity_check),
            "selected_motion_mode_check": callable(hooks.selected_motion_mode_check),
            "projection_overlay_check": callable(hooks.projection_overlay_check),
            "gt_axis_alignment_check": callable(hooks.gt_axis_alignment_check),
            "gt_occupancy_rasterizer": callable(hooks.gt_occupancy_rasterizer),
            "pairwise_bev_iou": callable(hooks.pairwise_bev_iou),
            "project_visible_support": callable(hooks.project_visible_support),
        }
        identifiers = {
            name: getattr(hooks, name)
            for name in (
                "occupancy_rasterizer_id",
                "support_projector_id",
                "gt_occupancy_rasterizer_id",
                "pairwise_bev_iou_policy_id",
                "patch_feature_hook_id",
            )
        }
        identifiers["gt_box_z_origin"] = hooks.gt_box_z_origin
        identifiers["decoded_box_z_origin"] = hooks.decoded_box_z_origin
        audit_results = {}
        external_hook_ids = {
            name: getattr(getattr(hooks, name), "production_hook_id", None)
            for name in EXTERNAL_PRODUCTION_HOOKS
        }
        for name in (
            "decoder_parity_check",
            "selected_motion_mode_check",
            "projection_overlay_check",
            "gt_axis_alignment_check",
        ):
            try:
                audit_results[name] = bool(getattr(hooks, name)())
            except Exception:
                audit_results[name] = False
        kind = hooks.integration_kind
        production_primitives_frozen = (
            hooks.perception_hook.occupancy_rasterizer
            is selected_mode_occupancy_callback_v1
            and hooks.gt_occupancy_rasterizer is rasterize_planningmetric_gt_v1
            and hooks.pairwise_bev_iou is pairwise_bev_iou_v1
            and hooks.project_visible_support is _production_visible_support
            and hooks.patch_feature_extractor is extract_evavit_patch_features_v1
            and hooks.occupancy_rasterizer_id == SELECTED_MODE_RASTERIZER_ID
            and hooks.gt_occupancy_rasterizer_id == GT_RASTERIZER_ID
            and hooks.pairwise_bev_iou_policy_id == PAIRWISE_BEV_IOU_POLICY_ID
            and hooks.support_projector_id == VISIBLE_SUPPORT_PROJECTION_VERSION
            and hooks.patch_feature_hook_id == PATCH_FEATURE_HOOK_ID
            and hooks.gt_box_z_origin == GT_BOX_Z_ORIGIN
            and hooks.decoded_box_z_origin == DECODED_BOX_Z_ORIGIN
        )
        audit_results["production_primitives_frozen"] = bool(
            production_primitives_frozen
        )
        builder_type = (
            hooks.branch_target_builder.__class__.__module__,
            hooks.branch_target_builder.__class__.__name__,
        )
        audit_results["concrete_production_branch_builder"] = (
            builder_type == PRODUCTION_BRANCH_BUILDER_TYPE
            and getattr(
                hooks.branch_target_builder, "production_hook_id", None
            )
            == "orion.actual-target-production-branch-builder/v1"
        )
        audit_results["external_production_hook_ids_present"] = all(
            isinstance(value, str) and bool(value.strip())
            for value in external_hook_ids.values()
        )
    suspicious = [
        "%s=%s" % (name, value)
        for name, value in identifiers.items()
        if isinstance(value, str)
        and any(word in value.lower() for word in ("mock", "placeholder", "todo"))
    ]
    ready = (
        kind == "real_orion"
        and all(connected.values())
        and all(identifiers.values())
        and all(audit_results.values())
        and not suspicious
    )
    return {
        "schema_version": RUNNER_INTEGRATION_SCHEMA_VERSION,
        "integration_kind": kind,
        "connected": connected,
        "identifiers": identifiers,
        "audit_results": audit_results,
        "external_production_hook_ids": external_hook_ids,
        "suspicious_identifier_values": suspicious,
        "real_execution_ready": bool(ready),
    }


def build_runner_preflight(
    plan: Mapping[str, Any],
    *,
    config_lineage: Mapping[str, Any],
    pipeline_audit: Mapping[str, Any],
    formatter_audit: Mapping[str, Any],
    box_z_origin_audit: Optional[Mapping[str, Any]] = None,
    hooks: Optional[ActualTargetRuntimeHooksV1] = None,
) -> Dict[str, Any]:
    if plan.get("schema_version") != REPLAY_PLAN_SCHEMA_VERSION:
        raise OrionActualTargetRunnerError("unsupported replay plan")
    route = plan.get("route")
    execution = plan.get("execution")
    source = plan.get("source_verification")
    if not all(isinstance(value, Mapping) for value in (route, execution, source)):
        raise OrionActualTargetRunnerError("replay plan sections are incomplete")
    assert isinstance(route, Mapping) and isinstance(execution, Mapping)
    assert isinstance(source, Mapping)
    exact_route = (
        route.get("canonical_route_key") == "Town04/Route214"
        and route.get("smoke_prefix_frame_range_inclusive") == [0, 63]
        and route.get("smoke_prefix_frame_count") == 64
        and execution.get("expected_frames_each_branch") == list(range(64))
        and execution.get("measurement_frame_count") == 43
        and len(execution.get("measurement_frames", [])) == 43
        and execution.get("persist_only_measurement_frames") is True
        and execution.get("branch_order") == ["clean", "observed"]
        and execution.get("expected_forward_count_total") == 128
    )
    source_ready = all(
        source.get(name) is True
        for name in (
            "source_info_sha256_verified",
            "exact_frame_zero_prefix_verified",
            "canonical_six_camera_metadata_verified",
            "camera_files_exist_for_all_frames",
            "annotation_files_exist_for_all_frames",
        )
    )
    hook_audit = runtime_hook_readiness(hooks)
    origin_audit = dict(box_z_origin_audit or {})
    exporter_dual_ids_ready = (
        BRANCH_BUNDLE_SCHEMA_VERSION.endswith("/v2")
        and hooks is not None
        and hooks.occupancy_rasterizer_id == SELECTED_MODE_RASTERIZER_ID
        and hooks.gt_occupancy_rasterizer_id == GT_RASTERIZER_ID
        and hooks.occupancy_rasterizer_id != hooks.gt_occupancy_rasterizer_id
    )
    checks = {
        "exact_route214_prefix0_63_contract": bool(exact_route),
        "source_and_files_verified": bool(source_ready),
        "dedicated_pipeline_mutation_verified": pipeline_audit.get("passed") is True,
        "deployment_config_not_mutated": (
            pipeline_audit.get("source_deployment_config_mutated") is False
        ),
        "formatter_fix_verified": formatter_audit.get("passed") is True,
        "distinct_gt_pred_box_z_origins_verified": origin_audit.get("passed") is True,
        "real_runtime_hooks_connected": hook_audit["real_execution_ready"] is True,
        "exporter_supports_distinct_pred_and_gt_occupancy_ids": bool(
            exporter_dual_ids_ready
        ),
    }
    execution_ready = all(checks.values())
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": RUNNER_PREFLIGHT_SCHEMA_VERSION,
        "plan_id": plan.get("plan_id"),
        "config_lineage": dict(config_lineage),
        "pipeline_audit": dict(pipeline_audit),
        "formatter_audit": dict(formatter_audit),
        "box_z_origin_audit": origin_audit,
        "runtime_hook_audit": hook_audit,
        "exporter_schema_audit": {
            "branch_bundle_schema_version": BRANCH_BUNDLE_SCHEMA_VERSION,
            "requires_v2": True,
            "predicted_rasterizer_id": (
                hooks.occupancy_rasterizer_id if hooks is not None else None
            ),
            "gt_rasterizer_id": (
                hooks.gt_occupancy_rasterizer_id if hooks is not None else None
            ),
            "distinct_ids_supported_and_configured": bool(exporter_dual_ids_ready),
        },
        "checks": checks,
        "execution_ready": execution_ready,
        "blockers": blockers,
        "job_submitted": False,
        "gpu_used": False,
        "carla_used": False,
        "training_performed": False,
        "failure_event_policy": {
            "calibration_policy_id": PILOT_CALIBRATION_POLICY_ID,
            "thresholds": [0.50] * 6,
            "minimum_patch_support": 0.01,
            "claim": "preregistered pilot thresholds; not calibration-optimal",
        },
        "object_matching_policy": {
            "policy_id": PILOT_MATCH_POLICY_ID,
            "minimum_prediction_score": PILOT_MINIMUM_PREDICTION_SCORE,
            "maximum_center_distance_m": PILOT_MAXIMUM_CENTER_DISTANCE_M,
            "sensitivity_maximum_center_distance_m": (
                PILOT_SENSITIVITY_CENTER_DISTANCE_M
            ),
            "claim": "preregistered pilot heuristic; not calibration-optimal",
        },
        "traffic_semantics": {
            "semantics_id": TRAFFIC_SEMANTICS_ID,
            "gt_state_label": "traffic_state[:,0]",
            "gt_validity": "traffic_state_mask & bool(traffic_state[:,1])",
            "predicted_state_probability": "sigmoid(all_traffic_states[...,:3])",
            "predicted_affects_ego": (
                "sigmoid(all_traffic_states[...,3]) retained for audit only"
            ),
            "forbidden": "softmax over all four logits",
            "route214_prefix_raw_annotation_audit": {
                "traffic_light_record_count": 1397,
                "state_affects_ego_counts": {
                    "state_0_affects_false": 1036,
                    "state_2_affects_false": 361,
                },
                "affects_ego_true_count": 0,
                "consequence": (
                    "traffic component is invalid for this prefix; never a negative label"
                ),
            },
        },
        "claim_boundary": {
            "preflight_is_g1": False,
            "supports_training": False,
            "supports_safety_claim": False,
        },
    }


def assert_real_execution_ready(preflight: Mapping[str, Any]) -> None:
    if preflight.get("schema_version") != RUNNER_PREFLIGHT_SCHEMA_VERSION:
        raise OrionActualTargetRunnerError("invalid runner preflight schema")
    if preflight.get("execution_ready") is not True:
        blockers = preflight.get("blockers", [])
        raise OrionActualTargetRunnerError(
            "real execution is fail-closed; unresolved blockers: %s"
            % ", ".join(str(item) for item in blockers)
        )


def _frame_runtime_audit(
    frame: PerceptionFrameV1,
    *,
    branch: str,
    expected_frame_idx: int,
    persisted: bool,
    target_adapter_ready: bool,
    eligibility_audit: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    canonical = frame.canonical_batch
    decoded = frame.adapted.frames[0]
    raw_shapes = {
        key: list(frame.raw_head_outputs[key].shape) for key in HEAD_OUTPUT_KEYS
    }
    classes = canonical.data["gt_labels_3d"]
    safety = torch.zeros_like(classes, dtype=torch.bool)
    for class_id in SAFETY_ACTOR_CLASS_IDS:
        safety |= classes == class_id
    affecting_light = (
        (classes == TRAFFIC_LIGHT_CLASS_ID)
        & canonical.traffic.state_valid_affects_ego
    )
    eligible = safety | affecting_light
    result = {
        "schema_version": RUNNER_FRAME_AUDIT_SCHEMA_VERSION,
        "frame_idx": canonical.frame_idx,
        "expected_frame_idx": expected_frame_idx,
        "scene_token": canonical.scene_token,
        "branch": branch,
        "model_forward_completed": True,
        "perception_only_path": (
            "extract_feat->prepare_location->position_embeding->pts_bbox_head"
        ),
        "llm_vae_planning_decode_skipped": True,
        "six_camera_images_loaded": canonical.data["img"].shape[1] == 6,
        "camera_order": list(canonical.camera_order),
        "traffic_state_shape_n_by_2": canonical.data["traffic_state"].ndim == 2
        and canonical.data["traffic_state"].shape[1] == 2,
        "traffic_state_mask_matches_objects": (
            canonical.data["traffic_state_mask"].shape[0]
            == canonical.data["traffic_state"].shape[0]
        ),
        "traffic_state_label_column": 0,
        "traffic_state_validity": "loader_mask_and_affects_ego_column_1",
        "raw_loader_valid_light_count": int(
            canonical.traffic.loader_valid.sum().item()
        ),
        "affects_ego_gt_valid_count": int(
            canonical.traffic.state_valid_affects_ego.sum().item()
        ),
        "gt_target_pre_filter_count": int(classes.numel()),
        "gt_target_post_filter_count": int(eligible.sum().item()),
        "gt_target_excluded_count": int((~eligible).sum().item()),
        "gt_target_safety_actor_count": int(safety.sum().item()),
        "gt_target_affecting_traffic_light_count": int(
            affecting_light.sum().item()
        ),
        "predicted_traffic_state_transform": "sigmoid_first_three_logits",
        "predicted_affects_ego_logit_policy": "retained_for_audit_only",
        "four_logit_softmax_used": False,
        "post_augmentation_lidar2img_count_is_6": (
            canonical.data["lidar2img"].shape[1] == 6
        ),
        "processed_image_shape_present": all(
            value > 0 for value in canonical.processed_image_hw
        ),
        "processed_image_hw": list(canonical.processed_image_hw),
        "raw_head_output_shapes": raw_shapes,
        "decoded_output_adapter_ready": True,
        "actual_target_adapter_ready": bool(target_adapter_ready),
        "selected_motion_mode_policy": decoded.motion_mode_policy,
        "traffic_logits_shape": list(decoded.traffic_state_logits.shape),
        "affects_ego_prediction_logit_retained": (
            decoded.traffic_state_logits.shape[-1] == 4
        ),
        "persisted": bool(persisted),
    }
    if eligibility_audit is None:
        result["gt_support_actor_axis_audited"] = False
        result["gt_support_actor_axis_audit_reason"] = (
            "not constructed on non-measurement warm-up frame"
        )
    else:
        required = (
            "pre_filter_count",
            "post_filter_count",
            "gt_and_support_actor_ids_exactly_equal",
            "all_axes_filtered_by_one_boolean_mask",
            "planningmetric_category_axis_matches_classes",
        )
        missing = [key for key in required if key not in eligibility_audit]
        if missing:
            raise OrionActualTargetRunnerError(
                "target builder eligibility audit missing: " + ", ".join(missing)
            )
        if (
            eligibility_audit["pre_filter_count"] != result["gt_target_pre_filter_count"]
            or eligibility_audit["post_filter_count"]
            != result["gt_target_post_filter_count"]
            or eligibility_audit["gt_and_support_actor_ids_exactly_equal"] is not True
            or eligibility_audit["all_axes_filtered_by_one_boolean_mask"] is not True
            or eligibility_audit["planningmetric_category_axis_matches_classes"] is not True
        ):
            raise OrionActualTargetRunnerError(
                "target builder GT/support eligibility audit disagrees with batch"
            )
        result["gt_support_actor_axis_audited"] = True
        result["gt_support_actor_axis_audit"] = dict(eligibility_audit)
    return result


def _is_actual_orion_model(model: Any) -> bool:
    core = _base_model(model)
    return core.__class__.__name__ == "Orion" and core.__class__.__module__.endswith(
        "models.detectors.orion"
    )


def run_chronological_actual_target_replay(
    *,
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    model: Any,
    branch_batches: Callable[[str], Iterable[Mapping[str, Any]]],
    hooks: ActualTargetRuntimeHooksV1,
) -> Dict[str, Any]:
    """Run exactly clean 0..63 then observed 0..63 and persist 43 pairs.

    This function has no model/dataloader constructor by design.  The real
    ORION environment must inject them after a successful preflight.  CPU
    mocks are rejected here so a unit test can never create a G1-looking run.
    """

    assert_real_execution_ready(preflight)
    if hooks.integration_kind != "real_orion" or not _is_actual_orion_model(model):
        raise OrionActualTargetRunnerError(
            "public runner accepts only the repository Orion class and real hooks"
        )
    if not callable(branch_batches):
        raise OrionActualTargetRunnerError("branch_batches must be callable")
    if preflight.get("plan_id") != plan.get("plan_id"):
        raise OrionActualTargetRunnerError("preflight plan_id mismatch")

    execution = plan["execution"]
    expected_frames = list(execution["expected_frames_each_branch"])
    measurement_frames = list(execution["measurement_frames"])
    measurement_set = set(measurement_frames)
    folder = plan["route"]["folder"]
    paired_replay_id = "route214-prefix63-%s" % plan["plan_id"][:16]
    policy = pilot_failure_event_policy()
    branch_outputs: Dict[str, Dict[int, Tuple[Any, torch.Tensor]]] = {}
    branch_reports: Dict[str, Any] = {}

    parity = bool(hooks.decoder_parity_check())
    selected_mode = bool(hooks.selected_motion_mode_check())
    projection = bool(hooks.projection_overlay_check())
    gt_axis_alignment = bool(hooks.gt_axis_alignment_check())
    if not (parity and selected_mode and projection and gt_axis_alignment):
        raise OrionActualTargetRunnerError(
            "decoder/motion/projection/GT-axis audits must pass before replay"
        )

    for branch in ("clean", "observed"):
        reset = reset_and_assert_orion_memory(model)
        history_id = "%s-%s-history" % (paired_replay_id, branch)
        frame_audits: List[Dict[str, Any]] = []
        measured: Dict[int, Tuple[Any, torch.Tensor]] = {}
        raw_loader_valid_light_count = 0
        affects_ego_valid_count = 0
        iterator = iter(branch_batches(branch))
        for expected_frame in expected_frames:
            try:
                raw_batch = next(iterator)
            except StopIteration as exc:
                raise OrionActualTargetRunnerError(
                    "%s replay ended before frame %d" % (branch, expected_frame)
                ) from exc
            if branch == "observed":
                raw_batch = hooks.corruption_transform(
                    raw_batch,
                    {
                        "branch": branch,
                        "frame_idx": expected_frame,
                        "corruption": plan["corruption"],
                    },
                )
            frame = hooks.perception_hook(model, raw_batch)
            canonical = frame.canonical_batch
            raw_loader_valid_light_count += int(
                canonical.traffic.loader_valid.sum().item()
            )
            affects_ego_valid_count += int(
                canonical.traffic.state_valid_affects_ego.sum().item()
            )
            if canonical.frame_idx != expected_frame or canonical.scene_token != folder:
                raise OrionActualTargetRunnerError(
                    "%s replay chronology mismatch at expected frame %d"
                    % (branch, expected_frame)
                )
            is_measurement = expected_frame in measurement_set
            eligibility_audit = None
            if is_measurement:
                context = {
                    "plan_id": plan["plan_id"],
                    "paired_replay_id": paired_replay_id,
                    "branch_history_id": history_id,
                    "branch": branch,
                    "frame_idx": expected_frame,
                    "previous_frame_idx": (
                        None if expected_frame == 0 else expected_frame - 1
                    ),
                    "failure_event_policy": policy,
                    "object_matching_policy_id": PILOT_MATCH_POLICY_ID,
                    "minimum_prediction_score": (
                        PILOT_MINIMUM_PREDICTION_SCORE
                    ),
                    "maximum_center_distance_m": (
                        PILOT_MAXIMUM_CENTER_DISTANCE_M
                    ),
                    "sensitivity_maximum_center_distance_m": (
                        PILOT_SENSITIVITY_CENTER_DISTANCE_M
                    ),
                    "matching_policy_claim": (
                        "preregistered_pilot_heuristic_not_calibrated_optimum"
                    ),
                    "gt_occupancy_rasterizer": hooks.gt_occupancy_rasterizer,
                    "pairwise_bev_iou": hooks.pairwise_bev_iou,
                    "project_visible_support": hooks.project_visible_support,
                    "gt_occupancy_rasterizer_id": (
                        hooks.gt_occupancy_rasterizer_id
                    ),
                    "predicted_occupancy_rasterizer_id": (
                        hooks.occupancy_rasterizer_id
                    ),
                    "pairwise_bev_iou_policy_id": (
                        hooks.pairwise_bev_iou_policy_id
                    ),
                    "support_projector_id": hooks.support_projector_id,
                    "gt_box_z_origin": hooks.gt_box_z_origin,
                    "decoded_box_z_origin": hooks.decoded_box_z_origin,
                    "box_z_origin_policy_id": BOX_Z_ORIGIN_POLICY_ID,
                    "post_augmentation_lidar2img_cpu": (
                        canonical.data["lidar2img"][0].detach().cpu()
                    ),
                    "processed_image_hw_by_view": (
                        [canonical.processed_image_hw] * len(ORION_CAMERA_ORDER)
                    ),
                    "traffic_semantics_id": TRAFFIC_SEMANTICS_ID,
                    "traffic_state_labels": canonical.traffic.state_labels,
                    "traffic_state_valid": (
                        canonical.traffic.state_valid_affects_ego
                    ),
                    "gt_actor_ids_aligned": canonical.data["gt_actor_ids"],
                    "gt_target_eligibility_policy": (
                        "safety-actors-plus-affecting-traffic-light/v1"
                    ),
                    "gt_target_eligibility_filter": filter_v1_gt_target_eligibility,
                    "gt_axis_alignment_check_passed": gt_axis_alignment,
                    "affects_ego_prediction_logit_policy": (
                        "retained_for_audit_only_not_state_component"
                    ),
                }
                built = hooks.branch_target_builder(
                    frame.adapted.frames[0], canonical, context
                )
                if not isinstance(built, BuiltBranchTargetV1):
                    raise OrionActualTargetRunnerError(
                        "branch_target_builder must return BuiltBranchTargetV1"
                    )
                bundle = built.bundle
                eligibility_audit = built.eligibility_audit
                if not isinstance(bundle, ActualTargetBranchBundleV1):
                    raise OrionActualTargetRunnerError(
                        "BuiltBranchTargetV1.bundle has the wrong type"
                    )
                if bundle.failure_event_policy.calibration_policy_id != PILOT_CALIBRATION_POLICY_ID:
                    raise OrionActualTargetRunnerError(
                        "real bundle must use the preregistered pilot threshold policy"
                    )
                if bundle.match_policy_id != PILOT_MATCH_POLICY_ID:
                    raise OrionActualTargetRunnerError(
                        "real bundle must use the preregistered match policy"
                    )
                if bundle.occupancy_rasterizer_id != SELECTED_MODE_RASTERIZER_ID:
                    raise OrionActualTargetRunnerError(
                        "bundle predicted occupancy must retain its production rasterizer ID"
                    )
                if bundle.gt_occupancy_rasterizer_id != GT_RASTERIZER_ID:
                    raise OrionActualTargetRunnerError(
                        "bundle GT occupancy must retain its distinct production rasterizer ID"
                    )
                if bundle.minimum_prediction_score != PILOT_MINIMUM_PREDICTION_SCORE:
                    raise OrionActualTargetRunnerError(
                        "real bundle must use minimum_prediction_score=0.50"
                    )
                if not torch.allclose(
                    bundle.max_center_distance,
                    torch.full_like(
                        bundle.max_center_distance,
                        PILOT_MAXIMUM_CENTER_DISTANCE_M,
                    ),
                ):
                    raise OrionActualTargetRunnerError(
                        "real bundle must use max_center_distance=4.0m"
                    )
                features = hooks.patch_feature_extractor(model, frame, context)
                if not isinstance(features, torch.Tensor) or features.ndim != 3:
                    raise OrionActualTargetRunnerError(
                        "patch_feature_extractor must return [V,P,D] tensor"
                    )
                measured[expected_frame] = (bundle, features.detach().cpu())
            frame_audits.append(
                _frame_runtime_audit(
                    frame,
                    branch=branch,
                    expected_frame_idx=expected_frame,
                    persisted=is_measurement,
                    target_adapter_ready=True,
                    eligibility_audit=eligibility_audit,
                )
            )
        try:
            extra = next(iterator)
        except StopIteration:
            extra = None
        if extra is not None:
            raise OrionActualTargetRunnerError(
                "%s replay yielded frames beyond prefix 63" % branch
            )
        if sorted(measured) != measurement_frames:
            raise OrionActualTargetRunnerError(
                "%s measurement output set is not exactly the plan" % branch
            )
        branch_outputs[branch] = measured
        branch_reports[branch] = {
            "frames_processed": expected_frames,
            "measurement_frames_persisted": measurement_frames,
            "reset_called_before_frame_zero": True,
            "reset_state_verified_empty": reset["all_audited_fields_are_none"],
            "reset_audit": reset,
            "no_other_branch_interleaving": True,
            "paired_replay_id": paired_replay_id,
            "branch_history_id": history_id,
            "per_frame_audit": frame_audits,
            "raw_loader_valid_light_count": raw_loader_valid_light_count,
            "affects_ego_valid_count": affects_ego_valid_count,
            "traffic_component_invalid_when_affects_count_zero": True,
        }

    persisted_records = []
    for frame_idx in measurement_frames:
        clean_bundle, clean_features = branch_outputs["clean"][frame_idx]
        observed_bundle, observed_features = branch_outputs["observed"][frame_idx]
        paired = pair_actual_target_branches(
            observed_bundle,
            clean_bundle,
            bundle_id="%s-frame-%05d" % (paired_replay_id, frame_idx),
            real_orion_hook_executed=True,
        )
        record = bridge_actual_target_bundle_to_v2_record(
            paired,
            observed_features,
            clean_features,
            record_id="route214-%05d" % frame_idx,
            pair_id="%s-%05d" % (paired_replay_id, frame_idx),
        )
        hooks.record_sink(
            record,
            paired,
            {
                "frame_idx": frame_idx,
                "plan_id": plan["plan_id"],
                "persist_only_measurement_frames": True,
            },
        )
        persisted_records.append(record.record_id)

    global_checks = {
        "source_info_sha256_verified": True,
        "camera_files_exist_for_all_frames": True,
        "annotation_files_exist_for_all_frames": True,
        "with_light_state_enabled": True,
        "traffic_state_not_overwritten_by_mask": True,
        "post_augmentation_matrices_verified": True,
        "camera_order_verified": True,
        "decoded_output_adapter_ready": True,
        "actual_target_adapter_ready": True,
        "decoder_parity_passed": parity,
        "selected_motion_mode_passed": selected_mode,
        "projection_overlay_passed": projection,
    }
    return {
        "schema_version": RUNNER_EXECUTION_SCHEMA_VERSION,
        "plan_id": plan["plan_id"],
        "persisted_record_ids": persisted_records,
        "persisted_record_count": len(persisted_records),
        "runtime_attestation": {
            "schema_version": RUNTIME_ATTESTATION_SCHEMA_VERSION,
            "plan_id": plan["plan_id"],
            "global_checks": global_checks,
            "branches": branch_reports,
            "runner_provenance": {
                "perception_path": (
                    "extract_feat->prepare_location->position_embeding->pts_bbox_head"
                ),
                "llm_vae_diffusion_executed": False,
                "traffic_semantics_id": TRAFFIC_SEMANTICS_ID,
                "failure_event_calibration_policy_id": (
                    PILOT_CALIBRATION_POLICY_ID
                ),
                "object_matching_policy": {
                    "policy_id": PILOT_MATCH_POLICY_ID,
                    "minimum_prediction_score": (
                        PILOT_MINIMUM_PREDICTION_SCORE
                    ),
                    "maximum_center_distance_m": (
                        PILOT_MAXIMUM_CENTER_DISTANCE_M
                    ),
                    "sensitivity_maximum_center_distance_m": (
                        PILOT_SENSITIVITY_CENTER_DISTANCE_M
                    ),
                    "claim": (
                        "preregistered pilot heuristic; not calibration-optimal"
                    ),
                },
                "runtime_hooks": runtime_hook_readiness(hooks),
                "geometry_integrations": {
                    "selected_mode_occupancy_rasterizer_id": (
                        hooks.occupancy_rasterizer_id
                    ),
                    "gt_occupancy_rasterizer_id": (
                        hooks.gt_occupancy_rasterizer_id
                    ),
                    "pairwise_bev_iou_policy_id": (
                        hooks.pairwise_bev_iou_policy_id
                    ),
                    "projected_visible_support_id": hooks.support_projector_id,
                    "camera_order": list(ORION_CAMERA_ORDER),
                    "patch_feature_hook_id": hooks.patch_feature_hook_id,
                    "gt_box_z_origin": hooks.gt_box_z_origin,
                    "decoded_box_z_origin": hooks.decoded_box_z_origin,
                    "box_z_origin_policy_id": BOX_Z_ORIGIN_POLICY_ID,
                    "distinct_occupancy_ids_are_not_forged_equal": True,
                },
                "gt_target_eligibility": {
                    "policy": "safety-actors-plus-affecting-traffic-light/v1",
                    "safety_actor_class_ids": list(SAFETY_ACTOR_CLASS_IDS),
                    "conditional_traffic_light_class_id": (
                        TRAFFIC_LIGHT_CLASS_ID
                    ),
                    "traffic_light_validity": (
                        "loader_mask_and_affects_ego_column_1"
                    ),
                    "excluded_class_ids": [4, 5, 8],
                    "gt_axis_alignment_check_passed": gt_axis_alignment,
                },
            },
        },
    }


__all__ = [
    "ACTUAL_TARGET_PIPELINE_ID",
    "ActualTargetRuntimeHooksV1",
    "CANONICAL_CAMERAS",
    "BuiltBranchTargetV1",
    "CanonicalORIONBatchV1",
    "FrozenORIONPerceptionHookV1",
    "GTEligibilityResultV1",
    "OrionActualTargetRunnerError",
    "PILOT_CALIBRATION_POLICY_ID",
    "PILOT_MATCH_POLICY_ID",
    "RUNNER_PREFLIGHT_SCHEMA_VERSION",
    "TRAFFIC_SEMANTICS_ID",
    "TrafficTargetV1",
    "assert_real_execution_ready",
    "build_production_runtime_hooks_v1",
    "build_runner_preflight",
    "canonicalize_orion_test_batch",
    "derive_v1_traffic_targets",
    "extract_evavit_patch_features_v1",
    "filter_v1_gt_target_eligibility",
    "load_stage3_agent_config",
    "mutate_stage3_agent_config_for_actual_targets",
    "pilot_failure_event_policy",
    "reset_and_assert_orion_memory",
    "run_chronological_actual_target_replay",
    "runtime_hook_readiness",
    "verify_box_z_origin_lineage",
    "verify_local_traffic_formatter_fix",
]
