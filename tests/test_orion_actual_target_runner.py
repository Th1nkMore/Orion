"""CPU-only tests for the fail-closed real ORION runner boundary."""

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from uq_estimator.orion_actual_target_runner import (
    CANONICAL_CAMERAS,
    PILOT_CALIBRATION_POLICY_ID,
    PILOT_MATCH_POLICY_ID,
    FrozenORIONPerceptionHookV1,
    OrionActualTargetRunnerError,
    assert_real_execution_ready,
    build_runner_preflight,
    build_production_runtime_hooks_v1,
    canonicalize_orion_test_batch,
    derive_v1_traffic_targets,
    extract_evavit_patch_features_v1,
    filter_v1_gt_target_eligibility,
    load_stage3_agent_config,
    mutate_stage3_agent_config_for_actual_targets,
    pilot_failure_event_policy,
    reset_and_assert_orion_memory,
    runtime_hook_readiness,
    verify_box_z_origin_lineage,
    verify_local_traffic_formatter_fix,
)
from uq_estimator.orion_decode_adapter import ORIONDecodeAdapterConfigV1
from uq_estimator.bev_target_rasterizer import (
    GT_RASTERIZER_ID,
    PAIRWISE_BEV_IOU_POLICY_ID,
    SELECTED_MODE_RASTERIZER_ID,
)
from uq_estimator.projected_visible_support import VISIBLE_SUPPORT_PROJECTION_VERSION
from uq_estimator.orion_replay_smoke import (
    build_replay_smoke_plan,
    load_pilot_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "adzoo" / "orion" / "configs" / "orion_stage3_agent.py"
PILOT = (
    REPO_ROOT
    / "configs"
    / "spatial_uq_route_manifests"
    / "b2d_val_exploratory_pilot10_seed20260826.json"
)


def _nodes(pipeline):
    for node in pipeline:
        yield node
        if "transforms" in node:
            yield from _nodes(node["transforms"])


def _mock_batch(frame_idx=0, scene_token="fixture-route"):
    count = 4
    return {
        "img": torch.zeros(1, 6, 3, 8, 8),
        "img_metas": [[{
            "frame_idx": frame_idx,
            "scene_token": scene_token,
            "camera_order": CANONICAL_CAMERAS,
            "pad_shape": [(8, 8, 3)] * 6,
        }]],
        "traffic_state": torch.tensor([[0, 1], [2, 0], [-1, -1], [-1, -1]]),
        "traffic_state_mask": torch.tensor([True, True, False, False]),
        "lidar2img": torch.eye(4).repeat(1, 6, 1, 1),
        "cam_intrinsic": torch.eye(4).repeat(1, 6, 1, 1),
        "timestamp": torch.tensor([0.1]),
        "ego_pose": torch.eye(4).unsqueeze(0),
        "ego_pose_inv": torch.eye(4).unsqueeze(0),
        "command": torch.tensor([1.0]),
        "can_bus": torch.zeros(1, 18),
        "gt_bboxes_3d": torch.zeros(count, 7),
        "gt_labels_3d": torch.tensor([0, 6, 4, 7]),
        "gt_attr_labels": torch.zeros(count, 8),
        "gt_actor_ids": torch.tensor([101, 102, 103, 104]),
    }


def test_dedicated_config_mutation_enables_and_collects_light_state_only_in_copy():
    source, lineage = load_stage3_agent_config(CONFIG)
    source_nodes = list(_nodes(source["test_pipeline"]))
    source_loader = next(node for node in source_nodes if node["type"] == "LoadAnnotations3D")
    source_collect = next(node for node in source_nodes if node["type"] == "CustomCollect3D")
    assert source_loader.get("with_light_state") is not True
    assert "traffic_state" not in source_collect["keys"]

    mutated, audit = mutate_stage3_agent_config_for_actual_targets(source)
    mutated_nodes = list(_nodes(mutated["test_pipeline"]))
    loader = next(node for node in mutated_nodes if node["type"] == "LoadAnnotations3D")
    collect = next(node for node in mutated_nodes if node["type"] == "CustomCollect3D")
    assert audit["passed"] is True
    assert audit["source_deployment_config_mutated"] is False
    assert loader["with_light_state"] is True
    assert loader["with_actor_ids"] is True
    assert "traffic_state" in collect["keys"]
    assert "traffic_state_mask" in collect["keys"]
    assert "gt_actor_ids" in collect["keys"]
    assert mutated["data"]["samples_per_gpu"] == 1
    assert lineage["sha256"]
    # Deep-copy contract: deployment config remains untouched.
    assert source_loader.get("with_light_state") is not True
    assert "traffic_state" not in source_collect["keys"]


def test_formatter_fix_has_static_and_functional_alignment_evidence():
    audit = verify_local_traffic_formatter_fix(REPO_ROOT)
    assert audit["passed"] is True
    assert audit["static_callsite_verified"] is True
    assert audit["functional_alignment_fixture_verified"] is True
    assert audit["actor_id_all_filter_stages_verified"] is True
    assert audit["actor_id_functional_alignment_verified"] is True


def test_traffic_v1_uses_state_column_and_loader_mask_intersect_affects_ego():
    result = derive_v1_traffic_targets(
        torch.tensor([[0, 1], [2, 0], [-1, -1]]),
        torch.tensor([True, True, False]),
    )
    assert result.state_labels.tolist() == [0, -1, -1]
    assert result.loader_valid.tolist() == [True, True, False]
    assert result.affects_ego.tolist() == [True, False, False]
    assert result.state_valid_affects_ego.tolist() == [True, False, False]

    route214_like = derive_v1_traffic_targets(
        torch.tensor([[0, 0], [2, 0]]), torch.tensor([True, True])
    )
    assert route214_like.loader_valid.sum().item() == 2
    assert route214_like.state_valid_affects_ego.sum().item() == 0
    assert route214_like.state_labels.tolist() == [-1, -1]


def test_gt_eligibility_filters_all_axes_and_checks_actor_id_order():
    classes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 6, 7, 8])
    count = classes.numel()
    state = torch.full((count, 2), -1, dtype=torch.long)
    mask = torch.zeros(count, dtype=torch.bool)
    state[6] = torch.tensor([0, 0])
    state[7] = torch.tensor([2, 1])
    mask[6:8] = True
    ids = ["actor-%02d" % index for index in range(count)]
    support = torch.zeros(6, 4, count)
    support[..., 7] = 0.5
    attr = torch.zeros(count, 34)
    attr[:, 27] = classes.to(torch.float32)
    result = filter_v1_gt_target_eligibility(
        boxes=torch.arange(count * 7, dtype=torch.float32).reshape(count, 7),
        classes=classes,
        gt_attr=attr,
        traffic_state=state,
        traffic_state_mask=mask,
        actor_ids=ids,
        projected_support=support,
        support_actor_ids=ids,
    )
    # car/van/truck/bicycle/pedestrian + only the affecting light.
    assert result.axes["classes"].tolist() == [0, 1, 2, 3, 6, 7]
    assert result.audit["pre_filter_count"] == 10
    assert result.audit["post_filter_count"] == 6
    assert result.audit["raw_loader_valid_light_count"] == 2
    assert result.audit["affects_ego_valid_count"] == 1
    assert result.audit["gt_and_support_actor_ids_exactly_equal"] is True
    assert result.axes["projected_support"].shape == (6, 4, 6)

    with pytest.raises(OrionActualTargetRunnerError, match="not exactly aligned"):
        filter_v1_gt_target_eligibility(
            boxes=torch.zeros(count, 7),
            classes=classes,
            gt_attr=attr,
            traffic_state=state,
            traffic_state_mask=mask,
            actor_ids=ids,
            projected_support=support,
            support_actor_ids=list(reversed(ids)),
        )
    bad_attr = attr.clone()
    bad_attr[0, 27] = 7
    with pytest.raises(OrionActualTargetRunnerError, match=r"gt_attr\[:,27\]"):
        filter_v1_gt_target_eligibility(
            boxes=torch.zeros(count, 7),
            classes=classes,
            gt_attr=bad_attr,
            traffic_state=state,
            traffic_state_mask=mask,
            actor_ids=ids,
            projected_support=support,
            support_actor_ids=ids,
        )


def test_canonical_batch_attests_post_aug_geometry_and_gt_alignment():
    result = canonicalize_orion_test_batch(_mock_batch(frame_idx=12))
    assert result.frame_idx == 12
    assert result.camera_order == CANONICAL_CAMERAS
    assert result.data["img"].shape == (1, 6, 3, 8, 8)
    assert result.data["lidar2img"].shape == (1, 6, 4, 4)
    assert result.traffic.state_valid_affects_ego.tolist() == [True, False, False, False]


class _FakeHead:
    def __init__(self):
        self.memory_embedding = torch.ones(1)
        self.memory_reference_point = torch.ones(1)
        self.memory_scene_query = torch.ones(1)
        self.scene_memory_timestamp = torch.ones(1)

    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_scene_query = None
        self.scene_memory_timestamp = None

    def __call__(self, img_metas, position, **data):
        probabilities = torch.tensor([[[[0.9, 0.1], [0.2, 0.8]]]])
        outs = {
            "all_cls_scores": torch.logit(probabilities),
            "all_bbox_preds": torch.zeros(1, 1, 2, 10),
            "all_traj_preds": torch.zeros(1, 1, 2, 2, 4),
            "all_traj_cls_scores": torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
            "all_traffic_states": torch.zeros(1, 1, 2, 4),
        }
        return outs, torch.zeros(1, 2, 4), None, None


class _FakeModel:
    def __init__(self):
        self.pts_bbox_head = _FakeHead()
        self.with_map_head = False
        self.training = True

    def eval(self):
        self.training = False
        return self

    def extract_feat(self, img):
        return torch.zeros(1, 6, 4, 40, 40)

    def prepare_location(self, img_metas, **data):
        return torch.zeros(6, 2, 2, 2)

    def position_embeding(self, data, location, img_metas):
        return torch.zeros(1, 24, 4)


def _occupancy(value):
    return torch.zeros(value.selected_deltas.shape[0], 2, 3, 3)


def test_perception_hook_captures_raw_head_and_connects_decode_adapter_only():
    model = _FakeModel()
    reset = reset_and_assert_orion_memory(model)
    assert reset["all_audited_fields_are_none"] is True
    hook = FrozenORIONPerceptionHookV1(
        ORIONDecodeAdapterConfigV1(
            num_classes=2,
            max_num=2,
            post_center_range=(-10, -10, -5, 10, 10, 5),
            class_mapping_id="fixture-class-map/v1",
            occupancy_rasterizer_id="fixture-selected-mode-raster/v1",
            with_light_state=True,
        ),
        _occupancy,
    )
    result = hook(model, _mock_batch())
    assert model.training is False
    assert set(result.raw_head_outputs) >= {
        "all_cls_scores",
        "all_bbox_preds",
        "all_traj_preds",
        "all_traj_cls_scores",
        "all_traffic_states",
    }
    assert len(result.adapted.frames) == 1
    assert result.adapted.frames[0].traffic_state_logits.shape[-1] == 4
    assert result.adapted.frames[0].traffic_probability_transform == "sigmoid"
    patches = extract_evavit_patch_features_v1(model, result, {})
    assert patches.shape == (6, 1600, 4)


def test_orion_head_reset_memory_clears_scene_query_buffers():
    source_path = REPO_ROOT / "mmcv/models/dense_heads/orion_head.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    orion_head = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OrionHead"
    )
    reset_memory = next(
        node
        for node in orion_head.body
        if isinstance(node, ast.FunctionDef) and node.name == "reset_memory"
    )
    cleared = {
        target.attr
        for node in ast.walk(reset_memory)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    assert {"memory_scene_query", "scene_memory_timestamp"} <= cleared


def _noop(*args, **kwargs):
    return None


def test_production_builder_hardwires_bev_projection_features_and_distinct_origins():
    config = ORIONDecodeAdapterConfigV1(
        num_classes=2,
        max_num=2,
        post_center_range=(-10, -10, -5, 10, 10, 5),
        class_mapping_id="fixture-class-map/v1",
        occupancy_rasterizer_id=SELECTED_MODE_RASTERIZER_ID,
        with_light_state=True,
    )
    hooks = build_production_runtime_hooks_v1(
        decode_config=config,
        branch_target_builder=_noop,
        corruption_transform=lambda batch, context: batch,
        record_sink=_noop,
        decoder_parity_check=lambda: True,
        selected_motion_mode_check=lambda: True,
        projection_overlay_check=lambda: False,
        gt_axis_alignment_check=lambda: True,
        gt_box_z_origin="bottom",
        decoded_box_z_origin="center",
    )
    assert hooks.occupancy_rasterizer_id == SELECTED_MODE_RASTERIZER_ID
    assert hooks.gt_occupancy_rasterizer_id == GT_RASTERIZER_ID
    assert hooks.pairwise_bev_iou_policy_id == PAIRWISE_BEV_IOU_POLICY_ID
    assert hooks.support_projector_id == VISIBLE_SUPPORT_PROJECTION_VERSION
    assert hooks.gt_box_z_origin == "bottom"
    assert hooks.decoded_box_z_origin == "center"
    assert runtime_hook_readiness(hooks)["real_execution_ready"] is False
    assert runtime_hook_readiness(hooks)["audit_results"]["projection_overlay_check"] is False
    assert runtime_hook_readiness(hooks)["audit_results"]["concrete_production_branch_builder"] is False
    assert runtime_hook_readiness(hooks)["audit_results"]["external_production_hook_ids_present"] is False
    source_config, config_lineage = load_stage3_agent_config(CONFIG)
    _, pipeline_audit = mutate_stage3_agent_config_for_actual_targets(source_config)
    pilot, pilot_lineage = load_pilot_manifest(PILOT)
    plan = build_replay_smoke_plan(pilot, pilot_lineage)
    preflight = build_runner_preflight(
        plan,
        config_lineage=config_lineage,
        pipeline_audit=pipeline_audit,
        formatter_audit=verify_local_traffic_formatter_fix(REPO_ROOT),
        box_z_origin_audit=verify_box_z_origin_lineage(REPO_ROOT),
        hooks=hooks,
    )
    assert preflight["checks"]["exporter_supports_distinct_pred_and_gt_occupancy_ids"] is True
    assert "exporter_supports_distinct_pred_and_gt_occupancy_ids" not in preflight["blockers"]

    matrices = torch.eye(4).repeat(6, 1, 1)
    for source, expected in (("privileged_gt", "bottom"), ("decoded_orion", "center")):
        projected = hooks.project_visible_support(
            torch.empty(0, 7),
            matrices,
            [(320, 640)] * 6,
            box_source=source,
            image_transform_id="fixture-post-aug/v1",
        )
        assert projected.support.shape == (6, 1600, 0)
        assert projected.projection_provenance.box_z_origin == expected

    with pytest.raises(OrionActualTargetRunnerError, match="explicit production origins"):
        build_production_runtime_hooks_v1(
            decode_config=config,
            branch_target_builder=_noop,
            corruption_transform=lambda batch, context: batch,
            record_sink=_noop,
            decoder_parity_check=lambda: True,
            selected_motion_mode_check=lambda: True,
            projection_overlay_check=lambda: True,
            gt_axis_alignment_check=lambda: True,
            gt_box_z_origin="center",
            decoded_box_z_origin="center",
        )


def test_preflight_is_false_without_real_raster_support_hooks_and_execute_refuses():
    config, lineage = load_stage3_agent_config(CONFIG)
    _, pipeline_audit = mutate_stage3_agent_config_for_actual_targets(config)
    formatter = verify_local_traffic_formatter_fix(REPO_ROOT)
    origins = verify_box_z_origin_lineage(REPO_ROOT)
    pilot, pilot_lineage = load_pilot_manifest(PILOT)
    plan = build_replay_smoke_plan(pilot, pilot_lineage)
    preflight = build_runner_preflight(
        plan,
        config_lineage=lineage,
        pipeline_audit=pipeline_audit,
        formatter_audit=formatter,
        box_z_origin_audit=origins,
        hooks=None,
    )
    assert preflight["execution_ready"] is False
    assert "source_and_files_verified" in preflight["blockers"]
    assert "real_runtime_hooks_connected" in preflight["blockers"]
    assert "exporter_supports_distinct_pred_and_gt_occupancy_ids" in preflight["blockers"]
    assert preflight["exporter_schema_audit"]["branch_bundle_schema_version"].endswith("/v2")
    assert preflight["box_z_origin_audit"]["gt_box_z_origin"] == "bottom"
    assert preflight["box_z_origin_audit"]["decoded_adapter_box_z_origin"] == "center"
    assert preflight["failure_event_policy"]["calibration_policy_id"] == PILOT_CALIBRATION_POLICY_ID
    assert preflight["object_matching_policy"]["policy_id"] == PILOT_MATCH_POLICY_ID
    assert preflight["object_matching_policy"]["minimum_prediction_score"] == 0.5
    assert preflight["traffic_semantics"]["route214_prefix_raw_annotation_audit"]["affects_ego_true_count"] == 0
    with pytest.raises(OrionActualTargetRunnerError, match="fail-closed"):
        assert_real_execution_ready(preflight)


def test_preflight_cli_dry_run_is_cpu_only_and_execute_flag_fails_closed():
    command = [sys.executable, "scripts/preflight_orion_actual_target_runner.py"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)
    assert result["execution_ready"] is False
    assert result["job_submitted"] is False
    assert result["gpu_used"] is False
    assert result["carla_used"] is False
    assert result["training_performed"] is False

    refused = subprocess.run(
        command + ["--execute"], check=False, capture_output=True, text=True
    )
    assert refused.returncode != 0
    assert "real execution is fail-closed" in refused.stderr


def test_real_policy_id_is_explicit_not_mock_or_optimal_claim():
    policy = pilot_failure_event_policy()
    assert policy.calibration_policy_id == PILOT_CALIBRATION_POLICY_ID
    assert "mock" not in policy.calibration_policy_id
    assert "unfitted" not in policy.calibration_policy_id
    assert policy.component_thresholds == (0.5,) * 6
    assert policy.minimum_patch_support == 0.01
