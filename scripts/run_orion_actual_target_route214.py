#!/usr/bin/env python3
"""Launch the bounded Route214 frozen-ORION actual-target smoke.

The default is a CPU dry-run.  Real model construction is entered only with
``--execute`` or the narrower ``--model-init-only`` diagnostic.  This script
never starts CARLA and never trains.  It is intentionally a thin production
launch layer around the fail-closed runner and an explicit integration
factory; target-building or QA semantics are not guessed here.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uq_estimator.bev_target_rasterizer import (  # noqa: E402
    SELECTED_MODE_RASTERIZER_ID,
)
from uq_estimator.corruptions import (  # noqa: E402
    corrupt_multiview_images_with_metadata,
)
from uq_estimator.orion_actual_target_runner import (  # noqa: E402
    OrionActualTargetRunnerError,
    assert_real_execution_ready,
    build_production_runtime_hooks_v1,
    build_runner_preflight,
    load_stage3_agent_config,
    mutate_stage3_agent_config_for_actual_targets,
    run_chronological_actual_target_replay,
    runtime_hook_readiness,
    verify_box_z_origin_lineage,
    verify_local_traffic_formatter_fix,
)
from uq_estimator.orion_decode_adapter import (  # noqa: E402
    ORIONDecodeAdapterConfigV1,
)
from uq_estimator.orion_replay_smoke import (  # noqa: E402
    build_replay_smoke_plan,
    load_pilot_manifest,
    normalize_folder,
    verify_source_infos,
)


LAUNCH_SCHEMA_VERSION = "orion-route214-production-launch/v1"
INTEGRATION_FACTORY_SCHEMA_VERSION = "orion-route214-production-integration/v1"
CORRUPTION_HOOK_ID = "route214-front-local-occlusion-fixed-seed-full-prefix/v1"
RECORD_SINK_ID = "route214-bounded-measurement-record-sink/v1"
BRANCH_BATCH_HOOK_ID = "route214-device-prepared-dual-loader-replay/v1"
DEFAULT_FACTORY = (
    "uq_estimator.orion_route214_production_integration:"
    "build_route214_production_integration_v1"
)
DEFAULT_CONFIG = REPO_ROOT / "adzoo/orion/configs/orion_stage3_agent.py"
DEFAULT_PILOT = (
    REPO_ROOT
    / "configs/spatial_uq_route_manifests/"
    "b2d_val_exploratory_pilot10_seed20260826.json"
)
DEFAULT_ASSET_ROOT = Path("/public/share/lidachuan/orion_assets")
DEFAULT_INFOS = DEFAULT_ASSET_ROOT / "data/infos/b2d_infos_val.pkl"
DEFAULT_DATASET_ROOT = DEFAULT_ASSET_ROOT / "data/bench2drive"
DEFAULT_MAP_ROOT = DEFAULT_DATASET_ROOT / "maps"
DEFAULT_MAP_FILE = DEFAULT_ASSET_ROOT / "data/infos/b2d_map_infos.pkl"
DEFAULT_CHECKPOINT = DEFAULT_ASSET_ROOT / "checkpoints/Orion.pth"
DEFAULT_LLM_PATH = DEFAULT_ASSET_ROOT / "checkpoints/pretrain_qformer"
DEFAULT_OUTPUT = DEFAULT_ASSET_ROOT / "spatial_uq_v1/route214_actual_target_smoke"
QA_KEYS = (
    "decoder_parity_check",
    "selected_motion_mode_check",
    "projection_overlay_check",
    "gt_axis_alignment_check",
)
REQUIRED_INTEGRATION_KEYS = ("branch_target_builder",) + QA_KEYS
RECORD_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class Route214LaunchError(RuntimeError):
    """Raised when the launch layer would otherwise guess or expand scope."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_audit(path: Path, kind: str) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    exists = resolved.is_file() if kind == "file" else resolved.is_dir()
    return {"path": str(resolved), "kind": kind, "exists": bool(exists)}


def _callable_identity(value: Any) -> Dict[str, Any]:
    return {
        "type": "%s.%s" % (value.__class__.__module__, value.__class__.__name__),
        "production_hook_id": getattr(value, "production_hook_id", None),
        "evidence_path": (
            str(getattr(value, "evidence_path"))
            if getattr(value, "evidence_path", None) is not None
            else None
        ),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Route214LaunchError("refusing to overwrite %s" % path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _replace_image_tensor(value: Any, new_tensor: torch.Tensor) -> Any:
    """Preserve the collate wrapper while replacing its sole image tensor."""

    if isinstance(value, torch.Tensor):
        return new_tensor
    if isinstance(value, list):
        if len(value) != 1:
            raise Route214LaunchError("raw image list must have one augmentation")
        result = list(value)
        result[0] = _replace_image_tensor(result[0], new_tensor)
        return result
    if isinstance(value, tuple):
        if len(value) != 1:
            raise Route214LaunchError("raw image tuple must have one augmentation")
        return (_replace_image_tensor(value[0], new_tensor),)
    if hasattr(value, "data"):
        result = copy.deepcopy(value)
        data = getattr(result, "data")
        replaced = _replace_image_tensor(data, new_tensor)
        if hasattr(result, "_data"):
            result._data = replaced
            return result
    raise Route214LaunchError("cannot replace the collated image tensor")


def _find_image_tensor(value: Any) -> torch.Tensor:
    seen = set()
    while not isinstance(value, torch.Tensor):
        identifier = id(value)
        if identifier in seen:
            raise Route214LaunchError("cyclic image wrapper")
        seen.add(identifier)
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        elif hasattr(value, "data"):
            value = value.data
        else:
            raise Route214LaunchError("cannot unwrap collated image tensor")
    if value.ndim not in (4, 5):
        raise Route214LaunchError("image tensor must be [6,3,H,W] or [1,6,3,H,W]")
    return value


@dataclass
class FixedRoute214LocalOcclusionV1:
    """One deterministic front-view local occlusion over observed frames 0..63."""

    seed: int = 20260826
    severity: int = 2
    front_view_index: int = 0
    production_hook_id: str = CORRUPTION_HOOK_ID

    def __post_init__(self) -> None:
        self._normalized_region: Optional[Tuple[float, float, float, float]] = None
        self._frame_metadata: Dict[int, Mapping[str, Any]] = {}

    def __call__(
        self,
        batch: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if context.get("branch") != "observed":
            raise Route214LaunchError("corruption hook is observed-branch only")
        frame_idx = context.get("frame_idx")
        if isinstance(frame_idx, bool) or not isinstance(frame_idx, int) or not 0 <= frame_idx <= 63:
            raise Route214LaunchError("corruption frame must lie in Route214 prefix 0..63")
        corruption = context.get("corruption")
        if not isinstance(corruption, Mapping):
            raise Route214LaunchError("runner corruption context is missing")
        expected = {
            "family": "local_occlusion",
            "severity": self.severity,
            "seed": self.seed,
            "event_window_frames_inclusive": [0, 63],
        }
        for key, value in expected.items():
            if corruption.get(key) != value:
                raise Route214LaunchError(
                    "corruption plan disagrees with fixed schedule at %s" % key
                )
        if frame_idx in self._frame_metadata:
            raise Route214LaunchError("observed frame was corrupted more than once")
        result_batch = copy.deepcopy(dict(batch))
        if "img" not in result_batch:
            raise Route214LaunchError("batch lacks img")
        source = _find_image_tensor(result_batch["img"])
        corrupted = corrupt_multiview_images_with_metadata(
            source,
            corruption="local_occlusion",
            severity=self.severity,
            view_indices=(self.front_view_index,),
            seed=self.seed,
            region=None,
        )
        region = tuple(corrupted.metadata.normalized_region)
        if self._normalized_region is None:
            self._normalized_region = region
        elif region != self._normalized_region:
            raise Route214LaunchError("fixed corruption region changed across frames")
        result_batch["img"] = _replace_image_tensor(
            result_batch["img"], corrupted.images
        )
        self._frame_metadata[frame_idx] = corrupted.metadata.to_dict()
        return result_batch

    def audit(self) -> Dict[str, Any]:
        return {
            "production_hook_id": self.production_hook_id,
            "family": "local_occlusion",
            "severity": self.severity,
            "seed": self.seed,
            "camera_name": "CAM_FRONT",
            "view_indices": [self.front_view_index],
            "fixed_region": (
                list(self._normalized_region)
                if self._normalized_region is not None
                else "deterministically_seeded_at_first_processed_image_shape"
            ),
            "event_window_frames_inclusive": [0, 63],
            "scope": "diagnostic_full_prefix_not_time_alignment_evidence",
            "processed_frame_count": len(self._frame_metadata),
            "processed_frames": sorted(self._frame_metadata),
        }


@dataclass
class BoundedRoute214RecordSinkV1:
    """Atomic, no-overwrite sink capped at the 43 measurement records."""

    output_root: Path
    maximum_records: int = 43
    production_hook_id: str = RECORD_SINK_ID

    def __post_init__(self) -> None:
        if self.maximum_records != 43:
            raise Route214LaunchError("Route214 sink cap must remain exactly 43")
        self.output_root = Path(self.output_root)
        self._records: Dict[str, Dict[str, Any]] = {}

    def __call__(self, record: Any, paired_bundle: Any, context: Mapping[str, Any]) -> None:
        record_id = getattr(record, "record_id", None)
        if not isinstance(record_id, str) or not RECORD_ID_PATTERN.fullmatch(record_id):
            raise Route214LaunchError("record_id is missing or unsafe")
        if record_id in self._records:
            raise Route214LaunchError("duplicate record_id %s" % record_id)
        if len(self._records) >= self.maximum_records:
            raise Route214LaunchError("bounded record sink exceeded 43 records")
        frame_idx = context.get("frame_idx")
        if isinstance(frame_idx, bool) or not isinstance(frame_idx, int):
            raise Route214LaunchError("record context lacks integer frame_idx")
        records_root = self.output_root / "records"
        records_root.mkdir(parents=True, exist_ok=True)
        target = records_root / (record_id + ".pt")
        if target.exists():
            raise Route214LaunchError("refusing to overwrite %s" % target)
        descriptor, temporary = tempfile.mkstemp(
            prefix=record_id + ".", suffix=".tmp", dir=str(records_root)
        )
        os.close(descriptor)
        temporary_path = Path(temporary)
        try:
            torch.save(
                {
                    "record": record,
                    "paired_actual_target_bundle": paired_bundle,
                    "context": dict(context),
                    "record_sink_id": self.production_hook_id,
                },
                temporary_path,
            )
            os.replace(str(temporary_path), str(target))
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        self._records[record_id] = {
            "record_id": record_id,
            "frame_idx": frame_idx,
            "path": str(target.resolve()),
            "size_bytes": target.stat().st_size,
            "sha256": _sha256(target),
        }

    def finalize(self, expected_record_ids: Sequence[str]) -> Dict[str, Any]:
        expected = list(expected_record_ids)
        if len(expected) != self.maximum_records or len(set(expected)) != len(expected):
            raise Route214LaunchError("runner must return exactly 43 unique record IDs")
        if list(self._records) != expected:
            raise Route214LaunchError("sink order/content disagrees with runner result")
        manifest = {
            "schema_version": "route214-bounded-record-manifest/v1",
            "production_hook_id": self.production_hook_id,
            "maximum_records": self.maximum_records,
            "record_count": len(self._records),
            "records": [self._records[record_id] for record_id in expected],
        }
        _atomic_json(self.output_root / "record_manifest.json", manifest)
        return manifest

    def audit(self) -> Dict[str, Any]:
        return {
            "production_hook_id": self.production_hook_id,
            "output_root": str(self.output_root.expanduser().resolve()),
            "maximum_records": self.maximum_records,
            "current_record_count": len(self._records),
            "overwrite_allowed": False,
            "persisted_frames_only": "preregistered_measurement_frames",
        }


def _move_value_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=False)
    if hasattr(value, "data"):
        result = copy.deepcopy(value)
        moved = _move_value_to_device(getattr(result, "data"), device)
        if not hasattr(result, "_data"):
            raise Route214LaunchError(
                "collate wrapper exposes data but has no writable _data field"
            )
        result._data = moved
        return result
    if isinstance(value, Mapping):
        return value.__class__(
            (key, _move_value_to_device(item, device))
            for key, item in value.items()
        )
    if isinstance(value, list):
        return [_move_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(item, device) for item in value)
    tensor = getattr(value, "tensor", None)
    if isinstance(tensor, torch.Tensor) and callable(getattr(value, "to", None)):
        return value.to(device)
    if hasattr(value, "_data") and hasattr(value, "data"):
        result = copy.deepcopy(value)
        result._data = _move_value_to_device(result.data, device)
        return result
    return value


@dataclass
class DevicePreparedRoute214BranchBatchesV1:
    """Re-iterable dataloader whose model inputs are explicitly CUDA-prepared."""

    data_loader: Iterable[Mapping[str, Any]]
    device: torch.device
    production_hook_id: str = BRANCH_BATCH_HOOK_ID

    def __post_init__(self) -> None:
        self._calls: list[str] = []

    def __call__(self, branch: str) -> Iterable[Mapping[str, Any]]:
        expected = ("clean", "observed")
        if len(self._calls) >= len(expected) or branch != expected[len(self._calls)]:
            raise Route214LaunchError(
                "branch batches must be requested once in clean->observed order"
            )
        self._calls.append(branch)
        for batch in self.data_loader:
            if not isinstance(batch, Mapping):
                raise Route214LaunchError("dataloader batch must be a mapping")
            # img_metas is deliberately kept CPU-only; all model/GT tensor axes
            # and LiDAR box wrappers are moved because the perception hook calls
            # the unwrapped Orion core and bypasses MMDataParallel scatter.
            prepared = {}
            for key, value in batch.items():
                prepared[key] = (
                    value
                    if key == "img_metas"
                    else _move_value_to_device(value, self.device)
                )
            yield prepared

    def audit(self) -> Dict[str, Any]:
        return {
            "production_hook_id": self.production_hook_id,
            "device": str(self.device),
            "branch_order": list(self._calls),
            "img_metas_cpu_only": True,
            "model_and_gt_tensors_device_prepared": True,
        }


def filter_dataset_to_route214_prefix(dataset: Any, folder: str) -> Dict[str, Any]:
    """Mutate only the newly built dataset to exact sorted frames 0..63."""

    infos = getattr(dataset, "data_infos", None)
    if not isinstance(infos, list):
        raise Route214LaunchError("built dataset lacks mutable data_infos list")
    selected: Dict[int, Mapping[str, Any]] = {}
    normalized_folder = normalize_folder(folder)
    for raw in infos:
        if not isinstance(raw, Mapping) or raw.get("folder") is None:
            continue
        if normalize_folder(raw["folder"]) != normalized_folder:
            continue
        frame_idx = raw.get("frame_idx")
        if isinstance(frame_idx, bool) or not isinstance(frame_idx, int):
            raise Route214LaunchError("Route214 data_info frame_idx must be integer")
        if not 0 <= frame_idx <= 63:
            continue
        if frame_idx in selected:
            raise Route214LaunchError("duplicate Route214 data_info frame")
        selected[frame_idx] = raw
    expected = list(range(64))
    if sorted(selected) != expected:
        missing = sorted(set(expected) - set(selected))
        raise Route214LaunchError(
            "dataset does not contain exact Route214 prefix; missing=%s" % missing[:10]
        )
    dataset.data_infos = [selected[index] for index in expected]
    dataset.flag = np.zeros(64, dtype=np.uint8)
    return {
        "folder": normalized_folder,
        "frame_range_inclusive": [0, 63],
        "frame_count": 64,
        "chronologically_sorted": True,
        "dataset_mutation_scope": "newly_constructed_smoke_dataset_only",
    }


def _config_nodes(pipeline: Sequence[Mapping[str, Any]]) -> Iterable[Dict[str, Any]]:
    for node in pipeline:
        if not isinstance(node, dict):
            raise Route214LaunchError("pipeline nodes must be dictionaries")
        yield node
        nested = node.get("transforms")
        if nested is not None:
            if not isinstance(nested, list):
                raise Route214LaunchError("nested transforms must be a list")
            yield from _config_nodes(nested)


def configure_runtime_paths(
    mutated: Mapping[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    """Copy path overrides into the already dedicated actual-target config."""

    result = copy.deepcopy(dict(mutated))
    data = result["data"]
    test = data["test"]
    test["ann_file"] = str(args.infos.expanduser().resolve())
    test["data_root"] = str(args.dataset_root.expanduser().resolve())
    test["map_root"] = str(args.map_root.expanduser().resolve())
    test["map_file"] = str(args.map_file.expanduser().resolve())
    test["test_mode"] = True
    data["samples_per_gpu"] = 1
    data["workers_per_gpu"] = args.workers
    model = result["model"]
    model["train_cfg"] = None
    model["pretrained"] = None
    model["frozen"] = True
    model["use_diff_decoder"] = False
    model["tokenizer"] = str(args.llm_path.expanduser().resolve())
    model["lm_head"] = str(args.llm_path.expanduser().resolve())
    for node in _config_nodes(result["test_pipeline"]):
        if node.get("type") == "LoadAnnoatationCriticalVQATest":
            node["tokenizer"] = str(args.llm_path.expanduser().resolve())
    return result


def build_decode_config(mutated_config: Mapping[str, Any]) -> ORIONDecodeAdapterConfigV1:
    model = mutated_config.get("model")
    if not isinstance(model, Mapping):
        raise Route214LaunchError("mutated config lacks model")
    head = model.get("pts_bbox_head")
    if not isinstance(head, Mapping):
        raise Route214LaunchError("mutated config lacks pts_bbox_head")
    coder = head.get("bbox_coder")
    if not isinstance(coder, Mapping):
        raise Route214LaunchError("mutated config lacks bbox_coder")
    classes = mutated_config.get("class_names")
    if not isinstance(classes, list) or not classes:
        raise Route214LaunchError("mutated config lacks class_names")
    class_digest = hashlib.sha256(
        json.dumps(classes, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return ORIONDecodeAdapterConfigV1(
        num_classes=int(coder["num_classes"]),
        max_num=int(coder["max_num"]),
        post_center_range=tuple(float(value) for value in coder["post_center_range"]),
        class_mapping_id="orion-stage3-nine-class-%s/v1" % class_digest,
        occupancy_rasterizer_id=SELECTED_MODE_RASTERIZER_ID,
        with_light_state=True,
        score_threshold=(
            float(coder["score_threshold"])
            if coder.get("score_threshold") is not None
            else None
        ),
    )


def _split_factory_spec(specification: str) -> Tuple[str, str]:
    module_name, separator, symbol = specification.partition(":")
    if not separator or not module_name.strip() or not symbol.strip():
        raise Route214LaunchError("integration factory must be module:function")
    return module_name.strip(), symbol.strip()


def load_production_integration(
    specification: str,
    context: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    """Import and validate the caller-owned production builder/QA boundary."""

    module_name, symbol = _split_factory_spec(specification)
    module = importlib.import_module(module_name)
    factory = getattr(module, symbol, None)
    if not callable(factory):
        raise Route214LaunchError("integration factory symbol is not callable")
    integration = factory(dict(context))
    if not isinstance(integration, Mapping):
        raise Route214LaunchError("integration factory must return a mapping")
    if integration.get("schema_version") != INTEGRATION_FACTORY_SCHEMA_VERSION:
        raise Route214LaunchError("unsupported production integration schema")
    missing = [key for key in REQUIRED_INTEGRATION_KEYS if key not in integration]
    if missing:
        raise Route214LaunchError(
            "production integration missing: " + ", ".join(missing)
        )
    try:
        builder_module = importlib.import_module(
            "uq_estimator.orion_actual_target_builder"
        )
        expected_builder_type = getattr(
            builder_module, "ProductionActualTargetBranchBuilderV1"
        )
    except (ImportError, AttributeError) as exc:
        raise Route214LaunchError(
            "ProductionActualTargetBranchBuilderV1 import is unavailable"
        ) from exc
    builder = integration["branch_target_builder"]
    if not isinstance(builder, expected_builder_type):
        raise Route214LaunchError(
            "branch_target_builder must be ProductionActualTargetBranchBuilderV1"
        )
    identities = {"branch_target_builder": _callable_identity(builder)}
    if not isinstance(getattr(builder, "production_hook_id", None), str) or not builder.production_hook_id.strip():
        raise Route214LaunchError("production builder must expose production_hook_id")
    for key in QA_KEYS:
        callback = integration[key]
        if not callable(callback):
            raise Route214LaunchError("%s must be callable" % key)
        hook_id = getattr(callback, "production_hook_id", None)
        evidence_path = getattr(callback, "evidence_path", None)
        if not isinstance(hook_id, str) or not hook_id.strip():
            raise Route214LaunchError("%s must expose production_hook_id" % key)
        if evidence_path is None or not str(evidence_path).strip():
            raise Route214LaunchError("%s must bind a persistent evidence_path" % key)
        identities[key] = _callable_identity(callback)
    return integration, {
        "factory": specification,
        "factory_type": "%s.%s" % (factory.__module__, factory.__name__),
        "callables": identities,
    }


def _prepare_plan(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pilot, pilot_lineage = load_pilot_manifest(args.pilot_manifest)
    source_verification = None
    source_error = None
    if args.infos.is_file() and args.dataset_root.is_dir():
        try:
            provisional = build_replay_smoke_plan(pilot, pilot_lineage)
            source_verification = verify_source_infos(
                args.infos,
                pilot,
                provisional["route"]["folder"],
                63,
                dataset_root=args.dataset_root,
            )
        except Exception as exc:
            source_error = "%s: %s" % (exc.__class__.__name__, exc)
            if args.execute or args.model_init_only:
                raise
    plan = build_replay_smoke_plan(
        pilot,
        pilot_lineage,
        corruption_family="local_occlusion",
        severity=2,
        seed=20260826,
        source_verification=source_verification,
    )
    return plan, {
        "pilot_manifest": dict(pilot_lineage),
        "source_verification_attempted": source_verification is not None or source_error is not None,
        "source_verification_error": source_error,
        "source_ready": all(
            plan["source_verification"].get(key) is True
            for key in (
                "source_info_sha256_verified",
                "exact_frame_zero_prefix_verified",
                "canonical_six_camera_metadata_verified",
                "camera_files_exist_for_all_frames",
                "annotation_files_exist_for_all_frames",
            )
        ),
    }


def _required_path_audits(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    return {
        "config": _path_audit(args.config, "file"),
        "pilot_manifest": _path_audit(args.pilot_manifest, "file"),
        "checkpoint": _path_audit(args.checkpoint, "file"),
        "infos": _path_audit(args.infos, "file"),
        "dataset_root": _path_audit(args.dataset_root, "directory"),
        "map_root": _path_audit(args.map_root, "directory"),
        "map_file": _path_audit(args.map_file, "file"),
        "llm_path": _path_audit(args.llm_path, "directory"),
    }


def _construct_model_and_loader(
    mutated_config: Mapping[str, Any],
    args: argparse.Namespace,
    plan: Mapping[str, Any],
) -> Tuple[Any, Any, DevicePreparedRoute214BranchBatchesV1, Dict[str, Any]]:
    if not torch.cuda.is_available():
        raise Route214LaunchError("CUDA is required for explicit real execution")
    device = torch.device("cuda", args.cuda_device)
    torch.cuda.set_device(device)
    from mmcv.datasets import build_dataloader, build_dataset
    from mmcv.models import build_model
    from mmcv.utils import Config, load_checkpoint, set_random_seed

    set_random_seed(args.seed, deterministic=True)
    cfg = Config(copy.deepcopy(dict(mutated_config)))
    dataset = build_dataset(cfg.data.test)
    dataset_audit = filter_dataset_to_route214_prefix(
        dataset, plan["route"]["folder"]
    )
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=args.seed,
        nonshuffler_sampler=cfg.data.get("nonshuffler_sampler"),
    )
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint_payload = load_checkpoint(
        model, str(args.checkpoint.expanduser().resolve()), map_location="cpu"
    )
    if isinstance(checkpoint_payload, Mapping):
        classes = checkpoint_payload.get("meta", {}).get("CLASSES")
        if classes is not None:
            model.CLASSES = classes
    model.requires_grad_(False)
    model.to(device).eval()
    core = getattr(model, "module", model)
    if hasattr(core, "pts_bbox_head") and hasattr(core.pts_bbox_head, "with_dn"):
        core.pts_bbox_head.with_dn = False
    branch_batches = DevicePreparedRoute214BranchBatchesV1(data_loader, device)
    return model, dataset, branch_batches, {
        "dataset": dataset_audit,
        "checkpoint": {
            "path": str(args.checkpoint.expanduser().resolve()),
            "sha256": _sha256(args.checkpoint.expanduser().resolve()),
            "checkpoint_meta_classes_present": hasattr(model, "CLASSES"),
        },
        "device": str(device),
        "workers": args.workers,
        "batch_size": 1,
        "shuffle": False,
        "model_eval": model.training is False,
        "model_parameters_require_grad": any(
            parameter.requires_grad for parameter in model.parameters()
        ),
        "branch_batches": branch_batches.audit(),
    }


def _factory_context(
    *,
    args: argparse.Namespace,
    plan: Mapping[str, Any],
    mutated_config: Mapping[str, Any],
    decode_config: ORIONDecodeAdapterConfigV1,
    dry_run: bool,
    model: Any = None,
    dataset: Any = None,
    branch_batches: Any = None,
    checkpoint_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": "orion-route214-production-factory-context/v1",
        "dry_run": dry_run,
        "repo_root": REPO_ROOT,
        "output_root": args.output_root.expanduser().resolve(),
        "plan": plan,
        "mutated_config": mutated_config,
        "decode_config": decode_config,
        "model": model,
        "dataset": dataset,
        "branch_batches": branch_batches,
        "checkpoint_path": args.checkpoint.expanduser().resolve(),
        "checkpoint_sha256": checkpoint_sha256 or args.checkpoint_sha256,
        "config_lineage": {
            "path": str(args.config.expanduser().resolve()),
            "sha256": _sha256(args.config.expanduser().resolve()),
        },
        "qa_evidence_paths": {
            "decoder_parity_check": args.decoder_parity_evidence.expanduser().resolve(),
            "selected_motion_mode_check": args.selected_mode_evidence.expanduser().resolve(),
            "projection_overlay_check": args.projection_overlay_evidence.expanduser().resolve(),
            "gt_axis_alignment_check": args.gt_axis_evidence.expanduser().resolve(),
        },
    }


def _build_hooks(
    integration: Mapping[str, Any],
    decode_config: ORIONDecodeAdapterConfigV1,
    corruption_hook: FixedRoute214LocalOcclusionV1,
    record_sink: BoundedRoute214RecordSinkV1,
):
    return build_production_runtime_hooks_v1(
        decode_config=decode_config,
        branch_target_builder=integration["branch_target_builder"],
        corruption_transform=corruption_hook,
        record_sink=record_sink,
        decoder_parity_check=integration["decoder_parity_check"],
        selected_motion_mode_check=integration["selected_motion_mode_check"],
        projection_overlay_check=integration["projection_overlay_check"],
        gt_axis_alignment_check=integration["gt_axis_alignment_check"],
        gt_box_z_origin="bottom",
        decoded_box_z_origin="center",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="default; no GPU/model/job")
    mode.add_argument("--execute", action="store_true", help="run the exact 128-forward smoke")
    mode.add_argument(
        "--model-init-only",
        action="store_true",
        help="load dataset/model/checkpoint and integration, but run zero frames",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pilot-manifest", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--infos", type=Path, default=DEFAULT_INFOS)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--map-file", type=Path, default=DEFAULT_MAP_FILE)
    parser.add_argument("--llm-path", type=Path, default=DEFAULT_LLM_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--integration-factory", default=DEFAULT_FACTORY)
    parser.add_argument(
        "--checkpoint-sha256",
        help="optional trusted lowercase digest; real mode recomputes and verifies it",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--decoder-parity-evidence",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--selected-mode-evidence",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--projection-overlay-evidence",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--gt-axis-evidence",
        type=Path,
        default=None,
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.workers < 0 or args.workers > 8:
        parser.error("--workers must lie in [0,8] for the 8-CPU allocation")
    if args.cuda_device < 0:
        parser.error("--cuda-device must be non-negative")
    if args.checkpoint_sha256 is not None and (
        len(args.checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in args.checkpoint_sha256)
    ):
        parser.error("--checkpoint-sha256 must be a lowercase SHA-256 digest")
    evidence_root = args.output_root / "evidence"
    if args.decoder_parity_evidence is None:
        args.decoder_parity_evidence = evidence_root / "decoder_parity.json"
    if args.selected_mode_evidence is None:
        args.selected_mode_evidence = evidence_root / "selected_motion_mode.json"
    if args.projection_overlay_evidence is None:
        args.projection_overlay_evidence = evidence_root / "projection_overlay.json"
    if args.gt_axis_evidence is None:
        args.gt_axis_evidence = evidence_root / "gt_axis_alignment.json"
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    real_requested = bool(args.execute or args.model_init_only)
    path_audits = _required_path_audits(args)
    if not path_audits["config"]["exists"] or not path_audits["pilot_manifest"]["exists"]:
        raise Route214LaunchError("local config and pilot manifest must exist")

    source_config, config_lineage = load_stage3_agent_config(args.config)
    mutated, pipeline_audit = mutate_stage3_agent_config_for_actual_targets(source_config)
    mutated = configure_runtime_paths(mutated, args)
    formatter_audit = verify_local_traffic_formatter_fix(REPO_ROOT)
    origin_audit = verify_box_z_origin_lineage(REPO_ROOT)
    plan, plan_audit = _prepare_plan(args)
    decode_config = build_decode_config(mutated)
    corruption_hook = FixedRoute214LocalOcclusionV1()
    record_sink = BoundedRoute214RecordSinkV1(args.output_root)

    model = None
    dataset = None
    branch_batches = None
    runtime_audit = None
    integration = None
    integration_audit: Dict[str, Any]
    integration_error = None
    hooks = None

    # Dry-run imports and describes the exact production dependency without
    # constructing ORION. Real mode rebuilds it after model/dataset creation so
    # evidence callbacks can bind the production objects.
    if not real_requested:
        try:
            integration, integration_audit = load_production_integration(
                args.integration_factory,
                _factory_context(
                    args=args,
                    plan=plan,
                    mutated_config=mutated,
                    decode_config=decode_config,
                    dry_run=True,
                ),
            )
            hooks = _build_hooks(
                integration, decode_config, corruption_hook, record_sink
            )
        except Exception as exc:
            integration_error = "%s: %s" % (exc.__class__.__name__, exc)
            integration_audit = {
                "factory": args.integration_factory,
                "loaded": False,
                "error": integration_error,
            }

    preflight = build_runner_preflight(
        plan,
        config_lineage=config_lineage,
        pipeline_audit=pipeline_audit,
        formatter_audit=formatter_audit,
        box_z_origin_audit=origin_audit,
        hooks=hooks,
    )

    if real_requested:
        missing_paths = [
            name for name, audit in path_audits.items() if audit["exists"] is not True
        ]
        if missing_paths:
            raise Route214LaunchError(
                "real execution paths are missing: " + ", ".join(missing_paths)
            )
        if plan_audit["source_ready"] is not True:
            raise Route214LaunchError("source/file verification did not pass")
        model, dataset, branch_batches, runtime_audit = _construct_model_and_loader(
            mutated, args, plan
        )
        computed_checkpoint_sha256 = runtime_audit["checkpoint"]["sha256"]
        if (
            args.checkpoint_sha256 is not None
            and args.checkpoint_sha256 != computed_checkpoint_sha256
        ):
            raise Route214LaunchError(
                "trusted --checkpoint-sha256 disagrees with the loaded checkpoint"
            )
        integration, integration_audit = load_production_integration(
            args.integration_factory,
            _factory_context(
                args=args,
                plan=plan,
                mutated_config=mutated,
                decode_config=decode_config,
                dry_run=False,
                model=model,
                dataset=dataset,
                branch_batches=branch_batches,
                checkpoint_sha256=computed_checkpoint_sha256,
            ),
        )
        hooks = _build_hooks(integration, decode_config, corruption_hook, record_sink)
        preflight = build_runner_preflight(
            plan,
            config_lineage=config_lineage,
            pipeline_audit=pipeline_audit,
            formatter_audit=formatter_audit,
            box_z_origin_audit=origin_audit,
            hooks=hooks,
        )
        # Model-init-only is a zero-forward infrastructure diagnostic. It may
        # report missing numerical/visual QA evidence; only the 128-forward
        # replay may cross the fail-closed G1 gate.
        if args.execute:
            assert_real_execution_ready(preflight)

    report: Dict[str, Any] = {
        "schema_version": LAUNCH_SCHEMA_VERSION,
        "mode": (
            "execute"
            if args.execute
            else "model_init_only"
            if args.model_init_only
            else "dry_run"
        ),
        "execution_requested": real_requested,
        "route": "Town04/Route214",
        "stage": "G1_actual_target_smoke_only",
        "stage_b": False,
        "carla_used": False,
        "training_performed": False,
        "slurm_job_submitted_by_this_script": False,
        "resource_contract": {
            "memory": "220G",
            "cpus": 8,
            "gpus": 1,
            "gpu_type": "A800",
            "time_limit": "02:00:00",
        },
        "paths": path_audits,
        "plan": {
            "plan_id": plan["plan_id"],
            "folder": plan["route"]["folder"],
            "frames_each_branch": plan["execution"]["expected_forward_count_each_branch"],
            "forward_count_total": plan["execution"]["expected_forward_count_total"],
            "measurement_record_count": plan["execution"]["expected_paired_target_record_count"],
            "audit": plan_audit,
        },
        "config": {
            "lineage": config_lineage,
            "pipeline_audit": pipeline_audit,
            "model_frozen": mutated["model"]["frozen"],
            "diffusion_enabled": mutated["model"]["use_diff_decoder"],
            "samples_per_gpu": mutated["data"]["samples_per_gpu"],
            "workers_per_gpu": mutated["data"]["workers_per_gpu"],
        },
        "decode_config": {
            "num_classes": decode_config.num_classes,
            "max_num": decode_config.max_num,
            "class_mapping_id": decode_config.class_mapping_id,
            "occupancy_rasterizer_id": decode_config.occupancy_rasterizer_id,
            "with_light_state": decode_config.with_light_state,
        },
        "production_integration": integration_audit,
        "production_integration_error": integration_error,
        "hook_readiness": runtime_hook_readiness(hooks),
        "corruption_hook": corruption_hook.audit(),
        "record_sink": record_sink.audit(),
        "runner_preflight": preflight,
        "runtime_construction": runtime_audit,
    }

    if not real_requested:
        if args.report is not None:
            _atomic_json(args.report.expanduser().resolve(), report)
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_root / "launch_preflight.json", report)
    if args.model_init_only:
        result = {
            "schema_version": LAUNCH_SCHEMA_VERSION,
            "mode": "model_init_only",
            "model_checkpoint_loaded": True,
            "dataset_prefix_built": True,
            "frames_processed": 0,
            "records_persisted": 0,
            "full_runner_execution_ready": preflight["execution_ready"],
            "full_runner_blockers": preflight["blockers"],
            "carla_used": False,
            "training_performed": False,
        }
        _atomic_json(args.output_root / "model_init_result.json", result)
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    assert model is not None and branch_batches is not None and hooks is not None
    runner_result = run_chronological_actual_target_replay(
        plan=plan,
        preflight=preflight,
        model=model,
        branch_batches=branch_batches,
        hooks=hooks,
    )
    sink_manifest = record_sink.finalize(runner_result["persisted_record_ids"])
    final_result = {
        "schema_version": LAUNCH_SCHEMA_VERSION,
        "mode": "execute",
        "runner_result": runner_result,
        "record_manifest": sink_manifest,
        "corruption_audit": corruption_hook.audit(),
        "branch_batch_audit": branch_batches.audit(),
        "carla_used": False,
        "training_performed": False,
    }
    _atomic_json(args.output_root / "run_result.json", final_result)
    json.dump(final_result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
