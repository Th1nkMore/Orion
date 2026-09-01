#!/usr/bin/env python3
"""Offline Stage-1 UQ extraction from saved clean closed-loop camera frames.

Only ORION's frozen EVAViT image backbone and the frozen task-agnostic
observation adapter are loaded.  The LLM, planning head, simulator and control
path are absent.  Raw three-component evidence is normalized with a frozen
route-prefix calibration and saved together with its scalar mean summary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_paired_spatial_features import (
    _build_real_backbone,
    _extract_tokens,
)
from scripts.scenario_factory_lib import sha256_file
from uq_estimator.counterfactual_evidence import EVIDENCE_COMPONENTS
from uq_estimator.online_observation_uq import load_frozen_pairwise_adapter


SCHEMA = "orion.stage1_observation_uq_sequence.v1"
MULTIFRAME_SCHEMA = "orion.stage1_observation_uq_multiframe.v1"
KEYFRAME_SCHEMA = "orion.scenario_event_keyframes.v1"
CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
CAMERA_DIRECTORIES = {
    "CAM_FRONT": "rgb_front_model_input",
    "CAM_FRONT_LEFT": "rgb_front_left",
    "CAM_FRONT_RIGHT": "rgb_front_right",
    "CAM_BACK": "rgb_back",
    "CAM_BACK_LEFT": "rgb_back_left",
    "CAM_BACK_RIGHT": "rgb_back_right",
}
RGB_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
RGB_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


def preprocess_saved_camera(path: Path) -> np.ndarray:
    """Reproduce ORION's deterministic test-time RGB image transform."""

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb = rgb.resize((640, 360), resample=Image.Resampling.BILINEAR)
        rgb = rgb.crop((0, 40, 640, 360))
        rgb = rgb.resize((640, 640), resample=Image.Resampling.BILINEAR)
        value = np.asarray(rgb, dtype=np.float32)
    value = (value - RGB_MEAN) / RGB_STD
    return np.ascontiguousarray(value.transpose(2, 0, 1))


def robust_component_calibration(
    raw_components: np.ndarray,
    baseline_indices: Sequence[int],
    *,
    relative_scale_floor: float = 0.05,
    absolute_scale_floor: float = 0.001,
    z_center: float = 4.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Normalize each view/component from a frozen prefix, never future data."""

    raw = np.asarray(raw_components, dtype=np.float32)
    if raw.ndim != 5 or raw.shape[-1] != len(EVIDENCE_COMPONENTS):
        raise ValueError("raw components must have shape [T,V,H,W,3]")
    selected = raw[np.asarray(baseline_indices, dtype=np.int64)]
    if selected.shape[0] < 4:
        raise ValueError("at least four saved prefix frames are required")
    median = np.median(selected, axis=(0, 2, 3))
    mad = np.median(
        np.abs(selected - median[None, :, None, None, :]),
        axis=(0, 2, 3),
    )
    scale = np.maximum.reduce((
        1.4826 * mad,
        relative_scale_floor * np.abs(median),
        np.full_like(median, absolute_scale_floor),
    ))
    z = (raw - median[None, :, None, None, :]) / scale[None, :, None, None, :]
    shifted = z - float(z_center)
    normalized = np.empty_like(shifted, dtype=np.float32)
    positive = shifted >= 0.0
    normalized[positive] = 1.0 / (1.0 + np.exp(-shifted[positive]))
    exponential = np.exp(shifted[~positive])
    normalized[~positive] = exponential / (1.0 + exponential)
    return normalized, {
        "schema": "orion.route_prefix_spatial_uq_calibration.v1",
        "scope": "per_view_per_component_over_prefix_frames_and_patches",
        "baseline_sequence_indices": list(map(int, baseline_indices)),
        "baseline_saved_frame_count": int(selected.shape[0]),
        "median": median.tolist(),
        "scale": scale.tolist(),
        "relative_scale_floor": float(relative_scale_floor),
        "absolute_scale_floor": float(absolute_scale_floor),
        "z_center": float(z_center),
        "formula": "sigmoid((raw - median) / robust_scale - z_center)",
        "uses_route_or_actor_inputs": False,
        "uses_corruption_metadata": False,
        "uses_future_frames": False,
    }


def _load_event_package(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "orion.scenario_event_package.v1":
        raise ValueError("unsupported event package schema")
    if value.get("qa_input_ready") is not True or value["runtime"]["valid"] is not True:
        raise ValueError("event package is not runtime-valid and QA-ready")
    if not value.get("critical_event"):
        raise ValueError("event package has no actor-grounded critical event")
    return value


def _available_frames(event_package: Mapping[str, Any]) -> List[int]:
    per_view = []
    for view in CAMERA_ORDER:
        root = Path(event_package["camera_inventory"][CAMERA_DIRECTORIES[view]]["path"])
        per_view.append({int(path.stem) for path in root.glob("*.png")})
    shared = sorted(set.intersection(*per_view))
    if not shared:
        raise ValueError("camera streams have no aligned saved frames")
    return shared


def _frame_tensor(
    event_package: Mapping[str, Any], frame_index: int
) -> Tuple[torch.Tensor, List[Dict[str, str]]]:
    arrays = []
    provenance = []
    for view in CAMERA_ORDER:
        root = Path(event_package["camera_inventory"][CAMERA_DIRECTORIES[view]]["path"])
        path = root / ("%04d.png" % frame_index)
        arrays.append(preprocess_saved_camera(path))
        provenance.append({
            "view": view,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        })
    return torch.from_numpy(np.stack(arrays)), provenance


def _keyframe_targets(
    keyframe_manifest_path: Path,
    event_package_path: Path,
    available_frames: Sequence[int],
) -> Tuple[List[int], Dict[int, Dict[str, Any]], Dict[str, Any]]:
    manifest = json.loads(keyframe_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != KEYFRAME_SCHEMA:
        raise ValueError("unsupported event-keyframe manifest schema")
    event_ref = manifest.get("provenance", {}).get("event_package", {})
    if event_ref.get("sha256") != sha256_file(event_package_path):
        raise ValueError("event-keyframe manifest event-package hash mismatch")
    if any(
        manifest.get("selection_policy", {}).get(key) is not False
        for key in (
            "uses_learned_uq",
            "uses_stage2_outputs",
            "uses_qa_answers",
            "uses_closed_loop_improvement",
        )
    ):
        raise ValueError("event-keyframe selection used a prohibited model outcome")
    rows = manifest.get("keyframes")
    if not isinstance(rows, list) or len(rows) < 3 or len(rows) > 5:
        raise ValueError("multi-frame Stage1 extraction requires three to five keyframes")
    by_frame: Dict[int, Dict[str, Any]] = {}
    for raw in rows:
        frame = int(raw["selected_saved_frame_index"])
        if frame in by_frame:
            raise ValueError("event-keyframe manifest duplicates a saved frame")
        if frame not in available_frames:
            raise ValueError("selected keyframe is absent from aligned camera streams")
        by_frame[frame] = dict(raw)
    targets = sorted(by_frame)
    return targets, by_frame, manifest


def _write_sequence(
    *,
    output_dir: Path,
    event_package_path: Path,
    adapter_checkpoint: Path,
    adapter_metadata: Mapping[str, Any],
    calibration: Mapping[str, Any],
    backbone_record: Mapping[str, Any],
    process_frames: Sequence[int],
    frame_inventory: Sequence[Mapping[str, Any]],
    normalized_components: np.ndarray,
    raw_components: np.ndarray,
    target_frame: int,
    context_frames: int,
    keyframe_reference: Mapping[str, Any] = None,
    keyframe_record: Mapping[str, Any] = None,
) -> Dict[str, Any]:
    eligible = [value for value in process_frames if value <= target_frame]
    context = eligible[-context_frames:]
    if len(context) != context_frames or context[-1] != target_frame:
        raise ValueError("insufficient temporal context ending at selected keyframe")
    context_indices = [process_frames.index(value) for value in context]
    selected_components = normalized_components[context_indices]
    selected_scalar = selected_components.mean(axis=-1).astype(np.float32)
    selected_raw = raw_components[context_indices]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stage1_observation_uq_sequence.npz"
    np.savez_compressed(
        output_path,
        uncertainty=selected_scalar,
        uncertainty_components=selected_components,
        raw_uncertainty_components=selected_raw,
    )
    inventory_by_frame = {
        int(row["saved_frame_index"]): row for row in frame_inventory
    }
    manifest = {
        "schema": SCHEMA,
        "status": "offline_frozen_stage1_output",
        "control_influence": False,
        "event_package": str(event_package_path.resolve()),
        "event_package_sha256": sha256_file(event_package_path),
        "latest_frame_index": target_frame,
        "context_saved_frame_indices": context,
        "camera_order": list(CAMERA_ORDER),
        "component_names": list(EVIDENCE_COMPONENTS),
        "checkpoint": str(adapter_checkpoint.resolve()),
        "checkpoint_sha256": adapter_metadata["sha256"],
        "checkpoint_schema": adapter_metadata["schema_version"],
        "normalization": "route_prefix_robust_sigmoid_v1",
        "normalization_metadata": calibration,
        "uncertainty": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "key": "uncertainty",
            "shape": list(selected_scalar.shape),
            "component_key": "uncertainty_components",
            "component_shape": list(selected_components.shape),
            "raw_component_key": "raw_uncertainty_components",
        },
        "processed_frame_inventory": [inventory_by_frame[value] for value in context],
        "backbone": dict(backbone_record),
        "forbidden_inputs": {
            "route": False,
            "actor_geometry": False,
            "ttc": False,
            "collision_outcome": False,
            "corruption_metadata": False,
        },
        "claim_boundary": (
            "Offline task-agnostic Stage-1 signal for QA construction; route-prefix "
            "normalization is operational calibration, not uncertainty ground truth."
        ),
    }
    if keyframe_reference is not None:
        manifest["keyframe_manifest"] = dict(keyframe_reference)
        manifest["keyframe_record"] = dict(keyframe_record or {})
    manifest_path = output_dir / "stage1_observation_uq_sequence.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def extract_sequence(
    *,
    event_package_path: Path,
    orion_config: Path,
    orion_checkpoint: Path,
    adapter_checkpoint: Path,
    adapter_checkpoint_sha256: str,
    output_dir: Path,
    context_frames: int,
    baseline_start_frame: int,
    baseline_end_frame: int,
    keyframe_manifest_path: Path = None,
) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite Stage-1 sequence output")
    if context_frames < 2:
        raise ValueError("context_frames must be at least two")
    if baseline_end_frame <= baseline_start_frame:
        raise ValueError("baseline frame interval is empty")
    event_package = _load_event_package(event_package_path)
    frames = _available_frames(event_package)
    keyframe_rows = {}
    keyframe_manifest = None
    if keyframe_manifest_path is None:
        expected = float(event_package["critical_event"]["step"]) / 10.0
        target_frames = [min(frames, key=lambda value: (abs(value - expected), value))]
    else:
        target_frames, keyframe_rows, keyframe_manifest = _keyframe_targets(
            keyframe_manifest_path, event_package_path, frames
        )
    latest_frame = max(target_frames)
    process_frames = [value for value in frames if value <= latest_frame]
    baseline_frames = [
        value for value in process_frames
        if baseline_start_frame <= value < baseline_end_frame
    ]
    if len(baseline_frames) < 4:
        raise ValueError("saved route does not contain four calibration-prefix frames")
    for target_frame in target_frames:
        eligible = [value for value in process_frames if value <= target_frame]
        if len(eligible[-context_frames:]) != context_frames:
            raise ValueError("insufficient temporal context ending at a selected frame")

    args = SimpleNamespace(config=orion_config, checkpoint=orion_checkpoint)
    _, backbone, backbone_metadata = _build_real_backbone(args)
    adapter, adapter_metadata = load_frozen_pairwise_adapter(
        adapter_checkpoint,
        expected_sha256=adapter_checkpoint_sha256,
        device="cuda",
    )
    raw_outputs = []
    frame_inventory = []
    previous_features = None
    for frame_index in process_frames:
        images, camera_files = _frame_tensor(event_package, frame_index)
        images = images.unsqueeze(0).cuda(non_blocking=True)
        with torch.inference_mode():
            tokens, height, width = _extract_tokens(backbone, images)
            features = tokens.reshape(
                1, len(CAMERA_ORDER), height, width, tokens.shape[-1]
            )
            previous_valid = torch.tensor(
                [previous_features is not None], dtype=torch.bool, device="cuda"
            )
            score = adapter(features, previous_features, previous_valid)
        raw_outputs.append(score[0].detach().float().cpu().numpy())
        previous_features = features.detach()
        frame_inventory.append({
            "saved_frame_index": frame_index,
            "camera_files": camera_files,
        })
        print(
            "[Stage1Offline] frame=%d/%d grid=%dx%d"
            % (frame_index, latest_frame, height, width),
            flush=True,
        )
    raw = np.stack(raw_outputs).astype(np.float32)
    baseline_sequence_indices = [process_frames.index(value) for value in baseline_frames]
    normalized_components, calibration = robust_component_calibration(
        raw, baseline_sequence_indices
    )
    backbone_record = {
        **backbone_metadata,
        "orion_checkpoint_sha256": sha256_file(orion_checkpoint),
        "loads_orion_llm": False,
        "loads_orion_planning_head": False,
    }
    if keyframe_manifest_path is None:
        return _write_sequence(
            output_dir=output_dir,
            event_package_path=event_package_path,
            adapter_checkpoint=adapter_checkpoint,
            adapter_metadata=adapter_metadata,
            calibration=calibration,
            backbone_record=backbone_record,
            process_frames=process_frames,
            frame_inventory=frame_inventory,
            normalized_components=normalized_components,
            raw_components=raw,
            target_frame=target_frames[0],
            context_frames=context_frames,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    keyframe_reference = {
        "path": str(keyframe_manifest_path.resolve()),
        "sha256": sha256_file(keyframe_manifest_path),
    }
    sequence_rows = []
    for target_frame in target_frames:
        frame_root = output_dir / ("frame_%04d" % target_frame)
        manifest = _write_sequence(
            output_dir=frame_root,
            event_package_path=event_package_path,
            adapter_checkpoint=adapter_checkpoint,
            adapter_metadata=adapter_metadata,
            calibration=calibration,
            backbone_record=backbone_record,
            process_frames=process_frames,
            frame_inventory=frame_inventory,
            normalized_components=normalized_components,
            raw_components=raw,
            target_frame=target_frame,
            context_frames=context_frames,
            keyframe_reference=keyframe_reference,
            keyframe_record=keyframe_rows[target_frame],
        )
        manifest_path = frame_root / "stage1_observation_uq_sequence.json"
        sequence_rows.append({
            "selected_saved_frame_index": target_frame,
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
            },
            "component_shape": manifest["uncertainty"]["component_shape"],
        })
    root_manifest = {
        "schema": MULTIFRAME_SCHEMA,
        "status": "offline_frozen_stage1_multiframe_output",
        "control_influence": False,
        "event_package": {
            "path": str(event_package_path.resolve()),
            "sha256": sha256_file(event_package_path),
        },
        "keyframe_manifest": keyframe_reference,
        "keyframe_count": len(sequence_rows),
        "sequences": sequence_rows,
        "shared_normalization": "route_prefix_robust_sigmoid_v1",
        "shared_normalization_metadata": calibration,
        "backbone": backbone_record,
        "selection_policy": keyframe_manifest["selection_policy"],
        "claim_boundary": (
            "One frozen Stage-1 replay shared across fixed temporal keyframes; "
            "no task, learned-UQ, Stage2, or outcome-adaptive frame selection."
        ),
    }
    root_path = output_dir / "stage1_observation_uq_multiframe.json"
    root_path.write_text(
        json.dumps(root_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return root_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-package", type=Path, required=True)
    parser.add_argument("--orion-config", type=Path, required=True)
    parser.add_argument("--orion-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--baseline-start-frame", type=int, default=2)
    parser.add_argument("--baseline-end-frame", type=int, default=8)
    parser.add_argument("--keyframe-manifest", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("offline Stage-1 extraction requires CUDA")
    manifest = extract_sequence(
        event_package_path=args.event_package.resolve(),
        orion_config=args.orion_config.resolve(),
        orion_checkpoint=args.orion_checkpoint.resolve(),
        adapter_checkpoint=args.adapter_checkpoint.resolve(),
        adapter_checkpoint_sha256=args.adapter_checkpoint_sha256,
        output_dir=args.output_dir.resolve(),
        context_frames=args.context_frames,
        baseline_start_frame=args.baseline_start_frame,
        baseline_end_frame=args.baseline_end_frame,
        keyframe_manifest_path=(
            args.keyframe_manifest.resolve() if args.keyframe_manifest else None
        ),
    )
    if manifest.get("schema") == MULTIFRAME_SCHEMA:
        summary = {
            "manifest": str((args.output_dir / "stage1_observation_uq_multiframe.json").resolve()),
            "keyframe_count": manifest["keyframe_count"],
            "selected_saved_frames": [
                row["selected_saved_frame_index"] for row in manifest["sequences"]
            ],
        }
    else:
        summary = {
            "manifest": str((args.output_dir / "stage1_observation_uq_sequence.json").resolve()),
            "shape": manifest["uncertainty"]["shape"],
            "component_shape": manifest["uncertainty"]["component_shape"],
            "latest_frame_index": manifest["latest_frame_index"],
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
