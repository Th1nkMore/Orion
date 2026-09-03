#!/usr/bin/env python3
"""Replay saved closed-loop cameras through the frozen Stage-1 U adapter.

This utility is deliberately diagnostic.  It loads only ORION's frozen
EVAViT image backbone and the frozen task-agnostic observation adapter.  It
does not load the language/planning stack and never modifies closed-loop
control.  The output retains the complete per-frame, per-view spatial map so
that camera and ground-plane-projected BEV visualizations can be rendered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_closedloop_stage1_uq_sequence import (  # noqa: E402
    CAMERA_ORDER,
    preprocess_saved_camera,
    robust_component_calibration,
)
from scripts.extract_paired_spatial_features import (  # noqa: E402
    _build_real_backbone,
    _extract_tokens,
)
from scripts.scenario_factory_lib import sha256_file  # noqa: E402
from uq_estimator.counterfactual_evidence import EVIDENCE_COMPONENTS  # noqa: E402
from uq_estimator.online_observation_uq import (  # noqa: E402
    load_frozen_pairwise_adapter,
)


SCHEMA = "orion.stage1_uq_visualization_replay.v1"
CAMERA_DIRECTORIES = {
    "CAM_FRONT": ("rgb_front_model_input", "rgb_front"),
    "CAM_FRONT_LEFT": ("rgb_front_left",),
    "CAM_FRONT_RIGHT": ("rgb_front_right",),
    "CAM_BACK": ("rgb_back",),
    "CAM_BACK_LEFT": ("rgb_back_left",),
    "CAM_BACK_RIGHT": ("rgb_back_right",),
}


def _scenario_dir(run_dir: Path) -> Path:
    traces = sorted(run_dir.glob("records_*/**/control_trace.jsonl"))
    if len(traces) != 1:
        raise ValueError(
            f"run must contain exactly one control trace, found {len(traces)}"
        )
    return traces[0].parent


def _camera_roots(scenario_dir: Path) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for camera in CAMERA_ORDER:
        candidates = [scenario_dir / value for value in CAMERA_DIRECTORIES[camera]]
        root = next(
            (value for value in candidates if value.is_dir() and any(value.glob("*.png"))),
            None,
        )
        if root is None:
            raise ValueError(f"saved camera stream missing for {camera}")
        roots[camera] = root
    return roots


def _aligned_frame_indices(roots: dict[str, Path]) -> list[int]:
    per_view = [
        {int(path.stem) for path in root.glob("*.png")}
        for root in roots.values()
    ]
    shared = sorted(set.intersection(*per_view))
    if not shared:
        raise ValueError("saved camera streams have no aligned frames")
    return shared


def _frame_tensor(
    roots: dict[str, Path], frame_index: int
) -> tuple[torch.Tensor, list[dict[str, str]]]:
    arrays = []
    provenance = []
    for camera in CAMERA_ORDER:
        path = roots[camera] / f"{frame_index:04d}.png"
        arrays.append(preprocess_saved_camera(path))
        provenance.append(
            {
                "view": camera,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        )
    return torch.from_numpy(np.stack(arrays)), provenance


def extract(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    scenario_dir = _scenario_dir(run_dir)
    camera_roots = _camera_roots(scenario_dir)
    available = _aligned_frame_indices(camera_roots)
    selected = available[:: args.frame_stride]
    if args.max_frames is not None:
        selected = selected[: args.max_frames]
    baseline_frames = [
        value
        for value in selected
        if args.baseline_start_frame <= value < args.baseline_end_frame
    ]
    if len(baseline_frames) < 4:
        raise ValueError("at least four selected calibration-prefix frames are required")

    backbone_args = SimpleNamespace(
        config=args.orion_config.resolve(), checkpoint=args.orion_checkpoint.resolve()
    )
    _, backbone, backbone_metadata = _build_real_backbone(backbone_args)
    adapter, adapter_metadata = load_frozen_pairwise_adapter(
        args.adapter_checkpoint.resolve(),
        expected_sha256=args.adapter_checkpoint_sha256,
        device="cuda",
    )

    previous_features = None
    raw_outputs = []
    frame_inventory = []
    for ordinal, frame_index in enumerate(selected, start=1):
        images, camera_files = _frame_tensor(camera_roots, frame_index)
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
        frame_inventory.append(
            {
                "saved_frame_index": frame_index,
                "camera_files": camera_files,
            }
        )
        print(
            "[Stage1UQVis] frame=%d ordinal=%d/%d grid=%dx%d"
            % (frame_index, ordinal, len(selected), height, width),
            flush=True,
        )

    raw = np.stack(raw_outputs).astype(np.float32)
    baseline_indices = [selected.index(value) for value in baseline_frames]
    normalized_components, calibration = robust_component_calibration(
        raw,
        baseline_indices,
        relative_scale_floor=args.relative_scale_floor,
        absolute_scale_floor=args.absolute_scale_floor,
        z_center=args.z_center,
    )
    uncertainty = normalized_components.mean(axis=-1).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stage1_uq_visualization_replay.npz"
    np.savez_compressed(
        output_path,
        frame_indices=np.asarray(selected, dtype=np.int32),
        uncertainty=uncertainty,
        uncertainty_components=normalized_components,
        raw_uncertainty_components=raw,
    )
    trace_path = scenario_dir / "control_trace.jsonl"
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "claim_boundary": (
            "Task-agnostic observation-evidence visualization only. Values are "
            "not probabilities, task risk, or a closed-loop safety result."
        ),
        "control_influence": False,
        "run_dir": str(run_dir),
        "scenario_dir": str(scenario_dir),
        "control_trace": {
            "path": str(trace_path.resolve()),
            "sha256": sha256_file(trace_path),
            "used_as_model_input": False,
        },
        "camera_order": list(CAMERA_ORDER),
        "camera_roots": {key: str(value.resolve()) for key, value in camera_roots.items()},
        "component_names": list(EVIDENCE_COMPONENTS),
        "frame_indices": selected,
        "frame_count": len(selected),
        "frame_inventory": frame_inventory,
        "calibration": calibration,
        "checkpoint": {
            "path": str(args.adapter_checkpoint.resolve()),
            "sha256": adapter_metadata["sha256"],
            "schema": adapter_metadata["schema_version"],
        },
        "backbone": {
            **backbone_metadata,
            "orion_checkpoint_sha256": sha256_file(args.orion_checkpoint),
            "loads_orion_llm": False,
            "loads_orion_planning_head": False,
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "uncertainty_shape": list(uncertainty.shape),
            "component_shape": list(normalized_components.shape),
            "raw_component_shape": list(raw.shape),
        },
        "forbidden_inputs": {
            "route": False,
            "actor_geometry": False,
            "ttc": False,
            "collision_outcome": False,
            "corruption_metadata": False,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--orion-config", type=Path, required=True)
    parser.add_argument("--orion-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-start-frame", type=int, default=2)
    parser.add_argument("--baseline-end-frame", type=int, default=8)
    parser.add_argument("--relative-scale-floor", type=float, default=0.05)
    parser.add_argument("--absolute-scale-floor", type=float, default=0.001)
    parser.add_argument("--z-center", type=float, default=4.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Stage-1 U visualization replay requires CUDA")
    if args.frame_stride <= 0 or (args.max_frames is not None and args.max_frames <= 0):
        raise SystemExit("frame-stride and max-frames must be positive")
    manifest = extract(args)
    print(
        json.dumps(
            {
                "manifest": str((args.output_dir / "manifest.json").resolve()),
                "frame_count": manifest["frame_count"],
                "shape": manifest["output"]["uncertainty_shape"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
