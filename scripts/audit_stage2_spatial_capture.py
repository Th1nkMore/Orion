#!/usr/bin/env python3
"""Audit that a closed-loop capture used the new spatial adapter path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uq_estimator.stage2_artifact_capture import (
    ARTIFACT_INDEX_SCHEMA,
    sha256_file,
)


SCHEMA_VERSION = "orion.stage2-spatial-capture-audit/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-stage1-sha256", required=True)
    parser.add_argument("--expected-warmup-frames", type=int, default=60)
    parser.add_argument("--expected-capture-stride-steps", type=int, default=10)
    parser.add_argument("--require-completed", action="store_true")
    return parser.parse_args()


def _tensor(path: str, key: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    value = payload[key] if isinstance(payload, dict) else payload
    if not isinstance(value, torch.Tensor):
        raise ValueError("%s does not contain %s" % (path, key))
    return value.detach().float()


def main() -> None:
    args = parse_args()
    if min(args.expected_warmup_frames, args.expected_capture_stride_steps) <= 0:
        raise ValueError("expected warmup and capture stride must be positive")
    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    index_path = run_dir / "stage2_artifacts" / "artifact_index.json"
    eval_path = run_dir / "eval_orion_traj_0.json"
    if not all(path.is_file() for path in (manifest_path, index_path, eval_path)):
        raise FileNotFoundError("run lacks manifest, artifact index, or evaluator result")
    manifest = json.loads(manifest_path.read_text())
    index = json.loads(index_path.read_text())
    evaluation = json.loads(eval_path.read_text())
    records = index.get("records") or []
    steps = [int(record["step"]) for record in records]
    checks = {
        "artifact_schema_v2": index.get("schema_version") == ARTIFACT_INDEX_SCHEMA,
        "artifact_count_matches": index.get("record_count") == len(records),
        "steps_follow_capture_stride": steps == list(range(
            0,
            args.expected_capture_stride_steps * len(steps),
            args.expected_capture_stride_steps,
        )),
        "new_spatial_source_learned_adapter": (
            manifest.get("orion_stage2_spatial_uq_source") == "learned_adapter"
            and index.get("uq_source") == "learned_stage1_spatial_uq"
        ),
        "stage1_checkpoint_attested": (
            manifest.get("orion_stage1_spatial_uq_checkpoint_sha256")
            == args.expected_stage1_sha256
            and index.get("stage1_checkpoint_sha256")
            == args.expected_stage1_sha256
        ),
        "legacy_density_disabled": manifest.get(
            "orion_enable_legacy_density_uq"
        ) == "0",
        "legacy_scalar_conditioning_disabled": manifest.get(
            "orion_closedloop_conditioning"
        ) == "none",
        "legacy_scalar_governor_disabled": manifest.get(
            "orion_closedloop_risk_mode"
        ) == "off",
        "effective_conditioning_is_new_path": manifest.get(
            "orion_effective_conditioning"
        ) == "spatial_stage1_to_vlm_stage2:learned_adapter",
        "warmup_contract_matches": int(
            manifest.get("orion_stage1_spatial_uq_warmup_frames", -1)
        ) == args.expected_warmup_frames,
        "capture_stride_contract_matches": int(
            manifest.get("orion_stage2_artifact_stride_steps", -1)
        ) == args.expected_capture_stride_steps,
    }
    tensor_checks = {
        "all_artifact_hashes_match": True,
        "all_tensors_finite": True,
        "all_uq_nonnegative": True,
        "planning_context_shape_256": True,
        "task_context_shape_89": True,
        "uq_three_components": True,
        "first_warmup_frames_exact_zero": True,
    }
    calibrated_frame_maxima = []
    raw_frame_maxima = []
    component_sum = torch.zeros(3)
    component_elements = 0
    for index_in_run, record in enumerate(records):
        for path_key, hash_key in (
            ("planning_context_path", "planning_context_sha256"),
            ("task_context_path", "task_context_sha256"),
            ("observation_uq_path", "observation_uq_sha256"),
            ("raw_observation_uq_path", "raw_observation_uq_sha256"),
        ):
            path = record.get(path_key)
            expected = record.get(hash_key)
            tensor_checks["all_artifact_hashes_match"] &= bool(
                path and expected and Path(path).is_file()
                and sha256_file(path) == expected
            )
        context = _tensor(record["planning_context_path"], "planning_context")
        task = _tensor(record["task_context_path"], "task_context")
        uq = _tensor(record["observation_uq_path"], "observation_uq")
        raw = _tensor(record["raw_observation_uq_path"], "raw_observation_uq")
        tensor_checks["all_tensors_finite"] &= all(
            bool(torch.isfinite(value).all()) for value in (context, task, uq, raw)
        )
        tensor_checks["all_uq_nonnegative"] &= bool(
            (uq >= 0).all() and (raw >= 0).all()
        )
        tensor_checks["planning_context_shape_256"] &= (
            context.ndim == 2 and context.shape[-1] == 256
        )
        tensor_checks["task_context_shape_89"] &= task.shape == (89,)
        tensor_checks["uq_three_components"] &= (
            uq.ndim == 4 and uq.shape[-1] == 3 and raw.shape == uq.shape
        )
        calibrated_frame_maxima.append(float(uq.max()))
        raw_frame_maxima.append(float(raw.max()))
        component_sum += uq.reshape(-1, 3).sum(dim=0)
        component_elements += int(uq.numel() // 3)
        if int(record["step"]) < args.expected_warmup_frames:
            tensor_checks["first_warmup_frames_exact_zero"] &= int(
                torch.count_nonzero(uq)
            ) == 0
    checks.update(tensor_checks)
    post_warmup = [
        value for value, record in zip(calibrated_frame_maxima, records)
        if int(record["step"]) >= args.expected_warmup_frames
    ]
    checks["post_warmup_frames_observed"] = bool(post_warmup)
    checks["post_warmup_spatial_response_nonzero"] = bool(
        post_warmup and max(post_warmup) > 0.0
    )
    checks["raw_adapter_response_nonzero"] = bool(
        raw_frame_maxima and max(raw_frame_maxima) > 0.0
    )
    completed = "Completed" in json.dumps(evaluation)
    if args.require_completed:
        checks["official_route_completed"] = completed
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "claim_boundary": (
            "This audit proves that the attested new spatial Stage-1 adapter "
            "produced same-frame tensors for Stage 2. It does not prove that "
            "the map is task-relevant or that learned-UQ improves safety."
        ),
        "checks": checks,
        "passed": all(checks.values()),
        "record_count": len(records),
        "warmup_frames": args.expected_warmup_frames,
        "capture_stride_steps": args.expected_capture_stride_steps,
        "post_warmup_frame_count": len(post_warmup),
        "calibrated_frame_max": {
            "overall": max(calibrated_frame_maxima, default=None),
            "post_warmup": max(post_warmup, default=None),
        },
        "raw_frame_max": max(raw_frame_maxima, default=None),
        "calibrated_component_mean": (
            (component_sum / component_elements).tolist()
            if component_elements else None
        ),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_index_path": str(index_path),
        "artifact_index_sha256": sha256_file(index_path),
        "evaluation_path": str(eval_path),
        "evaluation_sha256": sha256_file(eval_path),
    }
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite Stage-2 capture audit")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
