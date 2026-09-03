#!/usr/bin/env python3
"""Build the explicitly non-claim Route147 Stage-2 optimization smoke.

The captured learned Stage-1 map is used only to attest shapes and provenance.
Three controlled Stage-2 inputs are then built over the pre-registered event
window: an on-path observation-region oracle, an equal-area off-path oracle,
and exact zero UQ.  These are mechanism-training controls, not uncertainty
ground truth and not evidence that the learned adapter is correct.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uq_estimator.privileged_yield_labels import SCHEMA_VERSION as LABEL_SCHEMA
from uq_estimator.stage2_artifact_capture import (
    ARTIFACT_INDEX_SCHEMA,
    Stage2ArtifactWriter,
    sha256_file,
)
from uq_estimator.stage2_task_training import build_stage2_manifest


REPORT_SCHEMA = "orion.route147-stage2-optimization-smoke/v1"
VARIANTS = ("onpath_oracle", "offpath_oracle", "zero_uq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-artifact-index", required=True)
    parser.add_argument("--capture-trace", required=True)
    parser.add_argument("--event-spec", required=True)
    parser.add_argument("--mechanism-report", required=True)
    parser.add_argument("--failure-induction-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stride-steps", type=int, default=10)
    parser.add_argument("--oracle-strength", type=float, default=1.0)
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_tensor(path: str | Path, key: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    value = payload[key] if isinstance(payload, dict) else payload
    if not isinstance(value, torch.Tensor):
        raise ValueError("artifact does not contain tensor %s" % key)
    return value.detach().float()


def _normalized_region_to_slices(region, height: int, width: int):
    if not isinstance(region, list) or len(region) != 4:
        raise ValueError("event region must be normalized [x0,y0,x1,y1]")
    x0, y0, x1, y1 = (float(value) for value in region)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("event region lies outside the normalized image")
    column_start = min(width - 1, max(0, int(math.floor(x0 * width))))
    column_end = min(width, max(column_start + 1, int(math.ceil(x1 * width))))
    row_start = min(height - 1, max(0, int(math.floor(y0 * height))))
    row_end = min(height, max(row_start + 1, int(math.ceil(y1 * height))))
    return slice(row_start, row_end), slice(column_start, column_end)


def _identity_label(response: dict[str, Any], *, variant: str) -> dict[str, Any]:
    base = response["base_plan_cumulative_m"]
    conflict = response["conflict"]
    original = response["yield_label"]
    zeros = [[0.0, 0.0] for _ in base]
    return {
        "schema_version": LABEL_SCHEMA,
        "source": {},
        "supervision_contract": {
            "stage": "stage2_task_risk",
            "uses_observation_uq_target": False,
            "uses_density_uq": False,
            "uses_corruption_label": False,
            "source": "privileged_task_response_x_controlled_spatial_uq_location",
        },
        "base_plan_cumulative_m": base,
        "conflict": conflict,
        "yield_label": {
            "state": "go",
            "state_index": 0,
            "conflict_present": bool(original.get("conflict_present")),
            "imminent_conflict": bool(original.get("imminent_conflict")),
            "clearance_elapsed_seconds": 0.0,
            "release_elapsed_seconds": 0.0,
            "stop_path_distance_m": None,
            "reason": "%s_requires_exact_identity" % variant,
        },
        "safe_target_cumulative_m": base,
        "trajectory_residual_m": zeros,
    }


def _response_label(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": LABEL_SCHEMA,
        "source": {},
        "supervision_contract": {
            "stage": "stage2_task_risk",
            "uses_observation_uq_target": False,
            "uses_density_uq": False,
            "uses_corruption_label": False,
            "source": "exact_braking_aware_privileged_response_record",
        },
        "base_plan_cumulative_m": response["base_plan_cumulative_m"],
        "conflict": response["conflict"],
        "yield_label": response["yield_label"],
        "safe_target_cumulative_m": response["target_plan_cumulative_m"],
        "trajectory_residual_m": response["trajectory_residual_m"],
    }


def main() -> None:
    args = parse_args()
    if args.stride_steps <= 0:
        raise ValueError("stride-steps must be positive")
    if not math.isfinite(args.oracle_strength) or args.oracle_strength <= 0:
        raise ValueError("oracle-strength must be finite and positive")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite optimization-smoke directory")

    capture_path = Path(args.capture_artifact_index).resolve()
    trace_path = Path(args.capture_trace).resolve()
    event_path = Path(args.event_spec).resolve()
    mechanism_path = Path(args.mechanism_report).resolve()
    failure_path = Path(args.failure_induction_report).resolve()
    capture = _load_json(capture_path)
    event_spec = _load_json(event_path)
    mechanism = _load_json(mechanism_path)
    failure = _load_json(failure_path)
    if capture.get("schema_version") != ARTIFACT_INDEX_SCHEMA:
        raise ValueError("unexpected capture artifact schema")
    if capture.get("uq_source") != "learned_stage1_spatial_uq":
        raise ValueError("source capture must come from the new frozen Stage-1 adapter")
    if mechanism.get("primary_success") is not True:
        raise ValueError("Route147 planning mechanism did not pass")
    failure_pass = (failure.get("decision") or {}).get("failure_induction_pass")
    if failure_pass is None:
        failure_pass = (failure.get("gate") or {}).get("failure_induction_pass")
    if failure_pass is None:
        failure_pass = failure.get("failure_induction_pass")
    if failure_pass is not False:
        raise ValueError("this builder expects the disclosed failed glare-induction screen")
    event = event_spec["event"]
    if event.get("camera") != "CAM_FRONT":
        raise ValueError("Route147 event camera differs from CAM_FRONT")
    camera_order = capture.get("camera_order") or []
    if "CAM_FRONT" not in camera_order:
        raise ValueError("capture lacks the pre-registered front camera")
    front_index = camera_order.index("CAM_FRONT")
    # Do not leave a misleading empty result directory when any immutable
    # provenance input is missing or fails its scientific gate.
    output_dir.mkdir(parents=True)

    trace_by_step = {}
    with trace_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            trace_by_step[int(row["step"])] = row
    artifacts_by_step = {
        int(record["step"]): record for record in capture.get("records", [])
    }
    common_steps = sorted(set(trace_by_step).intersection(artifacts_by_step))
    selected_steps = [step for step in common_steps if step % args.stride_steps == 0]
    if not selected_steps:
        raise ValueError("capture and trace have no aligned 2-Hz samples")
    start_row = next((
        trace_by_step[step] for step in sorted(trace_by_step)
        if float(trace_by_step[step].get("route_progress") or 0.0)
        >= float(event["start_progress"])
    ), None)
    if start_row is None:
        raise ValueError("capture never reaches the frozen event progress")
    event_start_seconds = float(start_row["sim_time_seconds"])
    event_end_seconds = event_start_seconds + float(event["duration_seconds"])

    first_artifact = artifacts_by_step[selected_steps[0]]
    first_uq = _load_tensor(first_artifact["observation_uq_path"], "observation_uq")
    if first_uq.ndim != 4 or first_uq.shape[-1] != 3:
        raise ValueError("captured Stage-1 map shape differs")
    views, height, width, components = map(int, first_uq.shape)
    if views != len(camera_order):
        raise ValueError("captured map view count differs from camera order")
    regions = {
        "onpath_oracle": event["on_path_region"],
        "offpath_oracle": event["off_path_region"],
    }
    slices = {
        name: _normalized_region_to_slices(region, height, width)
        for name, region in regions.items()
    }
    variant_summaries = {}
    combined_records = []
    for variant in VARIANTS:
        route_group = "%s/%s" % (capture["route_group"], variant)
        variant_root = output_dir / variant
        writer = Stage2ArtifactWriter(
            variant_root / "artifacts",
            route_group=route_group,
            uq_source="oracle_spatial_uq",
            camera_order=camera_order,
        )
        labels_path = variant_root / "labels.jsonl"
        variant_root.mkdir(parents=True, exist_ok=True)
        state_counts = {state: 0 for state in ("go", "prepare_yield", "hold", "release")}
        active_samples = 0
        with labels_path.open("w") as label_handle:
            for step in selected_steps:
                trace = trace_by_step[step]
                response = trace.get("planning_response")
                if not isinstance(response, dict):
                    raise ValueError("capture trace lacks privileged response at step %d" % step)
                artifact = artifacts_by_step[step]
                context = _load_tensor(
                    artifact["planning_context_path"], "planning_context"
                )
                task = _load_tensor(
                    artifact["task_context_path"], "task_context"
                )
                timestamp = float(trace["sim_time_seconds"])
                active = event_start_seconds <= timestamp < event_end_seconds
                oracle = torch.zeros((views, height, width, components))
                if active and variant != "zero_uq":
                    row_slice, column_slice = slices[variant]
                    oracle[
                        front_index, row_slice, column_slice, :
                    ] = float(args.oracle_strength)
                    active_samples += 1
                writer.write(
                    step=step,
                    planning_context=context,
                    task_context=task,
                    observation_uq=oracle,
                    metadata={
                        "source_capture_step": step,
                        "event_active": active,
                        "oracle_variant": variant,
                        "synthetic_intervention_is_uq_truth": False,
                    },
                )
                if variant == "onpath_oracle" and active:
                    label = _response_label(response)
                else:
                    label = _identity_label(response, variant=variant)
                label["source"] = {
                    "step": step,
                    "sim_time_seconds": timestamp,
                    "route_progress": trace.get("route_progress"),
                    "capture_trace_path": str(trace_path),
                    "oracle_variant": variant,
                    "event_active": active,
                }
                state_counts[label["yield_label"]["state"]] += 1
                label_handle.write(json.dumps(label, sort_keys=True) + "\n")
        index_path = writer.finalize()
        manifest_path = variant_root / "stage2_manifest.jsonl"
        manifest_summary = build_stage2_manifest(
            labels_path,
            index_path,
            manifest_path,
            route_group=route_group,
            mechanism_report_path=mechanism_path,
        )
        records = [
            json.loads(line) for line in manifest_path.read_text().splitlines()
            if line.strip()
        ]
        combined_records.extend(records)
        variant_summaries[variant] = {
            "route_group": route_group,
            "sample_count": len(records),
            "event_active_map_samples": active_samples,
            "state_counts": state_counts,
            "artifact_index_path": str(index_path),
            "artifact_index_sha256": sha256_file(index_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_summary": manifest_summary,
        }
    combined_path = output_dir / "stage2_optimization_smoke_manifest.jsonl"
    with combined_path.open("w") as handle:
        for record in combined_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "optimization_smoke_nonclaim",
        "closed_loop_eligible": False,
        "claim_boundary": (
            "Route147 glare failed the failure-induction gate. These controlled "
            "oracle maps only test whether Stage 2 can learn location-dependent "
            "response while preserving off-path and zero-UQ identity."
        ),
        "density_uq_used": False,
        "learned_stage1_used_for_control": False,
        "source_capture_uses_new_stage1_adapter": True,
        "source_stage1_checkpoint_sha256": capture.get(
            "stage1_checkpoint_sha256"
        ),
        "source_capture_artifact_index": str(capture_path),
        "source_capture_artifact_index_sha256": sha256_file(capture_path),
        "source_trace": str(trace_path),
        "source_trace_sha256": sha256_file(trace_path),
        "event_spec": str(event_path),
        "event_spec_sha256": sha256_file(event_path),
        "failure_induction_report": str(failure_path),
        "failure_induction_report_sha256": sha256_file(failure_path),
        "failure_induction_pass": False,
        "mechanism_report": str(mechanism_path),
        "mechanism_report_sha256": sha256_file(mechanism_path),
        "event_trigger_step": int(start_row["step"]),
        "event_window_seconds": [event_start_seconds, event_end_seconds],
        "spatial_map_shape": [views, height, width, components],
        "sample_stride_steps": args.stride_steps,
        "combined_manifest_path": str(combined_path),
        "combined_manifest_sha256": sha256_file(combined_path),
        "combined_sample_count": len(combined_records),
        "variants": variant_summaries,
    }
    report_path = output_dir / "build_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
