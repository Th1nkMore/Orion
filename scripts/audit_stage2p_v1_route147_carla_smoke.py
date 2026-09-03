#!/usr/bin/env python3
"""Audit the single Route147 controlled-K CARLA vertical-slice smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


COLLISION_KEYS = (
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _trace(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("control trace is empty or malformed")
    return rows


def _residual(response: dict[str, Any]) -> list[list[float]]:
    value = response.get("trajectory_residual_m")
    if (
        not isinstance(value, list)
        or len(value) != 6
        or any(not isinstance(row, list) or len(row) != 2 for row in value)
    ):
        raise ValueError("Stage2-P residual shape differs")
    flattened = [number for row in value for number in row]
    if not all(isinstance(number, (int, float)) and math.isfinite(number) for number in flattened):
        raise ValueError("Stage2-P residual is non-finite")
    return [[float(number) for number in row] for row in value]


def _count(infractions: dict[str, Any], keys: tuple[str, ...]) -> int:
    return sum(
        len(infractions.get(key, []))
        for key in keys
        if isinstance(infractions.get(key, []), list)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite CARLA slice audit")
    prereg = _read(args.preregistration)
    submission = _read(args.submission)
    manifest_path = args.run_dir / "manifest.json"
    evaluation_path = args.run_dir / "eval_orion_traj_0.json"
    traces = list(args.run_dir.glob("records_orion_traj_0/*/control_trace.jsonl"))
    if not traces and (args.run_dir / "control_trace.jsonl").is_file():
        traces = [args.run_dir / "control_trace.jsonl"]
    if not manifest_path.is_file() or not evaluation_path.is_file() or len(traces) != 1:
        raise FileNotFoundError("CARLA slice terminal artifacts are incomplete")
    manifest = _read(manifest_path)
    evaluation = _read(evaluation_path)
    rows = _trace(traces[0])

    expected_environment = {
        "pilot_condition": "stage2p_controlled_k_smoke",
        "pilot_route_index": "147",
        "pilot_variant": "hazard",
        "orion_effective_conditioning": "controlled_k_to_stage2p_trajectory_response",
        "orion_enable_legacy_density_uq": "0",
        "orion_closedloop_corruption": "",
        "orion_closedloop_risk_mode": "off",
        "orion_planning_response_mode": "off",
        "orion_stage2_spatial_uq_source": "external_oracle",
        "orion_stage1_spatial_uq_checkpoint": "",
        "orion_observation_uq_checkpoint": "",
        "orion_stage2_engineering_smoke": "1",
        "orion_stage2_external_k_start_progress": "0.32",
        "orion_stage2_external_k_duration_seconds": "3.0",
        "orion_stage2_external_k_camera": "CAM_FRONT",
        "orion_stage2_external_k_region": "0.58,0.32,1.0,0.95",
        "orion_stage2_external_k_strength": "1.0",
        "orion_stage2_external_k_grid_size": "40",
        "orion_stage2_task_checkpoint_sha256": prereg["checkpoint"]["sha256"],
        "route_sha256": prereg["route"]["sha256"],
    }
    environment_checks = {
        key: manifest.get(key) == expected
        for key, expected in expected_environment.items()
    }
    implementation_paths = {
        "agent_config": "adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py",
        "orion_head": "mmcv/models/dense_heads/orion_head.py",
        "orion_detector": "mmcv/models/detectors/orion.py",
        "carla_agent": "team_code/orion_b2d_agent.py",
        "stage2p_module": "uq_estimator/stage2p_task_risk_trajectory.py",
        "closed_loop_runner": "scripts/run_closedloop_uq_pilot.sh",
    }
    implementation_checks = {
        name: manifest.get("source_sha256", {}).get(relative)
        == prereg["implementation_sha256"][name]
        for name, relative in implementation_paths.items()
    }

    active_indices: list[int] = []
    inactive_indices: list[int] = []
    lateral_max = 0.0
    longitudinal_max = 0.0
    active_nonzero = 0
    inactive_exact_zero = True
    active_gates: list[float] = []
    ttc_values: list[float] = []
    for index, row in enumerate(rows):
        response = row.get("planning_response")
        if not isinstance(response, dict):
            raise ValueError("live trace lacks Stage2-P planning response")
        if (
            response.get("mode") != "stage2p_controlled_k_engineering_smoke"
            or response.get("external_k_camera") != "CAM_FRONT"
            or response.get("external_k_region") != [0.58, 0.32, 1.0, 0.95]
            or response.get("external_k_strength") != 1.0
            or response.get("formal_stage2p_ready") is not False
            or response.get("closed_loop_safety_claim") is not False
            or row.get("corruption_active") is not False
            or row.get("risk", {}).get("mode") != "off"
        ):
            raise ValueError("live Stage2-P trace contract differs")
        residual = _residual(response)
        lateral_max = max(lateral_max, *(abs(item[0]) for item in residual))
        longitudinal_max = max(
            longitudinal_max, *(abs(item[1]) for item in residual)
        )
        maximum = max(abs(number) for item in residual for number in item)
        gate = float(response.get("global_gate"))
        if not math.isfinite(gate):
            raise ValueError("Stage2-P gate is non-finite")
        if response.get("external_k_active") is True:
            active_indices.append(index)
            active_gates.append(gate)
            active_nonzero += int(maximum > 0.0)
        else:
            inactive_indices.append(index)
            inactive_exact_zero = inactive_exact_zero and maximum == 0.0 and gate == 0.0
        safety = row.get("closedloop_safety") or {}
        value = safety.get("min_obb_collision_ttc_seconds")
        if isinstance(value, (int, float)) and math.isfinite(value):
            ttc_values.append(float(value))

    trace_checks = {
        "trace_has_records": bool(rows),
        "active_window_present": bool(active_indices),
        "active_window_contiguous": active_indices
        == list(range(active_indices[0], active_indices[-1] + 1)),
        "active_window_60_frames": len(active_indices) == 60,
        "active_gate_matches_unit_k": all(
            abs(gate - 0.8) <= 2e-7 for gate in active_gates
        ),
        "active_response_nonzero": active_nonzero > 0,
        "pre_window_identity_present": active_indices[0] > 0,
        "post_window_identity_present": active_indices[-1] < len(rows) - 1,
        "all_inactive_responses_exact_zero": inactive_exact_zero,
        "lateral_bound_2m": lateral_max <= 2.0,
        "longitudinal_bound_24m": longitudinal_max <= 24.0,
        "step_sequence_contiguous": [row.get("step") for row in rows]
        == list(range(rows[0]["step"], rows[0]["step"] + len(rows))),
    }
    hard_checks = {
        "preregistration_status": prereg.get("status")
        == "single_route_engineering_smoke_preregistered",
        "submission_job": submission.get("submission", {}).get("job_id")
        == "1123244",
        "submission_exhausted": submission.get("scope", {}).get(
            "remaining_submissions"
        )
        == 0,
        "manifest_environment": all(environment_checks.values()),
        "manifest_implementation_hashes": all(implementation_checks.values()),
        "trace_contract": all(trace_checks.values()),
    }
    if not all(hard_checks.values()):
        raise ValueError("CARLA vertical-slice hard check failed: %s" % hard_checks)

    checkpoint = evaluation.get("_checkpoint", {})
    official_records = checkpoint.get("records", [])
    if len(official_records) != 1:
        raise ValueError("official evaluator record count differs")
    official = official_records[0]
    exceptions = official.get("meta", {}).get("exceptions", [])
    if exceptions:
        raise RuntimeError("CARLA/agent runtime recorded an exception")
    infractions = official.get("infractions", {})
    scores = official.get("scores", {})
    value = {
        "schema": "orion.stage2p_v1_route147_carla_interface_audit.v1",
        "status": "vertical_slice_carla_interface_completed",
        "hard_checks_passed": True,
        "hard_checks": hard_checks,
        "environment_checks": environment_checks,
        "implementation_checks": implementation_checks,
        "trace_checks": trace_checks,
        "runtime": {
            "job_id": "1123244",
            "trace_records": len(rows),
            "active_records": len(active_indices),
            "active_first_step": rows[active_indices[0]]["step"],
            "active_last_step": rows[active_indices[-1]]["step"],
            "maximum_absolute_lateral_residual_m": lateral_max,
            "maximum_absolute_longitudinal_residual_m": longitudinal_max,
            "minimum_recorded_obb_ttc_seconds": min(ttc_values) if ttc_values else None,
            "manifest_sha256": _sha256(manifest_path),
            "control_trace_sha256": _sha256(traces[0]),
            "official_evaluation_sha256": _sha256(evaluation_path),
        },
        "soft_outcomes": {
            "official_status": official.get("status"),
            "route_completion_percent": scores.get("score_route"),
            "composed_score": scores.get("score_composed"),
            "collision_count": _count(infractions, COLLISION_KEYS),
            "min_speed_infraction_count": len(
                infractions.get("min_speed_infractions", [])
            ) if isinstance(infractions.get("min_speed_infractions", []), list) else None,
        },
        "interpretation": {
            "supported": "The hash-bound controlled K reached the live Stage2-P checkpoint, produced a finite bounded trajectory residual that was applied before PID control, and returned to exact zero-K identity after the window.",
            "not_supported": "This external K is an engineering oracle. One route cannot establish learned-U quality, task-relevance generalization, closed-loop benefit or safety."
        },
        "locks": {
            "formal_stage2p_ready": False,
            "closed_loop_safety_ready": False,
            "extra_training": False,
            "automatic_retry": False,
            "locked_test_read": False,
        },
        "claim_boundary": "Terminal audit of one controlled-K CARLA engineering vertical slice; route and safety outcomes remain soft and non-claim."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
