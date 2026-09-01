#!/usr/bin/env python3
"""Verify that a clean adapter trace was complete and control-invariant."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts.evaluate_native_collision_discovery import _load_terminal
from scripts.summarize_closedloop_safety import find_control_trace, load_records


SCHEMA_VERSION = "orion.clean_pairwise_trace_verification.v1"
CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=1e-12
        )
    except (TypeError, ValueError):
        return False


def validate_observation_record(
    observation: dict[str, Any], expected_checkpoint_sha256: str
) -> list[str]:
    """Return contract violations for one all-view observation record."""

    errors: list[str] = []
    if observation.get("checkpoint_sha256") != expected_checkpoint_sha256:
        errors.append("checkpoint_sha256")
    if tuple(observation.get("camera_order") or ()) != CAMERA_ORDER:
        errors.append("camera_order")
    grids = observation.get("pooled_grids")
    if not isinstance(grids, list) or len(grids) != len(CAMERA_ORDER):
        errors.append("pooled_grids_view_count")
        return errors
    for view_index, grid in enumerate(grids):
        if not isinstance(grid, list) or len(grid) != 10:
            errors.append(f"pooled_grid_{view_index}_height")
            continue
        for row in grid:
            if (
                not isinstance(row, list)
                or len(row) != 10
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in row
                )
            ):
                errors.append(f"pooled_grid_{view_index}_row")
                break
    return errors


def verify(run_dir: Path, preregistration: Path) -> dict[str, Any]:
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    trace_path = find_control_trace(run_dir)
    records = load_records(trace_path)
    _, eval_payload, terminal = _load_terminal(run_dir)
    expected_checkpoint = prereg["frozen_adapter"]["sha256"]

    manifest_checks = {
        "condition_clean_pairwise_trace": (
            manifest.get("pilot_condition") == "clean_pairwise_trace"
        ),
        "corruption_none": not bool(
            manifest.get("orion_closedloop_corruption")
        ),
        "risk_mode_off": (
            manifest.get("orion_closedloop_risk_mode") == "off"
        ),
        "uq_mode_none": (
            manifest.get("orion_closedloop_uq_mode") == "none"
        ),
        "legacy_density_uq_disabled": (
            manifest.get("orion_enable_legacy_density_uq") == "0"
        ),
        "checkpoint_hash_frozen": (
            manifest.get("orion_observation_uq_checkpoint_sha256")
            == expected_checkpoint
        ),
        "source_hashes_frozen": (
            manifest.get("source_sha256") == prereg["frozen_source_hashes"]
        ),
    }

    steps = [int(row["step"]) for row in records]
    trace_checks = {
        "steps_contiguous": steps
        == list(range(steps[0], steps[0] + len(steps))),
        "observation_uq_every_frame": all(
            isinstance(row.get("observation_uq"), dict) for row in records
        ),
        "legacy_density_score_absent_every_frame": all(
            row.get("density_uq_score") is None for row in records
        ),
        "corruption_never_active": all(
            row.get("corruption_active") is False for row in records
        ),
        "oracle_never_active": all(
            row.get("oracle_event_active") is False for row in records
        ),
        "risk_mode_always_off": all(
            (row.get("risk") or {}).get("mode") == "off" for row in records
        ),
        "risk_score_never_applied": all(
            (row.get("risk") or {}).get("applied_score") is None
            for row in records
        ),
        "risk_intensity_always_zero": all(
            _same_number((row.get("risk") or {}).get("intensity"), 0.0)
            for row in records
        ),
        "throttle_exact_passthrough": all(
            _same_number(
                (row.get("risk") or {}).get("throttle"),
                (row.get("risk") or {}).get("base_throttle"),
            )
            for row in records
        ),
        "brake_exact_passthrough": all(
            _same_number(
                (row.get("risk") or {}).get("brake"),
                (row.get("risk") or {}).get("base_brake"),
            )
            for row in records
        ),
    }

    observation_errors: dict[int, list[str]] = {}
    for row in records:
        observation = row.get("observation_uq")
        if not isinstance(observation, dict):
            observation_errors[int(row["step"])] = ["observation_uq_missing"]
            continue
        errors = validate_observation_record(observation, expected_checkpoint)
        if errors:
            observation_errors[int(row["step"])] = errors
    observation_checks = {
        "all_view_grid_contract_every_frame": not observation_errors,
        "baseline_reaches_minimum_frames": max(
            int(
                ((row.get("observation_uq") or {}).get("calibration") or {}).get(
                    "baseline_count", 0
                )
            )
            for row in records
        )
        >= int(prereg["causal_online_calibration"]["minimum_baseline_frames"]),
        "baseline_frozen_by_end": bool(
            ((records[-1].get("observation_uq") or {}).get("calibration") or {}).get(
                "baseline_frozen"
            )
        ),
    }
    endpoint_checks = {
        "evaluator_eligible": bool(eval_payload.get("eligible")),
        "terminal_status_recorded": bool(str(terminal.get("status", "")).strip()),
        "route_completion_recorded": (
            (terminal.get("scores") or {}).get("score_route") is not None
        ),
    }
    all_checks = {
        **manifest_checks,
        **trace_checks,
        **observation_checks,
        **endpoint_checks,
    }
    passed = all(all_checks.values())
    return {
        "schema": SCHEMA_VERSION,
        "run_dir": str(run_dir.resolve()),
        "preregistration": str(preregistration.resolve()),
        "trace_path": str(trace_path.resolve()),
        "trace_frames": len(records),
        "checks": all_checks,
        "observation_contract_errors": observation_errors,
        "terminal": {
            "status": terminal.get("status"),
            "scores": terminal.get("scores"),
            "infractions": terminal.get("infractions"),
        },
        "verification_passed": passed,
        "control_intervention": False if passed else None,
        "claim_boundary": prereg["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite clean pairwise verification")
    report = verify(args.run_dir, args.preregistration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "trace_frames": report["trace_frames"],
                "verification_passed": report["verification_passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["verification_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
