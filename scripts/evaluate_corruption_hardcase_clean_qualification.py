#!/usr/bin/env python3
"""Evaluate one terminal clean qualification run against frozen gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_clean_liveness_screen import longest_low_speed_interval
from scripts.summarize_closedloop_safety import find_control_trace, load_records


SCHEMA = "orion.corruption_hardcase_clean_qualification.v1"
PROTOCOL_SCHEMA = "orion.corruption_hardcase_wave1_clean_qualification.v1"
WAVE2_PREREG_SCHEMA = (
    "orion.corruption_hardcase_wave2_clean_qualification_preregistration.v1"
)
WAVE2_Q1_ACTIVATION_SCHEMA = (
    "orion.corruption_hardcase_wave2_clean_q1_activation.v1"
)
Q2_ACTIVATION_SCHEMA = "orion.corruption_hardcase_wave1_clean_q2_activation.v1"
Q1_RESULT_SCHEMA = "orion.corruption_hardcase_wave1_clean_q1_result.v1"
SAFETY_SCHEMA = "orion.closedloop_dynamic_actor_safety.v2"
HARD_INFRACTION_KEYS = (
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "route_dev",
    "vehicle_blocked",
    "scenario_timeouts",
    "route_timeout",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_one(root: Path, pattern: str) -> tuple[Path, dict[str, Any]]:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError("expected one %r below %s, found %d" % (pattern, root, len(paths)))
    return paths[0], json.loads(paths[0].read_text(encoding="utf-8"))


def _infraction_count(record: dict[str, Any], key: str) -> int:
    value = record.get("infractions", {}).get(key, [])
    if isinstance(value, list):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(bool(value))


def _repository_root(protocol_path: Path) -> Path:
    resolved = protocol_path.resolve()
    if resolved.parent.name != "scenario_factory" or resolved.parent.parent.name != "configs":
        raise ValueError("protocol path is not below configs/scenario_factory")
    return resolved.parent.parent.parent


def _load_hash_bound_reference(
    *, repository_root: Path, reference: dict[str, Any], expected_schema: str
) -> tuple[Path, dict[str, Any]]:
    path = Path(str(reference["path"]))
    if not path.is_absolute():
        path = repository_root / path
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if sha256(path) != reference.get("sha256"):
        raise ValueError("hash mismatch for %s" % path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != expected_schema:
        raise ValueError("unexpected schema for %s" % path)
    return path, payload


def _validate_phase_protocol(
    *, protocol_path: Path, protocol: dict[str, Any], phase: str, route_index: int
) -> tuple[str, bool]:
    if phase == "q1":
        if protocol.get("schema") == PROTOCOL_SCHEMA:
            if route_index not in protocol["selection"]["routes"]:
                raise ValueError("route is outside the frozen Q1 selection")
            return str(protocol["qualification_protocol"]["q1"]["run_id"]), False
        if protocol.get("schema") != WAVE2_Q1_ACTIVATION_SCHEMA:
            raise ValueError("unexpected Q1 clean qualification protocol")
        if protocol.get("status") != "authorized_after_user_resume":
            raise ValueError("Wave2 Q1 activation is not authorized")
        authorization = protocol.get("authorization", {})
        if (
            authorization.get("q1_clean_submission") is not True
            or authorization.get("user_resume_recorded") is not True
        ):
            raise ValueError("Wave2 Q1 activation authority is incomplete")
        scope = protocol["scope"]
        if route_index not in scope["routes"]:
            raise ValueError("route is outside the hash-bound Wave2 Q1 scope")
        if scope.get("condition") != "clean_off" or int(
            scope.get("runs_per_route", 0)
        ) != 1:
            raise ValueError("unexpected Wave2 Q1 scope contract")
        repository_root = _repository_root(protocol_path)
        _, base_prereg = _load_hash_bound_reference(
            repository_root=repository_root,
            reference=protocol["base_prereg"],
            expected_schema=WAVE2_PREREG_SCHEMA,
        )
        if route_index not in base_prereg["selection"]["routes"]:
            raise ValueError("Wave2 Q1 route is outside the base preregistration")
        if list(scope["routes"]) != list(base_prereg["selection"]["routes"]):
            raise ValueError("Wave2 Q1 scope differs from the frozen route order")
        if str(scope["run_id"]) != str(
            base_prereg["qualification_protocol"]["q1"]["run_id"]
        ):
            raise ValueError("Wave2 Q1 run ID differs from the base preregistration")
        return str(scope["run_id"]), True
    if phase != "q2":
        raise ValueError("unsupported clean qualification phase")
    if protocol.get("schema") != Q2_ACTIVATION_SCHEMA:
        raise ValueError("unexpected Q2 clean qualification activation")
    scope = protocol["scope"]
    if route_index not in scope["routes"]:
        raise ValueError("route is outside the hash-bound Q2 scope")
    if scope.get("condition") != "clean_off" or int(scope.get("runs_per_route", 0)) != 1:
        raise ValueError("unexpected Q2 scope contract")
    repository_root = _repository_root(protocol_path)
    _, base_protocol = _load_hash_bound_reference(
        repository_root=repository_root,
        reference=protocol["base_protocol"],
        expected_schema=PROTOCOL_SCHEMA,
    )
    if route_index not in base_protocol["selection"]["routes"]:
        raise ValueError("Q2 route is outside the base clean qualification protocol")
    _, q1_result = _load_hash_bound_reference(
        repository_root=repository_root,
        reference=protocol["q1_result"],
        expected_schema=Q1_RESULT_SCHEMA,
    )
    if route_index not in q1_result["decision"]["q2_exact_scope"]:
        raise ValueError("Q2 route was not authorized by the frozen Q1 result")
    return str(scope["run_id"]), False


def _sensor_bundle_timeout_count(run_dir: Path) -> int:
    count = 0
    for path in sorted(run_dir.glob("sensor_queue_diagnostics*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("event") == "sensor_bundle_timeout":
                count += 1
    return count


def evaluate(
    *, run_dir: Path, route_index: int, phase: str, protocol_path: Path
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_run_id, exact_speedometer_required = _validate_phase_protocol(
        protocol_path=protocol_path,
        protocol=protocol,
        phase=phase,
        route_index=route_index,
    )
    manifest_path, manifest = _load_one(run_dir, "manifest.json")
    eval_path, eval_payload = _load_one(run_dir, "eval*.json")
    terminal_records = eval_payload.get("_checkpoint", {}).get("records", [])
    if len(terminal_records) != 1:
        raise RuntimeError("expected one evaluator terminal record")
    terminal = terminal_records[0]
    trace_path = find_control_trace(run_dir)
    records = load_records(trace_path)
    if not records:
        raise RuntimeError("empty clean qualification trace")
    steps = [int(row["step"]) for row in records]
    times = [float(row["sim_time_seconds"]) for row in records]
    longest_stop = longest_low_speed_interval(
        records,
        speed_threshold_mps=0.25,
        ignore_before_seconds=2.0,
    )
    infraction_counts = {
        key: _infraction_count(terminal, key) for key in HARD_INFRACTION_KEYS
    }
    scores = terminal.get("scores", {})
    sensor_bundle_timeout_count = _sensor_bundle_timeout_count(run_dir)
    checks = {
        "route_index_matches": int(manifest.get("pilot_route_index")) == route_index,
        "run_id_matches_phase": manifest.get("pilot_run_id") == expected_run_id,
        "condition_clean_off": manifest.get("pilot_condition") == "clean_off",
        "uq_mode_none": manifest.get("orion_closedloop_uq_mode") == "none",
        "conditioning_none": manifest.get("orion_closedloop_conditioning") == "none",
        "risk_mode_off": manifest.get("orion_closedloop_risk_mode") == "off",
        "planning_response_off": manifest.get("orion_planning_response_mode") == "off",
        "stage2_source_disabled": (
            manifest.get("orion_stage2_spatial_uq_source") == "disabled"
        ),
        "legacy_density_disabled": (
            manifest.get("orion_enable_legacy_density_uq") == "0"
        ),
        "exact_frame_speedometer_enabled": (
            not exact_speedometer_required
            or manifest.get("orion_exact_frame_speedometer") == "1"
        ),
        "strict_sensor_queue_diagnostics_enabled": (
            not exact_speedometer_required
            or manifest.get("orion_sensor_queue_diagnostics") == "1"
        ),
        "no_sensor_bundle_timeout": sensor_bundle_timeout_count == 0,
        "evaluator_entry_finished": eval_payload.get("entry_status") == "Finished",
        "terminal_completed": terminal.get("status") == "Completed",
        "route_completion_100": float(scores.get("score_route", -1.0)) == 100.0,
        "zero_hard_infractions": all(value == 0 for value in infraction_counts.values()),
        "trace_steps_contiguous": steps == list(range(steps[0], steps[0] + len(steps))),
        "trace_times_strictly_monotonic": all(
            right > left for left, right in zip(times, times[1:])
        ),
        "safety_telemetry_complete": all(
            row.get("closedloop_safety", {}).get("available") is True
            and row.get("closedloop_safety", {}).get("schema") == SAFETY_SCHEMA
            for row in records
        ),
        "liveness_at_most_8s": float(longest_stop["duration_seconds"]) <= 8.0,
    }
    passed = all(checks.values())
    return {
        "schema": SCHEMA,
        "status": "clean_qualified" if passed else "clean_rejected",
        "phase": phase,
        "route_index": route_index,
        "run_dir": str(run_dir.resolve()),
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": sha256(protocol_path),
        },
        "source_files": {
            "manifest": str(manifest_path.resolve()),
            "evaluator": str(eval_path.resolve()),
            "control_trace": str(trace_path.resolve()),
        },
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "hard_infraction_counts": infraction_counts,
        "sensor_bundle_timeout_count": sensor_bundle_timeout_count,
        "exact_frame_speedometer_required": exact_speedometer_required,
        "route_completion_percent": float(scores.get("score_route", -1.0)),
        "trace_frames": len(records),
        "longest_low_speed_interval": longest_stop,
        "passed_clean_qualification": passed,
        "qualified_for_next_clean_repeat": passed if phase == "q1" else False,
        "qualified_for_corruption_screen": passed if phase == "q2" else False,
        "automatic_scope_locks": {
            "q2_submission": False,
            "corruption_submission": False,
            "heldout_confirmation": False,
            "stage2p": False,
            "formal_200_route_evaluation": False,
        },
        "claim_boundary": (
            "One clean Q1 replicate only. Passing Q1 permits a separately frozen "
            "Q2 clean repeat, not a corruption or method claim."
            if phase == "q1"
            else "One clean Q2 replicate only. Passing Q2 establishes stable-clean "
            "eligibility only when its frozen Q1 replicate also passed; corruption "
            "submission still requires a separate prospective result amendment."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--route-index", type=int, required=True)
    parser.add_argument("--phase", choices=("q1", "q2"), default="q1")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite clean qualification report")
    report = evaluate(
        run_dir=args.run_dir,
        route_index=args.route_index,
        phase=args.phase,
        protocol_path=args.protocol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": report["status"],
        "failed_checks": report["failed_checks"],
    }, indent=2, sort_keys=True))
    return 0 if report["passed_clean_qualification"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
