#!/usr/bin/env python3
"""Evaluate the preregistered Route197 task-event oracle upper bound."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from scripts.evaluate_native_collision_discovery import (
    COLLISION_KEYS,
    SERIOUS_INFRACTION_KEYS,
    _count_entries,
    _load_terminal,
)
from scripts.summarize_closedloop_safety import (
    find_control_trace,
    load_records,
    summarize_records,
)


SCHEMA_VERSION = "orion.route197_task_event_oracle_result.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rising_edges(values: list[bool]) -> int:
    return sum(value and (index == 0 or not values[index - 1]) for index, value in enumerate(values))


def _same(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def evaluate(
    run_dir: Path,
    preregistration: Path,
    clean_report_path: Path,
) -> dict[str, Any]:
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    clean_report = json.loads(clean_report_path.read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    trace_path = find_control_trace(run_dir)
    records = load_records(trace_path)
    summary = summarize_records(records)
    eval_path, eval_payload, terminal = _load_terminal(run_dir)
    scores = terminal.get("scores") or {}
    infractions = terminal.get("infractions") or {}
    if not isinstance(infractions, dict):
        infractions = {}
    collision_count = _count_entries(infractions, COLLISION_KEYS)
    serious_infraction_count = _count_entries(infractions, SERIOUS_INFRACTION_KEYS)

    frozen = prereg["frozen_condition"]
    window = prereg["window_selection_from_clean_geometry_only"]
    hashes = prereg["frozen_hashes"]
    clean_reference = prereg["clean_reference"]
    manifest_hashes = manifest.get("source_sha256") or {}
    manifest_checks = {
        "condition_native_event_oracle": manifest.get("pilot_condition") == "native_event_oracle",
        "route197": manifest.get("pilot_route_index") == "197",
        "corruption_none": not bool(manifest.get("orion_closedloop_corruption")),
        "uq_mode_none": manifest.get("orion_closedloop_uq_mode") == "none",
        "risk_mode_oracle": manifest.get("orion_closedloop_risk_mode") == "oracle",
        "legacy_density_uq_disabled": (
            manifest.get("orion_enable_legacy_density_uq") == "0"
        ),
        "observation_adapter_absent": not bool(manifest.get("orion_observation_uq_checkpoint")),
        "oracle_start_frozen": _same(
            manifest.get("orion_closedloop_risk_oracle_start_progress"),
            window["oracle_start_route_progress"],
        ),
        "oracle_duration_frozen": _same(
            manifest.get("orion_closedloop_risk_oracle_duration_seconds"),
            window["oracle_duration_seconds"],
        ),
        "risk_threshold_frozen": _same(manifest.get("orion_closedloop_risk_threshold"), frozen["risk_threshold"]),
        "risk_saturation_frozen": _same(manifest.get("orion_closedloop_risk_saturation"), frozen["risk_saturation"]),
        "risk_min_speed_frozen": _same(manifest.get("orion_closedloop_risk_min_speed"), frozen["risk_min_speed_mps"]),
        "risk_max_speed_frozen": _same(manifest.get("orion_closedloop_risk_max_speed"), frozen["risk_max_speed_mps"]),
        "risk_slowdown_margin_frozen": _same(manifest.get("orion_closedloop_risk_slowdown_margin"), frozen["risk_slowdown_margin_mps"]),
        "risk_brake_gain_frozen": _same(manifest.get("orion_closedloop_risk_brake_gain"), frozen["risk_brake_gain"]),
        "risk_max_brake_frozen": _same(manifest.get("orion_closedloop_risk_max_brake"), frozen["risk_max_brake"]),
        "agent_source_hash_frozen": manifest_hashes.get("team_code/orion_b2d_agent.py") == hashes["team_code/orion_b2d_agent.py"],
        "model_config_source_hash_frozen": manifest_hashes.get(
            "adzoo/orion/configs/orion_stage3_agent_uq.py"
        ) == hashes["adzoo/orion/configs/orion_stage3_agent_uq.py"],
        "model_head_source_hash_frozen": manifest_hashes.get(
            "mmcv/models/dense_heads/orion_head.py"
        ) == hashes["mmcv/models/dense_heads/orion_head.py"],
        "runner_source_hash_frozen": manifest_hashes.get("scripts/run_closedloop_uq_pilot.sh") == hashes["scripts/run_closedloop_uq_pilot.sh"],
        "risk_governor_source_hash_frozen": manifest_hashes.get("uq_estimator/risk_governor.py") == hashes["uq_estimator/risk_governor.py"],
        "route_xml_hash_frozen": manifest.get("route_sha256") == hashes["route_197_hazard.xml"],
        "clean_reference_report_hash_frozen": _sha256(clean_report_path)
        == clean_reference["report_sha256"],
        "clean_reference_values_frozen": (
            clean_report.get("endpoint", {}).get("collision_count")
            == clean_reference["official_vehicle_collisions"]
            and _same(
                clean_report.get("endpoint", {}).get("scores", {}).get(
                    "score_route"
                ),
                clean_reference["route_completion_percent"],
            )
            and _same(
                clean_report.get("endpoint", {}).get("scores", {}).get(
                    "score_composed"
                ),
                clean_reference["score_composed"],
            )
        ),
    }

    steps = [int(row["step"]) for row in records]
    active = [bool(row.get("oracle_event_active")) for row in records]
    active_indices = [index for index, value in enumerate(active) if value]
    if active_indices:
        active_times = [float(records[index]["sim_time_seconds"]) for index in active_indices]
        deltas = [right - left for left, right in zip(active_times, active_times[1:]) if right > left]
        step_seconds = statistics.median(deltas) if deltas else 0.05
        active_duration = active_times[-1] - active_times[0] + step_seconds
        first_active = records[active_indices[0]]
    else:
        active_duration = 0.0
        first_active = None

    inactive_rows = [row for row, is_active in zip(records, active) if not is_active]
    active_rows = [row for row, is_active in zip(records, active) if is_active]
    trace_checks = {
        "trace_steps_contiguous": steps == list(range(steps[0], steps[0] + len(steps))),
        "legacy_density_score_absent_every_frame": all(
            row.get("density_uq_score") is None for row in records
        ),
        "observation_adapter_output_absent_every_frame": all(
            row.get("observation_uq") is None for row in records
        ),
        "oracle_window_triggered": bool(active_rows),
        "oracle_window_one_rising_edge": rising_edges(active) == 1,
        "oracle_window_contiguous": active_indices == list(range(active_indices[0], active_indices[-1] + 1)) if active_indices else False,
        "oracle_window_duration_matches": abs(active_duration - float(window["oracle_duration_seconds"])) <= 0.11,
        "oracle_first_active_progress_at_or_after_frozen_start": (
            first_active is not None
            and float(first_active["route_progress"]) >= float(window["oracle_start_route_progress"])
        ),
        "risk_mode_oracle_every_frame": all((row.get("risk") or {}).get("mode") == "oracle" for row in records),
        "inactive_applied_score_zero": all(_same((row.get("risk") or {}).get("applied_score"), 0.0) for row in inactive_rows),
        "inactive_intensity_zero": all(_same((row.get("risk") or {}).get("intensity"), 0.0) for row in inactive_rows),
        "inactive_throttle_passthrough": all(_same((row.get("risk") or {}).get("throttle"), (row.get("risk") or {}).get("base_throttle")) for row in inactive_rows),
        "inactive_brake_passthrough": all(_same((row.get("risk") or {}).get("brake"), (row.get("risk") or {}).get("base_brake")) for row in inactive_rows),
        "active_applied_score_one": all(_same((row.get("risk") or {}).get("applied_score"), 1.0) for row in active_rows),
        "active_intensity_one": all(_same((row.get("risk") or {}).get("intensity"), 1.0) for row in active_rows),
        "active_control_intervention_observed": any(
            not _same((row.get("risk") or {}).get("throttle"), (row.get("risk") or {}).get("base_throttle"))
            or not _same((row.get("risk") or {}).get("brake"), (row.get("risk") or {}).get("base_brake"))
            for row in active_rows
        ),
    }

    endpoint_checks = {
        "runtime_valid_terminal_endpoint": bool(eval_payload.get("eligible")) and bool(str(terminal.get("status", "")).strip()),
        "official_vehicle_collision_count_zero": collision_count == 0,
        "route_completion_100": _same(scores.get("score_route"), 100.0),
        "new_serious_infraction_count_zero": serious_infraction_count == 0,
    }
    invariant_checks = {
        **manifest_checks,
        **{key: value for key, value in trace_checks.items() if key != "active_control_intervention_observed"},
        "active_control_intervention_observed": trace_checks["active_control_intervention_observed"],
        "runtime_valid_terminal_endpoint": endpoint_checks["runtime_valid_terminal_endpoint"],
    }
    primary_success = all(endpoint_checks.values()) and all(invariant_checks.values())
    return {
        "schema": SCHEMA_VERSION,
        "run_dir": str(run_dir.resolve()),
        "preregistration": str(preregistration.resolve()),
        "clean_report": str(clean_report_path.resolve()),
        "eval_path": str(eval_path.resolve()),
        "trace_path": str(trace_path.resolve()),
        "trace_sha256": _sha256(trace_path),
        "manifest_checks": manifest_checks,
        "trace_checks": trace_checks,
        "endpoint_checks": endpoint_checks,
        "oracle_window": {
            "first_active_step": int(first_active["step"]) if first_active else None,
            "first_active_time_seconds": float(first_active["sim_time_seconds"]) if first_active else None,
            "first_active_route_progress": float(first_active["route_progress"]) if first_active else None,
            "frames": len(active_rows),
            "duration_seconds": active_duration,
        },
        "endpoint": {
            "status": terminal.get("status"),
            "scores": scores,
            "official_collision_count": collision_count,
            "official_collision_entries": {key: infractions.get(key, []) for key in COLLISION_KEYS},
            "serious_infraction_count": serious_infraction_count,
        },
        "continuous_safety": summary,
        "clean_reference": {
            "official_collision_count": clean_report["endpoint"]["collision_count"],
            "scores": clean_report["endpoint"]["scores"],
            "continuous_safety": clean_report["continuous_safety"],
        },
        "primary_success": primary_success,
        "decision": (
            "task_response_mechanism_upper_bound_supported"
            if primary_success
            else "do_not_expand_learned_control_matrix"
        ),
        "claim_boundary": prereg["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--clean-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite task-event oracle report")
    report = evaluate(args.run_dir, args.preregistration, args.clean_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "primary_success": report["primary_success"], "decision": report["decision"]}, indent=2, sort_keys=True))
    return 0 if report["primary_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
