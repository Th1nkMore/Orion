#!/usr/bin/env python3
"""Audit a finite pedestrian-crossing planning oracle before A800 use.

This is a mechanism gate, not a closed-loop result.  It replays privileged
actor occupancy against the unmodified ORION plan saved by the valid Route147
clean run and checks braking room, finite release, and supervision provenance.
The historical source trace contains a passive legacy Density score, so the
report states that fact explicitly and never treats it as an input or target.
Any newly authorized run must use the current hard-disabled Density pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping


SCHEMA_VERSION = "orion.route147_bounded_crossing_offline_gate.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonlines(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _actor_category(meta: Mapping, actor_id: int | None) -> str | None:
    if actor_id is None:
        return None
    safety = meta.get("closedloop_safety") or {}
    for actor in safety.get("actors") or []:
        if int(actor.get("actor_id")) == int(actor_id):
            return str(actor.get("category", "")).strip().lower() or None
    return None


def audit(
    *,
    clean_gate_path: Path,
    trace_path: Path,
    labels_path: Path,
    meta_dir: Path,
    certified_deceleration_mps2: float = 3.0,
    reaction_seconds: float = 0.1,
    minimum_braking_margin_m: float = 2.0,
    maximum_response_duration_seconds: float = 3.0,
) -> dict:
    clean_gate = json.loads(clean_gate_path.read_text(encoding="utf-8"))
    trace = load_jsonlines(trace_path)
    labels = load_jsonlines(labels_path)
    trace_by_step = {int(row["step"]): row for row in trace}
    if not trace or not labels:
        raise ValueError("trace and labels must be non-empty")
    if certified_deceleration_mps2 <= 0:
        raise ValueError("certified_deceleration_mps2 must be positive")
    if reaction_seconds < 0:
        raise ValueError("reaction_seconds must be non-negative")

    non_go = [
        row for row in labels if row["yield_label"]["state"] != "go"
    ]
    hold = [row for row in labels if row["yield_label"]["state"] == "hold"]
    release = [
        row for row in labels if row["yield_label"]["state"] == "release"
    ]
    if not hold:
        raise ValueError("offline replay never entered hold")
    first_hold = hold[0]
    first_hold_step = int(first_hold["source"]["step"])
    first_hold_time = _finite(
        first_hold["source"]["sim_time_seconds"], "first_hold_time"
    )
    first_hold_trace = trace_by_step[first_hold_step]
    first_hold_speed = _finite(first_hold_trace["speed"], "first_hold_speed")
    stop_path_distance = _finite(
        first_hold["yield_label"]["stop_path_distance_m"],
        "stop_path_distance_m",
    )
    stopping_distance = (
        first_hold_speed * reaction_seconds
        + first_hold_speed * first_hold_speed
        / (2.0 * certified_deceleration_mps2)
    )
    braking_margin = stop_path_distance - stopping_distance

    critical = clean_gate["summary"]["safety"]["critical_frame"]
    critical_time = _finite(critical["sim_time_seconds"], "critical_time")
    critical_actor = critical["actor"]
    critical_actor_id = int(critical_actor["actor_id"])
    critical_ttc = _finite(
        critical_actor["obb_collision_ttc_seconds"], "critical_ttc"
    )

    actor_categories = []
    conflict_actor_ids = []
    for row in labels:
        actor_id = row["conflict"].get("critical_actor_id")
        if actor_id is None:
            continue
        step = int(row["source"]["step"])
        meta_path = meta_dir / f"{step // 10:04d}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        conflict_actor_ids.append(int(actor_id))
        actor_categories.append(_actor_category(meta, int(actor_id)))

    first_non_go_time = _finite(
        non_go[0]["source"]["sim_time_seconds"], "first_non_go_time"
    )
    following_go = next(
        (
            row for row in labels
            if _finite(row["source"]["sim_time_seconds"], "label_time")
            > first_non_go_time
            and row["yield_label"]["state"] == "go"
        ),
        None,
    )
    return_to_go_time = (
        _finite(following_go["source"]["sim_time_seconds"], "return_to_go")
        if following_go is not None else None
    )
    response_duration = (
        return_to_go_time - first_non_go_time
        if return_to_go_time is not None else None
    )

    density_nonnull = sum(
        row.get("density_uq_score") is not None for row in trace
    )
    adapter_nonnull = sum(row.get("observation_uq") is not None for row in trace)
    risk_modes = sorted({(row.get("risk") or {}).get("mode") for row in trace})
    supervision_contracts = {
        json.dumps(row["supervision_contract"], sort_keys=True)
        for row in labels
    }
    supervision_contract = (
        json.loads(next(iter(supervision_contracts)))
        if len(supervision_contracts) == 1 else None
    )

    checks = {
        "clean_reference_officially_valid": clean_gate.get("gate_passed") is True,
        "clean_reference_has_walker_near_miss": (
            critical_actor.get("category") == "walker" and critical_ttc <= 1.0
        ),
        "first_hold_targets_same_critical_walker": (
            int(first_hold["conflict"]["critical_actor_id"])
            == critical_actor_id
            and actor_categories
            and all(category == "walker" for category in actor_categories)
        ),
        "oracle_acts_before_baseline_critical_frame": first_hold_time < critical_time,
        "certified_braking_margin_sufficient": (
            braking_margin >= minimum_braking_margin_m
        ),
        "conflict_clears_and_releases": bool(release),
        "returns_to_go_with_bounded_delay": (
            response_duration is not None
            and response_duration <= maximum_response_duration_seconds
        ),
        "historical_density_was_passive_only": (
            density_nonnull == len(trace) and risk_modes == ["off"]
        ),
        "historical_new_adapter_absent": adapter_nonnull == 0,
        "stage2_labels_do_not_use_density_uq_or_corruption_targets": (
            supervision_contract is not None
            and supervision_contract.get("stage") == "stage2_task_risk"
            and supervision_contract.get("uses_density_uq") is False
            and supervision_contract.get("uses_corruption_label") is False
            and supervision_contract.get("uses_observation_uq_target") is False
        ),
    }
    passed = all(checks.values())
    return {
        "schema": SCHEMA_VERSION,
        "offline_gate_pass": passed,
        "decision": (
            "eligible_for_one_preregistered_route147_clean_oracle_pair"
            if passed else "do_not_submit_route147_bounded_crossing_pair"
        ),
        "scientific_role": (
            "planning-mechanism feasibility only; not adapter efficacy and not "
            "closed-loop safety evidence"
        ),
        "checks": checks,
        "thresholds": {
            "certified_deceleration_mps2": certified_deceleration_mps2,
            "reaction_seconds": reaction_seconds,
            "minimum_braking_margin_m": minimum_braking_margin_m,
            "maximum_response_duration_seconds": maximum_response_duration_seconds,
        },
        "evidence": {
            "critical_actor_id": critical_actor_id,
            "critical_actor_category": critical_actor.get("category"),
            "baseline_critical_time_seconds": critical_time,
            "baseline_min_walker_obb_ttc_seconds": critical_ttc,
            "first_hold_time_seconds": first_hold_time,
            "first_hold_lead_seconds": critical_time - first_hold_time,
            "first_hold_speed_mps": first_hold_speed,
            "first_hold_stop_path_distance_m": stop_path_distance,
            "certified_stopping_distance_m": stopping_distance,
            "certified_braking_margin_m": braking_margin,
            "conflict_actor_ids": sorted(set(conflict_actor_ids)),
            "conflict_actor_categories": sorted(set(actor_categories)),
            "first_release_time_seconds": (
                release[0]["source"]["sim_time_seconds"] if release else None
            ),
            "return_to_go_time_seconds": return_to_go_time,
            "response_duration_seconds": response_duration,
            "non_go_record_count_at_2hz": len(non_go),
            "historical_trace_density_score_nonnull_frames": density_nonnull,
            "historical_trace_observation_adapter_nonnull_frames": adapter_nonnull,
            "historical_trace_risk_modes": risk_modes,
            "historical_density_note": (
                "The valid clean reference predates Density retirement. Its "
                "score was logged but risk_mode=off, so it did not affect control. "
                "The authorized reruns must produce zero non-null Density frames."
            ),
        },
        "supervision_contract": supervision_contract,
        "artifacts": {
            "clean_gate_path": str(clean_gate_path),
            "clean_gate_sha256": sha256(clean_gate_path),
            "trace_path": str(trace_path),
            "trace_sha256": sha256(trace_path),
            "labels_path": str(labels_path),
            "labels_sha256": sha256(labels_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-gate", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--meta-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(
        clean_gate_path=args.clean_gate,
        trace_path=args.trace,
        labels_path=args.labels,
        meta_dir=args.meta_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
