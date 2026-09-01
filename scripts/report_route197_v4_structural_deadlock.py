#!/usr/bin/env python3
"""Record a non-terminal Route197 oracle run as a structural deadlock.

This report is deliberately separate from the terminal oracle evaluator.  A
resource-stopped run cannot establish an official collision or completion
outcome and must never authorize Stage-2 training.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess


SCHEMA_VERSION = "orion.route197_structural_deadlock.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def find_one(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name} under {root}, got {matches}")
    return matches[0]


def load_trace(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError("control trace is empty")
    steps = [int(row["step"]) for row in rows]
    if steps != list(range(steps[0], steps[0] + len(steps))):
        raise ValueError("control trace steps are not contiguous")
    return rows


def longest_true_interval(rows: list[dict], predicate) -> dict:
    best = None
    start = None
    for index in range(len(rows) + 1):
        active = index < len(rows) and bool(predicate(rows[index]))
        if active and start is None:
            start = index
        if not active and start is not None:
            end = index - 1
            duration = float(rows[end]["sim_time_seconds"]) - float(
                rows[start]["sim_time_seconds"]
            )
            candidate = (duration, start, end)
            if best is None or candidate > best:
                best = candidate
            start = None
    if best is None:
        return {"observed": False}
    duration, start, end = best
    first = rows[start]
    last = rows[end]
    return {
        "observed": True,
        "duration_seconds": duration,
        "start_step": int(first["step"]),
        "end_step": int(last["step"]),
        "start_time_seconds": float(first["sim_time_seconds"]),
        "end_time_seconds": float(last["sim_time_seconds"]),
        "start_speed_mps": float(first["speed"]),
        "end_speed_mps": float(last["speed"]),
        "start_route_progress": float(first["route_progress"]),
        "end_route_progress": float(last["route_progress"]),
        "route_progress_delta": float(last["route_progress"])
        - float(first["route_progress"]),
        "frame_count": end - start + 1,
    }


def parse_actor_flow_defaults(source: str) -> dict:
    speed_match = re.search(
        r"_flow_speed\s*=\s*get_value_parameter\([^\n]*?,\s*([0-9.]+)\s*\)",
        source,
    )
    interval_match = re.search(
        r"_source_dist_interval\s*=\s*get_interval_parameter\([^\n]*?"
        r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]\s*\)",
        source,
    )
    timeout_match = re.search(r"_scenario_timeout\s*=\s*([0-9.]+)", source)
    if not speed_match or not interval_match or not timeout_match:
        raise ValueError("could not parse frozen ActorFlow defaults")
    speed = float(speed_match.group(1))
    distances = [float(interval_match.group(1)), float(interval_match.group(2))]
    if speed <= 0 or distances[0] <= 0 or distances[1] < distances[0]:
        raise ValueError("invalid ActorFlow defaults")
    return {
        "flow_speed_mps": speed,
        "source_distance_interval_m": distances,
        "source_headway_interval_seconds": [value / speed for value in distances],
        "scenario_timeout_seconds": float(timeout_match.group(1)),
        "continuous_actor_flow_present": "ActorFlow(" in source,
        "initial_actors_enabled": "initial_actors=True" in source,
    }


def slurm_accounting(job_id: str) -> dict:
    command = [
        "sacct",
        "-j",
        job_id,
        "--format=JobID,State,ExitCode,Elapsed,NodeList,ReqCPUS,ReqMem,MaxRSS",
        "-P",
        "-n",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    rows = []
    keys = [
        "job_id",
        "state",
        "exit_code",
        "elapsed",
        "node_list",
        "requested_cpus",
        "requested_memory",
        "max_rss",
    ]
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        values = line.split("|")
        if len(values) != len(keys):
            raise ValueError(f"unexpected sacct row: {line!r}")
        rows.append(dict(zip(keys, values)))
    if not rows:
        raise ValueError(f"sacct returned no rows for {job_id}")
    return {"command": command, "rows": rows}


def build_report(
    run_dir: Path,
    geometry_path: Path,
    preregistration_path: Path,
    scenario_source_path: Path,
    job_id: str,
    stationary_threshold_mps: float,
) -> dict:
    manifest_path = run_dir / "manifest.json"
    eval_path = run_dir / "eval_orion_traj_0.json"
    trace_path = find_one(run_dir, "control_trace.jsonl")
    manifest = load_json(manifest_path)
    evaluator = load_json(eval_path)
    preregistration = load_json(preregistration_path)
    geometry = load_json(geometry_path)
    rows = load_trace(trace_path)
    source_text = scenario_source_path.read_text(encoding="utf-8")
    flow = parse_actor_flow_defaults(source_text)

    trace_sha = sha256_file(trace_path)
    frozen = preregistration["frozen_planning_response"]
    horizons = [float(value) for value in frozen["horizons_seconds"]]
    conflict_horizon = max(horizons)
    clearance_required = float(frozen["clearance_seconds"])
    max_source_headway = max(flow["source_headway_interval_seconds"])

    state_counts = Counter()
    clearance_values = []
    release_values = []
    density_absent = True
    adapter_absent = True
    risk_passthrough = True
    for row in rows:
        response = row.get("planning_response") or {}
        label = response.get("yield_label") or {}
        if label.get("state"):
            state_counts[str(label["state"])] += 1
        clearance_values.append(float(label.get("clearance_elapsed_seconds", 0.0)))
        release_values.append(float(label.get("release_elapsed_seconds", 0.0)))
        density_absent &= row.get("density_uq_score") is None
        adapter_absent &= row.get("observation_uq") is None
        risk = row.get("risk") or {}
        risk_passthrough &= (
            risk.get("mode") == "off"
            and risk.get("applied_score") is None
            and math.isclose(
                float(risk.get("throttle", math.nan)),
                float(risk.get("base_throttle", math.nan)),
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and math.isclose(
                float(risk.get("brake", math.nan)),
                float(risk.get("base_brake", math.nan)),
                rel_tol=0.0,
                abs_tol=1e-8,
            )
        )

    geometry_records = geometry.get("records") or []
    first_junction = next(
        (
            record
            for record in geometry_records
            if bool((record.get("ego_map_waypoint") or {}).get("is_junction"))
        ),
        None,
    )
    stationary = longest_true_interval(
        rows, lambda row: float(row["speed"]) < stationary_threshold_mps
    )
    max_clearance = max(clearance_values)
    metadata_conditioning_claim = manifest.get("orion_closedloop_conditioning")
    metadata_conditioning_defect = bool(
        metadata_conditioning_claim == "vision_adapter"
        and not manifest.get("orion_observation_uq_checkpoint")
        and adapter_absent
    )
    endpoint_terminal = bool(
        evaluator.get("eligible")
        and (evaluator.get("_checkpoint") or {}).get("records")
    )
    algebraic_incompatibility = max_source_headway < conflict_horizon
    observed_release_incompatible = max_clearance < clearance_required

    checks = {
        "trace_nonempty_and_contiguous": bool(rows),
        "geometry_schema_valid": (
            geometry.get("schema_version") == "orion.carla_junction_geometry.v1"
        ),
        "geometry_trace_hash_matches": geometry.get("trace_sha256") == trace_sha,
        "geometry_record_count_matches": len(geometry_records) == len(rows),
        "geometry_uses_raw_unmodified_plan_every_frame": bool(geometry_records)
        and all(
            record.get("base_plan_world_xy_source") == "raw_conflict"
            for record in geometry_records
        ),
        "legacy_density_absent_every_frame": density_absent,
        "new_observation_adapter_absent_every_frame": adapter_absent,
        "scalar_risk_governor_exact_passthrough": risk_passthrough,
        "official_terminal_endpoint_absent": not endpoint_terminal,
        "long_stationary_interval_observed": bool(stationary.get("observed"))
        and float(stationary.get("duration_seconds", 0.0)) >= 20.0,
        "release_state_never_reached": state_counts.get("release", 0) == 0,
        "observed_clearance_never_reaches_requirement": observed_release_incompatible,
        "continuous_flow_headway_shorter_than_conflict_horizon": (
            algebraic_incompatibility
        ),
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise ValueError(f"structural-deadlock evidence contract failed: {failed}")

    artifacts = {
        "control_trace": {"path": str(trace_path), "sha256": trace_sha},
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "evaluator": {"path": str(eval_path), "sha256": sha256_file(eval_path)},
        "geometry": {"path": str(geometry_path), "sha256": sha256_file(geometry_path)},
        "preregistration": {
            "path": str(preregistration_path),
            "sha256": sha256_file(preregistration_path),
        },
        "actor_flow_source": {
            "path": str(scenario_source_path),
            "sha256": sha256_file(scenario_source_path),
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "run_dir": str(run_dir),
        "classification": "resource_stopped_structural_deadlock",
        "resource_stop_reason": (
            "The non-terminal run was cancelled after a persistent no-release "
            "state exposed an incompatibility between a zero-conflict-horizon "
            "yield policy and the scenario's continuous ActorFlow."
        ),
        "slurm_accounting": slurm_accounting(job_id),
        "artifacts": artifacts,
        "trace": {
            "frame_count": len(rows),
            "first_step": int(rows[0]["step"]),
            "last_step": int(rows[-1]["step"]),
            "first_sim_time_seconds": float(rows[0]["sim_time_seconds"]),
            "last_sim_time_seconds": float(rows[-1]["sim_time_seconds"]),
            "last_route_progress": float(rows[-1]["route_progress"]),
            "stationary_threshold_mps": stationary_threshold_mps,
            "longest_stationary_interval": stationary,
            "first_ego_junction_record": first_junction,
            "state_counts": dict(state_counts),
            "maximum_clearance_elapsed_seconds": max_clearance,
            "maximum_release_elapsed_seconds": max(release_values),
        },
        "flow_release_incompatibility": {
            "actor_flow_defaults": flow,
            "conflict_horizons_seconds": horizons,
            "maximum_conflict_horizon_seconds": conflict_horizon,
            "required_conflict_free_clearance_seconds": clearance_required,
            "maximum_source_headway_seconds": max_source_headway,
            "maximum_source_headway_less_than_conflict_horizon": (
                algebraic_incompatibility
            ),
            "observed_maximum_clearance_seconds": max_clearance,
            "observed_clearance_less_than_required": observed_release_incompatible,
            "interpretation": (
                "In steady continuous flow, a new actor can enter the 3.0 s "
                "prediction horizon within at most 2.5 s. The policy additionally "
                "requires 1.0 s of uninterrupted clearance, so its release premise "
                "is structurally unsuitable for this merge task."
            ),
        },
        "signal_and_control_contract": {
            "legacy_density_enabled_manifest": manifest.get(
                "orion_enable_legacy_density_uq"
            ),
            "legacy_density_score_absent_every_frame": density_absent,
            "new_observation_adapter_checkpoint": manifest.get(
                "orion_observation_uq_checkpoint"
            ),
            "new_observation_adapter_absent_every_frame": adapter_absent,
            "scalar_risk_governor_exact_passthrough": risk_passthrough,
            "planning_response_mode": manifest.get("orion_planning_response_mode"),
        },
        "metadata_audit": {
            "recorded_conditioning": metadata_conditioning_claim,
            "conditioning_field_is_misleading": metadata_conditioning_defect,
            "effective_conditioning": "privileged_dynamics_aware_yield_only",
            "required_fix": (
                "Future manifests must derive conditioning from the loaded "
                "checkpoint and planning-response mode instead of retaining the "
                "legacy vision_adapter default."
            ),
        },
        "official_endpoint": {
            "eligible": evaluator.get("eligible"),
            "entry_status": evaluator.get("entry_status"),
            "record_count": len((evaluator.get("_checkpoint") or {}).get("records") or []),
            "collision_outcome": "not_determined",
            "route_completion_outcome": "not_determined",
        },
        "evidence_checks": checks,
        "primary_success": False,
        "stage2_eligible": False,
        "decision": "do_not_train_stage2_redesign_as_gap_acceptance_merge",
        "next_authorized_action": (
            "Offline gap-acceptance and merge-feasibility audit only. No learned "
            "adapter control, Stage-2 training, or additional A800 route matrix is "
            "authorized by this partial run."
        ),
        "claim_boundary": (
            "This partial trace proves neither zero collisions nor route success. "
            "It records a reproducible mechanism deadlock and the absence of both "
            "legacy Density UQ and the new learned observation adapter."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--scenario-source", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--stationary-threshold-mps", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite structural-deadlock report")
    report = build_report(
        args.run_dir,
        args.geometry,
        args.preregistration,
        args.scenario_source,
        args.job_id,
        args.stationary_threshold_mps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "classification": report["classification"],
                "primary_success": report["primary_success"],
                "stage2_eligible": report["stage2_eligible"],
                "decision": report["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
