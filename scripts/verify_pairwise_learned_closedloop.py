#!/usr/bin/env python3
"""Verify the frozen Route146 pairwise learned controlled-stop outcome."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "orion.pairwise-learned-closedloop-verification/v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--learned-run-dir", type=Path, required=True)
    parser.add_argument("--trace-gate-report", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    evaluation = json.loads(
        (run_dir / "eval_orion_traj_0.json").read_text(encoding="utf-8")
    )
    paths = glob.glob(
        str(run_dir / "records_*" / "**" / "control_trace.jsonl"),
        recursive=True,
    )
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one control trace below {run_dir}")
    trace = [
        json.loads(line)
        for line in Path(paths[0]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not trace:
        raise RuntimeError(f"empty control trace below {run_dir}")
    records = evaluation.get("_checkpoint", {}).get("records", [])
    if len(records) != 1:
        raise RuntimeError(f"expected exactly one evaluation record in {run_dir}")
    return evaluation, records[0], trace


def _event_interval(trace: list[dict[str, Any]]) -> tuple[int, int, float, float]:
    event_indices = [
        index for index, row in enumerate(trace) if bool(row["corruption_active"])
    ]
    if not event_indices:
        raise RuntimeError("trace never reached the registered corruption event")
    first = event_indices[0]
    last = event_indices[-1]
    if event_indices != list(range(first, last + 1)):
        raise RuntimeError("corruption event was not one contiguous interval")
    if last + 1 >= len(trace):
        raise RuntimeError("trace ended before the registered corruption event recovered")
    start = float(trace[first]["sim_time_seconds"])
    end = float(trace[last + 1]["sim_time_seconds"])
    return first, last, start, end


def _longest_low_speed(trace: list[dict[str, Any]], threshold: float = 0.25) -> float:
    times = [float(row["sim_time_seconds"]) for row in trace]
    deltas = [right - left for left, right in zip(times, times[1:]) if right > left]
    step = float(statistics.median(deltas)) if deltas else 0.05
    longest = 0.0
    start = None
    last = None
    for row in trace:
        current = float(row["sim_time_seconds"])
        if abs(float(row["speed"])) < threshold:
            if start is None:
                start = current
            last = current
            longest = max(longest, last - start + step)
        else:
            start = None
            last = None
    return longest


def _summarize_record(
    run_dir: Path,
    evaluation: dict[str, Any],
    record: dict[str, Any],
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir.resolve()),
        "eligible": evaluation.get("eligible"),
        "entry_status": evaluation.get("entry_status"),
        "record_status": record.get("status"),
        "route_completion": record["scores"]["score_route"],
        "score_composed": record["scores"]["score_composed"],
        "pedestrian_collisions": len(record["infractions"]["collisions_pedestrian"]),
        "vehicle_collisions": len(record["infractions"]["collisions_vehicle"]),
        "layout_collisions": len(record["infractions"]["collisions_layout"]),
        "duration_game": record["meta"]["duration_game"],
        "longest_below_0_25_mps_seconds": _longest_low_speed(trace),
    }


def _require_empty_infractions(record: dict[str, Any]) -> None:
    for key in (
        "collisions_layout",
        "collisions_vehicle",
        "collisions_pedestrian",
        "red_light",
        "stop_infraction",
        "outside_route_lanes",
        "scenario_timeouts",
        "route_dev",
        "vehicle_blocked",
        "route_timeout",
    ):
        if record["infractions"][key] != []:
            raise RuntimeError(f"learned controlled stop has non-empty {key}")


def verify(args: argparse.Namespace) -> dict[str, Any]:
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    trace_gate = json.loads(args.trace_gate_report.read_text(encoding="utf-8"))
    if trace_gate.get("gate_passed") is not True:
        raise RuntimeError("the frozen governor-off learned trace gate did not pass")
    if trace_gate.get("decision") != "submit_single_pairwise_controlled_stop":
        raise RuntimeError("trace-gate decision did not authorize the bounded learned run")

    off_job = int(prereg["fixed_references"]["transient_off_job"])
    oracle_job = int(prereg["fixed_references"]["controlled_stop_oracle_job"])
    off_dir = (
        args.results_root
        / "uqcl_p1_transient"
        / "raw"
        / f"route146_hazard_front_corrupt_transient_off-{off_job}"
    )
    oracle_dir = (
        args.results_root
        / "uqcl_p2_safe_stop"
        / "raw"
        / f"route146_hazard_front_corrupt_transient_oracle_stop-{oracle_job}"
    )
    run_dirs = {
        "transient_off": off_dir,
        "oracle_controlled_stop": oracle_dir,
        "learned_controlled_stop": args.learned_run_dir,
    }
    loaded = {name: _load_run(path) for name, path in run_dirs.items()}
    summaries = {
        name: _summarize_record(run_dirs[name], *loaded[name])
        for name in run_dirs
    }

    for name in ("transient_off", "oracle_controlled_stop"):
        evaluation, record, trace = loaded[name]
        if evaluation.get("eligible") is not True:
            raise RuntimeError(f"fixed reference {name} is not eligible")
        if evaluation.get("entry_status") != "Finished" or record.get("status") != "Completed":
            raise RuntimeError(f"fixed reference {name} is not completed")
        _, _, event_start, event_end = _event_interval(trace)
        if abs((event_end - event_start) - 5.0) > 1e-9:
            raise RuntimeError(f"fixed reference {name} did not use the five-second event")

    learned_eval, learned_record, learned_trace = loaded["learned_controlled_stop"]
    learned_manifest = json.loads(
        (args.learned_run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if learned_manifest.get("pilot_condition") != "front_corrupt_transient_pairwise_stop":
        raise RuntimeError("learned run condition differs from preregistration")
    if learned_manifest.get("orion_closedloop_risk_mode") != "aligned_learned":
        raise RuntimeError("learned run did not use the aligned learned governor")
    if (
        learned_manifest.get("orion_observation_uq_checkpoint_sha256")
        != prereg["signal_checkpoint"]["sha256"]
    ):
        raise RuntimeError("learned run checkpoint hash differs from preregistration")
    if learned_eval.get("eligible") is not True:
        raise RuntimeError("learned controlled stop is not eligible")
    if learned_eval.get("entry_status") != "Finished" or learned_record.get("status") != "Completed":
        raise RuntimeError("learned controlled stop did not complete")
    if float(learned_record["scores"]["score_route"]) != 100.0:
        raise RuntimeError("learned controlled stop did not complete 100% of the route")
    _require_empty_infractions(learned_record)
    if (
        float(learned_record["scores"]["score_composed"])
        <= float(loaded["transient_off"][1]["scores"]["score_composed"])
    ):
        raise RuntimeError("learned controlled stop did not improve on transient off")
    if summaries["learned_controlled_stop"]["longest_below_0_25_mps_seconds"] >= 8.0:
        raise RuntimeError("learned controlled stop violated the low-speed fast screen")

    first, last, event_start, event_end = _event_interval(learned_trace)
    if abs((event_end - event_start) - 5.0) > 1e-9:
        raise RuntimeError("learned run did not use the registered five-second event")
    threshold = float(prereg["controlled_stop_governor"]["threshold"])
    event_scores = [float(row["raw_uq_score"]) for row in learned_trace[first : last + 1]]
    event_intensities = [
        float(row["risk"]["intensity"]) for row in learned_trace[first : last + 1]
    ]
    if max(event_scores) < threshold or max(event_intensities) <= 0.0:
        raise RuntimeError("learned signal never activated the governor during the event")

    summaries["learned_controlled_stop"].update({
        "event_start_seconds": event_start,
        "event_end_seconds": event_end,
        "event_median_score": float(statistics.median(event_scores)),
        "event_max_governor_intensity": max(event_intensities),
    })
    return {
        "schema_version": SCHEMA_VERSION,
        "verified": True,
        "runs": summaries,
        "trace_gate_report": str(args.trace_gate_report.resolve()),
        "claim_boundary": {
            "supports": "one-route exploratory learned operational-trigger mechanism",
            "does_not_support": [
                "general spatial uncertainty validity",
                "corruption-family generalization of the closed-loop result",
                "paper-level multi-route safety improvement",
                "VLM semantic understanding of uncertainty",
            ],
        },
        "decision": "record_bounded_learned_mechanism_success_without_matrix_expansion",
    }


def main() -> int:
    args = _parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite verification report")
    report = verify(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "verified": report["verified"],
        "decision": report["decision"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
