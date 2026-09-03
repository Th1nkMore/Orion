#!/usr/bin/env python3
"""Verify the registered route-146 transient-dropout mechanism outcomes."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    return parser.parse_args()


def load_run(run_dir: Path):
    evaluation = json.loads((run_dir / "eval_orion_traj_0.json").read_text())
    trace_path = glob.glob(str(run_dir / "records_*" / "**" / "control_trace.jsonl"), recursive=True)[0]
    trace = [json.loads(line) for line in Path(trace_path).read_text().splitlines() if line]
    record = evaluation["_checkpoint"]["records"][0]
    return evaluation, record, trace


def toggles(trace):
    output = []
    previous = None
    for row in trace:
        active = bool(row["corruption_active"])
        if active != previous:
            output.append(row)
        previous = active
    return output


def longest_low_speed(trace, threshold=0.25):
    best = 0.0
    start = None
    for row in trace + [None]:
        below = row is not None and abs(float(row["speed"])) < threshold
        if below and start is None:
            start = float(row["sim_time_seconds"])
        elif not below and start is not None:
            end = float(trace[-1]["sim_time_seconds"] if row is None else previous["sim_time_seconds"])
            best = max(best, end - start)
            start = None
        if row is not None:
            previous = row
    return best


def main():
    root = parse_args().results_root
    run_dirs = {
        "off": root / "uqcl_p1_transient" / "raw" / "route146_hazard_front_corrupt_transient_off-1057306",
        "floor_oracle": root / "uqcl_p1_transient" / "raw" / "route146_hazard_front_corrupt_transient_oracle-1057314",
        "stop_oracle": root / "uqcl_p2_safe_stop" / "raw" / "route146_hazard_front_corrupt_transient_oracle_stop-1057566",
    }
    loaded = {name: load_run(path) for name, path in run_dirs.items()}

    summary = {}
    for name, (evaluation, record, trace) in loaded.items():
        assert evaluation["eligible"] is True
        assert evaluation["entry_status"] == "Finished"
        assert record["status"] == "Completed"
        assert record["scores"]["score_route"] == 100
        event_toggles = toggles(trace)
        assert [row["corruption_active"] for row in event_toggles] == [False, True, False]
        assert abs(
            event_toggles[2]["sim_time_seconds"]
            - event_toggles[1]["sim_time_seconds"]
            - 5.0
        ) < 1e-9
        summary[name] = {
            "job": int(run_dirs[name].name.rsplit("-", 1)[1]),
            "score_composed": record["scores"]["score_composed"],
            "route_completion": record["scores"]["score_route"],
            "pedestrian_collisions": len(record["infractions"]["collisions_pedestrian"]),
            "duration_game": record["meta"]["duration_game"],
            "event_start_s": event_toggles[1]["sim_time_seconds"],
            "event_end_s": event_toggles[2]["sim_time_seconds"],
            "longest_below_0_25_s": longest_low_speed(trace),
        }

    assert summary["off"]["score_composed"] == 50.0
    assert summary["off"]["pedestrian_collisions"] == 1
    assert summary["floor_oracle"]["score_composed"] == 50.0
    assert summary["floor_oracle"]["pedestrian_collisions"] == 1
    assert summary["stop_oracle"]["score_composed"] == 100.0
    assert summary["stop_oracle"]["pedestrian_collisions"] == 0
    assert summary["stop_oracle"]["longest_below_0_25_s"] < 8.0

    stop_record = loaded["stop_oracle"][1]
    for key in (
        "collisions_layout", "collisions_vehicle", "red_light", "stop_infraction",
        "outside_route_lanes", "scenario_timeouts", "route_dev", "vehicle_blocked",
        "route_timeout",
    ):
        assert stop_record["infractions"][key] == []

    print(json.dumps({"verified": True, "runs": summary}, indent=2))


if __name__ == "__main__":
    main()
