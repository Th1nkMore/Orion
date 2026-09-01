#!/usr/bin/env python3
"""Verify frozen hard-case progress windows against clean event packages."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def nearest(rows, target):
    return min(rows, key=lambda row: abs(float(row["sim_time_seconds"]) - float(target)))


def close(left, right, tolerance=1e-12):
    return abs(float(left) - float(right)) <= tolerance


def verify(windows_path: Path, project_root: Path):
    windows = load_json(windows_path)
    parent = windows["parent_funnel"]
    parent_path = project_root / parent["path"]
    assert sha256(parent_path) == parent["sha256"]
    funnel = load_json(parent_path)
    development_ids = {
        row["event_id"] for row in funnel["route_roles"]["development_screen"]
    }
    heldout_ids = {
        row["event_id"] for row in funnel["route_roles"]["heldout_confirmation"]
    }
    rows = windows["events"]
    assert {row["event_id"] for row in rows} == development_ids
    assert not ({row["event_id"] for row in rows} & heldout_ids)

    verified = []
    for frozen in rows:
        package_path = Path(frozen["event_package"])
        assert package_path.is_file(), package_path
        assert sha256(package_path) == frozen["event_package_sha256"]
        package = load_json(package_path)
        critical = package["critical_event"]
        expected_event_id = "route%d_step%d" % (
            package["route"]["route_index"], critical["step"]
        )
        assert frozen["event_id"] == expected_event_id
        assert frozen["clean_outcome"] == package["outcome_class"]
        assert close(frozen["route_progress_anchor"], critical["route_progress"])

        endpoint = package["official_endpoint"]
        if frozen["positive_case_eligible"]:
            assert endpoint["status"] == "Completed"
            assert endpoint["collision_count"] == 0
            assert endpoint["serious_infraction_count"] == 0
        else:
            assert frozen["route_index"] == 168
            assert endpoint["collision_count"] > 0

        trace_path = Path(package["source_files"]["control_trace"]["path"])
        assert trace_path.is_file(), trace_path
        assert sha256(trace_path) == frozen["control_trace_sha256"]
        trace_rows = [
            json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        source_start, source_end = critical["window_seconds"]
        selected = [
            nearest(trace_rows, source_start),
            nearest(trace_rows, critical["sim_time_seconds"]),
            nearest(trace_rows, source_end),
        ]
        assert [row["step"] for row in selected] == frozen["source_steps"]
        assert close(selected[0]["route_progress"], frozen["route_progress_window"][0])
        assert close(selected[1]["route_progress"], frozen["route_progress_anchor"])
        assert close(selected[2]["route_progress"], frozen["route_progress_window"][1])
        verified.append({
            "event_id": frozen["event_id"],
            "positive_case_eligible": frozen["positive_case_eligible"],
            "route_progress_window": frozen["route_progress_window"],
        })

    return {
        "schema": "orion.corruption_hardcase_event_windows_verification.v1",
        "passed": True,
        "event_count": len(verified),
        "positive_case_eligible_count": sum(
            int(row["positive_case_eligible"]) for row in verified
        ),
        "heldout_event_count": 0,
        "events": verified,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.windows, args.project_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
