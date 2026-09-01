#!/usr/bin/env python3
"""Freeze one clean-derived spatial event before degraded jobs are submitted."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-run-dir", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--route-file", type=Path, required=True)
    parser.add_argument("--start-progress", type=float, required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--on-path-region", type=float, nargs=4, required=True)
    parser.add_argument("--off-path-region", type=float, nargs=4, required=True)
    parser.add_argument("--family", choices=("local_blur", "local_glare", "local_occlusion"), default="local_glare")
    parser.add_argument("--severity", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--evidence-frame", type=int, action="append", required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {pattern!r} below {root}, found {len(matches)}"
        )
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_region(values: list[float], name: str) -> tuple[float, float, float, float]:
    top, left, bottom, right = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (top, left, bottom, right)):
        raise ValueError(f"{name} coordinates must be finite")
    if not (0.0 <= top < bottom <= 1.0 and 0.0 <= left < right <= 1.0):
        raise ValueError(f"{name} must be normalized and non-empty")
    return top, left, bottom, right


def area(region: tuple[float, float, float, float]) -> float:
    return (region[2] - region[0]) * (region[3] - region[1])


def overlap_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    height = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    width = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return height * width


def count_infractions(infractions: dict[str, Any], keys: tuple[str, ...]) -> int:
    total = 0
    for key in keys:
        value = infractions.get(key, [])
        total += len(value) if isinstance(value, list) else int(bool(value))
    return total


def longest_low_speed(rows: list[dict[str, Any]], threshold: float = 0.25) -> float:
    longest = 0.0
    start = None
    previous_time = None
    typical_step = 0.05
    positive_steps = [
        float(right["sim_time_seconds"]) - float(left["sim_time_seconds"])
        for left, right in zip(rows, rows[1:])
        if float(right["sim_time_seconds"]) > float(left["sim_time_seconds"])
    ]
    if positive_steps:
        positive_steps.sort()
        typical_step = positive_steps[len(positive_steps) // 2]
    for row in rows:
        current_time = float(row["sim_time_seconds"])
        low = abs(float(row["speed"])) < threshold and current_time >= 2.0
        if low and start is None:
            start = current_time
        if not low and start is not None:
            longest = max(longest, (previous_time or start) + typical_step - start)
            start = None
        previous_time = current_time
    if start is not None and previous_time is not None:
        longest = max(longest, previous_time + typical_step - start)
    return longest


def rows_before_progress(
    rows: list[dict[str, Any]], start_progress: float
) -> list[dict[str, Any]]:
    """Return the chronological trace prefix before the frozen event starts.

    The scenario-bank protocol uses the low-speed rule only as a pre-event
    liveness screen.  A legitimate stop caused by the hazard itself or by a
    later traffic light must not invalidate an otherwise usable clean
    reference.
    """
    prefix = []
    for row in rows:
        if float(row["route_progress"]) >= start_progress:
            break
        prefix.append(row)
    return prefix


def main() -> int:
    args = parse_args()
    if not (0.0 <= args.start_progress <= 1.0):
        raise ValueError("start-progress must lie in [0,1]")
    if not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0.0:
        raise ValueError("duration-seconds must be finite and positive")
    on_path = validate_region(args.on_path_region, "on-path region")
    off_path = validate_region(args.off_path_region, "off-path region")
    if not math.isclose(area(on_path), area(off_path), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("on-path and off-path regions must have identical area")
    if overlap_area(on_path, off_path) > 0.0:
        raise ValueError("on-path and off-path regions must not overlap")

    manifest_path = args.clean_run_dir / "manifest.json"
    eval_path = find_one(args.clean_run_dir, "eval_*.json")
    trace_path = find_one(args.clean_run_dir, "records_*/**/control_trace.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    review = json.loads(args.review_manifest.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if manifest.get("pilot_condition") != "clean_off":
        raise RuntimeError("event must be frozen from a clean_off run")
    if manifest.get("orion_closedloop_risk_mode") != "off":
        raise RuntimeError("clean reference did not use risk mode off")
    if manifest.get("orion_closedloop_corruption") not in (None, ""):
        raise RuntimeError("clean reference contains a corruption")
    if not evaluation.get("eligible") or evaluation.get("entry_status") != "Finished":
        raise RuntimeError("clean reference is not an eligible finished evaluation")
    records = evaluation.get("_checkpoint", {}).get("records", [])
    if len(records) != 1:
        raise RuntimeError("clean reference must contain exactly one route record")
    record = records[0]
    scores = record.get("scores", {})
    if record.get("status") != "Completed" or float(scores.get("score_route", 0)) != 100.0:
        raise RuntimeError("clean reference did not complete the route")
    if count_infractions(record.get("infractions", {}), COLLISION_KEYS) != 0:
        raise RuntimeError("clean reference contains a collision")
    if not rows or any("route_progress" not in row for row in rows):
        raise RuntimeError("clean trace is empty or lacks route progress")
    if not min(float(row["route_progress"]) for row in rows) <= args.start_progress <= max(
        float(row["route_progress"]) for row in rows
    ):
        raise RuntimeError("event start is outside the observed clean route progress")
    pre_event_rows = rows_before_progress(rows, args.start_progress)
    low_speed = longest_low_speed(pre_event_rows)
    if low_speed >= 8.0:
        raise RuntimeError(
            "clean reference violates the 8-second pre-event low-speed screen: "
            f"{low_speed:.2f}s"
        )
    review_run = Path(review["run_dir"]).resolve()
    if review_run != args.clean_run_dir.resolve():
        raise RuntimeError("review manifest does not belong to the clean run")
    selected_frames = {
        int(Path(item["front"]).stem) for item in review.get("selected", [])
    }
    if not set(args.evidence_frame).issubset(selected_frames):
        raise RuntimeError("every evidence frame must be present in the review manifest")

    payload = {
        "schema": "orion.closedloop_spatial_event_preregistration.v1",
        "status": "frozen_before_degraded_execution",
        "clean_reference": {
            "run_dir": str(args.clean_run_dir.resolve()),
            "job_id": manifest.get("slurm_job_id"),
            "route_index": manifest.get("pilot_route_index"),
            "variant": manifest.get("pilot_variant"),
            "eligible": True,
            "status": record.get("status"),
            "route_completion": float(scores["score_route"]),
            "collision_count": 0,
            "longest_below_0_25_mps_seconds": low_speed,
            "low_speed_screen_scope": "trace prefix before event start_progress",
        },
        "event": {
            "camera": "CAM_FRONT",
            "start_progress": args.start_progress,
            "duration_seconds": args.duration_seconds,
            "family": args.family,
            "severity": args.severity,
            "seed": args.seed,
            "on_path_region": list(on_path),
            "off_path_region": list(off_path),
            "matched_area": area(on_path),
            "evidence_frames": sorted(set(args.evidence_frame)),
            "rationale": args.rationale,
        },
        "execution_gate": {
            "first_degraded_condition": "hazard_on_path_response_off",
            "controls_authorized_only_after_failure_induction": True,
            "no_learned_or_oracle_control_yet": True,
        },
        "hashes": {
            "route_xml_sha256": sha256(args.route_file),
            "clean_manifest_sha256": sha256(manifest_path),
            "clean_eval_sha256": sha256(eval_path),
            "clean_trace_sha256": sha256(trace_path),
            "review_manifest_sha256": sha256(args.review_manifest),
        },
        "files": {
            "route_xml": str(args.route_file.resolve()),
            "clean_manifest": str(manifest_path.resolve()),
            "clean_eval": str(eval_path.resolve()),
            "clean_trace": str(trace_path.resolve()),
            "review_manifest": str(args.review_manifest.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
