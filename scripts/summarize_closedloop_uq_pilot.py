#!/usr/bin/env python3
"""Summarize official Bench2Drive outcomes and UQ intervention traces."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def count_infractions(infractions: dict, keys: tuple[str, ...]) -> int:
    total = 0
    for key in keys:
        value = infractions.get(key, [])
        total += len(value) if isinstance(value, list) else int(bool(value))
    return total


def load_trace(run_dir: Path) -> dict:
    paths = list(run_dir.glob("records_*/**/control_trace.jsonl"))
    if not paths:
        return {}
    records = [
        json.loads(line)
        for line in paths[0].read_text().splitlines()
        if line.strip()
    ]
    speeds = [float(record["speed"]) for record in records]
    raw_scores = [
        float(record["raw_uq_score"])
        for record in records
        if record.get("raw_uq_score") is not None
    ]
    intensities = [float(record["risk"]["intensity"]) for record in records]
    interventions = [
        abs(float(record["risk"]["throttle"]) - float(record["risk"]["base_throttle"]))
        + abs(float(record["risk"]["brake"]) - float(record["risk"]["base_brake"]))
        for record in records
    ]
    return {
        "trace_path": str(paths[0]),
        "trace_steps": len(records),
        "mean_speed": mean(speeds),
        "minimum_speed": min(speeds) if speeds else None,
        "mean_raw_uq": mean(raw_scores),
        "mean_risk_intensity": mean(intensities),
        "intervention_rate": (
            sum(value > 1e-6 for value in interventions) / len(interventions)
            if interventions else None
        ),
        "mean_control_delta": mean(interventions),
    }


def load_run(run_dir: Path) -> list[dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    eval_paths = list(run_dir.glob("eval_*.json"))
    base = {
        "run_dir": str(run_dir),
        "route_index": manifest.get("pilot_route_index"),
        "variant": manifest.get("pilot_variant"),
        "condition": manifest.get("pilot_condition"),
        "job_id": manifest.get("slurm_job_id"),
    }
    base.update(load_trace(run_dir))
    if not eval_paths:
        return [{**base, "status": "missing_eval"}]
    payload = json.loads(eval_paths[0].read_text())
    records = payload.get("_checkpoint", {}).get("records", [])
    if not records:
        return [{**base, "status": payload.get("entry_status", "no_record")}]

    rows = []
    for record in records:
        infractions = record.get("infractions", {})
        collision_count = count_infractions(
            infractions,
            ("collisions_layout", "collisions_pedestrian", "collisions_vehicle"),
        )
        traffic_count = count_infractions(
            infractions,
            ("red_light", "stop_infraction", "outside_route_lanes"),
        )
        mobility_count = count_infractions(
            infractions,
            ("min_speed_infractions", "vehicle_blocked", "route_timeout"),
        )
        scores = record.get("scores", {})
        meta = record.get("meta", {})
        rows.append(
            {
                **base,
                "route_id": record.get("route_id"),
                "scenario_name": record.get("scenario_name"),
                "status": record.get("status"),
                "collision_count": collision_count,
                "collision_any": int(collision_count > 0),
                "traffic_infraction_count": traffic_count,
                "mobility_infraction_count": mobility_count,
                "route_completion": scores.get("score_route"),
                "penalty_score": scores.get("score_penalty"),
                "driving_score": scores.get("score_composed"),
                "duration_game": meta.get("duration_game"),
                "duration_system": meta.get("duration_system"),
            }
        )
    return rows


def format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    rows = []
    for manifest in sorted(args.results_root.glob("*/manifest.json")):
        rows.extend(load_run(manifest.parent))
    if not rows:
        raise ValueError(f"No runs found below {args.results_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    columns = sorted({key for row in rows for key in row})
    csv_path = args.out_dir / "summary.csv"
    with csv_path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    display_columns = (
        "route_index", "variant", "condition", "status", "collision_count",
        "traffic_infraction_count", "mobility_infraction_count",
        "route_completion", "mean_speed", "mean_raw_uq",
        "mean_risk_intensity", "intervention_rate",
    )
    lines = [
        "# Closed-Loop UQ Pilot Summary",
        "",
        "| " + " | ".join(display_columns) + " |",
        "| " + " | ".join("---" for _ in display_columns) + " |",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            str(item.get("route_index")),
            str(item.get("variant")),
            str(item.get("condition")),
        ),
    ):
        lines.append(
            "| " + " | ".join(format_value(row.get(key)) for key in display_columns) + " |"
        )
    lines.extend(
        [
            "",
            "Interpret collision reduction together with route completion, mean speed,",
            "and intervention rate; collision-only wins are not sufficient.",
            "",
        ]
    )
    markdown_path = args.out_dir / "summary.md"
    markdown_path.write_text("\n".join(lines))
    print(csv_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
