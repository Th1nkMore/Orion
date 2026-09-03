#!/usr/bin/env python3
"""Fail-closed validation for one lightweight native-glare route capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCHEMA = "orion.native_glare_capture_validation.v1"


def validate_capture(root: Path, protocol: Path, profile: str, minimum_frames: int) -> dict:
    traces = list(root.rglob("capture_trace.jsonl"))
    if len(traces) != 1:
        raise RuntimeError("expected one capture trace below %s, found %d" % (root, len(traces)))
    trace = traces[0]
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < minimum_frames:
        raise RuntimeError("capture has only %d frames" % len(rows))
    spec = json.loads(protocol.read_text(encoding="utf-8"))
    expected_camera = spec["methods"]["carla_native_low_sun"]["camera_profiles"][profile]
    expected_weather = spec["methods"]["carla_native_low_sun"][
        "weather_shared_by_all_profiles"
    ]
    indices = [row["capture_index"] for row in rows]
    if indices != list(range(len(rows))):
        raise RuntimeError("capture indices are not contiguous")
    if any(row.get("profile") != profile for row in rows):
        raise RuntimeError("capture profile differs from requested profile")
    if any(row.get("orion_loaded") is not False for row in rows):
        raise RuntimeError("capture does not attest that ORION stayed unloaded")
    if any(row.get("camera_postprocess") != expected_camera for row in rows):
        raise RuntimeError("camera postprocess differs from frozen candidate")
    if any(row.get("weather") != expected_weather for row in rows):
        raise RuntimeError("capture weather differs from common low-sun condition")
    if any(not Path(row[view]).is_file() for row in rows for view in ("front", "bev")):
        raise RuntimeError("capture references a missing image")
    walker_rows = [
        row for row in rows
        if any(actor["type_id"].startswith("walker.pedestrian") for actor in row["nearby_actors"])
    ]
    if not walker_rows:
        raise RuntimeError("Route151 pedestrian never appeared in capture telemetry")
    progress = [float(row["route_progress"]) for row in rows]
    if max(progress) - min(progress) < 0.05:
        raise RuntimeError("ego did not traverse enough of Route151")
    payload = {
        "schema": SCHEMA,
        "profile": profile,
        "trace": str(trace.resolve()),
        "frame_count": len(rows),
        "walker_frame_count": len(walker_rows),
        "progress_min": min(progress),
        "progress_max": max(progress),
        "orion_loaded": False,
        "camera_postprocess": expected_camera,
        "weather": expected_weather,
        "claim_boundary": "visual capture validity only; no safety or UQ claim",
    }
    report = root / "capture_validation.json"
    if report.exists():
        raise FileExistsError("refusing to overwrite %s" % report)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--profile", choices=("clean", "light", "medium", "heavy"), required=True)
    parser.add_argument("--minimum-frames", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(validate_capture(
        args.root.resolve(),
        args.protocol.resolve(),
        args.profile,
        args.minimum_frames,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
