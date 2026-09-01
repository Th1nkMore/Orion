#!/usr/bin/env python3
"""Fail-closed validation for same-tick clean/medium/heavy glare capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


SCHEMA = "orion.native_glare_same_tick_validation.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(left, right, tolerance=1e-5):
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def validate_capture(root: Path, protocol: Path) -> dict:
    traces = list(root.rglob("capture_trace.jsonl"))
    if len(traces) != 1:
        raise RuntimeError("expected one capture trace below %s, found %d" % (root, len(traces)))
    trace = traces[0]
    rows = [
        json.loads(line)
        for line in trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    spec = json.loads(protocol.read_text(encoding="utf-8"))
    capture = spec["capture"]
    minimum = int(capture["minimum_saved_frames"])
    maximum = int(capture["maximum_saved_frames"])
    if not minimum <= len(rows) <= maximum:
        raise RuntimeError("capture frame count %d not in [%d,%d]" % (len(rows), minimum, maximum))
    indices = [int(row["capture_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise RuntimeError("capture indices are not contiguous")
    steps = [int(row["step"]) for row in rows]
    stride = int(capture["stride_simulator_ticks"])
    if any(right - left != stride for left, right in zip(steps[:-1], steps[1:])):
        raise RuntimeError("capture simulator steps are not continuous at the frozen stride")
    start, end = capture["route_progress_window"]
    if any(not float(start) <= float(row["route_progress"]) <= float(end) for row in rows):
        raise RuntimeError("capture includes a frame outside the frozen route progress window")
    if any(row.get("orion_loaded") is not False or row.get("adapter_loaded") is not False for row in rows):
        raise RuntimeError("capture does not attest that ORION and the adapter stayed unloaded")
    if any(row.get("same_tick") is not True for row in rows):
        raise RuntimeError("capture contains a non-same-tick triplet")
    if any(len(set(row["sensor_frames"].values())) != 1 for row in rows):
        raise RuntimeError("RGB sensor frame ids differ within a triplet")
    expected_profiles = spec["camera_profiles"]
    expected_ids = {value["sensor_id"] for value in expected_profiles.values()}
    for row in rows:
        if row.get("camera_profiles_requested") != expected_profiles:
            raise RuntimeError("requested camera profiles differ from the frozen protocol")
        if set(row["sensor_frames"]) != expected_ids:
            raise RuntimeError("captured RGB sensor ids differ from the frozen protocol")
        if row.get("weather_requested") != spec["weather"]:
            raise RuntimeError("requested weather differs from the frozen protocol")
        for field, expected in spec["weather"].items():
            actual = row["weather_readback"].get(field)
            if actual is None or not _close(actual, expected):
                raise RuntimeError("weather readback mismatch for %s" % field)
        readback = row["sensor_readback"]
        for profile_name, profile in expected_profiles.items():
            sensor_id = profile["sensor_id"]
            if sensor_id not in readback:
                raise RuntimeError("missing sensor readback for %s" % sensor_id)
            attributes = readback[sensor_id]["attributes"]
            if attributes.get("enable_postprocess_effects") != "true":
                raise RuntimeError("RGB postprocess is not enabled for %s" % sensor_id)
            for field in ("lens_flare_intensity", "bloom_intensity"):
                if field not in attributes or not _close(attributes[field], profile[field]):
                    raise RuntimeError("%s readback mismatch for %s" % (field, profile_name))
        bev_attributes = readback["bev"]["attributes"]
        if bev_attributes.get("enable_postprocess_effects") != "false":
            raise RuntimeError("BEV postprocess was not disabled")
        for path in list(row["front"].values()) + [row["bev"]]:
            if not Path(path).is_file():
                raise RuntimeError("capture references a missing image: %s" % path)
    nearby_pedestrian_frames = sum(
        any(actor.get("category") == "walker" for actor in row["nearby_actors"])
        for row in rows
    )
    minimum_pedestrians = int(capture["minimum_visible_pedestrian_frames"])
    if nearby_pedestrian_frames < minimum_pedestrians:
        raise RuntimeError("only %d frames contain a nearby pedestrian" % nearby_pedestrian_frames)
    payload = {
        "schema": SCHEMA,
        "trace": str(trace.resolve()),
        "trace_sha256": _sha256(trace),
        "protocol": str(protocol.resolve()),
        "protocol_sha256": _sha256(protocol),
        "frame_count": len(rows),
        "nearby_pedestrian_frame_count": nearby_pedestrian_frames,
        "first_step": steps[0],
        "last_step": steps[-1],
        "all_triplets_same_tick": True,
        "sensor_and_weather_readback_match": True,
        "orion_loaded": False,
        "adapter_loaded": False,
        "valid": True,
        "claim_boundary": "same-tick renderer capture integrity only; no model or safety claim",
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
    args = parser.parse_args()
    print(json.dumps(validate_capture(args.root.resolve(), args.protocol.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
