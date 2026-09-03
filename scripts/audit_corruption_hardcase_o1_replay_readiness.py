#!/usr/bin/env python3
"""Audit whether frozen development event packages can support O1 replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from PIL import Image


SCHEMA = "orion.corruption_hardcase_o1_replay_readiness.v1"
REQUIRED_CAMERAS = (
    "rgb_front",
    "rgb_front_left",
    "rgb_front_right",
    "rgb_back",
    "rgb_back_left",
    "rgb_back_right",
)
REPLAY_STATE_KEYS = {
    "can_bus",
    "command",
    "ego_pose",
    "ego_pose_inv",
    "lidar2img",
    "lidar2cam",
    "cam_intrinsic",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_dimensions(paths: list[Path]) -> list[list[int]]:
    if not paths:
        return []
    indices = sorted({0, len(paths) // 2, len(paths) - 1})
    dimensions = []
    for index in indices:
        with Image.open(paths[index]) as image:
            dimensions.append([int(image.width), int(image.height)])
    return dimensions


def audit_camera(record: dict[str, Any]) -> dict[str, Any]:
    directory = Path(record["path"])
    images = sorted(directory.glob("*.png")) if directory.is_dir() else []
    dimensions = sample_dimensions(images)
    return {
        "path": str(directory),
        "directory_exists": directory.is_dir(),
        "declared_frame_count": int(record.get("frame_count", -1)),
        "actual_frame_count": len(images),
        "frame_count_matches": len(images) == int(record.get("frame_count", -1)),
        "sample_dimensions": dimensions,
        "sample_dimensions_consistent": len({tuple(row) for row in dimensions}) <= 1,
        "first_frame": images[0].name if images else None,
        "last_frame": images[-1].name if images else None,
    }


def audit_event(event: dict[str, Any]) -> dict[str, Any]:
    package_path = Path(event["event_package"])
    package_hash = sha256(package_path) if package_path.is_file() else None
    package = read_json(package_path) if package_path.is_file() else {}
    inventory = package.get("camera_inventory", {})
    cameras = {
        name: audit_camera(inventory[name])
        for name in REQUIRED_CAMERAS
        if name in inventory
    }
    missing_cameras = [name for name in REQUIRED_CAMERAS if name not in inventory]
    counts = [row["actual_frame_count"] for row in cameras.values()]

    trace_record = package.get("source_files", {}).get("control_trace", {})
    trace_path = Path(trace_record.get("path", ""))
    trace_hash = sha256(trace_path) if trace_path.is_file() else None
    rows = []
    if trace_path.is_file():
        rows = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    steps = [int(row["step"]) for row in rows]
    times = [float(row["sim_time_seconds"]) for row in rows]
    step_contiguous = bool(steps) and steps == list(range(steps[0], steps[-1] + 1))
    time_deltas = [right - left for left, right in zip(times, times[1:])]
    model_interval_seconds = statistics.median(time_deltas) if time_deltas else None
    expected_saved_steps = (
        list(range(0, steps[-1] + 1, 10)) if steps and steps[0] == 0 else []
    )
    saved_count_matches_agent_cadence = bool(counts) and all(
        count == len(expected_saved_steps) for count in counts
    )
    saved_interval_seconds = (
        model_interval_seconds * 10.0
        if model_interval_seconds is not None and saved_count_matches_agent_cadence
        else None
    )
    trace_keys = set(rows[0]) if rows else set()
    missing_replay_state = sorted(REPLAY_STATE_KEYS - trace_keys)
    exact_model_tensor_available = "rgb_front_model_tensor" in inventory
    raw_resolution_1600x900 = bool(cameras) and all(
        row["sample_dimensions"]
        and all(size == [1600, 900] for size in row["sample_dimensions"])
        for row in cameras.values()
    )

    package_valid = bool(
        package_hash == event.get("event_package_sha256")
        and package.get("runtime", {}).get("valid") is True
        and not missing_cameras
        and all(row["frame_count_matches"] for row in cameras.values())
        and len(set(counts)) <= 1
        and trace_hash == trace_record.get("sha256")
        and trace_hash == event.get("control_trace_sha256")
        and step_contiguous
        and all(delta > 0.0 and math.isfinite(delta) for delta in time_deltas)
    )
    stale_delay_seconds = [0.2, 0.4]
    stale_cadence_ready = bool(
        saved_interval_seconds is not None
        and saved_interval_seconds <= min(stale_delay_seconds)
    )
    waterdrop_visual_reapply_ready = package_valid and raw_resolution_1600x900
    generic_orion_replay_ready = bool(
        package_valid
        and exact_model_tensor_available
        and not missing_replay_state
        and model_interval_seconds is not None
        and saved_interval_seconds == model_interval_seconds
    )

    return {
        "event_id": event["event_id"],
        "route_index": int(event["route_index"]),
        "event_package": str(package_path),
        "event_package_sha256": package_hash,
        "event_package_hash_matches": package_hash == event.get("event_package_sha256"),
        "package_valid": package_valid,
        "camera_inventory": cameras,
        "missing_required_cameras": missing_cameras,
        "camera_counts_consistent": len(set(counts)) <= 1,
        "trace": {
            "path": str(trace_path),
            "sha256": trace_hash,
            "hash_matches_package": trace_hash == trace_record.get("sha256"),
            "hash_matches_event_window": trace_hash == event.get("control_trace_sha256"),
            "rows": len(rows),
            "first_step": steps[0] if steps else None,
            "last_step": steps[-1] if steps else None,
            "steps_contiguous": step_contiguous,
            "model_interval_seconds": model_interval_seconds,
            "saved_interval_seconds": saved_interval_seconds,
            "saved_count_matches_agent_10_step_cadence": saved_count_matches_agent_cadence,
            "missing_replay_state_keys": missing_replay_state,
        },
        "exact_model_tensor_available": exact_model_tensor_available,
        "raw_six_view_resolution_1600x900": raw_resolution_1600x900,
        "family_readiness": {
            "front_stale": {
                "orion_offline_replay_ready": generic_orion_replay_ready and stale_cadence_ready,
                "saved_cadence_supports_200_or_400ms": stale_cadence_ready,
                "reason": (
                    "ready"
                    if generic_orion_replay_ready and stale_cadence_ready
                    else "saved cameras are diagnostic 10-step samples, not 20Hz history"
                ),
            },
            "lens_waterdrop_paired_template": {
                "visual_reapplication_ready": waterdrop_visual_reapply_ready,
                "orion_offline_replay_ready": generic_orion_replay_ready,
                "reason": (
                    "ready"
                    if generic_orion_replay_ready
                    else "raw frames can be transformed, but exact sequential ORION state/input metadata is absent"
                ),
            },
            "native_motion_blur": {
                "orion_offline_replay_ready": False,
                "reason": "native blur is a CARLA sensor-render condition and no paired blurred capture exists in the clean package",
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite O1 readiness audit")
    windows = read_json(args.event_windows)
    if windows.get("schema") != "orion.corruption_hardcase_event_windows.v1":
        raise ValueError("unexpected event-window schema")
    events = [audit_event(event) for event in windows["events"]]
    package_valid_count = sum(event["package_valid"] for event in events)
    family_counts = {
        family: sum(
            bool(event["family_readiness"][family]["orion_offline_replay_ready"])
            for event in events
        )
        for family in (
            "front_stale",
            "lens_waterdrop_paired_template",
            "native_motion_blur",
        )
    }
    output = {
        "schema": SCHEMA,
        "status": "audited_current_event_packages_not_ready_for_orion_offline_replay",
        "event_windows": {
            "path": str(args.event_windows),
            "sha256": sha256(args.event_windows),
        },
        "summary": {
            "events": len(events),
            "package_valid": package_valid_count,
            "all_packages_valid": package_valid_count == len(events),
            "orion_offline_replay_ready_by_family": family_counts,
            "waterdrop_visual_reapplication_ready": sum(
                bool(event["family_readiness"]["lens_waterdrop_paired_template"]["visual_reapplication_ready"])
                for event in events
            ),
        },
        "decision": {
            "reuse_existing_packages_for_front_stale_orion_replay": False,
            "reuse_existing_packages_for_waterdrop_orion_replay": False,
            "reuse_existing_packages_for_native_motion_blur_orion_replay": False,
            "allowed_reuse": "paired-template waterdrop visual/parameter diagnostics only",
            "required_next_step": "choose between high-rate sequential input-state capture followed by single-load replay, or a smaller prospective online ORION route screen after visual approval",
        },
        "events": events,
        "claim_boundary": "Readiness and provenance audit only; no corruption-conditioned ORION inference or safety outcome was produced.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0 if package_valid_count == len(events) else 1


if __name__ == "__main__":
    raise SystemExit(main())
