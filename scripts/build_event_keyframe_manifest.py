#!/usr/bin/env python3
"""Select fixed-offset temporal keyframes around an actor-grounded event.

Selection uses only the immutable clean event package and preregistered time
offsets.  Learned UQ, Stage2 outputs, collision improvement, and QA answers are
never consulted.  The default five frames are centered at -2, -1, 0, +1 and
+2 seconds relative to the actor-grounded critical-event center.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

try:
    from scripts.scenario_factory_lib import CAMERA_DIRECTORIES, load_jsonl, sha256_file
except ModuleNotFoundError:
    from scenario_factory_lib import CAMERA_DIRECTORIES, load_jsonl, sha256_file


SCHEMA = "orion.scenario_event_keyframes.v1"
DEFAULT_OFFSETS = (-2.0, -1.0, 0.0, 1.0, 2.0)


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _resolve_reference(reference: Mapping[str, Any], base: Path, name: str) -> Path:
    path = Path(str(reference.get("path", "")))
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.is_file():
        raise FileNotFoundError("%s is missing: %s" % (name, path))
    if sha256_file(path) != reference.get("sha256"):
        raise ValueError("%s SHA-256 mismatch" % name)
    return path


def _aligned_saved_frames(event: Mapping[str, Any]) -> List[int]:
    streams = []
    scenario_dir = None
    for directory in CAMERA_DIRECTORIES:
        root = Path(event["camera_inventory"][directory]["path"])
        streams.append({int(path.stem) for path in root.glob("*.png")})
        scenario_dir = root.parent
    if scenario_dir is None:
        raise ValueError("event package has no camera inventory")
    streams.append({int(path.stem) for path in (scenario_dir / "meta").glob("*.json")})
    aligned = sorted(set.intersection(*streams))
    if not aligned:
        raise ValueError("camera, BEV and meta streams have no aligned saved frames")
    return aligned


def _actor_at_row(row: Mapping[str, Any], actor_id: Any) -> Mapping[str, Any]:
    actors = (row.get("closedloop_safety") or {}).get("actors", [])
    for actor in actors:
        if actor.get("actor_id") == actor_id:
            return actor
    return {}


def select_event_keyframes(
    *, event_package_path: Path, offsets_seconds: Sequence[float]
) -> Dict[str, Any]:
    event_package_path = event_package_path.resolve()
    event = _load_json(event_package_path)
    if event.get("schema") != "orion.scenario_event_package.v1":
        raise ValueError("unsupported event-package schema")
    if event.get("qa_input_ready") is not True or event.get("runtime", {}).get("valid") is not True:
        raise ValueError("keyframes require a runtime-valid QA-ready event")
    critical = event.get("critical_event")
    if not isinstance(critical, dict):
        raise ValueError("keyframes require an actor-grounded critical event")
    offsets = tuple(float(value) for value in offsets_seconds)
    if not offsets or tuple(sorted(offsets)) != offsets or len(set(offsets)) != len(offsets):
        raise ValueError("keyframe offsets must be non-empty, unique and increasing")
    if 0.0 not in offsets:
        raise ValueError("keyframe offsets must include the event center at 0 seconds")

    trace_ref = event.get("source_files", {}).get("control_trace")
    if not isinstance(trace_ref, Mapping):
        raise ValueError("event package lacks control-trace provenance")
    trace_path = _resolve_reference(trace_ref, event_package_path.parent, "control trace")
    trace = load_jsonl(trace_path)
    aligned = _aligned_saved_frames(event)
    row_by_frame: Dict[int, Mapping[str, Any]] = {}
    for frame in aligned:
        row_by_frame[frame] = min(
            trace,
            key=lambda row: (abs(int(row["step"]) - frame * 10), int(row["step"])),
        )

    center = float(critical["sim_time_seconds"])
    unused = set(aligned)
    selected = []
    actor_id = critical.get("actor", {}).get("actor_id")
    for offset in offsets:
        target_time = center + offset
        if not unused:
            raise ValueError("insufficient distinct aligned frames for requested offsets")
        frame = min(
            unused,
            key=lambda value: (
                abs(float(row_by_frame[value]["sim_time_seconds"]) - target_time),
                abs(value - float(critical["step"]) / 10.0),
                value,
            ),
        )
        unused.remove(frame)
        row = row_by_frame[frame]
        actor = _actor_at_row(row, actor_id)
        selected.append({
            "requested_offset_seconds": offset,
            "requested_sim_time_seconds": target_time,
            "selected_saved_frame_index": frame,
            "selected_control_step": int(row["step"]),
            "selected_sim_time_seconds": float(row["sim_time_seconds"]),
            "selection_time_error_seconds": abs(float(row["sim_time_seconds"]) - target_time),
            "route_progress": float(row["route_progress"]),
            "speed_mps": float(row["speed"]),
            "critical_actor_snapshot": {
                "actor_id": actor_id,
                "present": bool(actor),
                "obb_ttc_seconds": actor.get("obb_collision_ttc_seconds"),
                "obb_separating_axis_gap_m": actor.get("obb_separating_axis_gap_m"),
            },
        })
    selected.sort(key=lambda row: row["selected_saved_frame_index"])
    if len({row["selected_saved_frame_index"] for row in selected}) != len(offsets):
        raise RuntimeError("keyframe selection produced duplicate frames")
    event_id = "route%s_step%s" % (
        event["route"]["route_index"], critical["step"]
    )
    return {
        "schema": SCHEMA,
        "status": "fixed_temporal_keyframes_selected",
        "event_id": event_id,
        "route_index": int(event["route"]["route_index"]),
        "critical_event_center": {
            "control_step": int(critical["step"]),
            "sim_time_seconds": center,
            "actor_id": actor_id,
        },
        "selection_policy": {
            "policy_id": "fixed_offsets_around_actor_grounded_event_v1",
            "offsets_seconds": list(offsets),
            "nearest_aligned_saved_frame": True,
            "distinct_frames_required": True,
            "uses_learned_uq": False,
            "uses_stage2_outputs": False,
            "uses_qa_answers": False,
            "uses_closed_loop_improvement": False,
        },
        "keyframes": selected,
        "provenance": {
            "event_package": {
                "path": str(event_package_path),
                "sha256": sha256_file(event_package_path),
            },
            "control_trace": {
                "path": str(trace_path),
                "sha256": sha256_file(trace_path),
            },
        },
        "claim_boundary": (
            "Fixed-offset temporal sampling for Stage2-L QA construction only; "
            "keyframe selection is not learned-UQ or closed-loop evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-package", type=Path, required=True)
    parser.add_argument("--offset-seconds", type=float, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite keyframe manifest")
    report = select_event_keyframes(
        event_package_path=args.event_package,
        offsets_seconds=args.offset_seconds or DEFAULT_OFFSETS,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "event_id": report["event_id"],
        "selected_saved_frames": [row["selected_saved_frame_index"] for row in report["keyframes"]],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
