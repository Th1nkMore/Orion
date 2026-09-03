#!/usr/bin/env python3
"""CPU-only geometry preflight for reusing Route196 as train coverage.

Route196 was frozen earlier as an engineering-overfit *train* event.  This
script applies the already frozen five temporal offsets to its original trace,
rebuilds task-relevance geometry from raw frame metadata, and reports camera
support.  It never reads a Route196 model report or any dev/locked-test result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.scenario_factory_lib import load_jsonl
from uq_estimator.task_relevance_geometry import CAMERA_ORDER, build_task_relevance_map


SCHEMA = "orion.stage2l_v12_route196_coverage_reuse_preflight.v1"
EVENT_RE = re.compile(r"^route(?P<route>\d+)_step(?P<step>\d+)$")
CAMERA_DIRECTORY_BY_VIEW = {
    "CAM_FRONT": "rgb_front_model_input",
    "CAM_FRONT_LEFT": "rgb_front_left",
    "CAM_FRONT_RIGHT": "rgb_front_right",
    "CAM_BACK": "rgb_back",
    "CAM_BACK_LEFT": "rgb_back_left",
    "CAM_BACK_RIGHT": "rgb_back_right",
}


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference(path: Path) -> Dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _closest_trace_rows_by_frame(
    trace: Sequence[Mapping[str, Any]], frames: Iterable[int]
) -> Dict[int, Mapping[str, Any]]:
    if not trace:
        raise ValueError("control trace is empty")
    return {
        frame: min(
            trace,
            key=lambda row: (abs(int(row["step"]) - frame * 10), int(row["step"])),
        )
        for frame in frames
    }


def select_fixed_frames(
    *,
    trace: Sequence[Mapping[str, Any]],
    available_frames: Sequence[int],
    critical_step: int,
    offsets_seconds: Sequence[float],
) -> Sequence[Dict[str, Any]]:
    if not available_frames:
        raise ValueError("no saved frames are available")
    offsets = tuple(float(value) for value in offsets_seconds)
    if tuple(sorted(offsets)) != offsets or len(offsets) != len(set(offsets)):
        raise ValueError("offsets must be unique and increasing")
    if 0.0 not in offsets:
        raise ValueError("offsets must include zero")
    center_row = min(
        trace,
        key=lambda row: (abs(int(row["step"]) - critical_step), int(row["step"])),
    )
    center_time = float(center_row["sim_time_seconds"])
    rows_by_frame = _closest_trace_rows_by_frame(trace, available_frames)
    unused = set(map(int, available_frames))
    selected = []
    for offset in offsets:
        if not unused:
            raise ValueError("insufficient distinct frames")
        target_time = center_time + offset
        frame = min(
            unused,
            key=lambda value: (
                abs(float(rows_by_frame[value]["sim_time_seconds"]) - target_time),
                abs(value - critical_step / 10.0),
                value,
            ),
        )
        unused.remove(frame)
        row = rows_by_frame[frame]
        selected.append(
            {
                "requested_offset_seconds": offset,
                "requested_sim_time_seconds": target_time,
                "selected_saved_frame_index": frame,
                "selected_control_step": int(row["step"]),
                "selected_sim_time_seconds": float(row["sim_time_seconds"]),
                "selection_time_error_seconds": abs(
                    float(row["sim_time_seconds"]) - target_time
                ),
            }
        )
    return sorted(selected, key=lambda row: row["selected_saved_frame_index"])


def _camera_integrity(meta_root: Path, frame: int) -> Sequence[Dict[str, Any]]:
    scenario_root = meta_root.parent
    rows = []
    for view in CAMERA_ORDER:
        directory = CAMERA_DIRECTORY_BY_VIEW[view]
        path = scenario_root / directory / ("%04d.png" % frame)
        if not path.is_file():
            raise FileNotFoundError("missing camera frame: %s" % path)
        rows.append({"view": view, "path": str(path), "sha256": _sha256(path)})
    return rows


def preflight(
    *,
    schedule_path: Path,
    formal_plan_path: Path,
    source_qa_dataset_path: Path,
    control_trace_path: Path,
    meta_root: Path,
) -> Dict[str, Any]:
    schedule = _read_json(schedule_path)
    formal_plan = _read_json(formal_plan_path)
    source_qa = _read_json(source_qa_dataset_path)
    if schedule.get("schema") != "orion.stage2_l.schedule.v1":
        raise ValueError("Stage2-L schedule schema differs")
    if formal_plan.get("schema") != "orion.stage2_l.formal_route_plan.v1":
        raise ValueError("formal route plan schema differs")
    event_id = str(schedule["engineering_overfit_smoke"]["event_id"])
    match = EVENT_RE.fullmatch(event_id)
    if match is None or int(match.group("route")) != 196:
        raise ValueError("frozen engineering event is not Route196")
    critical_step = int(match.group("step"))
    offsets = schedule["fixed_keyframe_policy"][
        "offsets_seconds_from_actor_grounded_event"
    ]
    minimum_frames = int(
        schedule["fixed_keyframe_policy"]["minimum_keyframes_per_event"]
    )
    if [float(value) for value in offsets] != [-2.0, -1.0, 0.0, 1.0, 2.0]:
        raise ValueError("Route196 fixed offsets differ")
    formal_events = {int(row["route_index"]): row for row in formal_plan["events"]}
    if 196 in formal_events:
        raise ValueError("Route196 unexpectedly overlaps the formal 24-event plan")
    if source_qa.get("schema") != "orion.uq_relevance_qa_dataset.v1":
        raise ValueError("Route196 source QA schema differs")
    records = source_qa.get("records", [])
    if not records or {str(row["event_id"]) for row in records} != {event_id}:
        raise ValueError("source QA does not identify only the frozen Route196 event")
    if {str(row["split"]) for row in records} != {"train"}:
        raise ValueError("Route196 source QA is not train-only")

    meta_root = meta_root.resolve()
    available = sorted(int(path.stem) for path in meta_root.glob("*.json"))
    trace = load_jsonl(control_trace_path.resolve())
    selected = select_fixed_frames(
        trace=trace,
        available_frames=available,
        critical_step=critical_step,
        offsets_seconds=offsets,
    )
    per_view_positive_frames = {view: 0 for view in CAMERA_ORDER}
    per_frame = []
    for row in selected:
        frame = int(row["selected_saved_frame_index"])
        meta_path = meta_root / ("%04d.json" % frame)
        meta = _read_json(meta_path)
        geometry = build_task_relevance_map(
            meta["plan"], meta["closedloop_safety"], patch_hw=(40, 40)
        )
        view_rows = {}
        for index, view in enumerate(CAMERA_ORDER):
            relevance_cells = int((geometry.relevance[index] > 0.0).sum())
            route_cells = int((geometry.route_corridor[index] > 0.0).sum())
            actor_cells = int((geometry.relevant_actor_support[index] > 0.0).sum())
            if relevance_cells:
                per_view_positive_frames[view] += 1
            view_rows[view] = {
                "relevance_cells": relevance_cells,
                "route_corridor_cells": route_cells,
                "conflict_actor_cells": actor_cells,
                "peak": float(geometry.relevance[index].max()),
            }
        per_frame.append(
            {
                **row,
                "meta": _reference(meta_path),
                "camera_files": _camera_integrity(meta_root, frame),
                "support_mode": geometry.provenance["support_mode"],
                "relevant_actor_ids": list(geometry.relevant_actor_ids),
                "route_point_coverage": float(geometry.route_point_coverage),
                "per_view": view_rows,
            }
        )
    eligible = len(per_frame) >= minimum_frames
    if not eligible:
        raise RuntimeError("Route196 fixed keyframes do not meet the frozen minimum")
    return {
        "schema": SCHEMA,
        "status": "route196_train_coverage_candidate_geometry_eligible",
        "passed": True,
        "gpu_used": False,
        "orion_forward_run": False,
        "training_started": False,
        "formal_bank_modified": False,
        "event_id": event_id,
        "route_index": 196,
        "source_role": "prior_engineering_train_event_not_held_out_evidence",
        "fixed_keyframes": per_frame,
        "fixed_keyframe_count": len(per_frame),
        "minimum_keyframes": minimum_frames,
        "per_view_positive_frame_count": per_view_positive_frames,
        "coverage_contribution": {
            "adds_one_independent_cam_front_right_train_event": (
                per_view_positive_frames["CAM_FRONT_RIGHT"] > 0
            ),
            "adds_one_independent_cam_front_left_train_event": (
                per_view_positive_frames["CAM_FRONT_LEFT"] > 0
            ),
            "adds_one_independent_cam_back_left_train_event": (
                per_view_positive_frames["CAM_BACK_LEFT"] > 0
            ),
            "fills_cam_back_right": per_view_positive_frames["CAM_BACK_RIGHT"] > 0,
        },
        "selection_boundary": {
            "selected_by_frozen_event_identity_and_fixed_offsets": True,
            "geometry_used_for_train_coverage_acceptance": True,
            "route196_model_reports_read": False,
            "dev_result_files_read": [],
            "locked_test_result_files_read": [],
            "observation_uq_or_qa_answers_used": False,
            "collision_or_closed_loop_improvement_used": False,
        },
        "inputs": {
            "schedule": _reference(schedule_path),
            "formal_plan": _reference(formal_plan_path),
            "source_qa_dataset": _reference(source_qa_dataset_path),
            "control_trace": _reference(control_trace_path),
        },
        "launch_locks": {
            "gpu_r_only_smoke": False,
            "language_bridge": False,
            "stage2p": False,
            "reason": "Route196 can add one side-view event but CAM_BACK_RIGHT remains uncovered.",
        },
        "claim_boundary": (
            "CPU geometry and source-integrity evidence for one train-only reuse "
            "candidate. It is not an independent event by frame count, cannot be "
            "held-out evidence, does not fill back-right coverage, and does not show "
            "R generalization, U semantics, planning, or safety improvement."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--formal-plan", type=Path, required=True)
    parser.add_argument("--source-qa-dataset", type=Path, required=True)
    parser.add_argument("--control-trace", type=Path, required=True)
    parser.add_argument("--meta-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Route196 coverage preflight")
    value = preflight(
        schedule_path=args.schedule,
        formal_plan_path=args.formal_plan,
        source_qa_dataset_path=args.source_qa_dataset,
        control_trace_path=args.control_trace,
        meta_root=args.meta_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": value["status"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
