#!/usr/bin/env python3
"""Scan accepted train-only raw frames for missing back-right R support."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uq_estimator.task_relevance_geometry import (
    CAMERA_ORDER,
    TaskRelevanceGeometryError,
    build_task_relevance_map,
)


SCHEMA = "orion.stage2l_v12_train_back_right_geometry_scan.v1"
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


def rank_candidates(rows, *, minimum_positive_frames: int):
    return sorted(
        [
            row
            for row in rows
            if row["per_view_positive_frame_count"]["CAM_BACK_RIGHT"]
            >= minimum_positive_frames
            and row["frozen_geometry_reselection_locked"] is False
        ],
        key=lambda row: (
            -row["per_view_positive_frame_count"]["CAM_BACK_RIGHT"],
            row["route_index"],
        ),
    )


def _aligned_frames(event_package: Mapping[str, Any]) -> tuple[Path, list[int]]:
    front_root = Path(event_package["camera_inventory"]["rgb_front"]["path"])
    scenario_root = front_root.parent
    meta_root = scenario_root / "meta"
    streams = [{int(path.stem) for path in meta_root.glob("*.json")}]
    for view in CAMERA_ORDER:
        directory = CAMERA_DIRECTORY_BY_VIEW[view]
        streams.append(
            {int(path.stem) for path in (scenario_root / directory).glob("*.png")}
        )
    aligned = sorted(set.intersection(*streams))
    if not aligned:
        raise ValueError("event has no aligned six-view/meta frames")
    return meta_root, aligned


def scan(
    *,
    protocol_path: Path,
    formal_plan_path: Path,
    accepted_bank_path: Path,
    route177_geometry_gate_path: Path,
) -> Dict[str, Any]:
    protocol = _read_json(protocol_path)
    formal_plan = _read_json(formal_plan_path)
    bank = _read_json(accepted_bank_path)
    route177_gate = _read_json(route177_geometry_gate_path)
    if protocol.get("schema") != "orion.stage2l_v12_train_coverage_repair_protocol.v1":
        raise ValueError("coverage repair protocol schema differs")
    if protocol.get("status") != "preregistered_cpu_geometry_scan_not_training_authority":
        raise ValueError("coverage repair protocol is not preregistered")
    refs = protocol["authoritative_inputs"]
    expected = {
        "formal_route_plan": formal_plan_path,
        "accepted_event_bank": accepted_bank_path,
    }
    for name, path in expected.items():
        if refs[name]["sha256"] != _sha256(path):
            raise ValueError("protocol input hash differs: %s" % name)
    if formal_plan.get("schema") != "orion.stage2_l.formal_route_plan.v1":
        raise ValueError("formal plan schema differs")
    if bank.get("schema") != "orion.stage2_l.formal_event_bank.v1":
        raise ValueError("accepted bank schema differs")
    if route177_gate.get("status") != "formal_event_qa_ineligible_under_frozen_geometry_gate":
        raise ValueError("Route177 geometry disposition differs")
    formal_by_route = {int(row["route_index"]): row for row in formal_plan["events"]}
    train_events = sorted(
        [row for row in bank["events"] if row["formal_split"] == "train"],
        key=lambda row: int(row["route_index"]),
    )
    if len(train_events) != 14:
        raise ValueError("accepted train event count differs")
    event_rows = []
    dereferenced_routes = []
    for event in train_events:
        route = int(event["route_index"])
        if formal_by_route.get(route, {}).get("formal_split") != "train":
            raise ValueError("accepted train event differs from formal plan")
        ref = event["event_package"]
        event_path = Path(ref["path"])
        if _sha256(event_path) != ref["sha256"]:
            raise ValueError("event package hash differs for route%d" % route)
        package = _read_json(event_path)
        if (
            package.get("schema") != "orion.scenario_event_package.v1"
            or package.get("runtime", {}).get("valid") is not True
            or package.get("qa_input_ready") is not True
        ):
            raise ValueError("accepted event package is not raw-scan eligible")
        dereferenced_routes.append(route)
        meta_root, frames = _aligned_frames(package)
        per_view = {view: 0 for view in CAMERA_ORDER}
        back_right_frames = []
        modes = Counter()
        geometry_errors = Counter()
        for frame in frames:
            meta_path = meta_root / ("%04d.json" % frame)
            meta = _read_json(meta_path)
            try:
                geometry = build_task_relevance_map(
                    meta["plan"], meta["closedloop_safety"], patch_hw=(40, 40)
                )
            except TaskRelevanceGeometryError as error:
                geometry_errors[str(error)] += 1
                continue
            modes[geometry.provenance["support_mode"]] += 1
            for index, view in enumerate(CAMERA_ORDER):
                if bool((geometry.relevance[index] > 0.0).any()):
                    per_view[view] += 1
                    if view == "CAM_BACK_RIGHT":
                        back_right_frames.append(frame)
        event_rows.append(
            {
                "route_index": route,
                "event_id": event["event_id"],
                "scenario_family": event["scenario_family"],
                "town": event["town"],
                "aligned_frame_count": len(frames),
                "geometry_valid_frame_count": int(sum(modes.values())),
                "per_view_positive_frame_count": per_view,
                "cam_back_right_positive_frames": back_right_frames,
                "support_mode_counts": dict(sorted(modes.items())),
                "geometry_error_counts": dict(sorted(geometry_errors.items())),
                "frozen_geometry_reselection_locked": route == 177,
                "event_package": _reference(event_path),
            }
        )
    minimum = int(
        protocol["candidate_gate"][
            "minimum_positive_saved_frames_in_one_independent_event"
        ]
    )
    candidates = rank_candidates(event_rows, minimum_positive_frames=minimum)
    return {
        "schema": SCHEMA,
        "status": (
            "existing_train_back_right_candidates_found"
            if candidates
            else "no_existing_train_back_right_candidate"
        ),
        "passed": True,
        "gpu_used": False,
        "orion_forward_run": False,
        "training_started": False,
        "formal_bank_modified": False,
        "inputs": {
            "protocol": _reference(protocol_path),
            "formal_plan": _reference(formal_plan_path),
            "accepted_bank": _reference(accepted_bank_path),
            "route177_geometry_gate": _reference(route177_geometry_gate_path),
        },
        "inspection_boundary": {
            "train_event_packages_dereferenced": dereferenced_routes,
            "dev_event_packages_dereferenced": [],
            "locked_test_event_packages_or_results_read": [],
            "model_predictions_or_losses_read": False,
            "observation_uq_or_qa_answers_read": False,
            "collision_or_infraction_outcomes_used": False,
        },
        "minimum_back_right_positive_frames": minimum,
        "events": event_rows,
        "ranked_candidates": candidates,
        "next_action": (
            "freeze top existing train candidate fixed keyframes for visual geometry review"
            if candidates
            else "activate the preregistered static XML fallback beginning with Route167"
        ),
        "launch_locks": {
            "gpu_r_only_smoke": False,
            "language_bridge": False,
            "stage2p": False,
        },
        "claim_boundary": "CPU train-only geometry inventory. Candidate discovery does not make selected frames formal data, does not read held-out outcomes, and does not establish R, U, language, planning, or safety performance.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--formal-plan", type=Path, required=True)
    parser.add_argument("--accepted-bank", type=Path, required=True)
    parser.add_argument("--route177-geometry-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite train back-right scan")
    value = scan(
        protocol_path=args.protocol,
        formal_plan_path=args.formal_plan,
        accepted_bank_path=args.accepted_bank,
        route177_geometry_gate_path=args.route177_geometry_gate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": value["status"], "candidates": [row["route_index"] for row in value["ranked_candidates"]], "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
