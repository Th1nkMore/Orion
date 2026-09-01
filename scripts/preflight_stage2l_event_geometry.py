#!/usr/bin/env python3
"""Preflight fixed Stage2-L keyframes before any Stage-1 GPU extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.scenario_factory_lib import sha256_file
from uq_estimator.task_relevance_geometry import (
    TaskRelevanceGeometryError,
    build_task_relevance_map,
)


SCHEMA = "orion.stage2l_event_geometry_preflight.v1"


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def preflight_geometry(
    *, event_package_path: Path, keyframe_manifest_path: Path,
    minimum_retained: int = 3,
) -> Dict[str, Any]:
    event_package_path = event_package_path.resolve()
    keyframe_manifest_path = keyframe_manifest_path.resolve()
    event = _load(event_package_path)
    keyframes = _load(keyframe_manifest_path)
    if event.get("schema") != "orion.scenario_event_package.v1":
        raise ValueError("unsupported event-package schema")
    if keyframes.get("schema") != "orion.scenario_event_keyframes.v1":
        raise ValueError("unsupported keyframe-manifest schema")
    event_ref = keyframes.get("provenance", {}).get("event_package", {})
    if event_ref.get("sha256") != sha256_file(event_package_path):
        raise ValueError("keyframe manifest and event package SHA-256 differ")
    expected_event_id = "route%s_step%s" % (
        event["route"]["route_index"], event["critical_event"]["step"]
    )
    if keyframes.get("event_id") != expected_event_id:
        raise ValueError("keyframe manifest and event package identify different events")
    if minimum_retained < 1:
        raise ValueError("minimum retained keyframes must be positive")

    front_root = Path(event["camera_inventory"]["rgb_front"]["path"])
    meta_root = front_root.parent / "meta"
    retained = []
    excluded = []
    for row in keyframes.get("keyframes", []):
        frame = int(row["selected_saved_frame_index"])
        meta_path = meta_root / ("%04d.json" % frame)
        if not meta_path.is_file():
            raise FileNotFoundError("fixed keyframe metadata is missing: %s" % meta_path)
        meta = _load(meta_path)
        try:
            geometry = build_task_relevance_map(
                meta["plan"], meta["closedloop_safety"], patch_hw=(40, 40)
            )
        except TaskRelevanceGeometryError as error:
            empty_support_errors = {
                "the ORION route has no visible camera support",
                "task relevance has no visible route or conflict-actor support",
            }
            if str(error) not in empty_support_errors:
                raise
            excluded.append({
                "selected_saved_frame_index": frame,
                "reason": "task_relevance_has_no_visible_support",
                "meta": {"path": str(meta_path), "sha256": sha256_file(meta_path)},
            })
            continue
        retained.append({
            "selected_saved_frame_index": frame,
            "route_point_coverage": geometry.route_point_coverage,
            "support_mode": getattr(geometry, "provenance", {}).get(
                "support_mode", "legacy_route_only"
            ),
            "relevant_actor_ids": list(getattr(geometry, "relevant_actor_ids", ())),
            "meta": {"path": str(meta_path), "sha256": sha256_file(meta_path)},
        })
    eligible = len(retained) >= minimum_retained
    return {
        "schema": SCHEMA,
        "status": (
            "eligible_before_stage1_extraction"
            if eligible else "ineligible_before_stage1_extraction"
        ),
        "eligible": eligible,
        "event_id": expected_event_id,
        "requested_keyframe_count": len(keyframes.get("keyframes", [])),
        "retained_keyframe_count": len(retained),
        "minimum_retained_keyframes": minimum_retained,
        "retained": retained,
        "excluded": excluded,
        "selection_and_label_inputs": {
            "fixed_keyframes_unchanged": True,
            "uses_observation_uq": False,
            "uses_stage1_adapter_outputs": False,
            "uses_stage2_outputs": False,
            "uses_qa_answers": False,
            "uses_collision_or_task_response_outcomes": False,
            "actor_only_task_relevance_allowed": True,
        },
        "provenance": {
            "event_package": {
                "path": str(event_package_path),
                "sha256": sha256_file(event_package_path),
            },
            "keyframe_manifest": {
                "path": str(keyframe_manifest_path),
                "sha256": sha256_file(keyframe_manifest_path),
            },
        },
        "claim_boundary": (
            "CPU-only geometric eligibility check before Stage1 extraction; "
            "not an uncertainty, language, planning, or safety result."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-package", type=Path, required=True)
    parser.add_argument("--keyframe-manifest", type=Path, required=True)
    parser.add_argument("--minimum-retained", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = preflight_geometry(
        event_package_path=args.event_package,
        keyframe_manifest_path=args.keyframe_manifest,
        minimum_retained=args.minimum_retained,
    )
    if args.output:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite Stage2-L geometry preflight")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["eligible"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
