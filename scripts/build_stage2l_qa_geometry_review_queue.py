#!/usr/bin/env python3
"""Build a hash-bound human review queue for multi-frame Stage2-L geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

try:
    from scripts.scenario_factory_lib import sha256_file
except ModuleNotFoundError:
    from scenario_factory_lib import sha256_file


QUEUE_SCHEMA = "orion.stage2_l.qa_geometry_review_queue.v1"
DECISIONS_SCHEMA = "orion.stage2_l.qa_geometry_review_decisions.v1"
FACTORY_SCHEMA = "orion.uq_relevance_multiframe_event_factory.v1"
VARIANTS = ("observed", "zero_uq", "on_path_uq", "off_path_uq", "view_shuffled_uq")
HUMAN_CHECKS = (
    "visual_artifact_integrity",
    "route_corridor_projection_plausible",
    "relevant_actor_support_alignment",
    "on_off_counterfactual_geometry",
    "map_text_consistency",
)


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _resolve(reference: Mapping[str, Any], base: Path, name: str) -> Path:
    path = Path(str(reference.get("path", "")))
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.is_file() or sha256_file(path) != reference.get("sha256"):
        raise ValueError("%s is absent or has a SHA-256 mismatch" % name)
    return path


def build_queue(report_paths: Sequence[Path]) -> Dict[str, Any]:
    if not report_paths:
        raise ValueError("at least one multi-frame factory report is required")
    review_order = []
    seen = set()
    for report_path in report_paths:
        report = _load(report_path)
        if report.get("schema") != FACTORY_SCHEMA or report.get("status") != "pending_multiframe_human_geometry_review":
            raise ValueError("factory report is not pending Stage2-L geometry review")
        event_id = str(report.get("event_id", ""))
        if not event_id or event_id in seen:
            raise ValueError("factory reports have absent or duplicate event ids")
        seen.add(event_id)
        keyframes = [int(value) for value in report.get("selected_saved_frames", [])]
        if len(keyframes) < 3 or len(keyframes) > 5 or len(keyframes) != int(report.get("keyframe_count", 0)):
            raise ValueError("factory report violates the fixed keyframe contract")
        visual_by_frame = {frame: {} for frame in keyframes}
        for row in report.get("visualizations", []):
            frame = int(row["selected_saved_frame_index"])
            variant = str(row["variant"])
            if frame not in visual_by_frame or variant in visual_by_frame[frame]:
                raise ValueError("factory visualization frame/variant is invalid or duplicated")
            manifest_path = _resolve(row["manifest"], report_path.parent, "visualization manifest")
            contact_sheet = _resolve(row["contact_sheet"], report_path.parent, "U/R/K contact sheet")
            visual_by_frame[frame][variant] = {
                "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
                "contact_sheet": {"path": str(contact_sheet), "sha256": sha256_file(contact_sheet)},
            }
        if any(set(rows) != set(VARIANTS) for rows in visual_by_frame.values()):
            raise ValueError("every keyframe must expose all five UQ variants for review")
        review_order.append({
            "event_id": event_id,
            "factory_report": {"path": str(report_path.resolve()), "sha256": sha256_file(report_path)},
            "event_package": report["event_package"],
            "keyframe_count": len(keyframes),
            "selected_saved_frames": keyframes,
            "visualizations": [
                {"selected_saved_frame_index": frame, "variants": visual_by_frame[frame]}
                for frame in keyframes
            ],
            "required_checks": list(HUMAN_CHECKS),
        })
    return {
        "schema": QUEUE_SCHEMA,
        "status": "pending_human_qa_geometry_review",
        "human_review_count": len(review_order),
        "review_order": review_order,
        "claim_boundary": "Geometry review validates QA construction integrity only; it does not validate Stage1 uncertainty or VLM understanding.",
    }


def decisions_template(queue: Mapping[str, Any], queue_path: Path) -> Dict[str, Any]:
    return {
        "schema": DECISIONS_SCHEMA,
        "status": "unreviewed_template",
        "reviewer": None,
        "reviewed_at": None,
        "review_queue": {"path": str(queue_path.resolve()), "sha256": sha256_file(queue_path)},
        "decisions": [
            {
                "event_id": row["event_id"],
                "factory_report_sha256": row["factory_report"]["sha256"],
                "decision": "pending",
                "checks": {name: "pending" for name in HUMAN_CHECKS},
                "rejection_basis": None,
                "notes": "",
            }
            for row in queue["review_order"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory-report", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite QA geometry review queue")
    queue = build_queue([path.resolve() for path in args.factory_report])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = args.output_dir / "qa_geometry_review_queue.json"
    queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    template = decisions_template(queue, queue_path)
    template_path = args.output_dir / "qa_geometry_review_decisions.template.json"
    template_path.write_text(json.dumps(template, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"queue": str(queue_path.resolve()), "decisions_template": str(template_path.resolve()), "human_review_count": queue["human_review_count"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
