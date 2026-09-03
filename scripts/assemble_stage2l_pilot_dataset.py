#!/usr/bin/env python3
"""Assemble frozen per-event QA and visual caches into one Stage2-L pilot."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

try:
    from scripts.scenario_factory_lib import sha256_file
except ModuleNotFoundError:
    from scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2_l.pilot_dataset.v1"
PILOT_BANK_STATUSES = {
    "orion.stage2_l.pilot_event_bank.v1": "frozen_before_stage2l_pilot_training",
    "orion.stage2_l.formal_pilot_event_bank.v1": "frozen_bank_training_still_locked",
}
SCHEDULE_SCHEMAS = {
    "orion.stage2_l.schedule.v1",
    "orion.stage2_l.schedule.v2",
}
FACTORY_SCHEMA = "orion.uq_relevance_multiframe_event_factory.v1"
VISUAL_CACHE_SCHEMA = "orion.stage2l_multiframe_visual_context_cache.v1"
QA_REVIEW_SCHEMA = "orion.stage2_l.qa_geometry_review_bank.v1"


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _resolve(reference: Mapping[str, Any], base: Path, name: str) -> Path:
    path = Path(str(reference.get("path", reference.get("output", ""))))
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.is_file() or sha256_file(path) != reference.get("sha256"):
        raise ValueError("%s is absent or has a SHA-256 mismatch" % name)
    return path


def _materialize_record_paths(
    row: Mapping[str, Any], records_path: Path
) -> Dict[str, Any]:
    """Keep per-event sidecars valid after records move into the pilot dataset."""
    result = copy.deepcopy(dict(row))
    sidecar = result.get("target", {}).get("map_sidecar")
    if isinstance(sidecar, dict):
        resolved = _resolve(sidecar, records_path.parent, "QA map sidecar")
        sidecar["path"] = str(resolved)
    return result


def assemble_pilot(
    *,
    pilot_bank_path: Path,
    schedule_path: Path,
    qa_review_bank_path: Path,
    factory_reports: Sequence[Path],
    visual_cache_manifests: Sequence[Path],
) -> Dict[str, Any]:
    bank = _load(pilot_bank_path)
    schedule = _load(schedule_path)
    qa_review = _load(qa_review_bank_path)
    bank_schema = str(bank.get("schema", ""))
    formal_split_mode = bank_schema == "orion.stage2_l.formal_pilot_event_bank.v1"
    if bank_schema not in PILOT_BANK_STATUSES or bank.get("status") != PILOT_BANK_STATUSES[bank_schema]:
        raise ValueError("Stage2-L pilot bank is not frozen")
    if schedule.get("schema") not in SCHEDULE_SCHEMAS:
        raise ValueError("unsupported Stage2-L schedule")
    if qa_review.get("schema") != QA_REVIEW_SCHEMA or qa_review.get("status") != "frozen_human_qa_geometry_review":
        raise ValueError("Stage2-L QA geometry review is not frozen")
    events = {str(row["event_id"]): row for row in bank.get("events", [])}
    if len(events) != len(bank.get("events", [])):
        raise ValueError("pilot bank duplicates event ids")
    gate = schedule["pilot_gate"]
    if len(events) != int(gate["minimum_independent_events"]):
        raise ValueError("pilot event count differs from the frozen schedule")
    if len(factory_reports) != len(events) or len(visual_cache_manifests) != len(events):
        raise ValueError("pilot requires one QA factory and one visual cache per event")

    factory_by_event = {}
    for path in factory_reports:
        report = _load(path)
        if report.get("schema") != FACTORY_SCHEMA:
            raise ValueError("unsupported multi-frame QA factory report")
        event_id = str(report.get("event_id", ""))
        if event_id in factory_by_event:
            raise ValueError("duplicate QA factory event")
        factory_by_event[event_id] = (path, report)
    cache_by_event = {}
    for path in visual_cache_manifests:
        manifest = _load(path)
        if manifest.get("schema") != VISUAL_CACHE_SCHEMA:
            raise ValueError("unsupported multi-frame visual cache manifest")
        report_path = _resolve(
            manifest["event_factory_report"], path.parent, "visual-cache factory report"
        )
        event_id = str(_load(report_path).get("event_id", ""))
        if event_id in cache_by_event:
            raise ValueError("duplicate visual-cache event")
        _resolve(manifest, path.parent, "visual-context cache")
        if any(manifest.get(key) is not False for key in (
            "privileged_safety_inputs_used",
            "stage1_uq_inputs_used",
            "task_relevance_targets_used",
            "qa_answers_used",
        )):
            raise ValueError("visual-context cache used prohibited pilot inputs")
        cache_by_event[event_id] = (path, manifest)
    if set(factory_by_event) != set(events) or set(cache_by_event) != set(events):
        raise ValueError("QA factories or visual caches do not match the frozen pilot events")
    reviewed = {str(row["event_id"]): row for row in qa_review.get("accepted", [])}
    if len(reviewed) != len(qa_review.get("accepted", [])) or set(reviewed) != set(events):
        raise ValueError("accepted QA geometry reviews do not exactly cover the pilot events")
    for event_id, (report_path, _) in factory_by_event.items():
        if reviewed[event_id]["factory_report"].get("sha256") != sha256_file(report_path):
            raise ValueError("accepted QA geometry review factory-report hash mismatch")

    combined: List[Dict[str, Any]] = []
    event_rows = []
    for event_id, event in sorted(events.items()):
        report_path, report = factory_by_event[event_id]
        cache_path, cache = cache_by_event[event_id]
        keyframes = int(report["keyframe_count"])
        if keyframes < 3 or keyframes > 5:
            raise ValueError("pilot event violates fixed keyframe count")
        records_path = _resolve(report["qa_dataset"]["records"], report_path.parent, "QA records")
        records = [
            _materialize_record_paths(json.loads(line), records_path)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) != keyframes * int(schedule["fixed_keyframe_policy"]["records_per_keyframe"]):
            raise ValueError("pilot event QA count differs from 20 per keyframe")
        split = str(event["pilot_split"])
        if formal_split_mode and (
            event.get("formal_split") != split
            or bank.get("selection_policy", {}).get("reassigns_frozen_splits") is not False
        ):
            raise ValueError("formal pilot dataset does not preserve frozen splits")
        if any(str(row.get("event_id")) != event_id or row.get("split") != split for row in records):
            raise ValueError("QA records disagree with the frozen event or pilot split")
        group_counts = Counter(str(row["counterfactual"]["group_id"]) for row in records)
        if len(group_counts) != keyframes or set(group_counts.values()) != {20}:
            raise ValueError("each fixed keyframe must contain five variants by four QA families")
        if set(cache["group_ids"]) != set(group_counts):
            raise ValueError("visual cache does not cover every QA keyframe group")
        combined.extend(records)
        event_rows.append({
            "event_id": event_id,
            "route_index": int(event["route_index"]),
            "split": split,
            "town": event["town"],
            "scenario_family": event["scenario_family"],
            "keyframe_count": keyframes,
            "qa_record_count": len(records),
            "qa_factory_report": {"path": str(report_path.resolve()), "sha256": sha256_file(report_path)},
            "visual_cache_manifest": {"path": str(cache_path.resolve()), "sha256": sha256_file(cache_path)},
            "visual_cache": {"path": cache["output"], "sha256": cache["sha256"]},
        })
    lower, upper = map(int, gate["expected_qa_records"])
    if not lower <= len(combined) <= upper:
        raise ValueError("combined pilot QA count is outside the frozen 480-800 range")
    combined.sort(key=lambda row: (
        row["split"], str(row["event_id"]), str(row["frame_id"]),
        str(row["counterfactual"]["variant"]), str(row["question_family"]),
    ))
    split_counts = Counter(row["split"] for row in combined)
    return {
        "schema": SCHEMA,
        "status": (
            "assembled_data_training_launch_locked"
            if formal_split_mode
            else "assembled_ready_for_stage2l_pilot_training"
        ),
        "pilot_training_ready": False if formal_split_mode else None,
        "formal_splits_preserved": formal_split_mode,
        "formal_training_ready": False,
        "records": combined,
        "event_count": len(events),
        "qa_record_count": len(combined),
        "qa_split_counts": dict(sorted(split_counts.items())),
        "events": event_rows,
        "provenance": {
            "pilot_event_bank": {"path": str(pilot_bank_path.resolve()), "sha256": sha256_file(pilot_bank_path)},
            "stage2l_schedule": {"path": str(schedule_path.resolve()), "sha256": sha256_file(schedule_path)},
            "qa_geometry_review_bank": {"path": str(qa_review_bank_path.resolve()), "sha256": sha256_file(qa_review_bank_path)},
        },
        "claim_boundary": "Assembled 6/2 Stage2-L pilot data only; no formal, trajectory, closed-loop, or safety claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-bank", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--qa-geometry-review-bank", type=Path, required=True)
    parser.add_argument("--factory-report", type=Path, action="append", required=True)
    parser.add_argument("--visual-cache-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite Stage2-L pilot dataset")
    result = assemble_pilot(
        pilot_bank_path=args.pilot_bank.resolve(),
        schedule_path=args.schedule.resolve(),
        qa_review_bank_path=args.qa_geometry_review_bank.resolve(),
        factory_reports=[path.resolve() for path in args.factory_report],
        visual_cache_manifests=[path.resolve() for path in args.visual_cache_manifest],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in result.pop("records")),
        encoding="utf-8",
    )
    result["records"] = {"path": str(records_path.resolve()), "sha256": sha256_file(records_path)}
    manifest_path = args.output_dir / "pilot_dataset_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path.resolve()), "event_count": result["event_count"], "qa_record_count": result["qa_record_count"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
