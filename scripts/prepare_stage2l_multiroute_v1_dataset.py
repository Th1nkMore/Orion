#!/usr/bin/env python3
"""Freeze the latest eight-event bank as a V5 multi-route smoke dataset.

This is a CPU-only transformation and integrity audit.  It deliberately does
not reinterpret a machine audit as formal human QA review.  The resulting
dataset is eligible only for the user-authorized bounded pre-experiment;
formal Stage2-L training and Stage2-P remain locked.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from scripts.audit_stage2l_v8_dataset_references import audit_references
from scripts.scenario_factory_lib import sha256_file
from scripts.upgrade_stage2l_v9_qa_records import (
    audit_records,
    upgrade_records,
)


SCHEMA = "orion.stage2l_multiroute_smoke_dataset.v1"
BANK_SCHEMA = "orion.stage2_l.formal_pilot_event_bank.v1"
QUEUE_SCHEMA = "orion.stage2_l.qa_geometry_review_queue.v1"
FACTORY_SCHEMA = "orion.uq_relevance_multiframe_event_factory.v1"
CACHE_SCHEMA = "orion.stage2l_multiframe_visual_context_cache.v1"
MACHINE_AUDIT_SCHEMA = "orion.stage2l_v6_dataset_contract_audit.v1"
EXPECTED_VARIANTS = {
    "observed",
    "zero_uq",
    "on_path_uq",
    "off_path_uq",
    "view_shuffled_uq",
}
EXPECTED_FAMILIES = {
    "observation_semantics",
    "epistemic_limitation",
    "task_relevance",
    "driving_implication",
}


def _read(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON artifact must contain an object: %s" % path)
    return value


def _resolve(
    reference: Mapping[str, Any], base: Path, name: str
) -> Path:
    path = Path(str(reference.get("path", reference.get("output", ""))))
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.is_file():
        raise FileNotFoundError("%s is missing: %s" % (name, path))
    expected = str(reference.get("sha256", ""))
    if expected and sha256_file(path) != expected:
        raise ValueError("%s SHA-256 mismatch" % name)
    return path


def _load_records(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _materialize_sidecar_paths(
    records: Sequence[Mapping[str, Any]], source_records: Path
) -> List[Dict[str, Any]]:
    """Make per-event relative sidecars valid after cross-event aggregation."""

    output: List[Dict[str, Any]] = []
    for source in records:
        row = copy.deepcopy(dict(source))
        sidecar = row["target"]["map_sidecar"]
        sidecar_path = _resolve(
            sidecar, source_records.parent, "task-relevance map sidecar"
        )
        sidecar["path"] = str(sidecar_path)
        output.append(row)
    return output


def _validate_bank_and_queue(
    bank: Mapping[str, Any], queue: Mapping[str, Any]
) -> Tuple[Dict[str, Mapping[str, Any]], Dict[str, Mapping[str, Any]]]:
    if (
        bank.get("schema") != BANK_SCHEMA
        or bank.get("status") != "frozen_bank_training_still_locked"
        or bank.get("selection_policy", {}).get("reassigns_frozen_splits")
        is not False
        or bank.get("checks", {}).get("locked_test_untouched") is not True
    ):
        raise ValueError("latest formal pilot bank is absent or not frozen")
    events = {
        str(row["event_id"]): row for row in bank.get("events", [])
    }
    if len(events) != 8 or len(events) != len(bank.get("events", [])):
        raise ValueError("multi-route smoke requires eight unique events")
    splits = Counter(str(row.get("pilot_split", "")) for row in events.values())
    towns = {str(row.get("town", "")) for row in events.values()}
    families = {
        str(row.get("scenario_family", "")) for row in events.values()
    }
    if splits != {"train": 6, "dev": 2} or len(towns) < 5 or len(families) < 7:
        raise ValueError("multi-route event coverage or 6/2 split changed")
    if any(
        row.get("qa_input_ready") is not True
        or row.get("human_review", {}).get("decision") != "accept"
        for row in events.values()
    ):
        raise ValueError("source event integrity review is incomplete")

    if (
        queue.get("schema") != QUEUE_SCHEMA
        or queue.get("status") != "pending_human_qa_geometry_review"
        or int(queue.get("human_review_count", 0)) != 8
    ):
        raise ValueError("QA geometry queue is absent or has changed state")
    queued = {
        str(row["event_id"]): row for row in queue.get("review_order", [])
    }
    if len(queued) != 8 or set(queued) != set(events):
        raise ValueError("QA geometry queue and event bank differ")
    return events, queued


def _validate_source_event(
    *,
    event_id: str,
    event: Mapping[str, Any],
    queued: Mapping[str, Any],
    queue_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    report_path = _resolve(
        queued["factory_report"], queue_path.parent, "event factory report"
    )
    report = _read(report_path)
    keyframes = int(report.get("keyframe_count", 0))
    if (
        report.get("schema") != FACTORY_SCHEMA
        or report.get("status")
        != "pending_multiframe_human_geometry_review"
        or str(report.get("event_id", "")) != event_id
        or keyframes != int(queued.get("keyframe_count", 0))
        or keyframes < 3
        or keyframes > 5
    ):
        raise ValueError("event factory contract changed: %s" % event_id)
    source_records = _resolve(
        report["qa_dataset"]["records"], report_path.parent, "event QA records"
    )
    rows = _load_records(source_records)
    expected_count = keyframes * 20
    if len(rows) != expected_count:
        raise ValueError("event QA count changed: %s" % event_id)
    groups = Counter(str(row["counterfactual"]["group_id"]) for row in rows)
    pairs = {
        (
            str(row["counterfactual"]["variant"]),
            str(row["question_family"]),
        )
        for row in rows
    }
    if (
        len(groups) != keyframes
        or set(groups.values()) != {20}
        or pairs
        != {
            (variant, family)
            for variant in EXPECTED_VARIANTS
            for family in EXPECTED_FAMILIES
        }
        or any(
            str(row.get("event_id", "")) != event_id
            or str(row.get("split", "")) != str(event["pilot_split"])
            for row in rows
        )
    ):
        raise ValueError("event matched-group/split contract changed: %s" % event_id)

    cache_manifest_path = report_path.parent / "orion_visual_contexts.json"
    cache_manifest = _read(cache_manifest_path)
    cache_path = _resolve(
        cache_manifest, cache_manifest_path.parent, "ORION visual cache"
    )
    cache_report = _resolve(
        cache_manifest["event_factory_report"],
        cache_manifest_path.parent,
        "visual-cache factory report",
    )
    if (
        cache_manifest.get("schema") != CACHE_SCHEMA
        or cache_manifest.get("status")
        != "immutable_multiframe_visual_context_cache"
        or cache_report != report_path
        or set(map(str, cache_manifest.get("group_ids", []))) != set(groups)
        or int(cache_manifest.get("keyframe_count", 0)) != keyframes
        or any(
            cache_manifest.get(key) is not False
            for key in (
                "privileged_safety_inputs_used",
                "stage1_uq_inputs_used",
                "task_relevance_targets_used",
                "qa_answers_used",
            )
        )
    ):
        raise ValueError("ORION visual cache contract changed: %s" % event_id)

    materialized = _materialize_sidecar_paths(rows, source_records)
    return materialized, {
        "event_id": event_id,
        "route_index": int(event["route_index"]),
        "split": str(event["pilot_split"]),
        "town": str(event["town"]),
        "scenario_family": str(event["scenario_family"]),
        "keyframe_count": keyframes,
        "qa_record_count": expected_count,
        "factory_report": {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
        },
        "source_records": {
            "path": str(source_records),
            "sha256": sha256_file(source_records),
        },
        "visual_cache_manifest": {
            "path": str(cache_manifest_path.resolve()),
            "sha256": sha256_file(cache_manifest_path),
        },
        "visual_cache": {
            "path": str(cache_path),
            "sha256": sha256_file(cache_path),
        },
    }


def prepare_dataset(
    *,
    event_bank_path: Path,
    qa_queue_path: Path,
    machine_audit_path: Path,
) -> Dict[str, Any]:
    bank = _read(event_bank_path)
    queue = _read(qa_queue_path)
    machine_audit = _read(machine_audit_path)
    events, queued = _validate_bank_and_queue(bank, queue)
    if (
        machine_audit.get("schema") != MACHINE_AUDIT_SCHEMA
        or machine_audit.get("status")
        != "passed_offline_dataset_contract_training_still_locked"
        or int(machine_audit.get("counts", {}).get("record_count", 0)) != 740
        or int(machine_audit.get("counts", {}).get("matched_group_count", 0))
        != 37
        or machine_audit.get("pilot_training_allowed") is not False
    ):
        raise ValueError("aggregate machine audit is absent, stale, or unlocked")

    source_rows: List[Dict[str, Any]] = []
    event_rows = []
    for event_id in sorted(events):
        rows, event_row = _validate_source_event(
            event_id=event_id,
            event=events[event_id],
            queued=queued[event_id],
            queue_path=qa_queue_path,
        )
        source_rows.extend(rows)
        event_rows.append(event_row)
    if len(source_rows) != 740:
        raise ValueError("aggregate source record count changed")
    upgraded, aggregate_audit = upgrade_records(source_rows)
    split_rows = {
        split: [row for row in upgraded if row["split"] == split]
        for split in ("train", "dev")
    }
    split_audits = {
        split: audit_records(rows) for split, rows in split_rows.items()
    }
    upgraded.sort(key=lambda row: (
        0 if row["split"] == "train" else 1,
        str(row["event_id"]),
        str(row["frame_id"]),
        str(row["counterfactual"]["variant"]),
        str(row["question_family"]),
    ))
    if (
        not aggregate_audit["passed"]
        or any(not value["passed"] for value in split_audits.values())
        or len(split_rows["train"]) != 540
        or len(split_rows["dev"]) != 200
    ):
        raise ValueError("V5 aggregate or split audit failed")
    return {
        "records": upgraded,
        "aggregate_audit": aggregate_audit,
        "split_audits": split_audits,
        "events": event_rows,
        "source": {
            "event_bank": {
                "path": str(event_bank_path.resolve()),
                "sha256": sha256_file(event_bank_path),
            },
            "qa_geometry_queue": {
                "path": str(qa_queue_path.resolve()),
                "sha256": sha256_file(qa_queue_path),
            },
            "machine_audit": {
                "path": str(machine_audit_path.resolve()),
                "sha256": sha256_file(machine_audit_path),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-bank", type=Path, required=True)
    parser.add_argument("--qa-geometry-queue", type=Path, required=True)
    parser.add_argument("--machine-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite multi-route dataset output")
    prepared = prepare_dataset(
        event_bank_path=args.event_bank.resolve(),
        qa_queue_path=args.qa_geometry_queue.resolve(),
        machine_audit_path=args.machine_audit.resolve(),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records_path = args.output_dir / "records.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in prepared["records"]
        ),
        encoding="utf-8",
    )
    aggregate_audit_path = args.output_dir / "audit.json"
    aggregate_audit_path.write_text(
        json.dumps(prepared["aggregate_audit"], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    split_audit_refs = {}
    for split, audit in prepared["split_audits"].items():
        path = args.output_dir / ("%s_audit.json" % split)
        path.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        split_audit_refs[split] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    reference_audit = audit_references(
        prepared["records"], records_parent=args.output_dir.resolve()
    )
    reference_audit_path = args.output_dir / "reference_audit.json"
    reference_audit_path.write_text(
        json.dumps(reference_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": SCHEMA,
        "status": (
            "bounded_multiroute_preexperiment_ready_training_not_started"
            if reference_audit["passed"]
            else "failed_reference_audit"
        ),
        "event_count": 8,
        "train_event_count": 6,
        "dev_event_count": 2,
        "keyframe_count": 37,
        "record_count": 740,
        "train_record_count": 540,
        "dev_record_count": 200,
        "events": prepared["events"],
        "records": {
            "path": str(records_path.resolve()),
            "sha256": sha256_file(records_path),
        },
        "audit": {
            "path": str(aggregate_audit_path.resolve()),
            "sha256": sha256_file(aggregate_audit_path),
        },
        "split_audits": split_audit_refs,
        "reference_audit": {
            "path": str(reference_audit_path.resolve()),
            "sha256": sha256_file(reference_audit_path),
            "passed": reference_audit["passed"],
        },
        "source": prepared["source"],
        "review_boundary": {
            "source_event_integrity_reviews_all_accepted": True,
            "aggregate_machine_qa_geometry_audit_passed": True,
            "formal_per_frame_human_qa_geometry_review_completed": False,
            "user_waived_per_experiment_review_for_bounded_preexperiments": True,
            "eligible_for_bounded_preexperiment": reference_audit["passed"],
            "eligible_for_formal_training": False,
        },
        "training_started": False,
        "formal_stage2l_training_allowed": False,
        "stage2p_allowed": False,
        "claim_boundary": (
            "Eight-event 6/2 V5 dataset for one bounded multi-route "
            "pre-experiment only; no formal-training, held-out "
            "generalization, planning, closed-loop, or safety claim."
        ),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(manifest_path.resolve()),
        "status": manifest["status"],
        "reference_audit_passed": reference_audit["passed"],
        "task_field_class_counts": prepared["aggregate_audit"][
            "task_field_class_counts"
        ],
    }, indent=2, sort_keys=True))
    return 0 if reference_audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
