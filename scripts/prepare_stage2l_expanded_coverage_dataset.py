#!/usr/bin/env python3
"""Build a bounded expanded-coverage Stage2-L diagnostic dataset.

The input is a fail-closed formal-inventory audit, but the output is never a
formal dataset.  It preserves the frozen train/dev identities, excludes every
locked-test event, verifies the underlying human event-integrity decisions,
and aggregates only hash-bound V5 QA records and visual caches.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

try:
    from scripts.audit_stage2l_v8_dataset_references import audit_references
    from scripts.scenario_factory_lib import sha256_file
    from scripts.upgrade_stage2l_v9_qa_records import audit_records
except ModuleNotFoundError:
    from audit_stage2l_v8_dataset_references import audit_references
    from scenario_factory_lib import sha256_file
    from upgrade_stage2l_v9_qa_records import audit_records


SCHEMA = "orion.stage2l_expanded_coverage_dataset.v1"
INVENTORY_SCHEMA = "orion.stage2_l.formal_inventory_audit.v1"
FORMAL_BANK_SCHEMA = "orion.stage2_l.formal_event_bank.v1"
FACTORY_SCHEMA = "orion.uq_relevance_multiframe_event_factory.v1"
CACHE_SCHEMA = "orion.stage2l_multiframe_visual_context_cache.v1"
QA_RECORD_SCHEMA = "orion.uq_relevance_qa_record.v5"
QA_CONTRACT_SCHEMA = "orion.stage2l_qa_contract.v5"
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


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _resolve(reference: Mapping[str, Any], base: Path, name: str) -> Path:
    path = Path(str(reference.get("path", reference.get("output", ""))))
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.is_file():
        raise FileNotFoundError("%s is missing: %s" % (name, path))
    if str(reference.get("sha256", "")) != sha256_file(path):
        raise ValueError("%s SHA-256 mismatch" % name)
    return path


def _records(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _materialize_sidecars(
    rows: Sequence[Mapping[str, Any]], source_records: Path
) -> List[Dict[str, Any]]:
    output = []
    for source in rows:
        row = copy.deepcopy(dict(source))
        sidecar = row.get("target", {}).get("map_sidecar", {})
        sidecar_path = _resolve(
            sidecar, source_records.parent, "task-relevance map sidecar"
        )
        sidecar["path"] = str(sidecar_path)
        output.append(row)
    return output


def _validate_inventory_and_bank(
    inventory: Mapping[str, Any], bank: Mapping[str, Any]
) -> Tuple[Dict[str, Mapping[str, Any]], Dict[str, Mapping[str, Any]]]:
    if (
        inventory.get("schema") != INVENTORY_SCHEMA
        or inventory.get("status") != "audited_available_subset_training_locked"
        or inventory.get("formal_training_ready") is not False
        or inventory.get("stage2p_allowed") is not False
        or int(inventory.get("audited_event_count", 0)) < 8
    ):
        raise ValueError("formal inventory is absent, stale, or not fail-closed")
    required_inventory_checks = {
        "all_available_factories_pass_v5_qa_contract",
        "all_available_caches_cover_their_qa_groups",
        "all_available_cache_lineage_is_current_or_reattested",
        "all_available_source_runs_are_clean_off",
        "all_available_stage1_sequences_use_frozen_checkpoint",
        "training_remains_fail_closed",
    }
    if any(
        inventory.get("checks", {}).get(key) is not True
        for key in required_inventory_checks
    ):
        raise ValueError("formal inventory has a failed available-data check")
    if bank.get("schema") != FORMAL_BANK_SCHEMA:
        raise ValueError("formal subset event bank schema changed")

    events = {
        str(row.get("event_id", "")): row for row in inventory.get("events", [])
    }
    bank_events = {
        str(row.get("event_id", "")): row for row in bank.get("events", [])
    }
    if (
        len(events) != int(inventory.get("audited_event_count", -1))
        or "" in events
        or len(events) != len(inventory.get("events", []))
        or not set(events).issubset(bank_events)
    ):
        raise ValueError("inventory event identities are incomplete or duplicated")
    for event_id, event in events.items():
        bank_event = bank_events[event_id]
        if (
            str(event.get("split", "")) not in {"train", "dev"}
            or str(event.get("split")) != str(bank_event.get("formal_split"))
            or int(event.get("route_index", -1))
            != int(bank_event.get("route_index", -2))
            or bank_event.get("qa_input_ready") is not True
            or bank_event.get("human_review", {}).get("decision") != "accept"
        ):
            raise ValueError(
                "expanded diagnostic includes an unreviewed/test/mismatched event: %s"
                % event_id
            )
    split_counts = Counter(str(row["split"]) for row in events.values())
    if not split_counts.get("train") or not split_counts.get("dev"):
        raise ValueError("expanded diagnostic needs non-empty train and dev events")
    return events, bank_events


def prepare_dataset(*, inventory_path: Path) -> Dict[str, Any]:
    inventory = _load(inventory_path)
    bank_path = _resolve(
        inventory.get("provenance", {}).get("partial_formal_event_bank", {}),
        inventory_path.parent,
        "reviewed partial formal event bank",
    )
    bank = _load(bank_path)
    events, _ = _validate_inventory_and_bank(inventory, bank)

    combined: List[Dict[str, Any]] = []
    event_rows = []
    for event_id, event in sorted(events.items()):
        report_path = _resolve(
            event.get("factory_report", {}),
            inventory_path.parent,
            "event factory report",
        )
        report = _load(report_path)
        if (
            report.get("schema") != FACTORY_SCHEMA
            or str(report.get("event_id", "")) != event_id
        ):
            raise ValueError("event factory identity changed: %s" % event_id)
        records_path = _resolve(
            report.get("qa_dataset", {}).get("records", {}),
            report_path.parent,
            "event V5 QA records",
        )
        rows = _materialize_sidecars(_records(records_path), records_path)
        keyframes = int(event.get("keyframe_count", 0))
        group_counts = Counter(
            str(row.get("counterfactual", {}).get("group_id", "")) for row in rows
        )
        pairs = {
            (
                str(row.get("counterfactual", {}).get("variant", "")),
                str(row.get("question_family", "")),
            )
            for row in rows
        }
        if (
            not 3 <= keyframes <= 5
            or len(rows) != keyframes * 20
            or len(group_counts) != keyframes
            or set(group_counts.values()) != {20}
            or pairs
            != {
                (variant, family)
                for variant in EXPECTED_VARIANTS
                for family in EXPECTED_FAMILIES
            }
            or any(
                row.get("schema") != QA_RECORD_SCHEMA
                or row.get("target", {}).get("qa_contract_schema")
                != QA_CONTRACT_SCHEMA
                or str(row.get("event_id", "")) != event_id
                or str(row.get("split", "")) != str(event["split"])
                for row in rows
            )
        ):
            raise ValueError("event V5 matched-group contract changed: %s" % event_id)
        event_audit = audit_records(rows)
        if event_audit.get("passed") is not True:
            raise ValueError("event V5 QA audit failed: %s" % event_id)

        cache_manifest_path = _resolve(
            event.get("visual_cache_manifest", {}),
            inventory_path.parent,
            "event visual-cache manifest",
        )
        cache_manifest = _load(cache_manifest_path)
        cache_path = _resolve(
            cache_manifest, cache_manifest_path.parent, "event visual cache"
        )
        if (
            cache_manifest.get("schema") != CACHE_SCHEMA
            or set(map(str, cache_manifest.get("group_ids", [])))
            != set(group_counts)
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
            raise ValueError("event visual-cache contract changed: %s" % event_id)
        combined.extend(rows)
        event_rows.append(
            {
                "event_id": event_id,
                "route_index": int(event["route_index"]),
                "split": str(event["split"]),
                "town": str(event["town"]),
                "scenario_family": str(event["scenario_family"]),
                "keyframe_count": keyframes,
                "qa_record_count": len(rows),
                "factory_report": {
                    "path": str(report_path),
                    "sha256": sha256_file(report_path),
                },
                "source_records": {
                    "path": str(records_path),
                    "sha256": sha256_file(records_path),
                },
                "visual_cache_manifest": {
                    "path": str(cache_manifest_path),
                    "sha256": sha256_file(cache_manifest_path),
                },
                "visual_cache": {
                    "path": str(cache_path),
                    "sha256": sha256_file(cache_path),
                },
                "qa_contract_audit": event_audit,
            }
        )
    combined.sort(
        key=lambda row: (
            0 if str(row["split"]) == "train" else 1,
            str(row["event_id"]),
            str(row["frame_id"]),
            str(row["counterfactual"]["variant"]),
            str(row["question_family"]),
        )
    )
    aggregate_audit = audit_records(combined)
    split_rows = {
        split: [row for row in combined if str(row["split"]) == split]
        for split in ("train", "dev")
    }
    split_audits = {split: audit_records(rows) for split, rows in split_rows.items()}
    if aggregate_audit.get("passed") is not True or any(
        value.get("passed") is not True for value in split_audits.values()
    ):
        raise ValueError("expanded aggregate/split V5 audit failed")
    return {
        "records": combined,
        "aggregate_audit": aggregate_audit,
        "split_audits": split_audits,
        "events": event_rows,
        "source": {
            "formal_inventory_audit": {
                "path": str(inventory_path.resolve()),
                "sha256": sha256_file(inventory_path),
            },
            "reviewed_partial_formal_event_bank": {
                "path": str(bank_path),
                "sha256": sha256_file(bank_path),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-inventory-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite expanded coverage dataset")
    prepared = prepare_dataset(inventory_path=args.formal_inventory_audit.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records_path = args.output_dir / "records.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            for row in prepared["records"]
        ),
        encoding="utf-8",
    )
    audit_refs = {}
    for name, audit in {
        "aggregate": prepared["aggregate_audit"],
        **prepared["split_audits"],
    }.items():
        path = args.output_dir / ("audit.json" if name == "aggregate" else "%s_audit.json" % name)
        path.write_text(
            json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        audit_refs[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    reference_audit = audit_references(
        prepared["records"], records_parent=args.output_dir.resolve()
    )
    reference_path = args.output_dir / "reference_audit.json"
    reference_path.write_text(
        json.dumps(reference_audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    split_event_counts = Counter(row["split"] for row in prepared["events"])
    split_record_counts = Counter(row["split"] for row in prepared["records"])
    keyframe_count = sum(int(row["keyframe_count"]) for row in prepared["events"])
    ready = reference_audit.get("passed") is True
    manifest = {
        "schema": SCHEMA,
        "status": (
            "bounded_expanded_coverage_preexperiment_ready_training_not_started"
            if ready
            else "failed_reference_audit"
        ),
        "event_count": len(prepared["events"]),
        "train_event_count": int(split_event_counts["train"]),
        "dev_event_count": int(split_event_counts["dev"]),
        "keyframe_count": keyframe_count,
        "record_count": len(prepared["records"]),
        "train_record_count": int(split_record_counts["train"]),
        "dev_record_count": int(split_record_counts["dev"]),
        "town_count": len({row["town"] for row in prepared["events"]}),
        "scenario_family_count": len(
            {row["scenario_family"] for row in prepared["events"]}
        ),
        "events": prepared["events"],
        "records": {"path": str(records_path.resolve()), "sha256": sha256_file(records_path)},
        "audit": audit_refs["aggregate"],
        "split_audits": {"train": audit_refs["train"], "dev": audit_refs["dev"]},
        "reference_audit": {
            "path": str(reference_path.resolve()),
            "sha256": sha256_file(reference_path),
            "passed": ready,
        },
        "source": prepared["source"],
        "review_boundary": {
            "source_event_integrity_reviews_all_accepted": True,
            "all_available_v5_machine_contracts_passed": True,
            "formal_per_frame_human_qa_geometry_review_completed": False,
            "user_waived_per_experiment_review_for_bounded_preexperiments": True,
            "eligible_for_bounded_preexperiment": ready,
            "eligible_for_formal_training": False,
            "locked_test_events_included": False,
        },
        "training_started": False,
        "formal_stage2l_training_allowed": False,
        "stage2p_allowed": False,
        "claim_boundary": (
            "Expanded train/dev coverage for one bounded engineering diagnostic. "
            "It is not the formal 24-event dataset and provides no locked-test, "
            "planning, closed-loop, generalization, or safety claim."
        ),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path.resolve()),
                "status": manifest["status"],
                "event_count": manifest["event_count"],
                "event_split": {
                    "train": manifest["train_event_count"],
                    "dev": manifest["dev_event_count"],
                },
                "record_count": manifest["record_count"],
                "reference_audit_passed": ready,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
