#!/usr/bin/env python3
"""Fail-closed audit of the currently available subset of formal Stage2-L data.

This deliberately cannot authorize training.  It applies the event-level
factory, Stage1, QA-contract and visual-cache checks used by the 24-event
assembler, while reporting which frozen events and human reviews are still
missing.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, Sequence

try:
    from scripts.assemble_stage2l_formal_dataset import (
        FACTORY_SCHEMA,
        FORMAL_BANK_SCHEMA,
        FORMAL_PLAN_SCHEMA,
        QA_CONTRACT_SCHEMA,
        QA_FACTORY_CONFIG_SCHEMA,
        QA_RECORD_SCHEMA,
        VISUAL_CACHE_SCHEMA,
        _load,
        _materialize_record_paths,
        _resolve,
        _validate_data_isolation_protocol,
        _validate_factory_source_lineage,
        _validate_group_records,
        audit_v5_records,
        validate_cache_manifest_for_factory,
    )
    from scripts.scenario_factory_lib import sha256_file
except ModuleNotFoundError:
    from assemble_stage2l_formal_dataset import (
        FACTORY_SCHEMA,
        FORMAL_BANK_SCHEMA,
        FORMAL_PLAN_SCHEMA,
        QA_CONTRACT_SCHEMA,
        QA_FACTORY_CONFIG_SCHEMA,
        QA_RECORD_SCHEMA,
        VISUAL_CACHE_SCHEMA,
        _load,
        _materialize_record_paths,
        _resolve,
        _validate_data_isolation_protocol,
        _validate_factory_source_lineage,
        _validate_group_records,
        audit_v5_records,
        validate_cache_manifest_for_factory,
    )
    from scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2_l.formal_inventory_audit.v1"


def audit_inventory(
    *,
    partial_bank_path: Path,
    formal_plan_path: Path,
    formal_data_protocol_path: Path,
    qa_factory_config_path: Path,
    factory_reports: Sequence[Path],
    visual_cache_manifests: Sequence[Path],
) -> Dict[str, Any]:
    bank = _load(partial_bank_path)
    plan = _load(formal_plan_path)
    protocol = _load(formal_data_protocol_path)
    qa_factory = _load(qa_factory_config_path)
    if bank.get("schema") != FORMAL_BANK_SCHEMA:
        raise ValueError("unsupported partial formal event bank")
    if plan.get("schema") != FORMAL_PLAN_SCHEMA or len(plan.get("events", [])) != 24:
        raise ValueError("formal route plan does not contain the frozen 24 routes")
    if qa_factory.get("schema") != QA_FACTORY_CONFIG_SCHEMA:
        raise ValueError("formal QA factory is not v5")
    stage1_checkpoint_sha256, orion_checkpoint_sha256 = (
        _validate_data_isolation_protocol(protocol)
    )

    bank_events = {str(row["event_id"]): row for row in bank.get("events", [])}
    if len(bank_events) != len(bank.get("events", [])):
        raise ValueError("partial formal event bank duplicates event ids")
    planned_by_route = {int(row["route_index"]): row for row in plan["events"]}

    factories = {}
    for path in factory_reports:
        report = _load(path)
        event_id = str(report.get("event_id", ""))
        if report.get("schema") != FACTORY_SCHEMA or not event_id:
            raise ValueError("unsupported partial QA factory report")
        if event_id in factories:
            raise ValueError("partial inventory duplicates a QA factory event")
        if report.get("qa_factory_config", {}).get("sha256") != sha256_file(
            qa_factory_config_path
        ):
            raise ValueError("partial event uses a different QA factory config")
        factories[event_id] = (path, report)

    caches = {}
    for path in visual_cache_manifests:
        manifest = _load(path)
        if manifest.get("schema") != VISUAL_CACHE_SCHEMA:
            raise ValueError("unsupported partial visual-cache manifest")
        report_path = _resolve(
            manifest.get("event_factory_report", {}),
            path.parent,
            "partial visual-cache factory report",
        )
        event_id = str(_load(report_path).get("event_id", ""))
        if not event_id or event_id in caches:
            raise ValueError("partial inventory duplicates a visual-cache event")
        caches[event_id] = (path, manifest)

    if set(factories) != set(caches):
        raise ValueError("partial QA factories and caches cover different events")
    if not set(factories).issubset(bank_events):
        raise ValueError("partial inventory contains an event outside the reviewed bank")

    event_rows = []
    qa_split_counts = Counter()
    total_records = 0
    for event_id in sorted(factories):
        event = bank_events[event_id]
        route_index = int(event["route_index"])
        planned = planned_by_route.get(route_index)
        if planned is None or str(planned["formal_split"]) != str(event["formal_split"]):
            raise ValueError("partial event differs from the frozen route plan")
        report_path, report = factories[event_id]
        cache_path, cache = caches[event_id]
        source_lineage = _validate_factory_source_lineage(
            report_path=report_path,
            report=report,
            event=event,
            expected_stage1_checkpoint_sha256=stage1_checkpoint_sha256,
        )
        cache_validation = validate_cache_manifest_for_factory(
            cache_manifest_path=cache_path,
            factory_report_path=report_path,
            expected_orion_checkpoint_sha256=orion_checkpoint_sha256,
        )
        records_path = _resolve(
            report.get("qa_dataset", {}).get("records", {}),
            report_path.parent,
            "partial formal QA records",
        )
        records = [
            _materialize_record_paths(json.loads(line), records_path)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(
            row.get("schema") != QA_RECORD_SCHEMA
            or row.get("target", {}).get("qa_contract_schema")
            != QA_CONTRACT_SCHEMA
            for row in records
        ):
            raise ValueError("partial inventory contains a non-v5 QA record")
        qa_audit = audit_v5_records(records)
        if qa_audit.get("passed") is not True:
            raise ValueError("partial event failed the v5 QA contract audit")
        split = str(event["formal_split"])
        group_counts = _validate_group_records(
            records=records, event_id=event_id, split=split
        )
        if set(cache.get("group_ids", [])) != set(group_counts):
            raise ValueError("partial visual cache does not cover every QA group")
        keyframes = int(report.get("keyframe_count", 0))
        if len(group_counts) != keyframes or len(records) != 20 * keyframes:
            raise ValueError("partial event violates the fixed QA/keyframe policy")
        total_records += len(records)
        qa_split_counts[split] += len(records)
        event_rows.append(
            {
                "event_id": event_id,
                "route_index": route_index,
                "split": split,
                "town": str(event["town"]),
                "scenario_family": str(event["scenario_family"]),
                "keyframe_count": keyframes,
                "qa_record_count": len(records),
                "factory_report": {
                    "path": str(report_path.resolve()),
                    "sha256": sha256_file(report_path),
                },
                "visual_cache_manifest": {
                    "path": str(cache_path.resolve()),
                    "sha256": sha256_file(cache_path),
                },
                "visual_cache_validation": cache_validation,
                "source_lineage": source_lineage,
                "qa_contract_audit": qa_audit,
            }
        )

    audited_routes = {row["route_index"] for row in event_rows}
    missing_plan = [
        {
            "route_index": int(row["route_index"]),
            "formal_split": str(row["formal_split"]),
            "town": str(row["town"]),
            "scenario_family": str(row["scenario_family"]),
        }
        for row in plan["events"]
        if int(row["route_index"]) not in audited_routes
    ]
    reviewed_without_usable_v5 = sorted(set(bank_events) - set(factories))
    return {
        "schema": SCHEMA,
        "status": "audited_available_subset_training_locked",
        "formal_training_ready": False,
        "stage2p_allowed": False,
        "audited_event_count": len(event_rows),
        "qa_record_count": total_records,
        "qa_split_counts": dict(sorted(qa_split_counts.items())),
        "town_count": len({row["town"] for row in event_rows}),
        "scenario_family_count": len(
            {row["scenario_family"] for row in event_rows}
        ),
        "events": event_rows,
        "reviewed_bank_events_without_usable_v5": reviewed_without_usable_v5,
        "frozen_plan_routes_without_audited_v5": missing_plan,
        "remaining_gates": {
            "exactly_24_events": len(event_rows) == 24,
            "frozen_16_4_4_split_complete": False,
            "formal_v5_qa_geometry_review_bank_complete": False,
            "immutable_formal_training_launch_amendment_present": False,
        },
        "checks": {
            "all_available_factories_pass_v5_qa_contract": True,
            "all_available_caches_cover_their_qa_groups": True,
            "all_available_cache_lineage_is_current_or_reattested": True,
            "all_available_source_runs_are_clean_off": True,
            "all_available_stage1_sequences_use_frozen_checkpoint": True,
            "training_remains_fail_closed": True,
        },
        "provenance": {
            "partial_formal_event_bank": {
                "path": str(partial_bank_path.resolve()),
                "sha256": sha256_file(partial_bank_path),
            },
            "formal_route_plan": {
                "path": str(formal_plan_path.resolve()),
                "sha256": sha256_file(formal_plan_path),
            },
            "formal_data_protocol": {
                "path": str(formal_data_protocol_path.resolve()),
                "sha256": sha256_file(formal_data_protocol_path),
            },
            "qa_factory_config": {
                "path": str(qa_factory_config_path.resolve()),
                "sha256": sha256_file(qa_factory_config_path),
            },
        },
        "claim_boundary": (
            "Diagnostic audit of the available formal-data subset only. It does "
            "not replace the 24-event assembler, human QA review, formal training "
            "authorization, model evaluation, planning or safety evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial-bank", type=Path, required=True)
    parser.add_argument("--formal-plan", type=Path, required=True)
    parser.add_argument("--formal-data-protocol", type=Path, required=True)
    parser.add_argument("--qa-factory-config", type=Path, required=True)
    parser.add_argument("--factory-report", type=Path, action="append", required=True)
    parser.add_argument(
        "--visual-cache-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite partial formal inventory audit")
    result = audit_inventory(
        partial_bank_path=args.partial_bank.resolve(),
        formal_plan_path=args.formal_plan.resolve(),
        formal_data_protocol_path=args.formal_data_protocol.resolve(),
        qa_factory_config_path=args.qa_factory_config.resolve(),
        factory_reports=[path.resolve() for path in args.factory_report],
        visual_cache_manifests=[
            path.resolve() for path in args.visual_cache_manifest
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "audited_event_count": result["audited_event_count"],
                "qa_record_count": result["qa_record_count"],
                "formal_training_ready": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
