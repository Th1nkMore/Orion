#!/usr/bin/env python3
"""Assemble the frozen 24-event Stage2-L dataset without unlocking training."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

try:
    from scripts.scenario_factory_lib import (
        EVENT_PACKAGE_SCHEMA,
        sha256_file,
        validate_clean_runtime_manifest,
    )
    from scripts.upgrade_stage2l_v9_qa_records import audit_records as audit_v5_records
    from scripts.reattest_stage2l_visual_cache import (
        validate_cache_manifest_for_factory,
    )
except ModuleNotFoundError:
    from scenario_factory_lib import (
        EVENT_PACKAGE_SCHEMA,
        sha256_file,
        validate_clean_runtime_manifest,
    )
    from upgrade_stage2l_v9_qa_records import audit_records as audit_v5_records
    from reattest_stage2l_visual_cache import validate_cache_manifest_for_factory


FORMAL_BANK_SCHEMA = "orion.stage2_l.formal_event_bank.v1"
FORMAL_BANK_STATUS = "formal_event_bank_complete_reviewed"
FORMAL_PLAN_SCHEMA = "orion.stage2_l.formal_route_plan.v1"
SCHEDULE_SCHEMA = "orion.stage2_l.schedule.v2"
FACTORY_SCHEMA = "orion.uq_relevance_multiframe_event_factory.v1"
VISUAL_CACHE_SCHEMA = "orion.stage2l_multiframe_visual_context_cache.v1"
QA_REVIEW_SCHEMA = "orion.stage2_l.qa_geometry_review_bank.v1"
OUTPUT_SCHEMA = "orion.stage2_l.formal_dataset.v1"
DATA_PROTOCOL_SCHEMA = "orion.stage2_l.formal_data_and_corruption_protocol.v1"
DATA_PROTOCOL_STATUS = "frozen_data_and_corruption_isolation_training_locked"
STAGE1_MULTIFRAME_SCHEMA = "orion.stage1_observation_uq_multiframe.v1"
STAGE1_SEQUENCE_SCHEMA = "orion.stage1_observation_uq_sequence.v1"
QA_FACTORY_CONFIG_SCHEMA = "orion.uq_relevance_qa_factory_config.v5"
QA_RECORD_SCHEMA = "orion.uq_relevance_qa_record.v5"
QA_CONTRACT_SCHEMA = "orion.stage2l_qa_contract.v5"
VARIANTS = {
    "observed",
    "zero_uq",
    "on_path_uq",
    "off_path_uq",
    "view_shuffled_uq",
}
QUESTION_FAMILIES = {
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
    if not path.is_file() or sha256_file(path) != reference.get("sha256"):
        raise ValueError("%s is absent or has a SHA-256 mismatch" % name)
    return path


def _materialize_record_paths(
    row: Mapping[str, Any], records_path: Path
) -> Dict[str, Any]:
    result = copy.deepcopy(dict(row))
    sidecar = result.get("target", {}).get("map_sidecar")
    if isinstance(sidecar, dict):
        resolved = _resolve(sidecar, records_path.parent, "QA map sidecar")
        sidecar["path"] = str(resolved)
    return result


def _indexed_unique(
    rows: Sequence[Mapping[str, Any]], key: str, name: str
) -> Dict[str, Mapping[str, Any]]:
    indexed = {str(row[key]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("%s duplicates %s" % (name, key))
    return indexed


def _validate_data_isolation_protocol(
    protocol: Mapping[str, Any],
) -> Tuple[str, str]:
    isolation = protocol.get("corruption_family_isolation", {})
    if isolation.get("stage2_l_image_training_corruptions") != []:
        raise ValueError("formal protocol must forbid Stage2-L image corruptions")
    if "clean visual observations" not in str(
        isolation.get("stage2_l_training_statement", "")
    ):
        raise ValueError("formal protocol does not bind Stage2-L to clean visuals")
    native = isolation.get("formal_unseen_family_primary", {})
    if not isinstance(native, Mapping) or any(
        native.get(key) is not False
        for key in (
            "adapter_training_allowed",
            "stage2_l_training_allowed",
            "checkpoint_selection_allowed",
        )
    ):
        raise ValueError("formal protocol does not keep native glare held out")
    stage1 = protocol.get("stage1_signal_role", {})
    checkpoint_sha256 = str(stage1.get("checkpoint", {}).get("sha256", ""))
    if len(checkpoint_sha256) != 64:
        raise ValueError("formal protocol lacks a frozen Stage1 checkpoint hash")
    if stage1.get("checkpoint_update_during_stage2_l") is not False:
        raise ValueError("formal protocol must freeze Stage1 during Stage2-L")
    visual_cache = protocol.get("visual_context_cache", {})
    orion_checkpoint_sha256 = str(
        visual_cache.get("orion_checkpoint", {}).get("sha256", "")
    )
    if len(orion_checkpoint_sha256) != 64:
        raise ValueError("formal protocol lacks a frozen ORION cache checkpoint hash")
    if any(
        visual_cache.get(key) is not False
        for key in (
            "stage1_uq_inputs_used",
            "task_relevance_targets_used",
            "qa_answers_used",
            "privileged_safety_inputs_used",
            "llm_run_during_cache",
            "trajectory_decoder_run_during_cache",
        )
    ):
        raise ValueError("formal protocol does not isolate ORION visual caching")
    source_condition = protocol.get("fixed_qa_construction", {}).get(
        "source_condition"
    )
    if not isinstance(source_condition, str) or not source_condition.startswith(
        "clean_off"
    ):
        raise ValueError("formal protocol source condition is not clean_off")
    locks = protocol.get("launch_locks", {})
    if any(
        locks.get(key) is not False
        for key in (
            "formal_stage2_l_allowed",
            "stage2_p_allowed",
            "closed_loop_matrix_allowed",
        )
    ):
        raise ValueError("formal protocol launch locks are not fail-closed")
    return checkpoint_sha256, orion_checkpoint_sha256


def _validate_factory_source_lineage(
    *,
    report_path: Path,
    report: Mapping[str, Any],
    event: Mapping[str, Any],
    expected_stage1_checkpoint_sha256: str,
) -> Dict[str, Any]:
    event_package_path = _resolve(
        report.get("event_package", {}), report_path.parent, "formal event package"
    )
    event_package = _load(event_package_path)
    if event_package.get("schema") != EVENT_PACKAGE_SCHEMA:
        raise ValueError("unsupported formal event-package schema")
    if (
        event_package.get("runtime", {}).get("valid") is not True
        or event_package.get("qa_input_ready") is not True
        or int(event_package.get("route", {}).get("route_index", -1))
        != int(event["route_index"])
    ):
        raise ValueError("formal event package is not runtime-valid and QA-ready")
    run_manifest_path = _resolve(
        event_package.get("source_files", {}).get("run_manifest", {}),
        event_package_path.parent,
        "formal source run manifest",
    )
    run_validation = validate_clean_runtime_manifest(_load(run_manifest_path))
    if run_validation.get("valid") is not True:
        raise ValueError("formal source run is not clean_off with a clean render condition")

    stage1_root_path = _resolve(
        report.get("stage1_multiframe_manifest", {}),
        report_path.parent,
        "formal Stage1 multiframe manifest",
    )
    stage1_root = _load(stage1_root_path)
    if (
        stage1_root.get("schema") != STAGE1_MULTIFRAME_SCHEMA
        or stage1_root.get("control_influence") is not False
        or stage1_root.get("event_package", {}).get("sha256")
        != sha256_file(event_package_path)
    ):
        raise ValueError("formal Stage1 multiframe lineage is invalid")
    sequences = stage1_root.get("sequences")
    if not isinstance(sequences, list) or not 3 <= len(sequences) <= 5:
        raise ValueError("formal Stage1 lineage requires three to five sequences")
    for row in sequences:
        sequence_path = _resolve(
            row.get("manifest", {}), stage1_root_path.parent, "formal Stage1 sequence"
        )
        sequence = _load(sequence_path)
        if (
            sequence.get("schema") != STAGE1_SEQUENCE_SCHEMA
            or sequence.get("status") != "offline_frozen_stage1_output"
            or sequence.get("control_influence") is not False
            or sequence.get("event_package_sha256") != sha256_file(event_package_path)
            or sequence.get("checkpoint_sha256")
            != expected_stage1_checkpoint_sha256
        ):
            raise ValueError("formal Stage1 sequence uses the wrong frozen checkpoint")
        forbidden = sequence.get("forbidden_inputs", {})
        if any(
            forbidden.get(key) is not False
            for key in (
                "route",
                "actor_geometry",
                "ttc",
                "collision_outcome",
                "corruption_metadata",
            )
        ):
            raise ValueError("formal Stage1 sequence used prohibited task inputs")
    return {
        "event_package": {
            "path": str(event_package_path),
            "sha256": sha256_file(event_package_path),
        },
        "run_manifest": {
            "path": str(run_manifest_path),
            "sha256": sha256_file(run_manifest_path),
        },
        "render_condition_attestation": run_validation[
            "render_condition_attestation"
        ],
        "stage1_multiframe_manifest": {
            "path": str(stage1_root_path),
            "sha256": sha256_file(stage1_root_path),
        },
        "stage1_checkpoint_sha256": expected_stage1_checkpoint_sha256,
    }


def _validate_group_records(
    *, records: Sequence[Mapping[str, Any]], event_id: str, split: str
) -> Counter:
    groups: Dict[str, list] = {}
    for row in records:
        if str(row.get("event_id")) != event_id or str(row.get("split")) != split:
            raise ValueError("QA records disagree with the frozen event or split")
        group_id = str(row.get("counterfactual", {}).get("group_id", ""))
        if not group_id:
            raise ValueError("QA record lacks a counterfactual group id")
        groups.setdefault(group_id, []).append(row)
    counts = Counter({group_id: len(rows) for group_id, rows in groups.items()})
    for group_id, rows in groups.items():
        pairs = {
            (
                str(row.get("counterfactual", {}).get("variant", "")),
                str(row.get("question_family", "")),
            )
            for row in rows
        }
        expected = {
            (variant, question)
            for variant in VARIANTS
            for question in QUESTION_FAMILIES
        }
        if len(rows) != 20 or pairs != expected:
            raise ValueError(
                "QA group %s must contain exactly five variants by four families"
                % group_id
            )
    return counts


def assemble_formal_dataset(
    *,
    formal_bank_path: Path,
    schedule_path: Path,
    formal_data_protocol_path: Path,
    qa_factory_config_path: Path,
    qa_review_bank_path: Path,
    factory_reports: Sequence[Path],
    visual_cache_manifests: Sequence[Path],
) -> Dict[str, Any]:
    bank = _load(formal_bank_path)
    schedule = _load(schedule_path)
    data_protocol = _load(formal_data_protocol_path)
    qa_review = _load(qa_review_bank_path)
    if (
        bank.get("schema") != FORMAL_BANK_SCHEMA
        or bank.get("status") != FORMAL_BANK_STATUS
        or bank.get("checks", {}).get("all_24_planned_routes_present") is not True
    ):
        raise ValueError("formal event bank is not complete and reviewed")
    if schedule.get("schema") != SCHEDULE_SCHEMA:
        raise ValueError("unsupported formal Stage2-L schedule")
    if (
        data_protocol.get("schema") != DATA_PROTOCOL_SCHEMA
        or data_protocol.get("status") != DATA_PROTOCOL_STATUS
    ):
        raise ValueError("formal data and corruption protocol is not frozen")
    (
        stage1_checkpoint_sha256,
        orion_checkpoint_sha256,
    ) = _validate_data_isolation_protocol(data_protocol)
    if (
        qa_review.get("schema") != QA_REVIEW_SCHEMA
        or qa_review.get("status") != "frozen_human_qa_geometry_review"
    ):
        raise ValueError("formal QA geometry review is not frozen")
    if not qa_factory_config_path.is_file():
        raise FileNotFoundError("formal QA factory config is missing")
    if _load(qa_factory_config_path).get("schema") != QA_FACTORY_CONFIG_SCHEMA:
        raise ValueError("formal QA factory must use the frozen v5 task-field schema")

    events = _indexed_unique(bank.get("events", []), "event_id", "formal bank")
    formal_gate = schedule.get("formal_gate", {})
    expected_events = int(formal_gate.get("independent_events", -1))
    if expected_events != 24 or len(events) != expected_events:
        raise ValueError("formal dataset requires exactly 24 independent events")
    expected_splits = {
        key: int(value)
        for key, value in formal_gate.get("event_level_split", {}).items()
    }
    actual_event_splits = Counter(
        str(row.get("formal_split")) for row in events.values()
    )
    if dict(actual_event_splits) != expected_splits:
        raise ValueError("formal event split differs from the frozen 16/4/4 split")

    plan_path = _resolve(
        bank.get("provenance", {}).get("formal_route_plan", {}),
        formal_bank_path.parent,
        "formal route plan",
    )
    plan = _load(plan_path)
    if plan.get("schema") != FORMAL_PLAN_SCHEMA:
        raise ValueError("unsupported formal route plan")
    protocol_plan = data_protocol.get("frozen_event_split", {}).get(
        "formal_route_plan", {}
    )
    protocol_qa = data_protocol.get("fixed_qa_construction", {}).get(
        "formal_qa_factory", {}
    )
    protocol_schedule = data_protocol.get("validated_sources", {}).get(
        "stage2l_schedule", {}
    )
    if protocol_plan.get("sha256") != sha256_file(plan_path):
        raise ValueError("formal data protocol references a different route plan")
    if protocol_qa.get("sha256") != sha256_file(qa_factory_config_path):
        raise ValueError("formal data protocol references a different QA factory")
    if protocol_schedule.get("sha256") != sha256_file(schedule_path):
        raise ValueError("formal data protocol references a different schedule")
    planned = {int(row["route_index"]): row for row in plan.get("events", [])}
    if len(planned) != 24:
        raise ValueError("formal route plan must contain 24 unique routes")
    for event in events.values():
        route = int(event["route_index"])
        if route not in planned or str(event["formal_split"]) != str(
            planned[route]["formal_split"]
        ):
            raise ValueError("formal event differs from the frozen route plan")

    if len(factory_reports) != 24 or len(visual_cache_manifests) != 24:
        raise ValueError("formal dataset requires one QA factory and cache per event")
    factory_by_event: Dict[str, tuple] = {}
    for path in factory_reports:
        report = _load(path)
        if report.get("schema") != FACTORY_SCHEMA:
            raise ValueError("unsupported multi-frame QA factory report")
        event_id = str(report.get("event_id", ""))
        if event_id in factory_by_event:
            raise ValueError("duplicate formal QA factory event")
        config_ref = report.get("qa_factory_config", {})
        if config_ref.get("sha256") != sha256_file(qa_factory_config_path):
            raise ValueError("formal event used a different QA factory config")
        factory_by_event[event_id] = (path, report)

    cache_by_event: Dict[str, tuple] = {}
    for path in visual_cache_manifests:
        manifest = _load(path)
        if manifest.get("schema") != VISUAL_CACHE_SCHEMA:
            raise ValueError("unsupported visual-context cache manifest")
        report_path = _resolve(
            manifest["event_factory_report"], path.parent, "visual-cache factory report"
        )
        event_id = str(_load(report_path).get("event_id", ""))
        if event_id in cache_by_event:
            raise ValueError("duplicate formal visual-cache event")
        cache_by_event[event_id] = (path, manifest)

    if set(factory_by_event) != set(events) or set(cache_by_event) != set(events):
        raise ValueError("formal factories or caches do not match the frozen events")
    reviewed = _indexed_unique(qa_review.get("accepted", []), "event_id", "QA review")
    if qa_review.get("rejected") or set(reviewed) != set(events):
        raise ValueError("accepted QA reviews do not exactly cover all formal events")
    for event_id, (report_path, _) in factory_by_event.items():
        if reviewed[event_id].get("factory_report", {}).get("sha256") != sha256_file(
            report_path
        ):
            raise ValueError("QA review factory-report hash mismatch")

    combined = []
    event_rows = []
    records_per_keyframe = int(schedule["fixed_keyframe_policy"]["records_per_keyframe"])
    for event_id, event in sorted(events.items()):
        report_path, report = factory_by_event[event_id]
        cache_path, cache = cache_by_event[event_id]
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
        keyframes = int(report.get("keyframe_count", 0))
        if not 3 <= keyframes <= 5:
            raise ValueError("formal event violates the fixed 3-5 keyframe policy")
        records_path = _resolve(
            report["qa_dataset"]["records"], report_path.parent, "formal QA records"
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
            raise ValueError("formal event contains legacy or non-v5 QA records")
        qa_contract_audit = audit_v5_records(records)
        if qa_contract_audit.get("passed") is not True:
            raise ValueError("formal event failed the v5 task-field QA contract audit")
        if len(records) != keyframes * records_per_keyframe:
            raise ValueError("formal event QA count differs from 20 per keyframe")
        split = str(event["formal_split"])
        group_counts = _validate_group_records(
            records=records, event_id=event_id, split=split
        )
        if len(group_counts) != keyframes or set(group_counts.values()) != {20}:
            raise ValueError("formal QA group count differs from keyframe count")
        if set(cache.get("group_ids", [])) != set(group_counts):
            raise ValueError("formal visual cache does not cover every QA group")
        combined.extend(records)
        event_rows.append(
            {
                "event_id": event_id,
                "route_index": int(event["route_index"]),
                "split": split,
                "town": str(event["town"]),
                "scenario_family": str(event["scenario_family"]),
                "keyframe_count": keyframes,
                "qa_record_count": len(records),
                "qa_factory_report": {
                    "path": str(report_path.resolve()),
                    "sha256": sha256_file(report_path),
                },
                "visual_cache_manifest": {
                    "path": str(cache_path.resolve()),
                    "sha256": sha256_file(cache_path),
                },
                "visual_cache": {
                    "path": str(cache["output"]),
                    "sha256": str(cache["sha256"]),
                },
                "visual_cache_validation": cache_validation,
                "source_lineage": source_lineage,
                "qa_contract_audit": qa_contract_audit,
            }
        )

    plan_lower, plan_upper = map(int, plan["expected_qa_records_after_geometry_gate"])
    schedule_lower, schedule_upper = map(
        int, formal_gate["target_structured_qa_records"]
    )
    lower = max(plan_lower, schedule_lower)
    upper = min(plan_upper, schedule_upper)
    if not lower <= len(combined) <= upper:
        raise ValueError("formal QA count is outside the frozen plan and schedule range")
    combined.sort(
        key=lambda row: (
            str(row["split"]),
            str(row["event_id"]),
            str(row["frame_id"]),
            str(row["counterfactual"]["variant"]),
            str(row["question_family"]),
        )
    )
    qa_split_counts = Counter(str(row["split"]) for row in combined)
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "assembled_formal_data_training_launch_locked",
        "formal_training_ready": False,
        "stage2p_allowed": False,
        "event_count": len(events),
        "qa_record_count": len(combined),
        "qa_split_counts": dict(sorted(qa_split_counts.items())),
        "town_count": len({row["town"] for row in event_rows}),
        "scenario_family_count": len(
            {row["scenario_family"] for row in event_rows}
        ),
        "events": event_rows,
        "records": combined,
        "checks": {
            "formal_24_event_bank_complete": True,
            "frozen_16_4_4_split_preserved": True,
            "every_event_has_3_to_5_keyframes": True,
            "every_keyframe_has_20_records": True,
            "all_qa_geometry_reviews_accepted_and_hash_bound": True,
            "visual_caches_exclude_stage1_uq_targets_and_answers": True,
            "visual_cache_lineage_is_current_or_reattested": True,
            "all_source_runs_are_clean_off_with_clean_render_conditions": True,
            "all_stage1_sequences_use_the_frozen_checkpoint": True,
            "formal_qa_record_range_passed": True,
            "training_launch_still_requires_immutable_amendment": True,
        },
        "provenance": {
            "formal_event_bank": {
                "path": str(formal_bank_path.resolve()),
                "sha256": sha256_file(formal_bank_path),
            },
            "formal_route_plan": {
                "path": str(plan_path),
                "sha256": sha256_file(plan_path),
            },
            "stage2l_schedule": {
                "path": str(schedule_path.resolve()),
                "sha256": sha256_file(schedule_path),
            },
            "formal_data_and_corruption_protocol": {
                "path": str(formal_data_protocol_path.resolve()),
                "sha256": sha256_file(formal_data_protocol_path),
            },
            "qa_factory_config": {
                "path": str(qa_factory_config_path.resolve()),
                "sha256": sha256_file(qa_factory_config_path),
            },
            "qa_geometry_review_bank": {
                "path": str(qa_review_bank_path.resolve()),
                "sha256": sha256_file(qa_review_bank_path),
            },
        },
        "claim_boundary": (
            "Hash-bound assembly of the frozen formal Stage2-L QA data only. "
            "It does not authorize training or establish understanding, planning, "
            "closed-loop, generalization, or safety evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-bank", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--formal-data-protocol", type=Path, required=True)
    parser.add_argument("--qa-factory-config", type=Path, required=True)
    parser.add_argument("--qa-geometry-review-bank", type=Path, required=True)
    parser.add_argument("--factory-report", type=Path, action="append", required=True)
    parser.add_argument(
        "--visual-cache-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite formal Stage2-L dataset")
    result = assemble_formal_dataset(
        formal_bank_path=args.formal_bank.resolve(),
        schedule_path=args.schedule.resolve(),
        formal_data_protocol_path=args.formal_data_protocol.resolve(),
        qa_factory_config_path=args.qa_factory_config.resolve(),
        qa_review_bank_path=args.qa_geometry_review_bank.resolve(),
        factory_reports=[path.resolve() for path in args.factory_report],
        visual_cache_manifests=[
            path.resolve() for path in args.visual_cache_manifest
        ],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in result.pop("records")
        ),
        encoding="utf-8",
    )
    result["records"] = {
        "path": str(records_path.resolve()),
        "sha256": sha256_file(records_path),
    }
    manifest_path = args.output_dir / "formal_dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path.resolve()),
                "event_count": result["event_count"],
                "qa_record_count": result["qa_record_count"],
                "status": result["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
