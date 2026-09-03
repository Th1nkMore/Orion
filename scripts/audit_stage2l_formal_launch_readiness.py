#!/usr/bin/env python3
"""Audit formal Stage2-L launch prerequisites without launching training.

The audit accepts either the current subset inventory or a completed formal
dataset manifest.  It deliberately treats a missing formal training protocol
or launch amendment as an ordinary, machine-readable failed gate.  A passing
report is still only an input to a separately hash-bound submit wrapper.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

try:
    from scripts.scenario_factory_lib import sha256_file
except ModuleNotFoundError:
    from scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2_l.formal_launch_readiness.v1"
DATA_PROTOCOL_SCHEMA = "orion.stage2_l.formal_data_and_corruption_protocol.v1"
DATA_PROTOCOL_STATUS = "frozen_data_and_corruption_isolation_training_locked"
FORMAL_DATASET_SCHEMA = "orion.stage2_l.formal_dataset.v1"
INVENTORY_SCHEMA = "orion.stage2_l.formal_inventory_audit.v1"
MR1_COMPARISON_SCHEMA = "orion.stage2l_mr1_duration_comparison.v1"
FORMAL_TRAINING_PROTOCOL_SCHEMA = "orion.stage2_l.formal_training_protocol.v1"
FORMAL_TRAINING_PROTOCOL_STATUS = "frozen_formal_training_protocol_launch_locked"
LAUNCH_AMENDMENT_SCHEMA = "orion.scenario_factory.amendment.v1"
EXPECTED_EVENT_SPLITS = {"train": 16, "dev": 4, "test": 4}
REQUIRED_DATASET_CHECKS = {
    "formal_24_event_bank_complete",
    "frozen_16_4_4_split_preserved",
    "every_event_has_3_to_5_keyframes",
    "every_keyframe_has_20_records",
    "all_qa_geometry_reviews_accepted_and_hash_bound",
    "visual_caches_exclude_stage1_uq_targets_and_answers",
    "visual_cache_lineage_is_current_or_reattested",
    "all_source_runs_are_clean_off_with_clean_render_conditions",
    "all_stage1_sequences_use_the_frozen_checkpoint",
    "formal_qa_record_range_passed",
    "training_launch_still_requires_immutable_amendment",
}


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _source_reference(path: Path) -> Dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}


def _reference_matches(
    reference: Mapping[str, Any], path: Path
) -> bool:
    return str(reference.get("sha256", "")) == sha256_file(path.resolve())


def _validate_data_isolation_protocol(protocol: Mapping[str, Any]) -> None:
    """Repeat the pure-JSON launch invariants without importing torch."""

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
    if len(str(stage1.get("checkpoint", {}).get("sha256", ""))) != 64:
        raise ValueError("formal protocol lacks a frozen Stage1 checkpoint hash")
    if stage1.get("checkpoint_update_during_stage2_l") is not False:
        raise ValueError("formal protocol must freeze Stage1 during Stage2-L")
    visual_cache = protocol.get("visual_context_cache", {})
    if len(str(visual_cache.get("orion_checkpoint", {}).get("sha256", ""))) != 64:
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
    if not str(
        protocol.get("fixed_qa_construction", {}).get("source_condition", "")
    ).startswith("clean_off"):
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


def _audit_data_source(
    *, source_path: Path, source: Mapping[str, Any], protocol_path: Path
) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    if source.get("schema") == INVENTORY_SCHEMA:
        events = list(source.get("events", []))
        split_counts = Counter(str(row.get("split", "")) for row in events)
        missing = list(source.get("frozen_plan_routes_without_audited_v5", []))
        checks = {
            "exactly_24_events": int(source.get("audited_event_count", -1)) == 24,
            "frozen_16_4_4_split_complete": dict(split_counts)
            == EXPECTED_EVENT_SPLITS,
            "formal_v5_qa_geometry_review_bank_complete": source.get(
                "remaining_gates", {}
            ).get("formal_v5_qa_geometry_review_bank_complete")
            is True,
            "formal_dataset_assembled_and_hash_bound": False,
            "source_protocol_hash_matches": _reference_matches(
                source.get("provenance", {}).get("formal_data_protocol", {}),
                protocol_path,
            ),
        }
        details = {
            "kind": "available_subset_inventory",
            "event_count": int(source.get("audited_event_count", 0)),
            "event_split_counts": dict(sorted(split_counts.items())),
            "qa_record_count": int(source.get("qa_record_count", 0)),
            "missing_frozen_routes": missing,
            "reviewed_bank_events_without_usable_v5": list(
                source.get("reviewed_bank_events_without_usable_v5", [])
            ),
        }
        return checks, details

    if source.get("schema") != FORMAL_DATASET_SCHEMA:
        raise ValueError("unsupported formal data source schema")
    events = list(source.get("events", []))
    split_counts = Counter(str(row.get("split", "")) for row in events)
    dataset_checks = source.get("checks", {})
    records_ref = source.get("records", {})
    records_path = Path(str(records_ref.get("path", "")))
    records_valid = (
        records_path.is_file()
        and str(records_ref.get("sha256", "")) == sha256_file(records_path)
    )
    protocol_ref = source.get("provenance", {}).get(
        "formal_data_and_corruption_protocol", {}
    )
    checks = {
        "exactly_24_events": int(source.get("event_count", -1)) == 24,
        "frozen_16_4_4_split_complete": dict(split_counts)
        == EXPECTED_EVENT_SPLITS,
        "formal_v5_qa_geometry_review_bank_complete": dataset_checks.get(
            "all_qa_geometry_reviews_accepted_and_hash_bound"
        )
        is True,
        "formal_dataset_assembled_and_hash_bound": (
            source.get("status") == "assembled_formal_data_training_launch_locked"
            and source.get("formal_training_ready") is False
            and source.get("stage2p_allowed") is False
            and records_valid
            and all(dataset_checks.get(key) is True for key in REQUIRED_DATASET_CHECKS)
        ),
        "source_protocol_hash_matches": _reference_matches(
            protocol_ref, protocol_path
        ),
    }
    details = {
        "kind": "assembled_formal_dataset",
        "event_count": int(source.get("event_count", 0)),
        "event_split_counts": dict(sorted(split_counts.items())),
        "qa_record_count": int(source.get("qa_record_count", 0)),
        "records": dict(records_ref),
        "missing_frozen_routes": [],
        "reviewed_bank_events_without_usable_v5": [],
    }
    return checks, details


def _audit_mr1_comparison(comparison: Mapping[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    controlled = comparison.get("controlled_comparison", {})
    decision = str(comparison.get("decision", ""))
    passed = bool(
        comparison.get("schema") == MR1_COMPARISON_SCHEMA
        and comparison.get("status") == "mr1_duration_compared_no_training_launched"
        and controlled.get("valid") is True
        and decision == "engineering_multievent_paradigm_passes"
    )
    return passed, {
        "controlled_comparison_valid": controlled.get("valid") is True,
        "decision": decision,
        "overfit_established": comparison.get("overfit_diagnostic", {}).get(
            "gate_count_overfit_established"
        )
        is True,
        "next_action": str(comparison.get("next_action", "")),
    }


def _audit_training_protocol(
    *, protocol_path: Optional[Path], data_source_path: Path,
    data_protocol_path: Path, mr1_comparison_path: Path
) -> Tuple[bool, Dict[str, Any]]:
    if protocol_path is None:
        return False, {"present": False}
    value = _load(protocol_path)
    inputs = value.get("validated_inputs", {})
    locks = value.get("launch_locks", {})
    passed = bool(
        value.get("schema") == FORMAL_TRAINING_PROTOCOL_SCHEMA
        and value.get("status") == FORMAL_TRAINING_PROTOCOL_STATUS
        and inputs.get("formal_data_source_sha256")
        == sha256_file(data_source_path)
        and inputs.get("formal_data_protocol_sha256")
        == sha256_file(data_protocol_path)
        and inputs.get("mr1_duration_comparison_sha256")
        == sha256_file(mr1_comparison_path)
        and locks.get("formal_stage2_l_allowed_after_amendment") is False
        and locks.get("stage2_p_allowed") is False
        and locks.get("immutable_launch_amendment_required") is True
        and value.get("optimization", {}).get("optimizer_steps")
        in range(1, 10001)
        and value.get("checkpoint_selection", {}).get("split") == "dev"
        and value.get("locked_test", {}).get("enters_optimizer") is False
        and value.get("locked_test", {}).get("used_for_checkpoint_selection")
        is False
        and isinstance(value.get("release_thresholds"), dict)
        and bool(value.get("release_thresholds"))
    )
    return passed, {
        "present": True,
        "path": str(protocol_path.resolve()),
        "sha256": sha256_file(protocol_path),
        "schema": value.get("schema"),
        "status": value.get("status"),
    }


def _audit_launch_amendment(
    *, amendment_path: Optional[Path], data_source_path: Path,
    data_protocol_path: Path, mr1_comparison_path: Path,
    training_protocol_path: Optional[Path]
) -> Tuple[bool, Dict[str, Any]]:
    if amendment_path is None:
        return False, {"present": False}
    value = _load(amendment_path)
    run = value.get("authorized_run", {})
    locks = value.get("launch_locks", {})
    inputs = value.get("validated_inputs", {})
    expected_inputs = {
        "formal_data_source_sha256": sha256_file(data_source_path),
        "formal_data_protocol_sha256": sha256_file(data_protocol_path),
        "mr1_duration_comparison_sha256": sha256_file(mr1_comparison_path),
        "formal_training_protocol_sha256": (
            sha256_file(training_protocol_path)
            if training_protocol_path is not None
            else None
        ),
    }
    passed = bool(
        training_protocol_path is not None
        and value.get("schema") == LAUNCH_AMENDMENT_SCHEMA
        and locks.get("formal_stage2l_training_allowed") is True
        and locks.get("stage2p_allowed") is False
        and int(run.get("maximum_submissions", 0)) == 1
        and run.get("automatic_retry_or_extension") is False
        and run.get("formal_training") is True
        and inputs == expected_inputs
    )
    return passed, {
        "present": True,
        "path": str(amendment_path.resolve()),
        "sha256": sha256_file(amendment_path),
        "schema": value.get("schema"),
    }


def audit_readiness(
    *,
    data_source_path: Path,
    formal_data_protocol_path: Path,
    mr1_duration_comparison_path: Path,
    formal_training_protocol_path: Optional[Path] = None,
    launch_amendment_path: Optional[Path] = None,
) -> Dict[str, Any]:
    source = _load(data_source_path)
    data_protocol = _load(formal_data_protocol_path)
    comparison = _load(mr1_duration_comparison_path)
    if (
        data_protocol.get("schema") != DATA_PROTOCOL_SCHEMA
        or data_protocol.get("status") != DATA_PROTOCOL_STATUS
    ):
        raise ValueError("formal data and corruption protocol is not frozen")
    _validate_data_isolation_protocol(data_protocol)

    data_checks, data_details = _audit_data_source(
        source_path=data_source_path,
        source=source,
        protocol_path=formal_data_protocol_path,
    )
    mr1_passed, mr1_details = _audit_mr1_comparison(comparison)
    training_protocol_passed, training_protocol_details = _audit_training_protocol(
        protocol_path=formal_training_protocol_path,
        data_source_path=data_source_path,
        data_protocol_path=formal_data_protocol_path,
        mr1_comparison_path=mr1_duration_comparison_path,
    )
    amendment_passed, amendment_details = _audit_launch_amendment(
        amendment_path=launch_amendment_path,
        data_source_path=data_source_path,
        data_protocol_path=formal_data_protocol_path,
        mr1_comparison_path=mr1_duration_comparison_path,
        training_protocol_path=formal_training_protocol_path,
    )
    checks = {
        **data_checks,
        "mr1_e_supports_multievent_learning_without_duration_overfit": mr1_passed,
        "formal_training_protocol_frozen": training_protocol_passed,
        "immutable_single_run_launch_amendment_valid": amendment_passed,
        "base_protocol_preserves_stage2p_lock": data_protocol.get(
            "launch_locks", {}
        ).get("stage2_p_allowed")
        is False,
    }
    ready = bool(checks and all(checks.values()))
    failed = sorted(key for key, value in checks.items() if not value)
    return {
        "schema": SCHEMA,
        "status": (
            "formal_stage2l_launch_inputs_passed_submit_wrapper_still_required"
            if ready
            else "formal_stage2l_launch_blocked"
        ),
        "formal_training_allowed": ready,
        "training_started": False,
        "stage2p_allowed": False,
        "checks": checks,
        "failed_gates": failed,
        "data": data_details,
        "mr1_duration_diagnostic": mr1_details,
        "formal_training_protocol": training_protocol_details,
        "launch_amendment": amendment_details,
        "provenance": {
            "data_source": _source_reference(data_source_path),
            "formal_data_and_corruption_protocol": _source_reference(
                formal_data_protocol_path
            ),
            "mr1_duration_comparison": _source_reference(
                mr1_duration_comparison_path
            ),
        },
        "claim_boundary": (
            "Fail-closed launch-readiness audit only. A passing report does not "
            "start training, unlock Stage2-P, or establish semantic, planning, "
            "closed-loop, generalization, or safety evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-source", type=Path, required=True)
    parser.add_argument("--formal-data-protocol", type=Path, required=True)
    parser.add_argument("--mr1-duration-comparison", type=Path, required=True)
    parser.add_argument("--formal-training-protocol", type=Path)
    parser.add_argument("--launch-amendment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite formal launch-readiness audit")
    result = audit_readiness(
        data_source_path=args.data_source.resolve(),
        formal_data_protocol_path=args.formal_data_protocol.resolve(),
        mr1_duration_comparison_path=args.mr1_duration_comparison.resolve(),
        formal_training_protocol_path=(
            args.formal_training_protocol.resolve()
            if args.formal_training_protocol is not None
            else None
        ),
        launch_amendment_path=(
            args.launch_amendment.resolve()
            if args.launch_amendment is not None
            else None
        ),
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
                "formal_training_allowed": result["formal_training_allowed"],
                "failed_gates": result["failed_gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 3 if args.require_ready and not result["formal_training_allowed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
