#!/usr/bin/env python3
"""Compare MR1-40 with the preregistered MR2 17-event coverage diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


SCHEMA = "orion.stage2l_mr2_coverage_comparison.v1"
MR1_SCHEMA = "orion.stage2l_mr1_multiroute_smoke.v1"
MR2_SCHEMA = "orion.stage2l_mr2_expanded_coverage_smoke.v1"
MR1_PROTOCOL_SCHEMA = "orion.stage2l_mr1_training_protocol.v1"
MR2_PROTOCOL_SCHEMA = "orion.stage2l_mr2_training_protocol.v1"


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gate_summary(checks: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    selected = {
        key: bool(value)
        for key, value in checks.items()
        if key.startswith(prefix + "_")
    }
    return {
        "passed": sum(selected.values()),
        "total": len(selected),
        "all_passed": bool(selected) and all(selected.values()),
        "failed": sorted(key for key, value in selected.items() if not value),
    }


def _snapshot(value: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "foreground_recall": float(
            value["relevance_support"]["foreground_recall"]
        ),
        "background_false_positive_rate": float(
            value["relevance_support"]["background_false_positive_rate"]
        ),
        "positive_order_fraction": float(
            value["ranking"]["positive_order_fraction"]
        ),
        "minimum_attained_fraction": float(
            value["ranking"]["minimum_attained_fraction"]
        ),
        "task_field_accuracy": float(value["task_fields"]["overall_accuracy"]),
        "supported_class_macro_recall": float(
            value["task_fields"]["supported_class_macro_recall"]
        ),
        "zero_uq_complete_field_accuracy": float(
            value["task_fields"]["zero_uq_complete_field_accuracy"]
        ),
        "stance_accuracy": float(
            value["task_fields"]["per_field_accuracy"]["stance"]
        ),
        "semantic_field_accuracy": float(
            value["deterministic_render"]["semantic_field_accuracy"]
        ),
        "semantic_answer_exact_match": float(
            value["deterministic_render"]["semantic_answer_exact_match"]
        ),
    }


def _metric_delta(
    reference: Mapping[str, float], expanded: Mapping[str, float]
) -> Dict[str, Any]:
    output = {}
    for key in sorted(reference):
        before = float(reference[key])
        after = float(expanded[key])
        lower_is_better = key == "background_false_positive_rate"
        signed = before - after if lower_is_better else after - before
        output[key] = {
            "mr1_8event_40step": before,
            "mr2_17event_40step": after,
            "raw_delta": after - before,
            "signed_improvement": signed,
            "direction": (
                "improved"
                if signed > 1e-9
                else "regressed"
                if signed < -1e-9
                else "unchanged"
            ),
        }
    return output


def _per_event_snapshot(value: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "foreground_recall": float(
            value["relevance_support"]["foreground_recall"]
        ),
        "background_false_positive_rate": float(
            value["relevance_support"]["background_false_positive_rate"]
        ),
        "positive_order_fraction": float(
            value["ranking"]["positive_order_fraction"]
        ),
        "minimum_attained_fraction": float(
            value["ranking"]["minimum_attained_fraction"]
        ),
    }


def _primary_event_presentation_counts(
    report: Mapping[str, Any], expected_events: set[str]
) -> Counter:
    counts: Counter = Counter()
    for row in report.get("history", []):
        event_ids = [str(value) for value in row.get("primary_event_ids", [])]
        if len(event_ids) != len(expected_events) or set(event_ids) != expected_events:
            return Counter()
        counts.update(event_ids)
    return counts


def _validate_protocol_pair(
    reference: Mapping[str, Any], expanded: Mapping[str, Any]
) -> Dict[str, bool]:
    return {
        "schemas_match_expected": (
            reference.get("schema") == MR1_PROTOCOL_SCHEMA
            and expanded.get("schema") == MR2_PROTOCOL_SCHEMA
        ),
        "optimizer_steps_fixed_at_40": (
            int(reference.get("bounded_preexperiment", {}).get("optimizer_steps", -1))
            == 40
            and int(expanded.get("bounded_preexperiment", {}).get("optimizer_steps", -1))
            == 40
        ),
        "losses_unchanged": reference.get("losses") == expanded.get("losses"),
        "release_gates_unchanged": (
            {
                key: value
                for key, value in reference.get("release_gates", {}).items()
                if key != "interpretation"
            }
            == {
                key: value
                for key, value in expanded.get("release_gates", {}).items()
                if key != "interpretation"
            }
        ),
        "fresh_initialization_in_both": (
            reference.get("bounded_preexperiment", {}).get(
                "fresh_initialization_from_original_orion_checkpoint"
            )
            is True
            and expanded.get("bounded_preexperiment", {}).get(
                "fresh_initialization_from_original_orion_checkpoint"
            )
            is True
        ),
        "formal_and_stage2p_locked": (
            reference.get("launch_locks", {}).get(
                "formal_stage2l_training_allowed"
            )
            is False
            and reference.get("launch_locks", {}).get("stage2p_allowed") is False
            and expanded.get("launch_locks", {}).get(
                "formal_stage2l_training_allowed"
            )
            is False
            and expanded.get("launch_locks", {}).get("stage2p_allowed") is False
        ),
    }


def compare_coverage(
    *,
    reference_report_path: Path,
    expanded_report_path: Path,
    reference_protocol_path: Path,
    expanded_protocol_path: Path,
) -> Dict[str, Any]:
    reference = _load(reference_report_path)
    expanded = _load(expanded_report_path)
    reference_protocol = _load(reference_protocol_path)
    expanded_protocol = _load(expanded_protocol_path)
    if (
        reference.get("schema") != MR1_SCHEMA
        or int(reference.get("optimizer_steps", -1)) != 40
        or len(reference.get("history", [])) != 40
    ):
        raise ValueError("reference is not the completed MR1 8-event/40-step run")
    if (
        expanded.get("schema") != MR2_SCHEMA
        or int(expanded.get("optimizer_steps", -1)) != 40
        or len(expanded.get("history", [])) != 40
        or expanded.get("diagnostic_identity", {}).get("not_formal_training")
        is not True
    ):
        raise ValueError("expanded report is not the completed MR2 coverage run")
    if (
        reference.get("engineering_preexperiment_only") is not True
        or expanded.get("engineering_preexperiment_only") is not True
        or reference.get("formal_training_ready") is not False
        or expanded.get("formal_training_ready") is not False
        or reference.get("stage2p_ready") is not False
        or expanded.get("stage2p_ready") is not False
    ):
        raise ValueError("MR comparison inputs do not preserve launch locks")

    protocol_checks = _validate_protocol_pair(reference_protocol, expanded_protocol)
    old_inputs = reference.get("provenance", {}).get("validated_inputs", {})
    new_inputs = expanded.get("provenance", {}).get("validated_inputs", {})
    old_train = set(map(str, reference.get("train_events", [])))
    old_dev = set(map(str, reference.get("dev_events", [])))
    new_train = set(map(str, expanded.get("train_events", [])))
    new_dev = set(map(str, expanded.get("dev_events", [])))
    reference_gate_keys = {
        split: {
            key for key in reference.get("checks", {}) if key.startswith(split + "_")
        }
        for split in ("train", "dev")
    }
    expanded_gate_keys = {
        split: {
            key for key in expanded.get("checks", {}) if key.startswith(split + "_")
        }
        for split in ("train", "dev")
    }
    old_presentations = _primary_event_presentation_counts(reference, old_train)
    new_presentations = _primary_event_presentation_counts(expanded, new_train)
    integrity = {
        **protocol_checks,
        "same_base_orion_checkpoint": (
            old_inputs.get("base_orion_checkpoint_sha256")
            == new_inputs.get("base_orion_checkpoint_sha256")
            and old_inputs.get("base_orion_checkpoint_sha256") is not None
        ),
        "same_orion_config": (
            old_inputs.get("orion_config_sha256")
            == new_inputs.get("orion_config_sha256")
            and old_inputs.get("orion_config_sha256") is not None
        ),
        "same_base_trainer": (
            old_inputs.get("trainer_sha256")
            == new_inputs.get("base_mr1_trainer_sha256")
            and old_inputs.get("trainer_sha256") is not None
        ),
        "reference_events_are_split_preserving_subset": (
            old_train < new_train
            and old_dev < new_dev
            and not (old_train & new_dev)
            and not (old_dev & new_train)
        ),
        "expected_event_counts": (
            len(old_train) == 6
            and len(old_dev) == 2
            and len(new_train) == 13
            and len(new_dev) == 4
        ),
        "train_dev_gate_keys_unchanged": (
            reference_gate_keys == expanded_gate_keys
            and all(reference_gate_keys.values())
        ),
        "every_step_covers_entire_train_split": bool(
            old_presentations and new_presentations
        ),
        "same_primary_presentations_per_train_event": (
            set(old_presentations) == old_train
            and set(new_presentations) == new_train
            and set(old_presentations.values()) == {40}
            and set(new_presentations.values()) == {40}
        ),
        "all_history_values_finite": all(
            row.get("finite_loss") is True
            and row.get("finite_gradient_norm") is True
            and row.get("finite_gradients") is True
            for report in (reference, expanded)
            for row in report.get("history", [])
        ),
        "trajectory_control_density_and_governor_disabled": (
            reference.get("checks", {}).get(
                "trajectory_control_density_and_governor_disabled"
            )
            is True
            and expanded.get("checks", {}).get(
                "trajectory_control_density_and_governor_disabled"
            )
            is True
        ),
    }
    comparison_valid = all(integrity.values())
    gates = {
        "mr1": {
            split: _gate_summary(reference["checks"], split)
            for split in ("train", "dev")
        },
        "mr2": {
            split: _gate_summary(expanded["checks"], split)
            for split in ("train", "dev")
        },
    }
    train_change = gates["mr2"]["train"]["passed"] - gates["mr1"]["train"]["passed"]
    dev_change = gates["mr2"]["dev"]["passed"] - gates["mr1"]["dev"]["passed"]
    if not comparison_valid:
        decision = "invalid_coverage_comparison"
        next_action = "Do not interpret MR2; repair protocol, lineage, split or runtime drift."
    elif gates["mr2"]["train"]["all_passed"] and gates["mr2"]["dev"]["all_passed"]:
        decision = "expanded_coverage_engineering_paradigm_passes"
        next_action = (
            "Use the result only to design the formal protocol after exact24 and "
            "human-review gates; do not launch formal training automatically."
        )
    elif train_change > 0 and dev_change < 0:
        decision = "expanded_coverage_overfit_or_objective_mismatch_stop"
        next_action = (
            "Stop repeated MR scaling and revise objective/capacity/label support "
            "before any further A800 training run."
        )
    elif dev_change > 0 and train_change >= 0:
        decision = "expanded_coverage_improves_but_remaining_gates_fail"
        next_action = (
            "Inspect only the remaining preregistered gates and absent class cells; "
            "formal training remains locked."
        )
    else:
        decision = "expanded_coverage_not_sufficient_revise_objective"
        next_action = (
            "Stop route/step scaling and revise relevance or structured-field "
            "objective before another training run."
        )

    common_events = {
        split: sorted(
            set(reference["after"][split]["per_event"])
            & set(expanded["after"][split]["per_event"])
        )
        for split in ("train", "dev")
    }
    common_event_changes = {
        split: {
            event: _metric_delta(
                _per_event_snapshot(reference["after"][split]["per_event"][event]),
                _per_event_snapshot(expanded["after"][split]["per_event"][event]),
            )
            for event in common_events[split]
        }
        for split in ("train", "dev")
    }
    return {
        "schema": SCHEMA,
        "status": "mr2_coverage_compared_no_further_training_launched",
        "decision": decision,
        "next_action": next_action,
        "controlled_comparison": {
            "valid": comparison_valid,
            "integrity_checks": integrity,
            "intended_change": "8-event/37-group -> 17-event/80-group coverage",
            "same_optimizer_steps": 40,
            "same_primary_presentations_per_train_event": 40,
            "common_events": common_events,
        },
        "coverage": {
            "mr1": {"train_events": 6, "dev_events": 2, "matched_groups": 37},
            "mr2": {"train_events": 13, "dev_events": 4, "matched_groups": 80},
        },
        "gate_summary": gates,
        "gate_count_change": {"train": train_change, "dev": dev_change},
        "split_distribution_metric_changes": {
            split: _metric_delta(
                _snapshot(reference["after"][split]),
                _snapshot(expanded["after"][split]),
            )
            for split in ("train", "dev")
        },
        "common_event_metric_changes": common_event_changes,
        "locks": {
            "automatic_retry_or_extension": False,
            "formal_stage2l_training_allowed": False,
            "stage2p_allowed": False,
        },
        "sources": {
            "mr1_40step_report": {
                "path": str(reference_report_path.resolve()),
                "sha256": _sha256(reference_report_path),
            },
            "mr2_17event_40step_report": {
                "path": str(expanded_report_path.resolve()),
                "sha256": _sha256(expanded_report_path),
            },
            "mr1_protocol": {
                "path": str(reference_protocol_path.resolve()),
                "sha256": _sha256(reference_protocol_path),
            },
            "mr2_protocol": {
                "path": str(expanded_protocol_path.resolve()),
                "sha256": _sha256(expanded_protocol_path),
            },
        },
        "claim_boundary": (
            "Engineering comparison of event/class coverage only. It is not a "
            "formal generalization, locked-test, planning, closed-loop or safety result."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--expanded-report", type=Path, required=True)
    parser.add_argument("--reference-protocol", type=Path, required=True)
    parser.add_argument("--expanded-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite MR2 coverage comparison")
    result = compare_coverage(
        reference_report_path=args.reference_report.resolve(),
        expanded_report_path=args.expanded_report.resolve(),
        reference_protocol_path=args.reference_protocol.resolve(),
        expanded_protocol_path=args.expanded_protocol.resolve(),
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
                "decision": result["decision"],
                "comparison_valid": result["controlled_comparison"]["valid"],
                "gate_count_change": result["gate_count_change"],
                "formal_stage2l_training_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["controlled_comparison"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
