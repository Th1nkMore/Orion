#!/usr/bin/env python3
"""Diagnose the completed Route151 Stage2-L v8 gate failure.

This is a CPU-only, evidence-bounded diagnosis.  It never authorizes another
training run and keeps the Stage2-L pilot and Stage2-P locks closed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


SCHEMA = "orion.stage2l_v8_route151_failure_diagnosis.v1"
EXPECTED_REPORT_SCHEMA = "orion.stage2l_v8_gradient_routed_smoke.v1"
EXPECTED_VALIDATION_SCHEMA = (
    "orion.stage2l_v8_route151_independent_validation.v1"
)
EXPECTED_ALIGNMENT_SCHEMA = (
    "orion.stage2l_v8_generation_prompt_alignment.v1"
)
EXPECTED_CHECKPOINT_AUDIT_SCHEMA = (
    "orion.stage2l_v8_checkpoint_tensor_audit.v1"
)


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON artifact must contain an object")
    return value


def _read_jsonl(path: Path) -> Sequence[Dict[str, Any]]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _failed(checks: Mapping[str, Any]) -> Sequence[str]:
    return tuple(sorted(str(key) for key, value in checks.items() if not value))


def _group_trajectories(history: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped = defaultdict(list)
    for row in history:
        grouped[str(row["group_id"])].append(row)
    output = {}
    for group_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda value: int(value["optimizer_step"]))
        fractions = [float(row["minimum_attained_fraction"]) for row in rows]
        ranking = [float(row["ranking_loss"]) for row in rows]
        map_losses = [
            float(row["foreground_balanced_relevance"]) for row in rows
        ]
        stance_losses = [
            float(row["dataset_frequency_balanced_stance"]) for row in rows
        ]
        output[group_id] = {
            "visit_count": len(rows),
            "optimizer_steps": [int(row["optimizer_step"]) for row in rows],
            "first_to_last": {
                "language_nll": [
                    float(rows[0]["language_nll"]),
                    float(rows[-1]["language_nll"]),
                ],
                "attained_fraction": [fractions[0], fractions[-1]],
                "ranking_loss": [ranking[0], ranking[-1]],
                "relevance_loss": [map_losses[0], map_losses[-1]],
                "stance_loss": [stance_losses[0], stance_losses[-1]],
            },
            "within_group_regression_observed": {
                "attained_fraction": any(
                    right < left
                    for left, right in zip(fractions, fractions[1:])
                ),
                "relevance_loss": any(
                    right > left
                    for left, right in zip(map_losses, map_losses[1:])
                ),
                "stance_loss": any(
                    right > left
                    for left, right in zip(stance_losses, stance_losses[1:])
                ),
            },
        }
    return output


def _label_contract_audit(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    hard = [
        row
        for row in records
        if row.get("loss_policy", {}).get("hard_language_target") is True
    ]
    zero = [
        row for row in hard
        if row["counterfactual"]["variant"] == "zero_uq"
    ]
    zero_groups = {
        str(row["counterfactual"]["group_id"]) for row in zero
    }
    arbitrary_zero_locations = []
    contradictory_zero_epistemic = []
    zero_risk_locations = []
    for row in zero:
        summary = row["target"]["structured_summary"]
        observation = summary["observation_uncertainty"]
        risk = summary["task_risk"]
        if (
            float(observation["peak_score"]) == 0.0
            and str(observation["peak_view"]).lower() != "none"
            and str(observation["peak_region"]).lower() != "none"
        ):
            arbitrary_zero_locations.append(str(row.get("sample_id", "")))
        answer = str(row["conversation"][1]["value"])
        if (
            row["question_family"] == "epistemic_limitation"
            and "evidence=unreliable" in answer
            and "hidden_content=unknown" in answer
        ):
            contradictory_zero_epistemic.append(
                str(row.get("sample_id", ""))
            )
        if (
            row["question_family"] == "task_relevance"
            and str(risk["level"]) == "low"
            and float(risk["peak_score"]) == 0.0
            and str(risk["peak_view"]).lower() != "none"
            and str(risk["peak_region"]).lower() != "none"
        ):
            zero_risk_locations.append(str(row.get("sample_id", "")))
    return {
        "hard_language_record_count": len(hard),
        "hard_language_by_variant": dict(sorted(Counter(
            str(row["counterfactual"]["variant"]) for row in hard
        ).items())),
        "hard_language_by_family": dict(sorted(Counter(
            str(row["question_family"]) for row in hard
        ).items())),
        "zero_uq_group_count": len(zero_groups),
        "zero_uq_hard_record_count": len(zero),
        "zero_uq_records_with_arbitrary_non_none_location": len(
            arbitrary_zero_locations
        ),
        "zero_uq_epistemic_answers_claiming_unreliable_unknown": len(
            contradictory_zero_epistemic
        ),
        "zero_uq_task_risk_answers_with_arbitrary_non_none_location": len(
            zero_risk_locations
        ),
        "semantic_contract_self_consistent": not (
            arbitrary_zero_locations
            or contradictory_zero_epistemic
            or zero_risk_locations
        ),
        "interpretation": (
            "Zero-UQ examples use argmax placeholder locations and still label "
            "evidence as unreliable/hidden content as unknown. These are not "
            "valid semantic targets for demonstrating that the VLM understands "
            "absence of flagged observation uncertainty."
        ),
    }


def diagnose(
    *,
    report: Mapping[str, Any],
    validation: Mapping[str, Any],
    prompt_alignment: Mapping[str, Any],
    checkpoint_audit: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if report.get("schema") != EXPECTED_REPORT_SCHEMA:
        raise ValueError("unexpected v8 report schema")
    if validation.get("schema") != EXPECTED_VALIDATION_SCHEMA:
        raise ValueError("unexpected independent-validation schema")
    if prompt_alignment.get("schema") != EXPECTED_ALIGNMENT_SCHEMA:
        raise ValueError("unexpected prompt-alignment schema")
    if checkpoint_audit.get("schema") != EXPECTED_CHECKPOINT_AUDIT_SCHEMA:
        raise ValueError("unexpected checkpoint-audit schema")
    if validation.get("integrity_valid") is not True:
        raise ValueError("cannot diagnose learning from an invalid artifact")
    if validation.get("smoke_passed") is not False:
        raise ValueError("failure diagnosis requires a failed smoke")
    checks = report.get("checks", {})
    if checks != validation.get("checks"):
        raise ValueError("report and independent validator disagree on gates")
    if prompt_alignment.get("status") != "alignment_pass" or not all(
        prompt_alignment.get("checks", {}).values()
    ):
        raise ValueError("prompt alignment must be resolved before learning diagnosis")
    if checkpoint_audit.get("status") != "finite_complete_checkpoint":
        raise ValueError("checkpoint is absent, incomplete, or non-finite")

    before = report["before"]
    after = report["after"]
    attained = [float(value) for value in after["ranking"]["attained_fraction"]]
    stance_recall = {
        str(key): float(value)
        for key, value in after["stance"]["per_target_class_recall"].items()
    }
    label_audit = _label_contract_audit(records)
    generation = after["generation_semantics"]
    findings = {
        "runtime_integrity_and_checkpoint": {
            "runtime_or_oom_failure": False,
            "independent_integrity_valid": True,
            "optimizer_steps": int(report["optimizer_steps"]),
            "saved_trainable_parameters": int(
                checkpoint_audit["total_saved_trainable_parameters"]
            ),
            "all_saved_tensors_finite": all(
                section["all_finite"]
                for section in checkpoint_audit["sections"].values()
            ),
        },
        "partial_structural_learnability": {
            "ranking_groups_at_or_above_0_8": sum(
                value >= 0.8 for value in attained
            ),
            "ranking_group_count": len(attained),
            "minimum_attained_fraction": min(attained),
            "positive_order_fraction": float(
                after["ranking"]["positive_order_fraction"]
            ),
            "foreground_recall": float(
                after["relevance_support"]["foreground_recall"]
            ),
            "background_false_positive_rate": float(
                after["relevance_support"]["background_false_positive_rate"]
            ),
            "stance_balanced_accuracy_before_after": [
                float(before["stance"]["balanced_accuracy"]),
                float(after["stance"]["balanced_accuracy"]),
            ],
            "stance_class_recall": stance_recall,
            "minimum_stance_target_probability": float(
                after["stance"]["minimum_target_probability"]
            ),
        },
        "language_result": {
            "teacher_forced_target_nll_before_after": [
                float(before["first_group_mean_hard_language_nll"]),
                float(after["first_group_mean_hard_language_nll"]),
            ],
            "same_family_margin_pass_before_after": [
                float(before["first_group_same_family_margin_pass_fraction"]),
                float(after["first_group_same_family_margin_pass_fraction"]),
            ],
            "semantic_parse_rate": float(generation["semantic_parse_rate"]),
            "semantic_field_accuracy": float(
                generation["semantic_field_accuracy"]
            ),
            "semantic_answer_exact_match": float(
                generation["semantic_answer_exact_match"]
            ),
            "nonrepeating_text_fraction": float(
                generation["nonrepeating_text_fraction"]
            ),
            "prompt_or_label_alignment_failure": False,
            "free_generation_semantics_learned": False,
            "interpretation": (
                "The aligned teacher-forced objective became much easier, but "
                "greedy generation learned no parseable semantic fields. This "
                "is a real autoregressive semantic/decoding failure, not a "
                "prompt-prefix or label-mask bug."
            ),
        },
        "label_contract": label_audit,
        "capacity_interpretation": {
            "insufficient_parameter_count_supported": False,
            "capacity_sufficiency_proven": False,
            "reason": (
                "R ranking, relevance recall, stance balanced accuracy, and "
                "teacher-forced NLL all improved substantially with 23.64M "
                "saved trainable parameters. The evidence instead points first "
                "to objective interference, sparse caution supervision, an "
                "inconsistent zero-UQ semantic contract, and unconstrained "
                "autoregressive field generation."
            ),
        },
    }
    return {
        "schema": SCHEMA,
        "status": "diagnosed_v8_gate_failure",
        "report_status": report["status"],
        "independent_validation_status": validation["status"],
        "passed_check_count": sum(bool(value) for value in checks.values()),
        "failed_check_count": len(_failed(checks)),
        "failed_checks": list(_failed(checks)),
        "findings": findings,
        "group_trajectories": _group_trajectories(report["history"]),
        "decision": {
            "retry_or_epoch_extension_allowed": False,
            "stage2l_pilot_training_allowed": False,
            "stage2p_allowed": False,
            "next_architecture": (
                "Keep Stage1 U frozen and task agnostic. Let the VLM predict "
                "explicit categorical semantic fields and the dense R map; "
                "render QA text deterministically from those predicted fields "
                "for the next learnability test. Free-form explanation remains "
                "auxiliary until field accuracy passes."
            ),
        },
        "next_design_constraints": [
            "Replace zero-UQ argmax locations with explicit none/not_applicable sentinels and do not label zero-UQ evidence as unreliable hidden content.",
            "Keep R and path/task relevance VLM-owned; do not move task relevance into the Stage1 adapter.",
            "Supervise value-bearing semantic fields with explicit categorical heads before rendering human-readable QA text.",
            "Retain greedy free generation only as a diagnostic until structured field prediction passes.",
            "Add an explicit background precision term or calibrated support threshold for R without sacrificing foreground recall.",
            "Increase genuine caution support across events; do not rely on reweighting only two caution labels in one event.",
            "Require every-group ranking, every stance class, R precision/recall, field accuracy, and zero-UQ invariance to pass before the eight-event pilot.",
            "Do not launch another A800 run without a new immutable amendment after CPU-only implementation and label audits pass.",
        ],
        "next_action": (
            "revise the zero-UQ QA contract and implement a VLM-owned "
            "structured field head plus deterministic renderer; CPU-test all "
            "contracts while keeping training locks closed"
        ),
        "claim_boundary": (
            "One-event engineering failure diagnosis only; no held-out, "
            "generalization, trajectory, closed-loop, or safety evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--prompt-alignment", type=Path, required=True)
    parser.add_argument("--checkpoint-audit", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(
        report=_read_json(args.report),
        validation=_read_json(args.validation),
        prompt_alignment=_read_json(args.prompt_alignment),
        checkpoint_audit=_read_json(args.checkpoint_audit),
        records=_read_jsonl(args.records),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
