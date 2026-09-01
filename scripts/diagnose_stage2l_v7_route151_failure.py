#!/usr/bin/env python3
"""Diagnose the completed Route151 v7 smoke without reopening training.

The diagnosis is intentionally evidence bounded.  It distinguishes runtime
integrity, partial structural learnability, optimization interference, and QA
format/semantic underfitting.  It does not turn a one-event smoke into a
generalization or safety claim and it never authorizes another GPU launch.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


SCHEMA = "orion.stage2l_v7_route151_failure_diagnosis.v1"
EXPECTED_REPORT_SCHEMA = "orion.stage2l_v7_calibrated_matched_smoke.v1"
EXPECTED_VALIDATION_SCHEMA = "orion.stage2l_v7_route151_independent_validation.v1"


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON artifact must contain an object")
    return value


def _failed(checks: Mapping[str, Any]) -> Sequence[str]:
    return tuple(sorted(str(key) for key, value in checks.items() if not value))


def _group_trajectories(history: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped = defaultdict(list)
    for row in history:
        grouped[str(row["group_id"])].append(row)
    output = {}
    for group_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda value: int(value["optimizer_step"]))
        fractions = [float(value["minimum_attained_fraction"]) for value in rows]
        map_losses = [float(value["foreground_balanced_relevance"]) for value in rows]
        stance_losses = [float(value["class_balanced_stance"]) for value in rows]
        output[group_id] = {
            "optimizer_steps": [int(value["optimizer_step"]) for value in rows],
            "first_to_last": {
                "language_nll": [
                    float(rows[0]["language_nll"]),
                    float(rows[-1]["language_nll"]),
                ],
                "attained_fraction": [fractions[0], fractions[-1]],
                "relevance_loss": [map_losses[0], map_losses[-1]],
                "stance_loss": [stance_losses[0], stance_losses[-1]],
            },
            "within_group_regression_observed": {
                "attained_fraction": any(
                    right < left for left, right in zip(fractions, fractions[1:])
                ),
                "relevance_loss": any(
                    right > left for left, right in zip(map_losses, map_losses[1:])
                ),
                "stance_loss": any(
                    right > left for left, right in zip(stance_losses, stance_losses[1:])
                ),
            },
        }
    return output


def diagnose(
    *,
    report: Mapping[str, Any],
    validation: Mapping[str, Any],
    tokenizer_audit: Mapping[str, Any],
    parameter_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    if report.get("schema") != EXPECTED_REPORT_SCHEMA:
        raise ValueError("unexpected v7 report schema")
    if validation.get("schema") != EXPECTED_VALIDATION_SCHEMA:
        raise ValueError("unexpected independent-validation schema")
    if validation.get("integrity_valid") is not True:
        raise ValueError("cannot diagnose learning from an invalid artifact")
    if validation.get("smoke_passed") is not False:
        raise ValueError("failure diagnosis requires a failed smoke")
    checks = report.get("checks", {})
    if checks != validation.get("checks"):
        raise ValueError("report and independent validator disagree on gates")

    before = report["before"]
    after = report["after"]
    attained = [float(value) for value in after["ranking"]["attained_fraction"]]
    required_fraction = 0.8
    groups_passing_ranking = sum(value >= required_fraction for value in attained)
    stance_before = float(before["stance"]["balanced_accuracy"])
    stance_after = float(after["stance"]["balanced_accuracy"])
    nll_before = float(before["first_group_mean_hard_language_nll"])
    nll_after = float(after["first_group_mean_hard_language_nll"])
    preference_before = float(
        before["first_group_same_family_margin_pass_fraction"]
    )
    preference_after = float(
        after["first_group_same_family_margin_pass_fraction"]
    )
    marker_lengths = {
        marker: int(value["token_count"])
        for marker, value in tokenizer_audit["family_markers"].items()
    }
    trainable = int(parameter_audit["total_saved_trainable_parameters"])

    findings = {
        "runtime_or_integrity_failure": False,
        "partial_structural_learnability": {
            "all_groups_positive_on_off_order": bool(
                checks["all_groups_positive_on_off_order"]
            ),
            "ranking_groups_at_or_above_0_8": groups_passing_ranking,
            "ranking_group_count": len(attained),
            "minimum_attained_fraction": min(attained),
            "foreground_recall": float(
                after["relevance_support"]["foreground_recall"]
            ),
            "stance_balanced_accuracy_before_after": [
                stance_before,
                stance_after,
            ],
        },
        "remaining_structural_failures": {
            "background_false_positive_rate": float(
                after["relevance_support"]["background_false_positive_rate"]
            ),
            "maximum_background_false_positive_rate": 0.05,
            "missing_stance_classes": sorted(
                key
                for key, value in after["stance"]["per_target_class_recall"].items()
                if float(value) < 1.0
            ),
            "minimum_stance_target_probability": float(
                after["stance"]["minimum_target_probability"]
            ),
        },
        "qa_underfit": {
            "mean_target_nll_before_after": [nll_before, nll_after],
            "same_family_margin_pass_before_after": [
                preference_before,
                preference_after,
            ],
            "family_tag_accuracy": float(
                after["generation_contract"]["family_tag_parse_and_accuracy"]
            ),
            "stance_parse_rate": float(
                after["generation_contract"]["hard_driving_stance_parse_rate"]
            ),
            "nonrepeating_fraction": float(
                after["generation_contract"]["nonrepeating_text_fraction"]
            ),
            "family_marker_subtoken_counts": marker_lengths,
        },
        "capacity_interpretation": {
            "saved_trainable_parameters": trainable,
            "insufficient_capacity_supported_by_this_smoke": False,
            "capacity_sufficiency_for_formal_training_proven": False,
            "reason": (
                "Large improvements in ranking, language NLL, and stance show "
                "that the current modules can learn the one-event task, while "
                "the smoke cannot establish formal multi-event capacity."
            ),
        },
    }
    return {
        "schema": SCHEMA,
        "status": "diagnosed_v7_gate_failure",
        "report_status": report["status"],
        "independent_validation_status": validation["status"],
        "optimizer_steps": int(report["optimizer_steps"]),
        "passed_check_count": sum(bool(value) for value in checks.values()),
        "failed_check_count": len(_failed(checks)),
        "failed_checks": list(_failed(checks)),
        "findings": findings,
        "group_trajectories": _group_trajectories(report["history"]),
        "next_design_constraints": [
            "Keep the Stage1 adapter frozen and task agnostic.",
            "Prevent QA-language gradients from changing the R logits and stance classifier while retaining explicit structural supervision.",
            "Separate deterministic response formatting from semantic field learning; do not count a forced or memorized family marker as understanding.",
            "Use common-vocabulary field prefixes and score the value-bearing semantic spans directly.",
            "Require R background FPR, every-group ranking, every stance class, semantic preference, and free-generation diagnostics to pass together.",
            "Do not unlock the eight-event pilot and do not launch another A800 run without a new immutable amendment.",
        ],
        "next_action": "prepare and CPU-test a gradient-routed Stage2-L v8 objective and QA contract; keep all training launch locks closed",
        "claim_boundary": (
            "One-event engineering failure diagnosis only; no held-out, "
            "generalization, trajectory, closed-loop, or safety evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--tokenizer-audit", type=Path, required=True)
    parser.add_argument("--parameter-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(
        report=_read_json(args.report),
        validation=_read_json(args.validation),
        tokenizer_audit=_read_json(args.tokenizer_audit),
        parameter_audit=_read_json(args.parameter_audit),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
