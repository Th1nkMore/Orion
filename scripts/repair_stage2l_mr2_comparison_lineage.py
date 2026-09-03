#!/usr/bin/env python3
"""Audit MR1 trainer drift and issue a secondary MR2 coverage interpretation.

The preregistered comparator correctly rejects an exact source-hash mismatch.
This tool does not overwrite or relabel that primary result.  It verifies that
the mismatch is solely the later admission guard that allowed an independently
authorized 80-step diagnostic, checks the pre-existing exact 40-step replay
evidence, and then applies the already-frozen decision rules in a separate,
explicitly post-hoc lineage-repaired engineering artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from scripts.scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2l_mr2_coverage_lineage_repair.v1"
OLD_GUARD = (
    "    if args.max_optimizer_steps != 40:\n"
    "        raise ValueError(\"MR1 is frozen at exactly 40 optimizer steps\")"
)
NEW_GUARD = (
    "    if args.max_optimizer_steps not in ALLOWED_BOUNDED_OPTIMIZER_STEPS:\n"
    "        raise ValueError(\"MR1 bounded diagnostics allow exactly 40 or 80 optimizer steps\")"
)
NEW_CONSTANT = "ALLOWED_BOUNDED_OPTIMIZER_STEPS = (40, 80)\n"


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def reconstruct_reference_trainer(current_source: str) -> str:
    if current_source.count(NEW_CONSTANT) != 1 or current_source.count(NEW_GUARD) != 1:
        raise ValueError("current trainer does not contain the exact admission-only drift")
    reconstructed = current_source.replace(NEW_CONSTANT, "", 1)
    reconstructed = reconstructed.replace(NEW_GUARD, OLD_GUARD, 1)
    if NEW_CONSTANT in reconstructed or NEW_GUARD in reconstructed:
        raise ValueError("trainer admission-guard reconstruction was incomplete")
    return reconstructed


def _shared_history_exact(
    reference_history: Sequence[Mapping[str, Any]],
    replay_history: Sequence[Mapping[str, Any]],
) -> bool:
    if len(reference_history) != 40 or len(replay_history) < 40:
        return False
    for old, new in zip(reference_history, replay_history[:40]):
        for key in set(old) & set(new):
            if old[key] != new[key]:
                return False
    return True


def _decision(gate_change: Mapping[str, int], expanded_checks: Mapping[str, bool]) -> Dict[str, str]:
    train = int(gate_change["train"])
    dev = int(gate_change["dev"])
    train_dev_gates = [
        value
        for key, value in expanded_checks.items()
        if key.startswith("train_") or key.startswith("dev_")
    ]
    if train_dev_gates and all(value is True for value in train_dev_gates):
        return {
            "decision": "expanded_coverage_engineering_paradigm_passes",
            "next_action": (
                "Use only to design formal training after exact24 and human-review "
                "gates; do not launch automatically."
            ),
        }
    if train > 0 and dev < 0:
        return {
            "decision": "expanded_coverage_overfit_or_objective_mismatch_stop",
            "next_action": (
                "Stop repeated MR scaling and revise objective, capacity or label support."
            ),
        }
    if dev > 0 and train >= 0:
        return {
            "decision": "expanded_coverage_improves_but_remaining_gates_fail",
            "next_action": (
                "Inspect only remaining preregistered gates and absent class cells; "
                "formal training stays locked."
            ),
        }
    return {
        "decision": "expanded_coverage_not_sufficient_revise_objective",
        "next_action": (
            "Stop route or step scaling and revise the relevance or structured-field objective."
        ),
    }


def repair_lineage(
    *,
    frozen_comparison_path: Path,
    frozen_plan_path: Path,
    reference_report_path: Path,
    expanded_report_path: Path,
    reference_protocol_path: Path,
    expanded_protocol_path: Path,
    current_base_trainer_path: Path,
    duration_comparison_path: Path,
    duration_report_path: Path,
) -> Dict[str, Any]:
    frozen = _load(frozen_comparison_path)
    plan = _load(frozen_plan_path)
    reference = _load(reference_report_path)
    expanded = _load(expanded_report_path)
    reference_protocol = _load(reference_protocol_path)
    expanded_protocol = _load(expanded_protocol_path)
    duration = _load(duration_comparison_path)
    duration_report = _load(duration_report_path)

    integrity = frozen.get("controlled_comparison", {}).get("integrity_checks", {})
    false_integrity = sorted(key for key, value in integrity.items() if value is not True)
    if (
        frozen.get("controlled_comparison", {}).get("valid") is not False
        or frozen.get("decision") != "invalid_coverage_comparison"
        or false_integrity != ["same_base_trainer"]
        or plan.get("status")
        != "comparison_inputs_and_decision_rules_frozen_before_mr2_result"
    ):
        raise ValueError("primary comparison failure is not isolated to trainer lineage")

    sources = frozen["sources"]
    expected_source_hashes = {
        "mr1_40step_report": sha256_file(reference_report_path),
        "mr2_17event_40step_report": sha256_file(expanded_report_path),
        "mr1_protocol": sha256_file(reference_protocol_path),
        "mr2_protocol": sha256_file(expanded_protocol_path),
    }
    if any(
        sources[name]["sha256"] != digest
        for name, digest in expected_source_hashes.items()
    ):
        raise ValueError("primary comparison sources do not match supplied artifacts")

    old_hash = reference_protocol["implementation_sources"][
        "scripts/train_stage2l_mr1_smoke.py"
    ]
    new_hash = expanded_protocol["implementation_sources"][
        "scripts/train_stage2l_mr1_smoke.py"
    ]
    current_source = current_base_trainer_path.read_text(encoding="utf-8")
    reconstructed = reconstruct_reference_trainer(current_source)
    reconstructed_hash = hashlib.sha256(reconstructed.encode("utf-8")).hexdigest()
    source_equivalence = {
        "current_trainer_matches_expanded_protocol": (
            sha256_file(current_base_trainer_path) == new_hash
        ),
        "admission_guard_reconstruction_matches_reference_protocol": (
            reconstructed_hash == old_hash
        ),
        "only_added_constant": NEW_CONSTANT.strip(),
        "only_replaced_guard": {
            "reference": OLD_GUARD.strip(),
            "expanded": NEW_GUARD.strip(),
        },
        "training_or_evaluation_body_changed": False,
        "both_compared_runs_use_40_steps": (
            int(reference.get("optimizer_steps", -1)) == 40
            and int(expanded.get("optimizer_steps", -1)) == 40
        ),
    }

    duration_control = duration.get("controlled_comparison", {})
    duration_sources = duration.get("sources", {})
    duration_evidence = {
        "audit_predates_mr2_result": (
            duration.get("status") == "mr1_duration_compared_no_training_launched"
        ),
        "controlled_comparison_valid": duration_control.get("valid") is True,
        "before_metrics_match": duration_control.get("before_metrics_match") is True,
        "first_40_step_replay_matches_within_1e_6": (
            duration_control.get("first_40_step_replay_matches_within_1e_6") is True
        ),
        "reported_first_40_deltas_are_zero": all(
            float(value) == 0.0
            for value in duration_control.get("first_40_step_max_abs_delta", {}).values()
        ),
        "all_stable_inputs_match": all(
            value is True
            for value in duration_control.get("stable_input_matches", {}).values()
        ),
        "duration_reference_is_same_mr1_report": (
            duration_sources.get("step40_report", {}).get("sha256")
            == sha256_file(reference_report_path)
        ),
        "duration_run_uses_expanded_base_trainer": (
            duration_report.get("provenance", {})
            .get("validated_inputs", {})
            .get("trainer_sha256")
            == new_hash
        ),
        "direct_shared_history_fields_exact_for_first_40_steps": _shared_history_exact(
            reference.get("history", []), duration_report.get("history", [])
        ),
    }
    lineage_repair_valid = all(
        value is True
        for value in list(source_equivalence.values())[:2]
        + [source_equivalence["both_compared_runs_use_40_steps"]]
        + list(duration_evidence.values())
    )
    if not lineage_repair_valid:
        raise ValueError("trainer lineage equivalence audit failed")

    outcome = _decision(frozen["gate_count_change"], expanded["checks"])
    return {
        "schema": SCHEMA,
        "status": "secondary_engineering_interpretation_completed_no_training_launched",
        "primary_preregistered_comparison": {
            "remains_invalid": True,
            "reason": "exact base-trainer source hash mismatch",
            "path": str(frozen_comparison_path.resolve()),
            "sha256": sha256_file(frozen_comparison_path),
            "must_not_be_overwritten": True,
        },
        "lineage_repair": {
            "valid_for_secondary_engineering_interpretation": True,
            "source_equivalence": source_equivalence,
            "preexisting_replay_evidence": duration_evidence,
            "reference_trainer_sha256": old_hash,
            "expanded_base_trainer_sha256": new_hash,
            "reconstructed_reference_trainer_sha256": reconstructed_hash,
        },
        "secondary_interpretation": {
            **outcome,
            "gate_count_change": frozen["gate_count_change"],
            "gate_summary": frozen["gate_summary"],
            "split_distribution_metric_changes": frozen[
                "split_distribution_metric_changes"
            ],
            "common_event_metric_changes": frozen["common_event_metric_changes"],
        },
        "locks": {
            "automatic_retry_or_extension": False,
            "additional_mr_training": False,
            "formal_stage2l_training_allowed": False,
            "stage2p_allowed": False,
            "locked_test_evaluation_allowed": False,
            "closed_loop_matrix_allowed": False,
        },
        "sources": {
            "frozen_plan": {
                "path": str(frozen_plan_path.resolve()),
                "sha256": sha256_file(frozen_plan_path),
            },
            "reference_report": {
                "path": str(reference_report_path.resolve()),
                "sha256": sha256_file(reference_report_path),
            },
            "expanded_report": {
                "path": str(expanded_report_path.resolve()),
                "sha256": sha256_file(expanded_report_path),
            },
            "duration_comparison": {
                "path": str(duration_comparison_path.resolve()),
                "sha256": sha256_file(duration_comparison_path),
            },
            "duration_report": {
                "path": str(duration_report_path.resolve()),
                "sha256": sha256_file(duration_report_path),
            },
            "current_base_trainer": {
                "path": str(current_base_trainer_path.resolve()),
                "sha256": sha256_file(current_base_trainer_path),
            },
        },
        "claim_boundary": (
            "Post-hoc, lineage-repaired engineering interpretation using an "
            "admission-only source diff and replay evidence that predates MR2. "
            "The preregistered exact-hash comparison remains invalid. This does "
            "not support formal generalization, VLM understanding, planning, "
            "closed-loop, or safety claims."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-comparison", type=Path, required=True)
    parser.add_argument("--frozen-plan", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--expanded-report", type=Path, required=True)
    parser.add_argument("--reference-protocol", type=Path, required=True)
    parser.add_argument("--expanded-protocol", type=Path, required=True)
    parser.add_argument("--current-base-trainer", type=Path, required=True)
    parser.add_argument("--duration-comparison", type=Path, required=True)
    parser.add_argument("--duration-report", type=Path, required=True)
    parser.add_argument("--lineage-amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite lineage-repaired comparison")
    amendment = _load(args.lineage_amendment)
    expected_inputs = {
        "lineage_auditor_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_comparison_plan_sha256": sha256_file(args.frozen_plan),
        "primary_invalid_comparison_sha256": sha256_file(args.frozen_comparison),
        "mr1_40_report_sha256": sha256_file(args.reference_report),
        "mr2_recovered_report_sha256": sha256_file(args.expanded_report),
        "preexisting_duration_comparison_sha256": sha256_file(
            args.duration_comparison
        ),
        "duration_80_report_sha256": sha256_file(args.duration_report),
    }
    locks = amendment.get("locks", {})
    if (
        amendment.get("schema")
        != "orion.stage2l_mr2_lineage_repair_amendment.v1"
        or amendment.get("validated_inputs") != expected_inputs
        or Path(str(amendment.get("authorized_audit", {}).get("output", ""))).resolve()
        != args.output.resolve()
        or amendment.get("authorized_audit", {}).get("optimizer_steps") != 0
        or amendment.get("authorized_audit", {}).get("may_modify_reports") is not False
        or amendment.get("trigger", {}).get("primary_comparison_must_remain_invalid")
        is not True
        or any(
            locks.get(name) is not False
            for name in (
                "automatic_retry_or_extension",
                "additional_mr_training",
                "formal_stage2l_training_allowed",
                "stage2p_allowed",
                "route203_glare_allowed",
                "locked_test_evaluation_allowed",
                "closed_loop_matrix_allowed",
            )
        )
    ):
        raise ValueError("lineage-repair amendment is absent, stale, or too broad")
    result = repair_lineage(
        frozen_comparison_path=args.frozen_comparison,
        frozen_plan_path=args.frozen_plan,
        reference_report_path=args.reference_report,
        expanded_report_path=args.expanded_report,
        reference_protocol_path=args.reference_protocol,
        expanded_protocol_path=args.expanded_protocol,
        current_base_trainer_path=args.current_base_trainer,
        duration_comparison_path=args.duration_comparison,
        duration_report_path=args.duration_report,
    )
    result["sources"]["lineage_amendment"] = {
        "path": str(args.lineage_amendment.resolve()),
        "sha256": sha256_file(args.lineage_amendment),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "primary_comparison_remains_invalid": True,
                "lineage_repair_valid": True,
                **result["secondary_interpretation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
