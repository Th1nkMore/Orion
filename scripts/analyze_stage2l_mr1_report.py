#!/usr/bin/env python3
"""Diagnose MR1 gate failures without launching another training run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence


SCHEMA = "orion.stage2l_mr1_diagnostic.v1"


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _target_distributions(records_path: Path) -> dict:
    unique = {}
    ignored_non_task_field_rows = 0
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        fields = dict(row["target"]["vlm_task_field_targets"])
        # Multiple QA families share each matched visual variant. Empty rows
        # have no task-field supervision, while driving_implication and
        # task_relevance carry disjoint partial dictionaries. Merge those
        # dictionaries at variant granularity and reject only true conflicts.
        if not fields:
            ignored_non_task_field_rows += 1
            continue
        key = (
            row["split"],
            row["counterfactual"]["group_id"],
            row["counterfactual"]["variant"],
        )
        merged = unique.setdefault(key, {})
        for field, value in fields.items():
            if field in merged and merged[field] != value:
                raise RuntimeError(
                    "conflicting task field within one matched variant: %s" % field
                )
            merged[field] = value
    distributions = {}
    for split in ("train", "dev"):
        counts = defaultdict(Counter)
        for (row_split, _, _), fields in unique.items():
            if row_split != split:
                continue
            for field, value in fields.items():
                counts[field][value] += 1
        distributions[split] = {
            field: dict(sorted(values.items()))
            for field, values in sorted(counts.items())
        }
    return {
        "unique_task_field_variant_count": len(unique),
        "ignored_non_task_field_qa_rows": ignored_non_task_field_rows,
        "by_split": distributions,
    }


def _metric_snapshot(split: Mapping[str, object]) -> dict:
    return {
        "foreground_recall": split["relevance_support"]["foreground_recall"],
        "background_false_positive_rate": split["relevance_support"][
            "background_false_positive_rate"
        ],
        "positive_order_fraction": split["ranking"]["positive_order_fraction"],
        "minimum_attained_fraction": split["ranking"]["minimum_attained_fraction"],
        "task_field_accuracy": split["task_fields"]["overall_accuracy"],
        "supported_class_macro_recall": split["task_fields"][
            "supported_class_macro_recall"
        ],
        "stance_accuracy": split["task_fields"]["per_field_accuracy"]["stance"],
        "zero_uq_complete_field_accuracy": split["task_fields"][
            "zero_uq_complete_field_accuracy"
        ],
        "semantic_answer_exact_match": split["deterministic_render"][
            "semantic_answer_exact_match"
        ],
        "semantic_field_accuracy": split["deterministic_render"][
            "semantic_field_accuracy"
        ],
    }


def _event_metrics(split: Mapping[str, object]) -> dict:
    result = {}
    for event, values in sorted(split["per_event"].items()):
        result[event] = {
            "positive_order_fraction": values["ranking"]["positive_order_fraction"],
            "minimum_attained_fraction": values["ranking"]["minimum_attained_fraction"],
            "foreground_recall": values["relevance_support"]["foreground_recall"],
            "background_false_positive_rate": values["relevance_support"][
                "background_false_positive_rate"
            ],
        }
    return result


def analyze(report_path: Path, protocol_path: Path, manifest_path: Path) -> dict:
    report = _read(report_path)
    protocol = _read(protocol_path)
    manifest = _read(manifest_path)
    if report.get("optimizer_steps") != 40 or len(report.get("history", [])) != 40:
        raise RuntimeError("diagnostic expects the completed 40-step MR1 report")
    if report.get("status") != "engineering_mr1_multiroute_smoke_failed_gate":
        raise RuntimeError("diagnostic is only for the failed-gate MR1 outcome")
    records_path = Path(manifest["records"]["path"]).resolve()
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    if _sha256(records_path) != manifest["records"]["sha256"]:
        raise RuntimeError("MR1 records hash differs from frozen manifest")

    history = report["history"]
    trend_fields = (
        "loss",
        "language_nll",
        "support_aligned_relevance",
        "background_support_hinge",
        "ranking_loss",
        "task_field_loss",
    )
    trend = {}
    for field in trend_fields:
        values = [float(row[field]) for row in history]
        first = _mean(values[:5])
        last = _mean(values[-5:])
        trend[field] = {
            "first_5_mean": first,
            "last_5_mean": last,
            "last_to_first_ratio": last / max(first, 1e-12),
        }

    primary = Counter(
        group for row in history for group in row["primary_group_ids"]
    )
    language = Counter(row["language_group_id"] for row in history)
    before = {
        split: _metric_snapshot(report["before"][split])
        for split in ("train", "dev")
    }
    after = {
        split: _metric_snapshot(report["after"][split])
        for split in ("train", "dev")
    }
    events = {
        split: _event_metrics(report["after"][split])
        for split in ("train", "dev")
    }
    train_bad_order = [
        event for event, values in events["train"].items()
        if values["positive_order_fraction"] < 1.0
    ]
    gates = protocol["release_gates"]
    failed_checks = sorted(key for key, value in report["checks"].items() if not value)
    optimization_not_saturated = all(
        trend[field]["last_to_first_ratio"] < 0.55
        for field in (
            "loss",
            "language_nll",
            "support_aligned_relevance",
            "ranking_loss",
            "task_field_loss",
        )
    )
    partial_heldout_transfer = (
        after["dev"]["positive_order_fraction"] > before["dev"]["positive_order_fraction"]
        and after["dev"]["task_field_accuracy"] > before["dev"]["task_field_accuracy"]
        and after["dev"]["zero_uq_complete_field_accuracy"]
        >= gates["dev_min_zero_uq_complete_field_accuracy"]
    )
    route164_concentrated = train_bad_order == ["route164_step522"]
    target_distribution = _target_distributions(records_path)

    return {
        "schema": SCHEMA,
        "status": "mr1_40step_diagnosed_no_training_launched",
        "sources": {
            "report": {"path": str(report_path), "sha256": _sha256(report_path)},
            "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
            "dataset_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            "records": {"path": str(records_path), "sha256": _sha256(records_path)},
        },
        "failed_checks": failed_checks,
        "before": before,
        "after": after,
        "after_per_event": events,
        "optimization_trend": trend,
        "presentation_coverage": {
            "primary_per_group_min": min(primary.values()),
            "primary_per_group_max": max(primary.values()),
            "language_group_steps_min": min(language.values()),
            "language_group_steps_max": max(language.values()),
            "primary_by_group": dict(sorted(primary.items())),
            "language_steps_by_group": dict(sorted(language.items())),
        },
        "target_distribution": target_distribution,
        "diagnosis": {
            "optimization_not_saturated_at_step_40": optimization_not_saturated,
            "partial_event_heldout_transfer_observed": partial_heldout_transfer,
            "train_ranking_failure_concentrated_in_route164": route164_concentrated,
            "capacity_limit_established": False,
            "loss_design_failure_established": False,
            "formal_generalization_established": False,
        },
        "next_experiment": {
            "name": "MR1-E fresh 80-step duration diagnostic",
            "justified": bool(optimization_not_saturated and partial_heldout_transfer),
            "fresh_initialization_from_original_orion": True,
            "optimizer_steps": 80,
            "must_hold_fixed": [
                "8-event 6/2 split",
                "all QA records and visual caches",
                "Stage1 adapter checkpoint",
                "architecture and loss weights",
                "learning rates and random seed",
                "release gates",
            ],
            "decision_rules": {
                "train_still_fails_after_80": "Stop increasing duration; inspect Route164 relevance objective/capacity.",
                "train_passes_but_dev_stalls": "Treat data diversity and task-field coverage as the next bottleneck.",
                "train_and_dev_pass": "Engineering paradigm passes; formal 24-event review/training remains separately locked.",
                "dev_regresses_while_train_improves": "Classify as overfitting and stop duration scaling."
            },
            "automatic_submission": False,
            "formal_training": False,
            "stage2p": False,
        },
        "claim_boundary": "Post-hoc engineering diagnosis of one bounded smoke; no formal generalization, planning, control, or safety claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite MR1 diagnostic")
    result = analyze(
        args.report.resolve(),
        args.protocol.resolve(),
        args.dataset_manifest.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "diagnosis": result["diagnosis"],
        "next_experiment": result["next_experiment"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
