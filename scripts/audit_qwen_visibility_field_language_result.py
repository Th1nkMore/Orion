#!/usr/bin/env python3
"""Independently audit an A1b field-language alignment report."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SCHEMA = "orion.qwen-visibility-field-language-audit/v1"
CONTROLS = ("true_u", "zero_u", "spatial_shuffle")
SPLITS = ("validation", "held_out")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _accuracy(rows: list[dict]) -> float:
    return sum(bool(row["canonical_exact"]) for row in rows) / len(rows)


def summarize(evaluations: list[dict]) -> dict:
    if not evaluations:
        raise ValueError("no evaluations")
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in evaluations:
        groups[(row["control"],)].append(row)
        groups[(row["control"], row["split"])].append(row)
        groups[(row["control"], row["task_field"])].append(row)
        groups[(row["control"], row["expected_answer"], "label")].append(row)

    by_control = {control: _accuracy(groups[(control,)]) for control in CONTROLS}
    true_by_split = {split: _accuracy(groups[("true_u", split)]) for split in SPLITS}
    fields = sorted({row["task_field"] for row in evaluations})
    true_by_field = {
        field: _accuracy(groups[("true_u", field)]) for field in fields
    }
    labels = sorted({row["expected_answer"] for row in evaluations})
    true_by_label = {
        label: _accuracy(groups[("true_u", label, "label")]) for label in labels
    }
    predictions = {
        control: dict(sorted(Counter(row["parsed"] for row in groups[(control,)]).items()))
        for control in CONTROLS
    }
    semantic = {
        control: sum(bool(row["control_semantic_exact"]) for row in groups[(control,)])
        / len(groups[(control,)])
        for control in CONTROLS
    }
    return {
        "count": len(evaluations),
        "accuracy_by_control_against_true_target": by_control,
        "true_u_accuracy_by_split": true_by_split,
        "true_u_accuracy_by_field": true_by_field,
        "true_u_accuracy_by_label": true_by_label,
        "prediction_counts_by_control": predictions,
        "control_semantic_accuracy": semantic,
        "true_minus_zero_accuracy_pp": 100.0 * (by_control["true_u"] - by_control["zero_u"]),
        "true_minus_shuffle_accuracy_pp": 100.0 * (
            by_control["true_u"] - by_control["spatial_shuffle"]
        ),
    }


def validate_rows(evaluations: list[dict], curriculum: dict) -> list[str]:
    errors = []
    expected_ids = set(curriculum["evaluation_example_ids"])
    keys = Counter((row["example_id"], row["control"]) for row in evaluations)
    expected_keys = {(example_id, control) for example_id in expected_ids for control in CONTROLS}
    if set(keys) != expected_keys:
        errors.append("evaluation example/control membership differs from curriculum")
    if any(count != 1 for count in keys.values()):
        errors.append("duplicate evaluation example/control rows")
    if len(expected_ids) != 160 or len(evaluations) != 480:
        errors.append("expected 160 examples and 480 control evaluations")
    if {row["split"] for row in evaluations} != set(SPLITS):
        errors.append("evaluation split differs from validation plus held_out")
    if {row["control"] for row in evaluations} != set(CONTROLS):
        errors.append("evaluation controls differ from preregistration")
    for row in evaluations:
        parsed = row["answer"].strip() if row["answer"].strip() in {"BELOW", "AT_OR_ABOVE"} else None
        if row["parsed"] != parsed:
            errors.append(f"parsed answer mismatch: {row['example_id']} {row['control']}")
            break
        if bool(row["canonical_exact"]) != (parsed == row["expected_answer"]):
            errors.append(f"canonical exact mismatch: {row['example_id']} {row['control']}")
            break
        if bool(row["control_semantic_exact"]) != (
            parsed == row["control_expected_answer"]
        ):
            errors.append(f"control semantic exact mismatch: {row['example_id']} {row['control']}")
            break
    return errors


def audit(report: dict, curriculum: dict) -> dict:
    metrics = summarize(report["evaluations"])
    errors = validate_rows(report["evaluations"], curriculum)
    scope = report["scope"]
    if report.get("optimizer_controls"):
        errors.append("control examples entered optimizer")
    if report.get("planning_expert_in_optimizer"):
        errors.append("planning expert entered optimizer")
    for name in (
        "vision_trainable_parameter_count",
        "planning_expert_trainable_parameter_count",
    ):
        if scope.get(name) != 0:
            errors.append(f"unexpected trainable scope: {name}")
    if scope.get("embedding_trainable") or scope.get("lm_head_trainable"):
        errors.append("embedding or lm_head unexpectedly trainable")
    gates = {
        "true_u_overall_at_least_80_percent": metrics[
            "accuracy_by_control_against_true_target"
        ]["true_u"] >= 0.80,
        "true_u_each_split_at_least_75_percent": min(
            metrics["true_u_accuracy_by_split"].values()
        ) >= 0.75,
        "true_u_each_field_at_least_60_percent": min(
            metrics["true_u_accuracy_by_field"].values()
        ) >= 0.60,
        "true_minus_zero_at_least_15pp": metrics["true_minus_zero_accuracy_pp"] >= 15.0,
    }
    return {
        "metrics": metrics,
        "integrity_errors": errors,
        "gates": gates,
        "gate_pass": not errors and all(gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite audit")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    protocol_path = Path(report["protocol_path"])
    curriculum_path = Path(report["curriculum_path"])
    manifest_path = Path(report["manifest_path"])
    for path, expected, label in (
        (protocol_path, report["protocol_sha256"], "protocol"),
        (curriculum_path, report["curriculum_sha256"], "curriculum"),
        (manifest_path, report["manifest_sha256"], "manifest"),
        (Path(report["checkpoint"]["path"]), report["checkpoint"]["sha256"], "checkpoint"),
    ):
        if sha256(path) != expected:
            raise ValueError(f"{label} hash changed")
    curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "report_sha256": sha256(args.report),
        "protocol_sha256": report["protocol_sha256"],
        "curriculum_sha256": report["curriculum_sha256"],
        "manifest_sha256": report["manifest_sha256"],
        "checkpoint_sha256": report["checkpoint"]["sha256"],
        **audit(report, curriculum),
        "claim_boundary": {
            "field_language_alignment_only": True,
            "visual_planning_or_safety_claim_allowed": False,
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
