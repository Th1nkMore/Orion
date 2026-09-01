#!/usr/bin/env python3
"""Compare frozen 40- and 80-step MR1 runs using preregistered stop rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

try:
    from scripts.analyze_stage2l_mr1_report import _event_metrics, _metric_snapshot
except ModuleNotFoundError:
    from analyze_stage2l_mr1_report import _event_metrics, _metric_snapshot


SCHEMA = "orion.stage2l_mr1_duration_comparison.v1"
TREND_FIELDS = (
    "loss",
    "language_nll",
    "support_aligned_relevance",
    "background_support_hinge",
    "ranking_loss",
    "task_field_loss",
)
STABLE_INPUT_KEYS = (
    "aggregate_audit_sha256",
    "base_orion_checkpoint_sha256",
    "dataset_manifest_sha256",
    "dev_audit_sha256",
    "orion_config_sha256",
    "records_sha256",
    "reference_audit_sha256",
    "train_audit_sha256",
    "visual_cache_sha256_by_event",
)


def _load(path: Path) -> dict:
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


def _gate_summary(checks: Mapping[str, Any], prefix: str) -> dict:
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


def _metric_delta(before: Mapping[str, float], after: Mapping[str, float]) -> dict:
    result = {}
    for key in sorted(set(before) & set(after)):
        old = float(before[key])
        new = float(after[key])
        lower_is_better = key == "background_false_positive_rate"
        signed_improvement = old - new if lower_is_better else new - old
        result[key] = {
            "step40": old,
            "step80": new,
            "raw_delta": new - old,
            "signed_improvement": signed_improvement,
            "direction": (
                "improved"
                if signed_improvement > 1e-9
                else "regressed"
                if signed_improvement < -1e-9
                else "unchanged"
            ),
        }
    return result


def _history_prefix_max_abs_delta(
    history40: Sequence[Mapping[str, Any]],
    history80: Sequence[Mapping[str, Any]],
) -> dict:
    result = {}
    for field in TREND_FIELDS:
        result[field] = max(
            abs(float(left[field]) - float(right[field]))
            for left, right in zip(history40, history80[:40])
        )
    return result


def _tail_trend(history: Sequence[Mapping[str, Any]]) -> dict:
    result = {}
    for field in TREND_FIELDS:
        middle = _mean([float(row[field]) for row in history[35:40]])
        final = _mean([float(row[field]) for row in history[-5:]])
        result[field] = {
            "steps_36_40_mean": middle,
            "steps_76_80_mean": final,
            "step80_tail_to_step40_tail_ratio": final / max(middle, 1e-12),
        }
    return result


def compare_duration(report40_path: Path, report80_path: Path) -> dict:
    report40 = _load(report40_path)
    report80 = _load(report80_path)
    if int(report40.get("optimizer_steps", -1)) != 40 or len(
        report40.get("history", [])
    ) != 40:
        raise ValueError("first report is not a completed 40-step MR1 run")
    if int(report80.get("optimizer_steps", -1)) != 80 or len(
        report80.get("history", [])
    ) != 80:
        raise ValueError("second report is not a completed 80-step MR1-E run")
    if report40.get("schema") != report80.get("schema"):
        raise ValueError("MR1 report schemas differ")

    inputs40 = report40.get("provenance", {}).get("validated_inputs", {})
    inputs80 = report80.get("provenance", {}).get("validated_inputs", {})
    stable_input_matches = {
        key: inputs40.get(key) == inputs80.get(key) and inputs40.get(key) is not None
        for key in STABLE_INPUT_KEYS
    }
    architecture_matches = report40.get("architecture") == report80.get(
        "architecture"
    )
    event_split_matches = (
        report40.get("train_events") == report80.get("train_events")
        and report40.get("dev_events") == report80.get("dev_events")
    )
    before_matches = report40.get("before") == report80.get("before")
    prefix_delta = _history_prefix_max_abs_delta(
        report40["history"], report80["history"]
    )
    prefix_replay_matches = max(prefix_delta.values()) <= 1e-6
    duration_only_comparison_valid = bool(
        all(stable_input_matches.values())
        and architecture_matches
        and event_split_matches
        and before_matches
        and prefix_replay_matches
    )

    gates40 = {
        split: _gate_summary(report40["checks"], split)
        for split in ("train", "dev")
    }
    gates80 = {
        split: _gate_summary(report80["checks"], split)
        for split in ("train", "dev")
    }
    train_improves = gates80["train"]["passed"] > gates40["train"]["passed"]
    dev_gate_regresses = gates80["dev"]["passed"] < gates40["dev"]["passed"]
    gate_count_overfit = bool(train_improves and dev_gate_regresses)
    global_failed80 = sorted(
        key
        for key, value in report80["checks"].items()
        if not value and not key.startswith("train_") and not key.startswith("dev_")
    )

    if not duration_only_comparison_valid:
        decision = "invalid_duration_only_comparison"
        next_action = (
            "Do not interpret the 40-to-80 change; audit input, architecture, "
            "baseline or deterministic-prefix drift."
        )
    elif gate_count_overfit:
        decision = "duration_overfit_stop"
        next_action = (
            "Stop duration scaling; retain the better development checkpoint only "
            "for diagnosis and increase event/class coverage before another run."
        )
    elif not gates80["train"]["all_passed"]:
        decision = "train_still_fails_stop_duration_scaling"
        next_action = (
            "Stop duration scaling and inspect the remaining per-event relevance "
            "objective, especially Route164, before changing capacity or data."
        )
    elif not gates80["dev"]["all_passed"]:
        decision = "train_passes_dev_stalls_data_coverage_bottleneck"
        next_action = (
            "Treat event, camera, region and task-field diversity as the bottleneck; "
            "do not add more steps on the same eight events."
        )
    elif global_failed80:
        decision = "global_integrity_or_render_gate_failed"
        next_action = (
            "Repair the remaining non-split integrity or deterministic-render gate "
            "without extending training duration."
        )
    else:
        decision = "engineering_multievent_paradigm_passes"
        next_action = (
            "Proceed with formal data completion and protocol freeze; formal Stage2-L "
            "still requires a separate immutable launch authorization."
        )

    metric_changes = {
        split: _metric_delta(
            _metric_snapshot(report40["after"][split]),
            _metric_snapshot(report80["after"][split]),
        )
        for split in ("train", "dev")
    }
    event_changes = {}
    for split in ("train", "dev"):
        event40 = _event_metrics(report40["after"][split])
        event80 = _event_metrics(report80["after"][split])
        if set(event40) != set(event80):
            raise ValueError("per-event metric identities differ")
        event_changes[split] = {
            event: _metric_delta(event40[event], event80[event])
            for event in sorted(event40)
        }

    return {
        "schema": SCHEMA,
        "status": "mr1_duration_compared_no_training_launched",
        "sources": {
            "step40_report": {
                "path": str(report40_path),
                "sha256": _sha256(report40_path),
            },
            "step80_report": {
                "path": str(report80_path),
                "sha256": _sha256(report80_path),
            },
        },
        "controlled_comparison": {
            "valid": duration_only_comparison_valid,
            "stable_input_matches": stable_input_matches,
            "architecture_matches": architecture_matches,
            "event_split_matches": event_split_matches,
            "before_metrics_match": before_matches,
            "first_40_step_replay_matches_within_1e_6": prefix_replay_matches,
            "first_40_step_max_abs_delta": prefix_delta,
        },
        "gate_summary": {
            "step40": gates40,
            "step80": gates80,
            "step80_global_failed": global_failed80,
        },
        "overfit_diagnostic": {
            "train_passed_gate_count_increased": train_improves,
            "dev_passed_gate_count_decreased": dev_gate_regresses,
            "gate_count_overfit_established": gate_count_overfit,
        },
        "metric_changes": metric_changes,
        "per_event_changes": event_changes,
        "optimization_tail": _tail_trend(report80["history"]),
        "decision": decision,
        "next_action": next_action,
        "locks": {
            "automatic_step_extension_allowed": False,
            "automatic_retry_allowed": False,
            "formal_stage2l_allowed": False,
            "stage2p_allowed": False,
        },
        "claim_boundary": (
            "Controlled engineering duration comparison on the same eight events. "
            "It provides no formal generalization, planning, closed-loop, or safety claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-40", type=Path, required=True)
    parser.add_argument("--report-80", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite MR1 duration comparison")
    result = compare_duration(args.report_40.resolve(), args.report_80.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "controlled_comparison_valid": result["controlled_comparison"][
                    "valid"
                ],
                "decision": result["decision"],
                "next_action": result["next_action"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
