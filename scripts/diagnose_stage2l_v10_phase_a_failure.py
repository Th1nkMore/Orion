#!/usr/bin/env python3
"""Diagnose a failed Stage2-L v10 Phase-A report without post-hoc relabeling."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Mapping


SCHEMA = "orion.stage2l_v10_phase_a_diagnosis.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _event_summary(per_group: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_group.values():
        grouped[str(row["event_id"])].append(row)
    result = {}
    for event_id, rows in sorted(grouped.items()):
        result[event_id] = {
            "group_count": len(rows),
            "mean_map_loss": statistics.mean(float(row["map_loss"]) for row in rows),
            "positive_order_fraction": statistics.mean(
                float(bool(row["positive_order"])) for row in rows
            ),
            "mean_learned_gap": statistics.mean(
                float(row["learned_gap"]) for row in rows
            ),
            "mean_attained_fraction": statistics.mean(
                float(row["attained_fraction"]) for row in rows
            ),
        }
    return result


def diagnose(report: Mapping[str, Any], *, report_sha256: str) -> dict[str, Any]:
    if (
        report.get("schema") != "orion.stage2l_v10_staged_smoke.v1"
        or report.get("status") != "stopped_after_phase_a_failed_gate"
        or report.get("completed_phases") != []
    ):
        raise ValueError("expected a scientifically stopped v10 Phase-A report")
    phase = report["phases"]["A_map_pretrain"]
    train = phase["metrics"]["train"]
    dev = phase["metrics"]["dev"]
    train_support = train["relevance_support"]
    dev_support = dev["relevance_support"]
    history = phase["history"]
    train_events = _event_summary(train["per_group"])
    dev_events = _event_summary(dev["per_group"])
    weak_dev_events = [
        event_id
        for event_id, values in dev_events.items()
        if values["positive_order_fraction"] < 0.8
    ]
    stable = bool(history) and all(row.get("finite") is True for row in history)
    loss_change = float(history[-1]["loss"]) - float(history[0]["loss"])
    return {
        "schema": SCHEMA,
        "terminal_status": report["status"],
        "report_sha256": report_sha256,
        "evidence": {
            "optimization_stable": stable and loss_change < 0.0,
            "logged_loss_change_step40_minus_step1": loss_change,
            "background_suppression_healthy": (
                float(train_support["background_false_positive_rate"]) <= 0.1
                and float(dev_support["background_false_positive_rate"]) <= 0.1
            ),
            "heldout_foreground_background_gap_positive": float(
                dev_support["foreground_background_probability_gap"]
            ) > 0.0,
            "train_foreground_recall": float(train_support["foreground_recall"]),
            "dev_foreground_recall": float(dev_support["foreground_recall"]),
            "foreground_recall_generalization_gap_train_minus_dev": float(
                train_support["foreground_recall"]
                - dev_support["foreground_recall"]
            ),
            "dev_events_below_0_8_positive_order_fraction": weak_dev_events,
            "dev_event_failure_is_heterogeneous": bool(weak_dev_events)
            and len(weak_dev_events) < len(dev_events),
        },
        "per_event": {"train": train_events, "dev": dev_events},
        "diagnosis": {
            "numerical_divergence_supported": False,
            "insufficient_parameter_capacity_supported": False,
            "pure_threshold_calibration_issue_proven": False,
            "foreground_underactivation_supported": True,
            "heldout_spatial_transfer_gap_supported": True,
            "why_not_just_lower_gate": (
                "The frozen gate failed and held-out ordering is weak on a subset "
                "of events; a post-hoc lower threshold would not establish that R "
                "localized the right task-relevant support."
            ),
        },
        "next_bounded_diagnostic": {
            "kind": "evaluation_only_checkpoint_replay",
            "optimizer_steps": 0,
            "checkpoint": "phase_a.pt",
            "required_outputs": [
                "per-event threshold sweep of recall, precision and background FPR",
                "threshold-free average precision",
                "foreground/background probability quantiles",
                "R-map overlays for every held-out event",
                "support-label overlays beside R-map overlays",
            ],
            "decision_after_replay": {
                "good_ordering_but_miscalibrated": "revise calibration/objective in a new v10.1 protocol",
                "wrong_spatial_support_on_weak_events": "repair route/task conditioning or support labels before more steps",
                "uniformly_improving_but_underfit": "authorize a new longer Phase-A-only smoke; never extend v10 in place",
            },
        },
        "locks": {
            "v10_reclassified_as_pass": False,
            "automatic_step_extension": False,
            "phase_b": False,
            "phase_c": False,
            "formal_stage2l": False,
            "stage2p": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Phase-A diagnosis")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = diagnose(report, report_sha256=_sha256(args.report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "evidence": result["evidence"],
        "diagnosis": result["diagnosis"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
