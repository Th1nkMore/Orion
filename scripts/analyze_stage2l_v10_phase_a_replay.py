#!/usr/bin/env python3
"""Summarize the evaluation-only Stage2-L v10 Phase-A checkpoint replay.

This report-only utility combines threshold-free metrics with a hash-bound
manual review of representative six-view overlays.  It never loads ORION,
constructs an optimizer, changes a checkpoint, or unlocks later stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "orion.stage2l_v10_phase_a_replay_analysis.v1"
REPORT_SCHEMA = "orion.stage2l_v10_phase_a_checkpoint_replay.v1"
REVIEW_SCHEMA = "orion.stage2l_v10_phase_a_replay_visual_review.v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_feasibility(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_recall: float,
    maximum_background_fpr: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("threshold sweep is empty")
    normalized = [
        {
            "threshold": float(row["threshold"]),
            "recall": float(row["recall"]),
            "precision": float(row["precision"]),
            "background_fpr": float(row["background_fpr"]),
        }
        for row in rows
    ]
    feasible = [
        row
        for row in normalized
        if row["recall"] >= minimum_recall
        and row["background_fpr"] <= maximum_background_fpr
    ]
    fpr_bounded = [
        row for row in normalized if row["background_fpr"] <= maximum_background_fpr
    ]
    recall_bounded = [row for row in normalized if row["recall"] >= minimum_recall]
    return {
        "minimum_recall": float(minimum_recall),
        "maximum_background_fpr": float(maximum_background_fpr),
        "feasible": bool(feasible),
        "feasible_rows": feasible,
        "maximum_recall_under_fpr_ceiling": max(
            (row["recall"] for row in fpr_bounded), default=0.0
        ),
        "minimum_background_fpr_at_recall_floor": min(
            (row["background_fpr"] for row in recall_bounded), default=1.0
        ),
    }


def _metric_summary(diagnostics: Mapping[str, Any]) -> dict[str, float]:
    cells = int(diagnostics["cell_count"])
    foreground = int(diagnostics["foreground_cell_count"])
    if cells <= 0 or not 0 < foreground < cells:
        raise ValueError("diagnostic prevalence is malformed")
    prevalence = foreground / cells
    average_precision = float(diagnostics["average_precision"])
    return {
        "average_precision": average_precision,
        "foreground_prevalence": prevalence,
        "average_precision_lift_over_prevalence": average_precision / prevalence,
    }


def analyze(
    report: Mapping[str, Any],
    review: Mapping[str, Any],
    *,
    report_path: Path,
    review_path: Path,
) -> dict[str, Any]:
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("status")
        != "phase_a_checkpoint_replay_complete_evaluation_only"
        or report.get("optimizer_steps") != 0
        or report.get("checkpoint_updated") is not False
        or report.get("event_count") != 17
        or report.get("group_count") != 80
    ):
        raise ValueError("expected the complete evaluation-only v10 replay")
    if (
        review.get("schema") != REVIEW_SCHEMA
        or review.get("status")
        != "frozen_manual_review_of_hash_bound_replay_maps"
    ):
        raise ValueError("visual review is absent or not frozen")

    per_group = report.get("per_group", {})
    reviewed_rows = []
    for row in review.get("rows", []):
        group_id = str(row.get("group_id", ""))
        replay_row = per_group.get(group_id)
        if not isinstance(replay_row, Mapping):
            raise ValueError("review references an unknown replay group: %s" % group_id)
        local_map = (
            report_path.parent
            / "maps"
            / str(row["event_id"])
            / (group_id + ".png")
        ).resolve()
        if (
            replay_row.get("event_id") != row.get("event_id")
            or not local_map.is_file()
            or _sha256(local_map) != row.get("map_sha256")
        ):
            raise ValueError("reviewed replay map provenance differs: %s" % group_id)
        reviewed_rows.append(
            {
                **dict(row),
                "map_path": str(local_map),
                "group_metrics": _metric_summary(replay_row["diagnostics"]),
                "frozen_target_relative_threshold": dict(
                    replay_row["diagnostics"]["frozen_target_relative_threshold"]
                ),
            }
        )

    split_metrics = {
        split: _metric_summary(report["per_split"][split])
        for split in ("train", "dev")
    }
    threshold = {
        split: threshold_feasibility(
            report["per_split"][split]["absolute_threshold_sweep"],
            minimum_recall=0.8,
            maximum_background_fpr=0.1,
        )
        for split in ("train", "dev")
    }
    event_metrics = {
        event_id: {
            "split": value["split"],
            **_metric_summary(value["diagnostics"]),
        }
        for event_id, value in sorted(report["per_event"].items())
    }
    dev_aps = [
        value["average_precision"]
        for value in event_metrics.values()
        if value["split"] == "dev"
    ]
    wrong_view = any(
        row.get("finding") == "wrong_camera_view_selection"
        for row in reviewed_rows
    )
    multi_view_collapse = any(
        row.get("finding") == "multi_view_support_collapsed_to_front"
        for row in reviewed_rows
    )
    no_feasible_threshold = not threshold["train"]["feasible"] and not threshold[
        "dev"
    ]["feasible"]
    dev_signal = (
        split_metrics["dev"]["average_precision_lift_over_prevalence"] >= 2.0
    )
    heterogeneous_dev = max(dev_aps) / max(min(dev_aps), 1e-12) >= 3.0

    return {
        "schema": SCHEMA,
        "status": "v10_phase_a_replay_diagnosed_training_remains_locked",
        "optimizer_steps": 0,
        "checkpoint_updated": False,
        "split_metrics": split_metrics,
        "threshold_feasibility": threshold,
        "event_metrics": event_metrics,
        "reviewed_representatives": reviewed_rows,
        "diagnosis": {
            "threshold_free_signal_above_chance": dev_signal,
            "dev_event_performance_heterogeneous": heterogeneous_dev,
            "pure_threshold_calibration_supported": not no_feasible_threshold,
            "view_selection_mismatch_observed": wrong_view,
            "multi_view_support_collapse_observed": multi_view_collapse,
            "systematic_view_order_bug_proven": False,
            "longer_training_alone_supported": False,
            "phase_a_interface_or_conditioning_audit_required": True,
        },
        "decision": {
            "v10_gate_revision": False,
            "continue_v10_in_place": False,
            "immediate_next_action": (
                "Audit canonical camera order and whether the 529 cached ORION "
                "det/map tokens expose an explicit camera/grid binding to the "
                "six relevance-query grids. Only after that audit may a new "
                "Phase-A-only v10.1 smoke be prepared."
            ),
            "new_phase_a_only_step_range_after_audit": [100, 300],
            "phase_b": False,
            "phase_c": False,
            "formal_stage2l": False,
            "stage2p": False,
            "closed_loop": False,
        },
        "provenance": {
            "replay_report": {
                "path": str(report_path.resolve()),
                "sha256": _sha256(report_path.resolve()),
            },
            "visual_review": {
                "path": str(review_path.resolve()),
                "sha256": _sha256(review_path.resolve()),
            },
        },
        "claim_boundary": (
            "Diagnostic analysis of a failed engineering checkpoint only; not a "
            "gate revision, formal generalization result, planning result, "
            "closed-loop result, or safety claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--visual-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite replay analysis")
    value = analyze(
        _read_json(args.report.resolve()),
        _read_json(args.visual_review.resolve()),
        report_path=args.report.resolve(),
        review_path=args.visual_review.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value["diagnosis"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
