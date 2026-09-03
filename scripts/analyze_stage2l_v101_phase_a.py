#!/usr/bin/env python3
"""Report-only analysis for the terminal v10.1 view-aligned Phase-A smoke."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


SCHEMA = "orion.stage2l_v101_phase_a_analysis.v1"
REPORT_SCHEMA = "orion.stage2l_v101_view_aligned_phase_a.v1"
BASELINE_SCHEMA = "orion.stage2l_v10_phase_a_replay_analysis.v1"
CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


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


def group_view_diagnostics(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    support_fraction: float,
) -> dict[str, Any]:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if probability.shape != (1, 6, 10, 10) or target.shape != probability.shape:
        raise ValueError("group view diagnostic shape differs")
    if not np.isfinite(probability).all() or not np.isfinite(target).all():
        raise ValueError("group view diagnostic contains non-finite values")
    peak = float(target.max())
    if peak <= 0.0:
        raise ValueError("group target has no positive support")
    support = target[0] >= peak * float(support_fraction)
    support_counts = support.sum(axis=(1, 2))
    target_mass = target[0].sum(axis=(1, 2))
    predicted_mass = probability[0].sum(axis=(1, 2))
    supported = tuple(
        CAMERA_ORDER[index] for index in np.flatnonzero(support_counts > 0)
    )
    target_top = int(np.argmax(target_mass))
    prediction_top = int(np.argmax(predicted_mass))
    return {
        "target_supported_views": list(supported),
        "target_supported_view_count": len(supported),
        "target_dominant_view": CAMERA_ORDER[target_top],
        "prediction_dominant_view": CAMERA_ORDER[prediction_top],
        "prediction_top1_hits_any_target_supported_view": (
            CAMERA_ORDER[prediction_top] in supported
        ),
        "prediction_top1_matches_target_dominant_view": prediction_top == target_top,
        "target_view_mass": {
            CAMERA_ORDER[index]: float(value)
            for index, value in enumerate(target_mass)
        },
        "prediction_view_mass": {
            CAMERA_ORDER[index]: float(value)
            for index, value in enumerate(predicted_mass)
        },
    }


def _aggregate_view_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("view diagnostic aggregate is empty")
    multi = [row for row in rows if row["target_supported_view_count"] > 1]
    target_counts = Counter(row["target_dominant_view"] for row in rows)
    predicted_counts = Counter(row["prediction_dominant_view"] for row in rows)
    return {
        "group_count": len(rows),
        "top1_hits_any_target_supported_view_fraction": float(
            np.mean(
                [row["prediction_top1_hits_any_target_supported_view"] for row in rows]
            )
        ),
        "top1_matches_target_dominant_view_fraction": float(
            np.mean([row["prediction_top1_matches_target_dominant_view"] for row in rows])
        ),
        "multi_view_target_group_count": len(multi),
        "multi_view_top1_hit_fraction": (
            float(
                np.mean(
                    [
                        row["prediction_top1_hits_any_target_supported_view"]
                        for row in multi
                    ]
                )
            )
            if multi
            else None
        ),
        "target_dominant_view_counts": dict(sorted(target_counts.items())),
        "prediction_dominant_view_counts": dict(sorted(predicted_counts.items())),
    }


def analyze(
    *,
    report: Mapping[str, Any],
    baseline: Mapping[str, Any],
    maps: Mapping[str, Any],
    report_path: Path,
    baseline_path: Path,
    maps_path: Path,
    support_fraction: float,
) -> dict[str, Any]:
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("status")
        not in (
            "phase_a_gate_passed_engineering_only",
            "phase_a_stopped_without_gate_pass",
        )
        or report.get("phase_a_only") is not True
        or report.get("formal_stage2l_ready") is not False
        or report.get("stage2p_ready") is not False
        or report.get("locks", {}).get("stage1_uq_loaded") is not False
    ):
        raise ValueError("v10.1 Phase-A report is absent or not terminal")
    if baseline.get("schema") != BASELINE_SCHEMA:
        raise ValueError("v10 replay baseline analysis differs")
    if not maps or len(maps) != 80:
        raise ValueError("v10.1 final spatial maps must contain 80 groups")
    final_step = int(report["optimizer_steps"])
    if maps_path.stem != "spatial_maps_step%03d" % final_step:
        raise ValueError("v10.1 spatial map checkpoint step differs")

    event_split = {}
    for split in ("train", "dev"):
        for event_id in report["final_metrics"][split]["per_event"]:
            if event_id in event_split:
                raise ValueError("event appears in both train and dev")
            event_split[event_id] = split
    per_group = {}
    by_split = defaultdict(list)
    by_event = defaultdict(list)
    for group_id, value in sorted(maps.items()):
        event_id = str(group_id).rsplit("_saved_", 1)[0]
        if event_id not in event_split:
            raise ValueError("spatial map event is absent from final metrics")
        row = group_view_diagnostics(
            value["probability"].numpy(),
            value["target"].numpy(),
            support_fraction=support_fraction,
        )
        row.update({"event_id": event_id, "split": event_split[event_id]})
        per_group[str(group_id)] = row
        by_split[event_split[event_id]].append(row)
        by_event[event_id].append(row)

    split_view = {split: _aggregate_view_rows(rows) for split, rows in by_split.items()}
    event_view = {
        event_id: {"split": event_split[event_id], **_aggregate_view_rows(rows)}
        for event_id, rows in sorted(by_event.items())
    }
    v10_train = baseline["split_metrics"]["train"]
    v10_dev = baseline["split_metrics"]["dev"]
    final_train = report["final_metrics"]["train"]
    final_dev = report["final_metrics"]["dev"]
    train_delta = float(final_train["average_precision"] - v10_train["average_precision"])
    dev_delta = float(final_dev["average_precision"] - v10_dev["average_precision"])
    gate_passed = all(bool(value) for value in report["final_checks"].values())
    milestones = [
        {
            "optimizer_step": int(row["optimizer_step"]),
            "train_average_precision": float(row["metrics"]["train"]["average_precision"]),
            "dev_average_precision": float(row["metrics"]["dev"]["average_precision"]),
            "passed": bool(row["passed"]),
        }
        for row in report["evaluations"]
    ]
    if gate_passed:
        next_action = (
            "Freeze the v10.1 Phase-A result and propose one bounded Phase-B "
            "risk-alignment smoke; do not automatically launch it."
        )
    elif dev_delta > 0.0:
        next_action = (
            "Keep later stages locked. Inspect per-event/view residuals and decide "
            "whether one targeted R-interface repair is justified."
        )
    else:
        next_action = (
            "Stop the current R-interface direction; additional epochs alone are "
            "not justified."
        )
    return {
        "schema": SCHEMA,
        "status": "terminal_v101_phase_a_analyzed_later_stages_locked",
        "optimizer_steps": final_step,
        "stop_reason": report["stop_reason"],
        "gate_passed": gate_passed,
        "average_precision_comparison": {
            "v10_train": float(v10_train["average_precision"]),
            "v101_train": float(final_train["average_precision"]),
            "train_delta": train_delta,
            "v10_dev": float(v10_dev["average_precision"]),
            "v101_dev": float(final_dev["average_precision"]),
            "dev_delta": dev_delta,
        },
        "milestones": milestones,
        "view_binding_metrics": {
            "per_split": split_view,
            "per_event": event_view,
            "per_group": per_group,
        },
        "decision": {
            "next_action": next_action,
            "phase_b_automatically_authorized": False,
            "phase_c": False,
            "formal_stage2l": False,
            "stage2p": False,
            "closed_loop": False,
            "route203_native_glare_submission": False,
        },
        "provenance": {
            "v101_report": {
                "path": str(report_path.resolve()),
                "sha256": _sha256(report_path.resolve()),
            },
            "v10_replay_analysis": {
                "path": str(baseline_path.resolve()),
                "sha256": _sha256(baseline_path.resolve()),
            },
            "final_spatial_maps": {
                "path": str(maps_path.resolve()),
                "sha256": _sha256(maps_path.resolve()),
            },
        },
        "claim_boundary": (
            "Engineering Phase-A diagnosis only; not formal generalization, "
            "planning, closed-loop, or safety evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--v10-replay-analysis", type=Path, required=True)
    parser.add_argument("--spatial-maps", type=Path, required=True)
    parser.add_argument("--support-fraction", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite v10.1 Phase-A analysis")
    value = analyze(
        report=_read_json(args.report.resolve()),
        baseline=_read_json(args.v10_replay_analysis.resolve()),
        maps=torch.load(args.spatial_maps.resolve(), map_location="cpu"),
        report_path=args.report.resolve(),
        baseline_path=args.v10_replay_analysis.resolve(),
        maps_path=args.spatial_maps.resolve(),
        support_fraction=args.support_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": value["status"],
                "gate_passed": value["gate_passed"],
                "average_precision_comparison": value[
                    "average_precision_comparison"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
