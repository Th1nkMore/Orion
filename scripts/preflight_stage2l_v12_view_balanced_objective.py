#!/usr/bin/env python3
"""CPU preflight for the Stage2-L v12 view-balanced R-only objective."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uq_estimator.stage2l_view_balanced_objective_v12 import (
    SCHEMA as OBJECTIVE_SCHEMA,
    view_balanced_relevance_terms_v12,
    view_balanced_weight_summary,
)
from uq_estimator.task_relevance_geometry import CAMERA_ORDER


SCHEMA = "orion.stage2l_v12_view_balanced_objective_preflight.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return float(np.mean(values)) if values else None


def summarize_weight_redistribution(
    targets: torch.Tensor,
    *,
    support_fraction: float,
) -> Dict[str, Any]:
    summary = view_balanced_weight_summary(
        targets, support_fraction_of_peak=support_fraction
    )
    current = summary["current_per_group_view_mass"].cpu().numpy()
    proposed = summary["proposed_per_group_view_mass"].cpu().numpy()
    active = summary["active_foreground_views"].cpu().numpy()
    per_view = {}
    for index, view in enumerate(CAMERA_ORDER):
        active_rows = active[:, index]
        per_view[view] = {
            "positive_group_count": int(active_rows.sum()),
            "current_mean_foreground_weight_share": float(current[:, index].mean()),
            "proposed_mean_foreground_weight_share": float(proposed[:, index].mean()),
            "proposed_mean_share_when_active": (
                float(proposed[active_rows, index].mean())
                if bool(active_rows.any())
                else None
            ),
        }
    return {
        "per_view": per_view,
        "foreground_weight_sum_minimum": float(
            summary["foreground_weight_sum"].min().item()
        ),
        "foreground_weight_sum_maximum": float(
            summary["foreground_weight_sum"].max().item()
        ),
        "background_weight_sum_minimum": float(
            summary["background_weight_sum"].min().item()
        ),
        "background_weight_sum_maximum": float(
            summary["background_weight_sum"].max().item()
        ),
        "mean_active_foreground_views": float(active.sum(axis=1).mean()),
    }


def preflight(
    *,
    spatial_maps_path: Path,
    r_binding_audit_path: Path,
    support_fraction: float = 0.1,
) -> Dict[str, Any]:
    spatial_maps_path = spatial_maps_path.resolve()
    r_binding_audit_path = r_binding_audit_path.resolve()
    audit = _read_json(r_binding_audit_path)
    if (
        audit.get("schema") != "orion.stage2l_v11_1_r_binding_audit.v1"
        or audit.get("passed") is not True
        or audit.get("inputs", {}).get("spatial_maps", {}).get("sha256")
        != _sha256(spatial_maps_path)
    ):
        raise ValueError("R-binding audit or spatial-map lineage differs")
    maps = torch.load(spatial_maps_path, map_location="cpu")
    train_group_ids = sorted(
        group_id
        for group_id, row in audit["per_group"].items()
        if row["split"] == "train"
    )
    if len(train_group_ids) != 60 or set(train_group_ids) - set(maps):
        raise ValueError("v12 preflight requires the frozen 60 train groups")
    targets = torch.cat(
        [maps[group_id]["target"].detach().float().cpu() for group_id in train_group_ids],
        dim=0,
    )
    if tuple(targets.shape) != (60, 6, 10, 10):
        raise ValueError("v12 preflight target stack differs")
    redistribution = summarize_weight_redistribution(
        targets, support_fraction=support_fraction
    )
    signature = inspect.signature(view_balanced_relevance_terms_v12)
    allowed_parameters = {
        "relevance_logits",
        "relevance_target",
        "support_fraction_of_peak",
        "calibration_bce_weight",
        "support_hinge_weight",
    }
    if set(signature.parameters) != allowed_parameters:
        raise ValueError("v12 objective interface expands beyond R logits/target")
    exact_weight_sums = (
        abs(redistribution["foreground_weight_sum_minimum"] - 1.0) <= 1e-6
        and abs(redistribution["foreground_weight_sum_maximum"] - 1.0) <= 1e-6
        and abs(redistribution["background_weight_sum_minimum"] - 1.0) <= 1e-6
        and abs(redistribution["background_weight_sum_maximum"] - 1.0) <= 1e-6
    )
    per_view = redistribution["per_view"]
    missing_views = [
        view for view in CAMERA_ORDER if per_view[view]["positive_group_count"] == 0
    ]
    low_event_views = [
        view
        for view in CAMERA_ORDER
        if int(audit["per_view"][view]["train_positive_event_count"]) < 3
    ]
    return {
        "schema": SCHEMA,
        "status": "v12_view_balanced_objective_cpu_preflight_pass_coverage_locked",
        "passed": exact_weight_sums,
        "gpu_used": False,
        "orion_forward_run": False,
        "training_started": False,
        "objective_schema": OBJECTIVE_SCHEMA,
        "objective_interface_parameters": sorted(signature.parameters),
        "objective_reads_only_contextual_r_logits_and_soft_r_target": True,
        "forbidden_inputs_used": {
            "observation_uq": False,
            "qa_answer": False,
            "ttc_or_outcome": False,
            "corruption_label": False,
            "trajectory_or_control": False,
        },
        "inputs": {
            "spatial_maps": {
                "path": str(spatial_maps_path),
                "sha256": _sha256(spatial_maps_path),
            },
            "r_binding_audit": {
                "path": str(r_binding_audit_path),
                "sha256": _sha256(r_binding_audit_path),
            },
        },
        "train_group_count": len(train_group_ids),
        "support_fraction_of_peak": float(support_fraction),
        "weight_redistribution": redistribution,
        "coverage_gate": {
            "formal_nonzero_all_views_passed": not missing_views,
            "views_with_zero_positive_groups": missing_views,
            "descriptive_views_with_fewer_than_three_independent_events": low_event_views,
            "formal_coverage_ready": False,
            "reason": "Objective integrity can pass independently, but current accepted train coverage remains zero for CAM_BACK_RIGHT and below three events for CAM_FRONT_RIGHT.",
        },
        "launch_locks": {
            "automatic_gpu_submission": False,
            "language_bridge_training": False,
            "formal_stage2l": False,
            "stage2p": False,
            "learned_u_closed_loop": False,
        },
        "claim_boundary": "CPU objective/weighting preflight only. It does not show that retraining improves held-out R, learned language semantics, planning, or safety.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-maps", type=Path, required=True)
    parser.add_argument("--r-binding-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-fraction", type=float, default=0.1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite v12 objective preflight")
    value = preflight(
        spatial_maps_path=args.spatial_maps,
        r_binding_audit_path=args.r_binding_audit,
        support_fraction=args.support_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": value["status"], "passed": value["passed"], "output": str(args.output.resolve())},
            sort_keys=True,
        )
    )
    return 0 if value["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
