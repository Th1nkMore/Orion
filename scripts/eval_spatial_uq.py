#!/usr/bin/env python3
"""Evaluate and calibrate a Stage-1 spatial UQ student on route-held-out data.

Real evaluation:

    python scripts/eval_spatial_uq.py \
        --checkpoint checkpoints/spatial_uq/stage1.pt \
        --records paired_features.pt \
        --manifest route_disjoint_manifest.json \
        --report results/spatial_uq_stage1/evaluation.json

Dependency-free CPU smoke:

    python scripts/eval_spatial_uq.py --mock --report /tmp/spatial_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uq_estimator.spatial_evaluation import (  # noqa: E402
    evaluate_stage1_checkpoint,
    save_evaluation_report,
)
from uq_estimator.spatial_training import (  # noqa: E402
    RouteDisjointManifest,
    load_paired_feature_records,
    make_mock_paired_records,
    run_stage1_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit monotonic calibration on the calibration routes and report "
            "validation/calibration/held-out Stage-1 spatial UQ metrics"
        )
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/spatial_uq_stage1/evaluation.json"),
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--feature-dim", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=12)
    parser.add_argument(
        "--failure-event-threshold",
        type=float,
        default=0.5,
        help="Binarize the explicit failure_event_target for discrimination metrics.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _mock_manifest() -> RouteDisjointManifest:
    # Both calibration and held-out contain one even and one odd route so the
    # actual-failure and proxy calibrators remain separate but executable.
    return RouteDisjointManifest(
        splits={
            "train": ("route_000", "route_001", "route_002", "route_003"),
            "validation": ("route_004", "route_005"),
            "calibration": ("route_006", "route_007"),
            "held_out": ("route_008", "route_009"),
        },
        seed=0,
    )


def _mock_records(feature_dim: int, seed: int):
    """Make a separable mock with both target provenances in calibration."""
    records = make_mock_paired_records(
        feature_dim=feature_dim,
        n_routes=10,
        pairs_per_route=2,
        seed=seed,
    )
    revised = []
    for record in records:
        mask = record.corruption_mask.bool()
        observed = record.clean_patch_features.clone()
        if record.severity > 0:
            # Antipodal features yield an unambiguous cosine-error proxy in
            # the masked cells while leaving the other cells exactly clean.
            observed = torch.where(
                mask.unsqueeze(-1), -record.clean_patch_features, observed
            )
        route_number = int(record.route_id.rsplit("_", 1)[1])
        actual_severity = None
        actual_event = None
        actual_valid = None
        clean_severity = None
        clean_event = None
        clean_valid = None
        if route_number % 2 == 0:
            actual_severity = (
                mask.float()
                if record.severity > 0
                else torch.full_like(mask.float(), 0.05)
            )
            actual_event = (actual_severity >= 0.5).float()
            actual_valid = torch.ones_like(mask, dtype=torch.bool)
            clean_severity = torch.full_like(mask.float(), 0.05)
            clean_event = torch.zeros_like(mask.float())
            clean_valid = torch.ones_like(mask, dtype=torch.bool)
        revised.append(
            replace(
                record,
                observed_patch_features=observed,
                error_severity_target=actual_severity,
                failure_event_target=actual_event,
                target_valid_mask=actual_valid,
                clean_error_severity_target=clean_severity,
                clean_failure_event_target=clean_event,
                clean_target_valid_mask=clean_valid,
            )
        )
    return revised


def _run(
    args: argparse.Namespace,
    checkpoint: Path,
    records,
    manifest: RouteDisjointManifest,
):
    report = evaluate_stage1_checkpoint(
        checkpoint_path=checkpoint,
        records=records,
        supplied_manifest=manifest,
        failure_event_threshold=args.failure_event_threshold,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        device="cpu",
    )
    save_evaluation_report(report, args.report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mock:
        if args.checkpoint is not None or args.records is not None or args.manifest is not None:
            raise SystemExit("--mock cannot be combined with checkpoint/records/manifest")
        records = _mock_records(args.feature_dim, args.seed)
        manifest = _mock_manifest()
        with tempfile.TemporaryDirectory(prefix="orion-spatial-uq-eval-") as directory:
            checkpoint = Path(directory) / "mock_stage1.pt"
            run_stage1_training(
                records=records,
                manifest=manifest,
                output_path=checkpoint,
                feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                ensemble_members=3,
                teacher_epochs=1,
                student_epochs=1,
                batch_size=12,
                seed=args.seed,
                device="cpu",
            )
            report = _run(args, checkpoint, records, manifest)
    else:
        if args.checkpoint is None or args.records is None:
            raise SystemExit("real evaluation requires --checkpoint and --records")
        records = load_paired_feature_records(args.records)
        manifest = (
            RouteDisjointManifest.load(args.manifest)
            if args.manifest is not None
            else None
        )
        report = evaluate_stage1_checkpoint(
            checkpoint_path=args.checkpoint,
            records=records,
            supplied_manifest=manifest,
            failure_event_threshold=args.failure_event_threshold,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
            device="cpu",
        )
        save_evaluation_report(report, args.report)

    summary = {
        "report": str(args.report.resolve()),
        "schema_version": report["schema_version"],
        "evaluated_splits": report["evaluated_splits"],
        "train_split_evaluated": report["train_split_evaluated"],
        "held_out_used_for_calibration": report["calibration"][
            "held_out_used_for_fitting"
        ],
        "pooled_cross_provenance_metrics": "prohibited",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
