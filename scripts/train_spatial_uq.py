#!/usr/bin/env python3
"""Train the standalone Stage-1 spatial UQ heads.

Examples:

    python scripts/train_spatial_uq.py --mock --smoke \
        --output /tmp/spatial_uq_smoke.pt

    python scripts/train_spatial_uq.py \
        --records paired_features.pt \
        --manifest route_disjoint_manifest.json \
        --feature-dim 1024 \
        --output checkpoints/spatial_uq/stage1.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch

# Keep this entry point runnable without installing the repository as a package.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uq_estimator.spatial_training import (  # noqa: E402
    RouteDisjointManifest,
    build_route_disjoint_manifest,
    load_paired_feature_records,
    make_mock_paired_records,
    run_stage1_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train route-disjoint spatial UQ ensemble and student"
    )
    parser.add_argument("--records", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/spatial_uq/stage1.pt"))
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--ensemble-members", type=int, default=3)
    parser.add_argument("--min-log-variance", type=float, default=-6.0)
    parser.add_argument("--max-log-variance", type=float, default=3.0)
    parser.add_argument("--teacher-epochs", type=int, default=5)
    parser.add_argument("--student-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mock:
        records = make_mock_paired_records(
            feature_dim=args.feature_dim,
            n_routes=7,
            pairs_per_route=2,
            seed=args.seed,
        )
        manifest = build_route_disjoint_manifest(records, seed=args.seed)
    else:
        if args.records is None or args.manifest is None:
            raise SystemExit("real training requires both --records and --manifest")
        records = load_paired_feature_records(args.records)
        manifest = RouteDisjointManifest.load(args.manifest)

    if args.smoke:
        args.teacher_epochs = 1
        args.student_epochs = 1
        args.batch_size = min(args.batch_size, 8)
        args.hidden_dim = min(args.hidden_dim, 32)
        args.device = "cpu"

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")

    checkpoint = run_stage1_training(
        records=records,
        manifest=manifest,
        output_path=args.output,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        ensemble_members=args.ensemble_members,
        min_log_variance=args.min_log_variance,
        max_log_variance=args.max_log_variance,
        teacher_epochs=args.teacher_epochs,
        student_epochs=args.student_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
    )
    summary = {
        "checkpoint": str(args.output.resolve()),
        "schema_version": checkpoint["schema_version"],
        "target_contract_schema_version": checkpoint["target_contract_schema_version"],
        "ensemble_members": checkpoint["model_config"]["ensemble_members"],
        "target_provenance": checkpoint["target_provenance"],
        "claim_boundary": checkpoint["claim_boundary"],
        "last_validation": (
            checkpoint["history"]["validation"][-1]
            if checkpoint["history"]["validation"]
            else None
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
