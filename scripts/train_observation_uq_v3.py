#!/usr/bin/env python3
"""Train the clean-conditional observation-UQ v3 prototype.

Mock convergence:

    python scripts/train_observation_uq_v3.py --mock \
        --output results/observation_uq_v3/mock.pt

Existing real paired cache (old targets are intentionally ignored):

    python scripts/train_observation_uq_v3.py \
        --records paired.pt --manifest route_manifest.json \
        --patch-height 40 --patch-width 40 \
        --train-family local_blur --train-family local_dark \
        --heldout-family local_glare --output observation_uq_v3.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uq_estimator.observation_uq_v3 import (  # noqa: E402
    examples_from_paired_records,
    make_mock_examples,
    route_splits_from_manifest,
    run_observation_uq_training,
    validate_family_protocol,
)
from uq_estimator.spatial_training import load_paired_feature_records  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-height", type=int, default=40)
    parser.add_argument("--patch-width", type=int, default=40)
    parser.add_argument("--train-family", action="append", dest="train_families")
    parser.add_argument("--heldout-family", action="append", dest="heldout_families")
    parser.add_argument("--feature-dim", type=int)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--teacher-members", type=int, default=2)
    parser.add_argument("--teacher-epochs", type=int, default=10)
    parser.add_argument("--adapter-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--disagreement-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--mock-routes", type=int, default=12)
    parser.add_argument("--mock-frames-per-route", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _load_examples(args: argparse.Namespace):
    if args.mock:
        feature_dim = args.feature_dim or 16
        args.patch_height = 8
        args.patch_width = 8
        examples = make_mock_examples(
            feature_dim=feature_dim,
            routes=args.mock_routes,
            frames_per_route=args.mock_frames_per_route,
            height=args.patch_height,
            width=args.patch_width,
            seed=args.seed,
        )
        return examples, feature_dim
    if args.records is None or args.manifest is None:
        raise SystemExit("real training requires --records and --manifest")
    records = load_paired_feature_records(args.records)
    manifest_payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    examples = examples_from_paired_records(
        records,
        route_splits_from_manifest(manifest_payload),
        args.patch_height,
        args.patch_width,
    )
    feature_dim = args.feature_dim or int(examples[0].current.shape[-1])
    return examples, feature_dim


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.train_families = args.train_families or ["local_blur", "local_dark"]
    args.heldout_families = args.heldout_families or ["local_glare"]
    validate_family_protocol(args.train_families, args.heldout_families)
    if args.patch_height <= 0 or args.patch_width <= 0:
        raise SystemExit("patch dimensions must be positive")
    if args.output.exists():
        raise SystemExit("output already exists; refusing to overwrite: %s" % args.output)
    if args.smoke:
        args.teacher_epochs = min(args.teacher_epochs, 2)
        args.adapter_epochs = min(args.adapter_epochs, 2)
        args.teacher_members = min(args.teacher_members, 1)
        args.hidden_dim = min(args.hidden_dim, 24)
        args.batch_size = min(args.batch_size, 8)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    examples, feature_dim = _load_examples(args)
    checkpoint = run_observation_uq_training(
        examples=examples,
        train_families=args.train_families,
        heldout_families=args.heldout_families,
        output_path=args.output,
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        teacher_members=args.teacher_members,
        teacher_epochs=args.teacher_epochs,
        adapter_epochs=args.adapter_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        disagreement_weight=args.disagreement_weight,
        seed=args.seed,
        device=device,
    )
    output = {
        "checkpoint": str(args.output.resolve()),
        "report": str(args.output.with_suffix(".report.json").resolve()),
        "schema_version": checkpoint["schema_version"],
        "claim_boundary": checkpoint["claim_boundary"],
        "data_attestation": checkpoint["data_attestation"],
        "teacher_first_loss": checkpoint["history"]["teacher_train"][0]["loss"],
        "teacher_last_loss": checkpoint["history"]["teacher_train"][-1]["loss"],
        "adapter_first_loss": checkpoint["history"]["adapter_train"][0]["loss"],
        "adapter_last_loss": checkpoint["history"]["adapter_train"][-1]["loss"],
        "evaluations": checkpoint["evaluations"],
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
