#!/usr/bin/env python3
"""Train or resume the clean-only v3.1 adapter from a passed Teacher."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uq_estimator.observation_uq_shard import (  # noqa: E402
    examples_from_feature_shard,
    load_feature_shard,
)
from uq_estimator.observation_uq_v3 import (  # noqa: E402
    run_clean_only_adapter_training,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--adapter-epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20270826)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    shard = load_feature_shard(args.shard)
    examples = examples_from_feature_shard(shard)
    teacher_checkpoint = torch.load(args.teacher, map_location="cpu")
    checkpoint = run_clean_only_adapter_training(
        examples=examples,
        teacher_checkpoint=teacher_checkpoint,
        output_path=args.output,
        adapter_epochs=args.adapter_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume_path=args.resume,
        seed=args.seed,
        device=device,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.output.resolve()),
                "report": str(args.output.with_suffix(".report.json").resolve()),
                "schema_version": checkpoint["schema_version"],
                "checkpoint_selection": checkpoint["checkpoint_selection"],
                "data_attestation": checkpoint["data_attestation"],
                "evaluations": checkpoint["evaluations"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
