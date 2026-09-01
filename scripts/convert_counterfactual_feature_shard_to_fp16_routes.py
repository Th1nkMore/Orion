#!/usr/bin/env python3
"""Convert a monolithic FP16 counterfactual shard to lossless route shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uq_estimator.counterfactual_sharded_dataset import (  # noqa: E402
    write_fp16_route_shards_from_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument(
        "--max-routes",
        type=int,
        help="Write a partial probe only; omitted means all routes.",
    )
    parser.add_argument("--target-batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if len(args.source_sha256) != 64:
        raise ValueError("--source-sha256 must contain 64 hex characters")
    payload = torch.load(args.input, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError("input feature shard must be a mapping")
    manifest = write_fp16_route_shards_from_payload(
        payload,
        args.output_dir,
        source_feature_shard_sha256=args.source_sha256,
        max_routes=args.max_routes,
        target_batch_size=args.target_batch_size,
    )
    print(json.dumps({
        "status": manifest["status"],
        "output_dir": str(args.output_dir.resolve()),
        "written_route_count": manifest["written_route_count"],
        "written_clean_count": manifest["written_clean_count"],
        "written_observed_count": manifest["written_observed_count"],
        "shard_bytes": sum(row["size_bytes"] for row in manifest["shards"]),
        "source_fp16_features_bitwise_preserved": all(
            row["source_fp16_features_bitwise_preserved"]
            for row in manifest["shards"]
        ),
        "target_computed_from_source_fp16": manifest[
            "storage_contract"
        ]["targets_computed_from_source_fp16"],
        "adapter_training_performed": False,
        "heldout_data_read": False,
        "stage_b_performed": False,
    }, indent=2, sort_keys=True))
    print("COUNTERFACTUAL_FP16_ROUTE_SHARD_OK=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
