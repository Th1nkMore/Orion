"""Build coarse localization labels from Density-UQ embedding sensitivity.

The label is the front-camera block whose masking causes the largest change in
Density-UQ active embedding. It is intended as a lightweight pseudo target for
testing whether injected active embeddings can be aligned to location language.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.visualize_density_embedding_sensitivity import compute_sensitivity
from uq_estimator.density import DensityUQEstimator
from uq_estimator.risk_qa import (
    reliability_level,
    reliability_percentile,
    select_balanced_sample_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--descriptor-cache", required=True, type=Path)
    parser.add_argument("--density-checkpoint", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--balanced-per-level", type=int, required=True)
    parser.add_argument("--grid", type=int, default=10)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--mask-mode", default="zero")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def region_from_block(row: int, col: int, grid: int) -> str:
    third = grid / 3.0
    vertical = "upper" if row < third else "middle" if row < 2 * third else "lower"
    horizontal = "left" if col < third else "center" if col < 2 * third else "right"
    return f"{vertical}-{horizontal}"


def main() -> None:
    args = parse_args()
    cache = torch.load(args.descriptor_cache, map_location="cpu", weights_only=False)
    density_payload = torch.load(args.density_checkpoint, map_location="cpu", weights_only=True)
    assignment = density_payload["split_assignment"]
    estimator = DensityUQEstimator.from_checkpoint(density_payload).eval()

    selected_indices = [
        index
        for index, route in enumerate(cache["routes"])
        if assignment.get(route) == args.split
        and (args.feature_dir / cache["filenames"][index]).exists()
    ]
    if not selected_indices:
        raise RuntimeError(f"No cached feature samples found for split {args.split!r}")

    score_by_id: dict[str, float] = {}
    with torch.no_grad():
        for start in range(0, len(selected_indices), 512):
            part = selected_indices[start : start + 512]
            _, scores, _, _ = estimator.encode_descriptor(
                cache["descriptors"][part].float()
            )
            for index, score in zip(part, scores.flatten().tolist()):
                score_by_id[cache["filenames"][index]] = float(score)

    selected_ids, counts = select_balanced_sample_ids(
        score_by_id,
        args.balanced_per_level,
        args.seed,
    )

    labels: dict[str, dict[str, object]] = {}
    for offset, sample_id in enumerate(selected_ids, start=1):
        data = torch.load(args.feature_dir / sample_id, map_location="cpu", weights_only=False)
        tokens = data["tokens"].float()
        _, _, stats = compute_sensitivity(
            estimator,
            tokens,
            camera_index=args.camera_index,
            grid=args.grid,
            mask_mode=args.mask_mode,
        )
        row = int(stats["max_row"])
        col = int(stats["max_col"])
        score = score_by_id[sample_id]
        labels[sample_id] = {
            "region": region_from_block(row, col, args.grid),
            "max_row": row,
            "max_col": col,
            "max_l2": float(stats["max_l2"]),
            "mean_l2": float(stats["mean_l2"]),
            "score": score,
            "level": reliability_level(reliability_percentile(score)),
        }
        print(
            f"[{offset}/{len(selected_ids)}] {sample_id}: "
            f"{labels[sample_id]['region']} max_l2={stats['max_l2']:.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "split": args.split,
                "balanced_counts": counts,
                "grid": args.grid,
                "camera_index": args.camera_index,
                "mask_mode": args.mask_mode,
                "labels": labels,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
