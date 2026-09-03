#!/usr/bin/env python3
"""Export compact UQ score arrays from eval_openloop .pt for figure scripts.

Usage:
    python scripts/export_openloop_scores.py \\
        --input results/eval_openloop_v3.pt \\
        --output results/eval_openloop_v3_scores.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uq_estimator.roc_plot import compute_auroc, export_openloop_scores, records_to_arrays


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Export open-loop UQ scores to .npz')
    p.add_argument('--input', required=True, help='eval_openloop .pt file')
    p.add_argument(
        '--output',
        default=None,
        help='Output .npz (default: <input>_scores.npz)',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(
        in_path.stem + '_scores.npz'
    )

    data = torch.load(in_path, map_location='cpu', weights_only=False)
    records = data['records']
    scores, labels = records_to_arrays(records)
    auroc = compute_auroc(labels, scores)

    export_openloop_scores(records, out_path, source=str(in_path))
    print(f'Exported {len(scores)} scores to {out_path}')
    print(f'  Normal: {(labels == 0).sum()}, Adverse: {(labels == 1).sum()}')
    print(f'  AUROC: {auroc:.4f}')


if __name__ == '__main__':
    main()
