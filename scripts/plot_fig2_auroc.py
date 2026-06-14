#!/usr/bin/env python3
"""Regenerate fig2_auroc.pdf from eval_openloop scores.

Priority:
  1. --scores / --input explicit paths
  2. results/eval_openloop_v3_scores.npz
  3. results/eval_openloop_v3.pt
  4. --fallback-v3-json (approximate; use only when .pt is unavailable)

Usage:
    python scripts/plot_fig2_auroc.py
    python scripts/plot_fig2_auroc.py --input results/eval_openloop_v3.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uq_estimator.roc_plot import (
    calibrate_scores_from_v3_summary,
    export_openloop_scores,
    load_openloop_scores,
    plot_auroc_curve,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Plot ROC figure for presentation')
    p.add_argument('--scores', default=None, help='Scores .npz or eval .pt')
    p.add_argument('--input', default=None, help='Alias for --scores')
    p.add_argument(
        '--out',
        default='results/figures/baseline/fig2_auroc.pdf',
        help='Output figure path',
    )
    p.add_argument(
        '--fallback-v3-json',
        default='results/eval_openloop_v3.json',
        help='Approximate scores from v3 summary when .pt/.npz missing',
    )
    p.add_argument(
        '--save-scores',
        default=None,
        help='Optional path to save resolved scores as .npz',
    )
    return p.parse_args()


def resolve_scores(args: argparse.Namespace) -> tuple:
    root = Path(__file__).resolve().parents[1]
    candidate = args.scores or args.input
    if candidate:
        scores, labels = load_openloop_scores(root / candidate if not Path(candidate).is_absolute() else candidate)
        return scores, labels, f'file:{candidate}'

    for rel in ('results/eval_openloop_v3_scores.npz', 'results/eval_openloop_v3.pt'):
        path = root / rel
        if path.is_file():
            scores, labels = load_openloop_scores(path)
            return scores, labels, f'file:{rel}'

    json_path = root / args.fallback_v3_json
    if json_path.is_file():
        print(
            f'WARNING: using calibrated scores from {json_path.name}; '
            'copy eval_openloop_v3.pt from server and re-run for exact curve.'
        )
        scores, labels, auroc = calibrate_scores_from_v3_summary(json_path)
        return scores, labels, f'calibrated:{json_path.name} (AUROC≈{auroc:.4f})'

    raise FileNotFoundError(
        'No eval_openloop_v3.pt/.npz found. '
        'Copy results/eval_openloop_v3.pt from server, then:\n'
        '  python scripts/export_openloop_scores.py --input results/eval_openloop_v3.pt\n'
        '  python scripts/plot_fig2_auroc.py --scores results/eval_openloop_v3_scores.npz'
    )


def main() -> None:
    args = parse_args()
    scores, labels, source = resolve_scores(args)
    auroc = plot_auroc_curve(scores, labels, args.out)
    print(f'Saved: {args.out}  (AUROC={auroc:.4f}, source={source})')

    if args.save_scores:
        import numpy as np

        out = Path(args.save_scores)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, uq_score=scores, is_adverse=labels, source=np.array(source))
        print(f'Saved scores: {out}')


if __name__ == '__main__':
    main()
