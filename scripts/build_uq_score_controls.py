#!/usr/bin/env python3
"""Create marginally matched constant and shuffled controls from a UQ trace."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.trace.read_text().splitlines()
        if line.strip()
    ]
    scores = [
        float(record["raw_uq_score"])
        for record in records
        if record.get("raw_uq_score") is not None
    ]
    if not scores:
        raise ValueError(f"No raw UQ scores in {args.trace}")
    shuffled = list(scores)
    random.Random(args.seed).shuffle(shuffled)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    shuffled_path = args.out_dir / f"shuffled_scores_seed{args.seed}.json"
    summary_path = args.out_dir / "score_controls.json"
    shuffled_path.write_text(
        json.dumps({"seed": args.seed, "scores": shuffled}, indent=2) + "\n"
    )
    summary = {
        "source_trace": str(args.trace),
        "count": len(scores),
        "mean": statistics.fmean(scores),
        "median": statistics.median(scores),
        "minimum": min(scores),
        "maximum": max(scores),
        "shuffled_trace": str(shuffled_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
