#!/usr/bin/env python3
"""Export or validate CPU-only oracle-to-adapter supervision datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uq_estimator.oracle_adapter_dataset import (
    DATASET_VERSION,
    SCHEMA_VERSION,
    build_run_samples,
    export_dataset,
    validate_dataset,
)


DEFAULT_SCHEMA = (
    PROJECT_ROOT
    / "configs"
    / "oracle_adapter_dataset"
    / "sample_schema_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join closed-loop trace, poses, frames, and outcomes without crossing rollouts"
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        help="closed-loop run directory; repeat to concatenate independent rollouts",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--relevance", choices=("on_path", "off_path"))
    parser.add_argument(
        "--allow-partial-horizon",
        action="store_true",
        help="retain terminal samples with a trailing-zero future mask",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="perform all joins and validation but write no dataset files",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        metavar="DATASET_DIR",
        help="validate an already exported dataset instead of exporting",
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate is not None:
        if args.run_dir or args.out_dir or args.relevance or args.dry_run:
            raise SystemExit("--validate cannot be combined with export arguments")
        print(json.dumps(validate_dataset(args.validate), indent=2, sort_keys=True))
        return

    if not args.run_dir:
        raise SystemExit("at least one --run-dir is required")
    if args.relevance is None:
        raise SystemExit("--relevance is required; it is never inferred from camera name")
    if not args.schema.is_file():
        raise SystemExit(f"schema file does not exist: {args.schema}")
    require_full_horizon = not args.allow_partial_horizon

    if args.dry_run:
        summaries = []
        total = 0
        for run_dir in args.run_dir:
            samples, summary = build_run_samples(
                run_dir,
                relevance=args.relevance,
                require_full_horizon=require_full_horizon,
            )
            total += len(samples)
            summaries.append(summary)
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "writes_performed": False,
                    "dataset_version": DATASET_VERSION,
                    "sample_schema_version": SCHEMA_VERSION,
                    "sample_count": total,
                    "runs": summaries,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.out_dir is None:
        raise SystemExit("--out-dir is required unless --dry-run is used")
    manifest = export_dataset(
        args.run_dir,
        args.out_dir,
        relevance=args.relevance,
        require_full_horizon=require_full_horizon,
        schema_path=args.schema,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
