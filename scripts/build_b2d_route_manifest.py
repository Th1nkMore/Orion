#!/usr/bin/env python3
"""Build an auditable Stage-1 route split from trusted B2D infos metadata."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
UQ_MODULE_ROOT = REPO_ROOT / "uq_estimator"
# Import the pure-stdlib module directly so Python does not execute
# uq_estimator/__init__.py (which correctly imports PyTorch for normal model
# use, but is irrelevant to this metadata-only login-node command).
if str(UQ_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(UQ_MODULE_ROOT))

from b2d_route_manifest import (  # noqa: E402
    B2DManifestError,
    build_b2d_route_manifest,
    load_b2d_infos,
    load_exclude_list,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--infos",
        type=Path,
        required=True,
        help="Trusted .pkl/.pickle or .json Bench2Drive infos file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Manifest JSON path; required unless --dry-run is used.",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        help="Optional second file containing only lineage_audit.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--calibration-ratio", type=float, default=0.1)
    parser.add_argument("--held-out-ratio", type=float, default=0.1)
    parser.add_argument("--minimum-canonical-routes", type=int, default=12)
    parser.add_argument("--min-routes-per-split", type=int, default=2)
    parser.add_argument(
        "--no-town-disjoint-heldout",
        action="store_true",
        help="Disable the preferred whole-town held-out selection.",
    )
    parser.add_argument(
        "--exclude-folder",
        action="append",
        default=[],
        help=(
            "B2D offline folder to exclude. Its whole canonical Town/RouteN group "
            "is removed. Repeat as needed."
        ),
    )
    parser.add_argument(
        "--exclude-list",
        action="append",
        default=[],
        type=Path,
        help="JSON or newline-delimited offline folder exclusion list.",
    )
    parser.add_argument(
        "--allow-unmatched-excludes",
        action="store_true",
        help="Do not fail if an exclusion matches no offline folder/route.",
    )
    parser.add_argument(
        "--closed-loop-development-route",
        action="append",
        default=[],
        help="External label for audit only; never mapped to an offline route ID.",
    )
    parser.add_argument(
        "--closed-loop-headline-route",
        action="append",
        default=[],
        help="External label for audit only; never mapped to an offline route ID.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the manifest without writing any file.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.dry_run and args.output is None:
        raise B2DManifestError("--output is required unless --dry-run is used")
    infos, source = load_b2d_infos(args.infos)
    exclusions = list(args.exclude_folder)
    for path in args.exclude_list:
        exclusions.extend(load_exclude_list(path))

    payload = build_b2d_route_manifest(
        infos,
        seed=args.seed,
        ratios={
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "calibration": args.calibration_ratio,
            "held_out": args.held_out_ratio,
        },
        exclude_folders=exclusions,
        allow_unmatched_excludes=args.allow_unmatched_excludes,
        minimum_canonical_routes=args.minimum_canonical_routes,
        min_routes_per_split=args.min_routes_per_split,
        prefer_town_disjoint_heldout=not args.no_town_disjoint_heldout,
        closed_loop_development_routes=args.closed_loop_development_route,
        closed_loop_headline_routes=args.closed_loop_headline_route,
        source_lineage=source,
    )
    payload["lineage_audit"]["writes_performed"] = not args.dry_run

    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if not args.dry_run:
        assert args.output is not None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        if args.audit_output is not None:
            args.audit_output.parent.mkdir(parents=True, exist_ok=True)
            args.audit_output.write_text(
                json.dumps(payload["lineage_audit"], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
