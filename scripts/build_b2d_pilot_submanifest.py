#!/usr/bin/env python3
"""Select an auditable 8-12-route, 800-1000-state B2D Stage-1 pilot."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
UQ_MODULE_ROOT = REPO_ROOT / "uq_estimator"
if str(UQ_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(UQ_MODULE_ROOT))

from b2d_pilot_sampling import (  # noqa: E402
    B2DPilotSamplingError,
    build_b2d_pilot_submanifest,
    load_b2d_infos,
    load_parent_manifest,
    pilot_summary,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infos", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--route-count", type=int, default=10)
    parser.add_argument("--target-states", type=int, default=900)
    parser.add_argument("--minimum-routes-per-split", type=int, default=2)
    parser.add_argument("--max-candidate-distance-m", type=float, default=50.0)
    parser.add_argument("--max-candidate-lateral-m", type=float, default=15.0)
    parser.add_argument("--minimum-candidate-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-background-fraction", type=float, default=0.25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print compact audit statistics instead of hundreds of frame records.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.dry_run and args.output is None:
        raise B2DPilotSamplingError("--output is required unless --dry-run is used")
    infos, info_lineage = load_b2d_infos(args.infos)
    parent, parent_lineage = load_parent_manifest(args.parent_manifest)
    parent_lineage["info_source"] = info_lineage
    payload = build_b2d_pilot_submanifest(
        infos,
        parent,
        parent_lineage,
        seed=args.seed,
        route_count=args.route_count,
        target_states=args.target_states,
        minimum_routes_per_split=args.minimum_routes_per_split,
        max_candidate_distance_m=args.max_candidate_distance_m,
        max_candidate_lateral_m=args.max_candidate_lateral_m,
        minimum_candidate_fraction=args.minimum_candidate_fraction,
        minimum_background_fraction=args.minimum_background_fraction,
    )
    payload["writes_performed"] = not args.dry_run
    if not args.dry_run:
        assert args.output is not None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    rendered = pilot_summary(payload) if args.summary_only else payload
    json.dump(rendered, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
