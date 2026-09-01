#!/usr/bin/env python3
"""Build or audit a fail-closed chronological ORION actual-target smoke plan."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
UQ_MODULE_ROOT = REPO_ROOT / "uq_estimator"
if str(UQ_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(UQ_MODULE_ROOT))

from orion_replay_smoke import (  # noqa: E402
    DEFAULT_PREFIX_END,
    DEFAULT_ROUTE_KEY,
    OrionReplayPlanError,
    build_replay_smoke_plan,
    evaluate_runtime_attestation,
    load_pilot_manifest,
    replay_plan_summary,
    verify_source_infos,
)


DEFAULT_PILOT = (
    REPO_ROOT
    / "configs"
    / "spatial_uq_route_manifests"
    / "b2d_val_exploratory_pilot10_seed20260826.json"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-manifest", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--route", default=DEFAULT_ROUTE_KEY)
    parser.add_argument("--prefix-end", type=int, default=DEFAULT_PREFIX_END)
    parser.add_argument("--corruption", default="local_occlusion")
    parser.add_argument("--severity", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--infos",
        type=Path,
        help="Trusted B2D info pickle; verifies the parent SHA and exact frame-0 prefix.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="With --infos, also verify every six-view image and raw anno file.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON plan output.")
    parser.add_argument(
        "--runtime-attestation",
        type=Path,
        help="Evaluate an attestation from a future real exporter against this plan.",
    )
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.dataset_root is not None and args.infos is None:
        raise OrionReplayPlanError("--dataset-root requires --infos")
    pilot, lineage = load_pilot_manifest(args.pilot_manifest)
    source_verification = None
    if args.infos is not None:
        # Resolve the route/folder once without claiming source verification,
        # then rebuild the final plan with the file audit attached.
        provisional = build_replay_smoke_plan(
            pilot,
            lineage,
            canonical_route_key=args.route,
            prefix_end=args.prefix_end,
            corruption_family=args.corruption,
            severity=args.severity,
            seed=args.seed,
        )
        source_verification = verify_source_infos(
            args.infos,
            pilot,
            provisional["route"]["folder"],
            args.prefix_end,
            dataset_root=args.dataset_root,
        )
    plan = build_replay_smoke_plan(
        pilot,
        lineage,
        canonical_route_key=args.route,
        prefix_end=args.prefix_end,
        corruption_family=args.corruption,
        severity=args.severity,
        seed=args.seed,
        source_verification=source_verification,
    )
    result = replay_plan_summary(plan) if args.summary_only else plan
    if args.runtime_attestation is not None:
        attestation = json.loads(args.runtime_attestation.read_text(encoding="utf-8"))
        if not isinstance(attestation, dict):
            raise OrionReplayPlanError("runtime attestation root must be an object")
        result = {
            "plan": result,
            "g1_evaluation": evaluate_runtime_attestation(plan, attestation),
        }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
