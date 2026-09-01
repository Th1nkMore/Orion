#!/usr/bin/env python3
"""Collect the frozen nine Wave0 pair decisions into one immutable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "orion.corruption_hardcase_wave_decision_collection.v1"
ROUTES = ("151", "180", "194")
CONDITIONS = (
    "front_stale_200ms",
    "waterdrop_medium",
    "native_motion_blur_medium",
)


def decision_filename(route: str, condition: str) -> str:
    return "route%s_%s_decision.json" % (route, condition)


def collect(analysis_root: Path) -> dict[str, Any]:
    rows = []
    for route in ROUTES:
        for condition in CONDITIONS:
            path = analysis_root / decision_filename(route, condition)
            if not path.is_file():
                raise FileNotFoundError(str(path))
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != (
                "orion.corruption_hardcase_wave_pair_decision.v1"
            ):
                raise ValueError("unexpected pair schema in %s" % path)
            if payload.get("route_index") != route:
                raise ValueError("route mismatch in %s" % path)
            if payload.get("condition") != condition:
                raise ValueError("condition mismatch in %s" % path)
            rows.append({
                "route_index": route,
                "condition": condition,
                "path": str(path.resolve()),
                "valid": payload["validity"]["valid"],
                "positive_case": payload["decision"]["positive_case"],
                "evidence_tier": payload["decision"]["evidence_tier"],
                "hard_endpoint_degraded": payload["hard_endpoint"]["degraded"],
                "continuous_margin_degraded": payload[
                    "continuous_safety_margin"
                ]["degraded"],
            })
    invalid = [row for row in rows if not row["valid"]]
    positive = [row for row in rows if row["positive_case"]]
    by_condition = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        by_condition[condition] = {
            "pairs": len(selected),
            "valid_pairs": sum(row["valid"] for row in selected),
            "positive_pairs": sum(row["positive_case"] for row in selected),
            "positive_routes": [
                row["route_index"] for row in selected if row["positive_case"]
            ],
        }
    return {
        "schema": SCHEMA,
        "analysis_root": str(analysis_root.resolve()),
        "expected_pairs": len(ROUTES) * len(CONDITIONS),
        "observed_pairs": len(rows),
        "all_pairs_present": len(rows) == len(ROUTES) * len(CONDITIONS),
        "all_pairs_valid": not invalid,
        "valid_pairs": sum(row["valid"] for row in rows),
        "invalid_pairs": invalid,
        "positive_pairs": positive,
        "positive_pair_count": len(positive),
        "by_condition": by_condition,
        "pairs": rows,
        "decision_boundary": {
            "heldout_confirmation_automatically_authorized": False,
            "stage2p_automatically_authorized": False,
            "formal_200_route_evaluation_automatically_authorized": False,
            "reason": (
                "This collection reports failure-induction candidates only; "
                "the next experimental scope must be selected from the frozen "
                "funnel after reviewing valid positives and visual artifacts."
            ),
        },
        "claim_boundary": (
            "Development failure-induction evidence only; no learned-UQ, "
            "task-relevance, control-benefit, held-out, or formal benchmark claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite Wave0 decision collection")
    report = collect(args.analysis_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "all_pairs_valid": report["all_pairs_valid"],
        "positive_pair_count": report["positive_pair_count"],
    }, indent=2, sort_keys=True))
    return 0 if report["all_pairs_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
