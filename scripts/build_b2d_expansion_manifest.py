#!/usr/bin/env python3
"""Build a formal infos-backed manifest from a frozen B2D expansion plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uq_estimator.b2d_expansion_manifest import build_b2d_expansion_manifest  # noqa: E402
from uq_estimator.b2d_route_manifest import load_b2d_infos  # noqa: E402


def lineage(path: Path, kind: str) -> dict:
    raw = path.read_bytes()
    return {
        "kind": kind,
        "path": str(path.resolve()),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infos", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--expansion-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite manifest %s" % args.output)
    infos, source = load_b2d_infos(args.infos)
    baseline = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    plan = json.loads(args.expansion_plan.read_text(encoding="utf-8"))
    payload = build_b2d_expansion_manifest(
        infos,
        baseline,
        plan,
        source_lineage=source,
        baseline_lineage=lineage(args.baseline_manifest, "baseline_route_manifest"),
        expansion_plan_lineage=lineage(args.expansion_plan, "frozen_expansion_plan"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "infos_sha256": source["sha256"],
        "split_counts": {
            split: len(value["route_ids"])
            for split, value in payload["splits"].items()
        },
        "heldout_towns": payload["lineage_audit"]["selection"]["heldout_towns"],
        "leakage_checks": payload["lineage_audit"]["leakage_checks"],
    }, indent=2, sort_keys=True))
    print("B2D_EXPANSION_MANIFEST_OK=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
