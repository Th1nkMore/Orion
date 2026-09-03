#!/usr/bin/env python3
"""Verify component partitions and aggregates in the frozen raw inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.scan_stage2l_v12_existing_raw_actor_support_inventory import (
    CAMERA_ORDER,
    COMPONENTS,
    SCHEMA as INVENTORY_SCHEMA,
    summarize_routes,
)


SCHEMA = "orion.stage2l_v12_existing_raw_actor_support_inventory_verification.v1"


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_row(row: Dict[str, Any]) -> None:
    counts = row["component_positive_frame_count"]
    frames = row["component_positive_frames"]
    for component in COMPONENTS:
        if set(counts[component]) != set(CAMERA_ORDER):
            raise ValueError("component count camera order differs")
        if set(frames[component]) != set(CAMERA_ORDER):
            raise ValueError("component frame camera order differs")
        for view in CAMERA_ORDER:
            values = list(map(int, frames[component][view]))
            if values != sorted(set(values)):
                raise ValueError("component frames are not sorted and unique")
            if int(counts[component][view]) != len(values):
                raise ValueError("component count differs from frame list")
    for view in CAMERA_ORDER:
        route = set(frames["route_support"][view])
        actor = set(frames["actor_support"][view])
        union = set(frames["union_relevance"][view])
        route_only = set(frames["route_only"][view])
        actor_only = set(frames["actor_only"][view])
        both = set(frames["route_and_actor"][view])
        if union != route | actor:
            raise ValueError("union frames differ from route union actor")
        if route_only != route - actor:
            raise ValueError("route-only frames differ")
        if actor_only != actor - route:
            raise ValueError("actor-only frames differ")
        if both != route & actor:
            raise ValueError("route-and-actor frames differ")
        if (route_only | actor_only | both) != union:
            raise ValueError("exclusive support partitions do not cover union")


def verify_inventory(value: Dict[str, Any]) -> Dict[str, Any]:
    if value.get("schema") != INVENTORY_SCHEMA:
        raise ValueError("inventory schema differs")
    rows = value["deduplicated_routes"]
    if len(rows) != int(value["inventory"]["deduplicated_route_count"]):
        raise ValueError("deduplicated route count differs")
    route_indices = [int(row["route_index"]) for row in rows]
    if route_indices != sorted(set(route_indices)):
        raise ValueError("deduplicated route identities are not sorted and unique")
    for row in value["all_geometry_runs"]:
        _verify_row(row)
    for row in rows:
        _verify_row(row)
        if row.get("formal_split") in ("dev", "test"):
            raise ValueError("formal held-out route leaked into inventory")
    expected_summary = summarize_routes(rows)
    if value["summary"] != expected_summary:
        raise ValueError("frozen aggregate summary differs from route rows")
    expected_actor_routes = {
        view: [
            int(row["route_index"])
            for row in rows
            if row["component_positive_frame_count"]["actor_support"][view] > 0
        ]
        for view in CAMERA_ORDER
    }
    if value["actor_support_route_indices_by_view"] != expected_actor_routes:
        raise ValueError("actor-support route index summary differs")
    return {
        "deduplicated_route_count": len(rows),
        "geometry_run_count": len(value["all_geometry_runs"]),
        "zero_actor_support_views": value["summary"]["zero_actor_support_views"],
        "zero_union_support_views": value["summary"]["zero_union_support_views"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite inventory verification")
    value = _read_json(args.inventory)
    verified = verify_inventory(value)
    output = {
        "schema": SCHEMA,
        "status": "passed",
        "inventory": {
            "path": str(args.inventory.resolve()),
            "sha256": _sha256(args.inventory),
        },
        "verified": verified,
        "checks": {
            "all_run_component_counts_match_frame_lists": True,
            "route_actor_union_identity_exact": True,
            "exclusive_component_partition_exact": True,
            "deduplicated_summary_recomputed_exactly": True,
            "formal_dev_or_locked_test_absent": True,
            "actor_route_indices_recomputed_exactly": True,
        },
        "claim_boundary": "Internal consistency verification only; it does not validate the semantic correctness of the weak geometry target.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", **verified}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
