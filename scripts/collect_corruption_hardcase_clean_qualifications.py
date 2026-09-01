#!/usr/bin/env python3
"""Collect one frozen Q1 clean-qualification wave."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "orion.corruption_hardcase_clean_qualification_collection.v1"
REPORT_SCHEMA = "orion.corruption_hardcase_clean_qualification.v1"
PROTOCOL_SCHEMA = "orion.corruption_hardcase_wave1_clean_qualification.v1"
WAVE2_Q1_ACTIVATION_SCHEMA = (
    "orion.corruption_hardcase_wave2_clean_q1_activation.v1"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report_filename(route_index: int) -> str:
    return "route%d_clean_q1_qualification.json" % route_index


def collect(*, analysis_root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") == PROTOCOL_SCHEMA:
        routes = list(protocol["selection"]["routes"])
    elif protocol.get("schema") == WAVE2_Q1_ACTIVATION_SCHEMA:
        if protocol.get("status") != "authorized_after_user_resume":
            raise ValueError("Wave2 Q1 activation is not authorized")
        routes = list(protocol["scope"]["routes"])
    else:
        raise ValueError("unexpected clean qualification protocol")
    rows = []
    for route_index in routes:
        path = analysis_root / report_filename(route_index)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != REPORT_SCHEMA:
            raise ValueError("unexpected report schema in %s" % path)
        if payload.get("phase") != "q1" or payload.get("route_index") != route_index:
            raise ValueError("route/phase mismatch in %s" % path)
        if payload.get("protocol", {}).get("sha256") != sha256(protocol_path):
            raise ValueError("protocol hash mismatch in %s" % path)
        rows.append({
            "route_index": route_index,
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "status": payload["status"],
            "qualified_for_next_clean_repeat": payload[
                "qualified_for_next_clean_repeat"
            ],
            "failed_checks": payload["failed_checks"],
            "route_completion_percent": payload["route_completion_percent"],
            "longest_low_speed_interval_seconds": payload[
                "longest_low_speed_interval"
            ]["duration_seconds"],
            "hard_infraction_counts": payload["hard_infraction_counts"],
        })
    passers = [row["route_index"] for row in rows if row["qualified_for_next_clean_repeat"]]
    rejected = [row["route_index"] for row in rows if not row["qualified_for_next_clean_repeat"]]
    return {
        "schema": SCHEMA,
        "status": "q1_collection_complete",
        "analysis_root": str(analysis_root.resolve()),
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": sha256(protocol_path),
        },
        "expected_routes": routes,
        "observed_reports": len(rows),
        "all_reports_present": len(rows) == len(routes),
        "q1_pass_routes": passers,
        "q1_rejected_routes": rejected,
        "q1_pass_count": len(passers),
        "q1_rejected_count": len(rejected),
        "routes": rows,
        "next_stage": {
            "q2_candidate_routes": passers,
            "q2_automatically_authorized": False,
            "corruption_automatically_authorized": False,
            "required_before_q2": (
                "Freeze a result amendment binding this collection and one exact "
                "Q2 clean-repeat submission plan."
            ),
        },
        "claim_boundary": (
            "Q1 clean reproducibility screen only. Passers require an independent "
            "Q2 clean repeat before any corruption-conditioned job."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Q1 qualification collection")
    report = collect(
        analysis_root=args.analysis_root,
        protocol_path=args.protocol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "q1_pass_routes": report["q1_pass_routes"],
        "q1_rejected_routes": report["q1_rejected_routes"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
