#!/usr/bin/env python3
"""Write replay-alignment diagnostics for a completed clean/degraded pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.summarize_closedloop_safety import (
    build_paired_event_report,
    find_control_trace,
    load_records,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-run", type=Path, required=True)
    parser.add_argument("--degraded-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite prefix-alignment report")
    clean_trace = find_control_trace(args.clean_run)
    degraded_trace = find_control_trace(args.degraded_run)
    paired = build_paired_event_report(
        load_records(clean_trace), load_records(degraded_trace)
    )
    report = {
        "schema": "orion.closedloop_pair_prefix_alignment.v1",
        "clean_run": str(args.clean_run.resolve()),
        "degraded_run": str(args.degraded_run.resolve()),
        "clean_trace": str(clean_trace.resolve()),
        "degraded_trace": str(degraded_trace.resolve()),
        "clean_trace_sha256": sha256(clean_trace),
        "degraded_trace_sha256": sha256(degraded_trace),
        "pre_event_alignment": paired["pre_event_alignment"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
