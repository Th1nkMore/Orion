#!/usr/bin/env python3
"""Build one immutable scenario-factory event package from a clean replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scenario_factory_lib import ALLOWED_SPLITS, build_event_package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=ALLOWED_SPLITS, required=True)
    parser.add_argument("--batch-manifest", type=Path)
    parser.add_argument("--visualization-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite scenario event package")
    report = build_event_package(
        args.run_dir,
        split=args.split,
        batch_manifest_path=args.batch_manifest,
        visualization_manifest_path=args.visualization_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "runtime_valid": report["runtime"]["valid"],
                "outcome_class": report["outcome_class"],
                "qa_input_ready": report["qa_input_ready"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
