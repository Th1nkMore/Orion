#!/usr/bin/env python3
"""Audit split leakage and semantic consistency in a Stage2-L QA dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uq_relevance_qa_factory_lib import audit_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--config",
        default="configs/scenario_factory/qa_factory_v1.json",
    )
    parser.add_argument("--output")
    parser.add_argument("--require-formal-gates", action="store_true")
    args = parser.parse_args()
    dataset_path = Path(args.dataset).resolve()
    config_path = Path(args.config).resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    audit = audit_dataset(dataset, config=config, dataset_dir=dataset_path.parent)
    rendered = json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        output = Path(args.output).resolve()
        if output.exists():
            raise FileExistsError("refusing to overwrite QA audit output")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.require_formal_gates and not audit["formal_training_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
