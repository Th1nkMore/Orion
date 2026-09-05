#!/usr/bin/env python3
"""Audit a completed V1c report without loading Qwen or its checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = PROJECT_ROOT / "uq_estimator" / "qwen_visibility_grounding_evaluation.py"
    spec = importlib.util.spec_from_file_location(
        "_qwen_visibility_grounding_evaluation_audit", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module = _load_module()
    audit = module.audit_grounding_overfit_report(args.report, args.output)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

