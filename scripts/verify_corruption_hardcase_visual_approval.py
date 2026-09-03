#!/usr/bin/env python3
"""Verify one hard-case corruption's visual approval and immutable lineage."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "uq_estimator"
    / "corruption_visual_approval.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "orion_corruption_visual_approval_contract", _CONTRACT_PATH
)
_CONTRACT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CONTRACT
_SPEC.loader.exec_module(_CONTRACT)
VisualApprovalError = _CONTRACT.VisualApprovalError
verify_visual_approval = _CONTRACT.verify_visual_approval


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--condition")
    parser.add_argument("--allow-pending", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        record = verify_visual_approval(
            gate_path=args.gate,
            repository_root=args.repository_root,
            family=args.family,
            condition=args.condition,
            require_approved=not args.allow_pending,
        )
        payload = {
            "schema": "orion.corruption_hardcase_visual_approval_verification.v1",
            "status": (
                "lineage_valid_pending_human_review"
                if record.decision_status == "pending"
                else "approved_lineage_valid"
            ),
            "record": record.to_dict(),
        }
        exit_code = 0
    except (VisualApprovalError, ValueError, KeyError, json.JSONDecodeError) as error:
        payload = {
            "schema": "orion.corruption_hardcase_visual_approval_verification.v1",
            "status": "rejected_fail_closed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        exit_code = 2
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite approval verification")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
