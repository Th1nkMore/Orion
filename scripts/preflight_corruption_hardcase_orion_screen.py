#!/usr/bin/env python3
"""Resolve one hard-case runtime condition and enforce visual approval.

This command is intentionally lightweight: submitters call it before sbatch,
and runners call it again before CARLA or ORION can be loaded.
"""

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


STALE_DELAY_BY_SEVERITY = {1: 100, 2: 200, 3: 400}
RETIRED_CONDITIONS = {"lens_waterdrop_transient_off"}


def resolve_condition(
    *,
    pilot_condition: str,
    corruption_severity: int,
    paired_waterdrop_profile: str,
    native_motion_blur_profile: str,
) -> tuple[str, str, dict[str, str]] | None:
    """Return approval family, exact condition, and frozen runtime variables."""

    if pilot_condition in RETIRED_CONDITIONS:
        raise VisualApprovalError(
            "lens_waterdrop_transient_off names the retired failed v1 "
            "implementation; use lens_waterdrop_paired_template_transient_off"
        )
    if pilot_condition == "front_stale_transient_off":
        try:
            delay_ms = STALE_DELAY_BY_SEVERITY[int(corruption_severity)]
        except (KeyError, ValueError) as error:
            raise VisualApprovalError(
                "front stale severity must be 1, 2, or 3"
            ) from error
        return (
            "front_stale",
            "delay_ms:%d" % delay_ms,
            {
                "ORION_CLOSEDLOOP_CORRUPTION": "front_stale",
                "ORION_CLOSEDLOOP_CORRUPTION_SEVERITY": str(corruption_severity),
                "ORION_CLOSEDLOOP_CORRUPTION_VIEWS": "front",
            },
        )
    if pilot_condition == "lens_waterdrop_paired_template_transient_off":
        if paired_waterdrop_profile not in {"light", "medium", "heavy"}:
            raise VisualApprovalError(
                "paired waterdrop profile must explicitly be light, medium, or heavy"
            )
        return (
            "lens_waterdrop_paired_template",
            "profile:%s" % paired_waterdrop_profile,
            {
                "ORION_CLOSEDLOOP_CORRUPTION": "lens_waterdrop_paired_template",
                "ORION_PAIRED_WATERDROP_PROFILE": paired_waterdrop_profile,
                "ORION_CLOSEDLOOP_CORRUPTION_VIEWS": "front",
            },
        )
    if pilot_condition == "native_motion_blur_off":
        if native_motion_blur_profile not in {"light", "medium", "heavy"}:
            raise VisualApprovalError(
                "native motion blur profile must explicitly be light, medium, or heavy"
            )
        return (
            "native_motion_blur",
            "profile:%s" % native_motion_blur_profile,
            {
                "ORION_CLOSEDLOOP_CORRUPTION": "",
                "ORION_NATIVE_MOTION_BLUR_PROFILE": native_motion_blur_profile,
            },
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--pilot-condition", required=True)
    parser.add_argument("--corruption-severity", type=int, default=1)
    parser.add_argument("--paired-waterdrop-profile", default="")
    parser.add_argument("--native-motion-blur-profile", default="none")
    parser.add_argument("--allow-pending", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        resolved = resolve_condition(
            pilot_condition=args.pilot_condition,
            corruption_severity=args.corruption_severity,
            paired_waterdrop_profile=args.paired_waterdrop_profile,
            native_motion_blur_profile=args.native_motion_blur_profile,
        )
        if resolved is None:
            payload = {
                "schema": "orion.corruption_hardcase_orion_preflight.v1",
                "status": "not_a_hardcase_condition",
                "pilot_condition": args.pilot_condition,
            }
        else:
            family, exact_condition, runtime_environment = resolved
            record = verify_visual_approval(
                gate_path=args.gate,
                repository_root=args.repository_root,
                family=family,
                condition=exact_condition,
                require_approved=not args.allow_pending,
            )
            payload = {
                "schema": "orion.corruption_hardcase_orion_preflight.v1",
                "status": (
                    "approved_ready_for_submission"
                    if record.decision_status == "approved"
                    else "lineage_valid_submission_locked_pending_human_review"
                ),
                "pilot_condition": args.pilot_condition,
                "family": family,
                "condition": exact_condition,
                "runtime_environment": runtime_environment,
                "approval": record.to_dict(),
                "architecture_locks": {
                    "ORION_CLOSEDLOOP_UQ_MODE": "none",
                    "ORION_CLOSEDLOOP_CONDITIONING": "none",
                    "ORION_CLOSEDLOOP_RISK_MODE": "off",
                    "ORION_PLANNING_RESPONSE_MODE": "off",
                },
            }
        exit_code = 0
    except (VisualApprovalError, ValueError, KeyError, json.JSONDecodeError) as error:
        payload = {
            "schema": "orion.corruption_hardcase_orion_preflight.v1",
            "status": "rejected_fail_closed",
            "pilot_condition": args.pilot_condition,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        exit_code = 2

    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite preflight record")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
