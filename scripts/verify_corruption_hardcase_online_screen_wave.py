#!/usr/bin/env python3
"""Verify the prospective online hard-case wave without submitting work."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "uq_estimator/corruption_visual_approval.py"
SPEC = importlib.util.spec_from_file_location(
    "orion_corruption_visual_approval_contract", CONTRACT_PATH
)
CONTRACT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONTRACT
SPEC.loader.exec_module(CONTRACT)
VisualApprovalError = CONTRACT.VisualApprovalError
verify_visual_approval = CONTRACT.verify_visual_approval


SCHEMA = "orion.corruption_hardcase_online_screen_wave.v1"
ARCHITECTURE_LOCKS = {
    "ORION_CLOSEDLOOP_UQ_MODE": "none",
    "ORION_CLOSEDLOOP_CONDITIONING": "none",
    "ORION_CLOSEDLOOP_RISK_MODE": "off",
    "ORION_PLANNING_RESPONSE_MODE": "off",
    "legacy_density_uq": False,
    "learned_uq_control": False,
    "stage2_control": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def condition_runtime(family: str, exact_condition: str) -> dict[str, str]:
    if family == "front_stale":
        if not exact_condition.startswith("delay_ms:"):
            raise ValueError("front stale condition must use delay_ms:<value>")
        delay = int(exact_condition[len("delay_ms:"):])
        severity_by_delay = {100: 1, 200: 2, 400: 3}
        if delay not in severity_by_delay:
            raise ValueError("unsupported stale delay")
        return {
            "PILOT_CONDITION": "front_stale_transient_off",
            "ORION_CLOSEDLOOP_CORRUPTION_SEVERITY": str(severity_by_delay[delay]),
        }
    if family == "lens_waterdrop_paired_template":
        if not exact_condition.startswith("profile:"):
            raise ValueError("waterdrop condition must use profile:<value>")
        profile = exact_condition[len("profile:"):]
        if profile not in {"light", "medium", "heavy"}:
            raise ValueError("unsupported paired waterdrop profile")
        return {
            "PILOT_CONDITION": "lens_waterdrop_paired_template_transient_off",
            "ORION_PAIRED_WATERDROP_PROFILE": profile,
        }
    if family == "native_motion_blur":
        if not exact_condition.startswith("profile:"):
            raise ValueError("motion-blur condition must use profile:<value>")
        profile = exact_condition[len("profile:"):]
        if profile not in {"light", "medium", "heavy"}:
            raise ValueError("unsupported native motion blur profile")
        return {
            "PILOT_CONDITION": "native_motion_blur_off",
            "ORION_NATIVE_MOTION_BLUR_PROFILE": profile,
        }
    raise ValueError("unknown hard-case family")


def verify_hash_record(record: dict[str, Any]) -> str:
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(str(path))
    actual = sha256(path)
    if actual != record["sha256"]:
        raise ValueError("hash differs: %s" % path)
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--activation", type=Path)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--allow-pending", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if config.get("schema") != SCHEMA:
            raise ValueError("unexpected online-screen schema")
        if config.get("architecture_locks") != ARCHITECTURE_LOCKS:
            raise ValueError("architecture locks differ")
        route_indices = [int(row["route_index"]) for row in config["routes"]]
        if route_indices != config["selection"]["routes"]:
            raise ValueError("route selection and route records differ")
        if len(route_indices) != len(set(route_indices)):
            raise ValueError("duplicate route in online screen")
        if config["selection"].get("heldout_routes_included") is not False:
            raise ValueError("heldout route leaked into development wave")
        if config["selection"].get("route203_included") is not False:
            raise ValueError("Route203 is locked")
        if config["execution_locks"].get("slurm_submission") is not False:
            raise ValueError("pending config unexpectedly enables submission")
        activation = None
        activation_primary = {}
        if args.activation is not None:
            activation = json.loads(args.activation.read_text(encoding="utf-8"))
            if activation.get("schema") != (
                "orion.corruption_hardcase_online_screen_wave_activation.v1"
            ):
                raise ValueError("unexpected wave activation schema")
            if activation["base_wave"]["sha256"] != sha256(args.config):
                raise ValueError("activation base-wave hash differs")
            expected_base_path = str(args.config)
            if args.config.is_absolute():
                try:
                    expected_base_path = str(
                        args.config.resolve().relative_to(
                            args.repository_root.resolve()
                        )
                    )
                except ValueError:
                    pass
            if activation["base_wave"]["path"] != expected_base_path:
                raise ValueError("activation base-wave path differs")
            gate_activation_path = (
                args.repository_root / activation["visual_gate"]["path"]
            )
            if sha256(gate_activation_path) != activation["visual_gate"]["sha256"]:
                raise ValueError("activation visual-gate hash differs")
            authorization_path = (
                args.repository_root / activation["authorization"]["path"]
            )
            if sha256(authorization_path) != activation["authorization"]["sha256"]:
                raise ValueError("activation authorization hash differs")
            if activation.get("architecture_locks") != ARCHITECTURE_LOCKS:
                raise ValueError("activation architecture locks differ")
            if activation["matrix"]["routes"] != route_indices:
                raise ValueError("activation route matrix differs")
            if activation["matrix"].get("total_jobs") != 12:
                raise ValueError("activation total job count differs")
            if activation["execution_authority"].get("wave0_slurm_submission") is not True:
                raise ValueError("activation does not authorize Wave0 submission")
            for locked_scope in (
                "heldout_confirmation", "route203", "stage2p",
                "formal_200_route_evaluation",
            ):
                if activation["execution_authority"].get(locked_scope) is not False:
                    raise ValueError("activation broadens locked scope: %s" % locked_scope)
            activation_primary = activation["primary_exact_conditions"]
        route_hashes = {
            str(row["route_index"]): verify_hash_record(row["route_xml"])
            for row in config["routes"]
        }
        template = config["condition_templates"]["lens_waterdrop_paired_template"]
        bank = Path(template["template_bank"]["path"])
        template_hashes = {}
        for name, expected in template["template_bank"]["files"].items():
            path = bank / name
            if not path.is_file():
                raise FileNotFoundError(str(path))
            actual = sha256(path)
            if actual != expected:
                raise ValueError("waterdrop template hash differs: %s" % path)
            template_hashes[name] = actual

        gate_path = args.repository_root / config["execution_locks"]["visual_gate"]
        approvals = {}
        primary_runtime = {}
        for family in (
            "front_stale",
            "lens_waterdrop_paired_template",
            "native_motion_blur",
        ):
            template = config["condition_templates"][family]
            candidates = template["candidate_exact_conditions"]
            primary = activation_primary.get(
                family, template.get("primary_exact_condition")
            )
            if primary is not None and primary not in candidates:
                raise ValueError("primary condition is not a candidate")
            if not args.allow_pending and primary is None:
                raise VisualApprovalError(
                    "exact primary condition is not frozen for %s" % family
                )
            verified = []
            for exact in candidates:
                record = verify_visual_approval(
                    gate_path=gate_path,
                    repository_root=args.repository_root,
                    family=family,
                    condition=exact,
                    require_approved=(not args.allow_pending and exact == primary),
                )
                verified.append({
                    "condition": exact,
                    "decision_status": record.decision_status,
                    "exact_condition_approved": exact in record.approved_conditions,
                    "implementation_sha256": record.implementation_sha256,
                })
            approvals[family] = verified
            if primary is not None:
                primary_runtime[family] = condition_runtime(family, primary)

        status = (
            "dry_run_valid_submission_locked_pending_visual_approval"
            if args.allow_pending
            else "approved_wave_ready_for_explicit_submission_authorization"
        )
        payload = {
            "schema": "orion.corruption_hardcase_online_screen_wave_verification.v1",
            "status": status,
            "config": {"path": str(args.config), "sha256": sha256(args.config)},
            "activation": (
                {"path": str(args.activation), "sha256": sha256(args.activation)}
                if args.activation is not None else None
            ),
            "route_sha256": route_hashes,
            "template_sha256": template_hashes,
            "approvals": approvals,
            "primary_runtime": primary_runtime,
            "jobs_submitted": 0,
            "orion_loaded": False,
        }
        exit_code = 0
    except (
        VisualApprovalError,
        FileNotFoundError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        payload = {
            "schema": "orion.corruption_hardcase_online_screen_wave_verification.v1",
            "status": "rejected_fail_closed",
            "error_type": type(error).__name__,
            "error": str(error),
            "jobs_submitted": 0,
            "orion_loaded": False,
        }
        exit_code = 2
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite wave verification")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
