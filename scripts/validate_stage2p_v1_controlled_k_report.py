#!/usr/bin/env python3
"""Validate the terminal controlled-K Stage2-P engineering slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from uq_estimator.stage2p_task_risk_trajectory import load_checkpoint


REPORT_SCHEMA = "orion.stage2p_v1_controlled_k_interface_smoke.v1"
CHECKPOINT_SCHEMA = "orion.stage2p_task_risk_trajectory.v1"
EXPECTED_STATUS = "controlled_k_interface_completed_with_soft_quality_failures"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--retry-authorization", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = (
        args.report,
        args.checkpoint,
        args.protocol,
        args.preflight,
        args.launch,
        args.retry_authorization,
        args.submission,
    )
    if not all(path.is_file() for path in inputs):
        raise FileNotFoundError("Stage2-P terminal validation input is missing")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage2-P validation")

    report = _read(args.report)
    protocol = _read(args.protocol)
    preflight = _read(args.preflight)
    launch = _read(args.launch)
    retry = _read(args.retry_authorization)
    submission = _read(args.submission)
    expected_inputs = preflight.get("validated_inputs")
    history = report.get("history", [])
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("status") != EXPECTED_STATUS
        or report.get("optimizer_steps") != 80
        or len(history) != 80
        or [row.get("optimizer_step") for row in history] != list(range(1, 81))
        or any(row.get("finite") is not True for row in history)
        or any(
            row.get("trainable_scope")
            != "task_risk_projector_and_trajectory_response_only"
            for row in history
        )
        or report.get("hard_checks_passed") is not True
        or not all(report.get("hard_checks", {}).values())
        or report.get("soft_checks_passed") is not False
        or report.get("validated_inputs") != expected_inputs
        or not _all_finite(report)
    ):
        raise ValueError("Stage2-P report contract differs")
    locks = report.get("locks", {})
    if (
        locks.get("formal_stage2p_ready") is not False
        or locks.get("closed_loop_eligible") is not False
        or locks.get("locked_test_read") is not False
        or locks.get("automatic_extension") is not False
        or locks.get("benchmark_claim") is not False
    ):
        raise ValueError("Stage2-P report locks differ")
    if (
        report.get("protocol_sha256") != _sha256(args.protocol)
        or report.get("preflight_sha256") != _sha256(args.preflight)
        or report.get("launch_sha256") != _sha256(args.launch)
        or launch.get("validated_inputs") != expected_inputs
        or launch.get("authorized_run", {}).get("optimizer_steps") != 80
        or launch.get("authorized_run", {}).get("maximum_submissions") != 1
        or launch.get("authorized_run", {}).get("automatic_retry") is not False
    ):
        raise ValueError("Stage2-P frozen lineage differs")
    if (
        retry.get("status")
        != "single_replacement_authorized_after_preoptimization_launch_failure"
        or retry.get("invalid_job", {}).get("job_id") != "1123186"
        or retry.get("invalid_job", {}).get("optimizer_steps") != 0
        or retry.get("invalid_job", {}).get("training_output_written") is not False
        or retry.get("authorization", {}).get("maximum_replacement_submissions")
        != 1
        or submission.get("status")
        != "single_authorized_stage2p_replacement_submitted"
        or submission.get("submission", {}).get("job_id") != "1123187"
        or submission.get("scope", {}).get("remaining_replacement_submissions")
        != 0
        or submission.get("hash_binding", {}).get("retry_launch_sha256")
        != _sha256(args.launch)
    ):
        raise ValueError("Stage2-P replacement attestation differs")

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    contract = checkpoint.get("responsibility_contract", {})
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("status") != EXPECTED_STATUS
        or checkpoint.get("optimizer_steps") != 80
        or checkpoint.get("engineering_smoke_only") is not True
        or checkpoint.get("formal_stage2p_ready") is not False
        or checkpoint.get("closed_loop_eligible") is not False
        or checkpoint.get("validated_inputs") != expected_inputs
        or contract.get("forward_inputs")
        != ["frozen_orion_planning_context", "task_risk_k"]
        or contract.get("raw_observation_u_forward") is not False
        or contract.get("privileged_task_context_forward") is not False
        or contract.get("route_actor_ttc_outcome_forward") is not False
        or contract.get("privileged_labels_supervision_only") is not True
    ):
        raise ValueError("Stage2-P checkpoint contract differs")
    projector, response, metadata = load_checkpoint(
        args.checkpoint,
        expected_sha256=_sha256(args.checkpoint),
        device="cpu",
    )
    projector.eval()
    response.eval()
    torch.manual_seed(7)
    planning_context = torch.randn(1, 256, 256)
    zero_k = torch.zeros(1, 6, 40, 40)
    unit_k = zero_k.clone()
    unit_k[0, 0, 23:40, 12:38] = 1.0
    with torch.no_grad():
        zero_output = response(planning_context, projector(zero_k))
        unit_output = response(planning_context, projector(unit_k))
    runtime_checks = {
        "checkpoint_loader_passed": metadata["schema"] == CHECKPOINT_SCHEMA,
        "native_context_exact_identity": torch.equal(
            zero_output.conditioned_context, planning_context
        ),
        "zero_k_trajectory_exact_identity": int(
            torch.count_nonzero(zero_output.trajectory_residual)
        ) == 0,
        "unit_k_response_finite": bool(
            torch.isfinite(unit_output.trajectory_residual).all()
        ),
        "unit_k_lateral_within_2m": float(
            unit_output.trajectory_residual[..., 0].abs().max()
        ) <= 2.0,
        "unit_k_longitudinal_within_24m": float(
            unit_output.trajectory_residual[..., 1].abs().max()
        ) <= 24.0,
    }
    if not all(runtime_checks.values()):
        raise ValueError("Stage2-P checkpoint runtime check failed")

    value = {
        "schema": "orion.stage2p_v1_controlled_k_terminal_validation.v1",
        "status": "validated_integrity_pass_soft_specificity_failures",
        "integrity_valid": True,
        "job_id": "1123187",
        "optimizer_steps": 80,
        "artifact_hashes": {
            "report.json": _sha256(args.report),
            "stage2p_controlled_k_response.pt": _sha256(args.checkpoint),
        },
        "runtime_checks": runtime_checks,
        "quality_diagnosis": {
            "positive_target_mae_m": report["after"]["positive_target_mae_m"],
            "positive_nonzero_response_fraction": report["after"][
                "positive_nonzero_response_fraction"
            ],
            "hard_identity_mean_response_m": report["after"][
                "hard_identity_mean_response_m"
            ],
            "irrelevant_k_max_response_m": report["after"]["by_variant"][
                "irrelevant_k"
            ]["maximum_absolute_response_m"],
            "view_shuffled_k_max_response_m": report["after"]["by_variant"][
                "view_shuffled_k"
            ]["maximum_absolute_response_m"],
            "failed_soft_checks": sorted(
                key for key, passed in report["soft_checks"].items() if not passed
            ),
            "interpretation": (
                "The response learned the four positive controlled-K targets and "
                "preserved exact zero-K identity, but its worst low-K irrelevant and "
                "view-shuffled responses exceeded the preregistered 0.2 m soft bound."
            ),
        },
        "decision": "run_one_hash_bound_route147_carla_interface_smoke",
        "locks": {
            "formal_stage2p_ready": False,
            "closed_loop_safety_ready": False,
            "locked_test_read": False,
            "extra_training": False,
            "automatic_retry": False,
        },
        "claim_boundary": (
            "Independent integrity and specificity diagnosis for one controlled-K "
            "trajectory-interface engineering smoke; not a learned-U, generalization, "
            "closed-loop benefit or safety result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
