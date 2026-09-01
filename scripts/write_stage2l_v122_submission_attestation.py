#!/usr/bin/env python3
"""Write the hash-bound attestation for one Stage2-L v12.2 semantic slice."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict

from scripts.scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2l_v12_2_submission_attestation.v1"
PROTOCOL_SCHEMA = "orion.stage2l_v12_2_vertical_slice_semantic_protocol.v1"
PREFLIGHT_SCHEMA = "orion.stage2l_v12_2_vertical_slice_semantic_preflight.v1"
LAUNCH_SCHEMA = "orion.stage2l_v12_2_vertical_slice_semantic_launch.v1"
AUTHORIZATION_STATUS = "immutable_single_soft_gate_semantic_slice_authorization"
JOB_NAME = "s2l_v122_semantic"


def _read(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _validated_inputs(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "trainer_sha256": sha256_file(args.trainer.resolve()),
        "runtime_sha256": sha256_file(args.runtime.resolve()),
        "factorized_relevance_sha256": sha256_file(
            args.factorized_relevance.resolve()
        ),
        "identifiability_audit_sha256": sha256_file(
            args.identifiability_audit.resolve()
        ),
        "soft_gate_policy_sha256": sha256_file(args.soft_gate_policy.resolve()),
        "dataset_manifest_sha256": sha256_file(args.dataset_manifest.resolve()),
        "v11_records_sha256": sha256_file(args.v11_records.resolve()),
        "dataset_audit_report_sha256": sha256_file(
            args.dataset_audit_report.resolve()
        ),
        "view_feature_cache_sha256": sha256_file(
            args.view_feature_cache.resolve()
        ),
        "u_tokenizer_checkpoint_sha256": sha256_file(
            args.u_tokenizer_checkpoint.resolve()
        ),
        "v121_checkpoint_sha256": sha256_file(args.v121_checkpoint.resolve()),
        "v121_report_sha256": sha256_file(args.v121_report.resolve()),
        "v121_terminal_validation_sha256": sha256_file(
            args.v121_terminal_validation.resolve()
        ),
        "orion_config_sha256": sha256_file(args.orion_config.resolve()),
        "orion_checkpoint_sha256": sha256_file(
            args.orion_checkpoint.resolve()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--factorized-relevance", type=Path, required=True)
    parser.add_argument("--identifiability-audit", type=Path, required=True)
    parser.add_argument("--soft-gate-policy", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--v11-records", type=Path, required=True)
    parser.add_argument("--dataset-audit-report", type=Path, required=True)
    parser.add_argument("--view-feature-cache", type=Path, required=True)
    parser.add_argument("--u-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--v121-checkpoint", type=Path, required=True)
    parser.add_argument("--v121-report", type=Path, required=True)
    parser.add_argument("--v121-terminal-validation", type=Path, required=True)
    parser.add_argument("--orion-config", type=Path, required=True)
    parser.add_argument("--orion-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--remote-log", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not str(args.job_id).isdigit():
        raise ValueError("Slurm job id must be numeric")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite v12.2 submission attestation")
    protocol = _read(args.protocol.resolve())
    preflight = _read(args.preflight.resolve())
    launch = _read(args.launch.resolve())
    actual = _validated_inputs(args)
    run = launch.get("authorized_run", {})
    locks = launch.get("locks", {})
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("passed") is not True
        or preflight.get("training_started") is not False
        or preflight.get("validated_inputs") != actual
        or launch.get("schema") != LAUNCH_SCHEMA
        or launch.get("status") != AUTHORIZATION_STATUS
        or launch.get("validated_inputs") != actual
        or launch.get("protocol_sha256") != sha256_file(args.protocol.resolve())
        or launch.get("preflight_sha256") != sha256_file(args.preflight.resolve())
        or run.get("output_root") != str(args.output_root.resolve())
        or run.get("maximum_submissions") != 1
        or run.get("automatic_retry") is not False
        or run.get("optimizer_steps") != 40
        or locks.get("bounded_v122_semantic_slice_allowed") is not True
        or locks.get("formal_stage2l_allowed") is not False
        or locks.get("stage2p_allowed") is not False
        or locks.get("closed_loop_allowed") is not False
    ):
        raise ValueError("v12.2 submission inputs or locks differ from lineage")
    value = {
        "schema": SCHEMA,
        "status": "single_v12_2_semantic_slice_submitted_and_attested",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_id": str(args.job_id),
        "job_name": JOB_NAME,
        "remote_log": str(args.remote_log.resolve()),
        "authorized_output_root": str(args.output_root.resolve()),
        "maximum_submissions": 1,
        "automatic_retry": False,
        "optimizer_steps": 40,
        "only_trainable_module": "TaskRiskLanguageBridge",
        "formal_stage2l": False,
        "stage2p": False,
        "closed_loop": False,
        "validated_inputs": actual,
        "protocol": {
            "path": str(args.protocol.resolve()),
            "sha256": sha256_file(args.protocol.resolve()),
        },
        "preflight": {
            "path": str(args.preflight.resolve()),
            "sha256": sha256_file(args.preflight.resolve()),
        },
        "launch": {
            "path": str(args.launch.resolve()),
            "sha256": sha256_file(args.launch.resolve()),
        },
        "claim_boundary": (
            "Submission lineage for one controlled-U semantic engineering "
            "slice only; not a model-quality, planning or safety result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
