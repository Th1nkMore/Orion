#!/usr/bin/env python3
"""Write one hash-bound Stage2-L v13 capacity-smoke attestation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict

from scripts.scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2l_v13_process_qa_submission_attestation.v1"
PROTOCOL_SCHEMA = "orion.stage2l_v13_process_qa_capacity_protocol.v1"
PREFLIGHT_SCHEMA = "orion.stage2l_v13_process_qa_preflight.v1"
LAUNCH_SCHEMA = "orion.stage2l_v13_process_qa_launch.v1"
LAUNCH_STATUS = "immutable_two_arm_process_qa_smoke_authorization"
ARMS = {
    "lora": "s2l_v13_lora",
    "partial_unfreeze": "s2l_v13_p4",
}


def _read(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _validated_inputs(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "trainer_sha256": sha256_file(args.trainer.resolve()),
        "process_module_sha256": sha256_file(args.process_module.resolve()),
        "v122_lineage_helper_sha256": sha256_file(
            args.v122_lineage_helper.resolve()
        ),
        "factorized_relevance_sha256": sha256_file(
            args.factorized_relevance.resolve()
        ),
        "dataset_manifest_sha256": sha256_file(
            args.dataset_manifest.resolve()
        ),
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
        "v121_checkpoint_sha256": sha256_file(
            args.v121_checkpoint.resolve()
        ),
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
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--process-module", type=Path, required=True)
    parser.add_argument("--v122-lineage-helper", type=Path, required=True)
    parser.add_argument("--factorized-relevance", type=Path, required=True)
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
        raise FileExistsError("refusing to overwrite v13 attestation")
    protocol = _read(args.protocol.resolve())
    preflight = _read(args.preflight.resolve())
    launch = _read(args.launch.resolve())
    actual = _validated_inputs(args)
    arm = launch.get("authorized_arms", {}).get(args.arm, {})
    locks = launch.get("locks", {})
    architecture = launch.get("architecture_invariants", {})
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("passed") is not True
        or preflight.get("training_started") is not False
        or preflight.get("training_arm") != args.arm
        or preflight.get("validated_inputs") != actual
        or preflight.get("protocol_sha256")
        != sha256_file(args.protocol.resolve())
        or launch.get("schema") != LAUNCH_SCHEMA
        or launch.get("status") != LAUNCH_STATUS
        or launch.get("validated_inputs") != actual
        or launch.get("protocol_sha256")
        != sha256_file(args.protocol.resolve())
        or arm.get("preflight_sha256")
        != sha256_file(args.preflight.resolve())
        or arm.get("output_root") != str(args.output_root.resolve())
        or arm.get("maximum_submissions") != 1
        or arm.get("optimizer_steps") != 200
        or launch.get("automatic_retry") is not False
        or architecture.get("task_risk_language_bridge_present") is not False
        or architecture.get("k_used_as_model_input") is not False
        or locks.get("bounded_v13_capacity_comparison_allowed") is not True
        or locks.get("formal_stage2l_allowed") is not False
        or locks.get("stage2p_allowed") is not False
        or locks.get("closed_loop_allowed") is not False
    ):
        raise ValueError("v13 submission inputs or locks differ from lineage")
    value = {
        "schema": SCHEMA,
        "status": "single_v13_capacity_arm_submitted_and_attested",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_arm": args.arm,
        "job_id": str(args.job_id),
        "job_name": ARMS[args.arm],
        "remote_log": str(args.remote_log.resolve()),
        "authorized_output_root": str(args.output_root.resolve()),
        "optimizer_steps": 200,
        "maximum_submissions": 1,
        "automatic_retry": False,
        "task_risk_language_bridge_present": False,
        "k_used_as_model_input": False,
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
            "Submission lineage for one controlled-U direct-token capacity "
            "smoke only; not formal generalization, planning or safety."
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
