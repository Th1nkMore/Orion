#!/usr/bin/env python3
"""Write the hash-bound attestation for one Stage2-L v12.1 R-only job."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict

from scripts.scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2l_v12_1_factorized_r_submission_attestation.v1"
PROTOCOL_SCHEMA = "orion.stage2l_v12_1_factorized_r_smoke_protocol.v1"
PREFLIGHT_SCHEMA = "orion.stage2l_v12_1_factorized_r_smoke_preflight.v1"
LAUNCH_SCHEMA = "orion.stage2l_v12_1_factorized_r_smoke_launch.v1"
AUTHORIZATION_STATUS = "immutable_single_factorized_r_only_authorization"
JOB_NAME = "s2l_v121_factr"


def _read(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _validated_inputs(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "trainer_sha256": sha256_file(args.trainer.resolve()),
        "factorized_module_sha256": sha256_file(
            args.factorized_module.resolve()
        ),
        "dataset_manifest_sha256": sha256_file(
            args.dataset_manifest.resolve()
        ),
        "view_feature_cache_sha256": sha256_file(
            args.view_feature_cache.resolve()
        ),
        "orion_config_sha256": sha256_file(args.orion_config.resolve()),
        "orion_checkpoint_sha256": sha256_file(
            args.orion_checkpoint.resolve()
        ),
        "v101_checkpoint_sha256": sha256_file(
            args.v101_checkpoint.resolve()
        ),
        "v101_report_sha256": sha256_file(args.v101_report.resolve()),
        "factorized_cpu_report_sha256": sha256_file(
            args.factorized_cpu_report.resolve()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--factorized-module", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--view-feature-cache", type=Path, required=True)
    parser.add_argument("--orion-config", type=Path, required=True)
    parser.add_argument("--orion-checkpoint", type=Path, required=True)
    parser.add_argument("--v101-checkpoint", type=Path, required=True)
    parser.add_argument("--v101-report", type=Path, required=True)
    parser.add_argument("--factorized-cpu-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--launch-amendment", type=Path, required=True)
    parser.add_argument("--remote-log", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not str(args.job_id).isdigit():
        raise ValueError("Slurm job id must be numeric")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite v12.1 submission attestation")

    protocol = _read(args.protocol.resolve())
    preflight = _read(args.preflight.resolve())
    launch = _read(args.launch_amendment.resolve())
    actual = _validated_inputs(args)
    run = launch.get("authorized_run", {})
    locks = launch.get("launch_locks", {})
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status")
        != "frozen_bounded_r_only_protocol_launch_locked"
        or protocol.get("validated_inputs") != actual
        or preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("passed") is not True
        or preflight.get("training_started") is not False
        or preflight.get("gpu_used") is not False
        or preflight.get("validated_inputs") != actual
        or launch.get("schema") != LAUNCH_SCHEMA
        or launch.get("status") != AUTHORIZATION_STATUS
        or launch.get("validated_inputs") != actual
        or launch.get("protocol_sha256")
        != sha256_file(args.protocol.resolve())
        or launch.get("preflight_sha256")
        != sha256_file(args.preflight.resolve())
        or run.get("output_root") != str(args.output_root.resolve())
        or run.get("maximum_submissions") != 1
        or run.get("automatic_retry") is not False
        or run.get("maximum_optimizer_steps") != 40
        or run.get("only_trainable_paths")
        != ["existing_orion_lora", "same_view_relevance_queries", "factorized_relevance_head"]
        or locks.get("bounded_r_only_smoke_allowed") is not True
        or any(
            locks.get(name) is not False
            for name in (
                "stage1_uq_input",
                "u_tokenizer",
                "language_training",
                "trajectory_or_control",
                "formal_stage2l",
                "stage2p",
                "closed_loop",
                "locked_test_read",
            )
        )
    ):
        raise ValueError("v12.1 submission inputs or locks differ from lineage")

    value = {
        "schema": SCHEMA,
        "status": "single_factorized_r_only_job_submitted_and_attested",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_id": str(args.job_id),
        "job_name": JOB_NAME,
        "remote_log": str(args.remote_log.resolve()),
        "authorized_output_root": str(args.output_root.resolve()),
        "maximum_submissions": 1,
        "automatic_retry": False,
        "maximum_optimizer_steps": 40,
        "validated_inputs": actual,
        "protocol": {
            "path": str(args.protocol.resolve()),
            "sha256": sha256_file(args.protocol.resolve()),
        },
        "preflight": {
            "path": str(args.preflight.resolve()),
            "sha256": sha256_file(args.preflight.resolve()),
        },
        "launch_amendment": {
            "path": str(args.launch_amendment.resolve()),
            "sha256": sha256_file(args.launch_amendment.resolve()),
        },
        "locks": {
            "stage1_uq_loaded": False,
            "u_tokenizer_loaded": False,
            "language_training": False,
            "trajectory_or_control": False,
            "formal_stage2l": False,
            "stage2p": False,
            "closed_loop": False,
            "locked_test_read": False,
        },
        "claim_boundary": (
            "Submission lineage for one bounded factorized-R-only engineering "
            "job. It is not a model result, semantic-U result, language result, "
            "planning result or safety claim."
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
