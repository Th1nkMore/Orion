#!/usr/bin/env python3
"""Write the immutable hash-bound attestation for one Stage2-L v11 job."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict

from scripts.scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2l_v11_submission_attestation.v1"


def _read(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _validated_inputs(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "trainer_sha256": sha256_file(args.trainer.resolve()),
        "runtime_sha256": sha256_file(args.runtime.resolve()),
        "identifiability_audit_sha256": sha256_file(
            args.identifiability_audit.resolve()
        ),
        "parent_contract_sha256": sha256_file(args.parent_contract.resolve()),
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
        "v101_checkpoint_sha256": sha256_file(
            args.v101_checkpoint.resolve()
        ),
        "v101_report_sha256": sha256_file(args.v101_report.resolve()),
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
    parser.add_argument("--identifiability-audit", type=Path, required=True)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--v11-records", type=Path, required=True)
    parser.add_argument("--dataset-audit-report", type=Path, required=True)
    parser.add_argument("--view-feature-cache", type=Path, required=True)
    parser.add_argument("--u-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--v101-checkpoint", type=Path, required=True)
    parser.add_argument("--v101-report", type=Path, required=True)
    parser.add_argument("--orion-config", type=Path, required=True)
    parser.add_argument("--orion-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--remote-log", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not str(args.job_id).isdigit():
        raise ValueError("Slurm job id must be numeric")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite v11 submission attestation")
    protocol = _read(args.protocol.resolve())
    preflight = _read(args.preflight.resolve())
    amendment = _read(args.amendment.resolve())
    actual = _validated_inputs(args)
    run = amendment.get("authorized_run", {})
    locks = amendment.get("launch_locks", {})
    if (
        protocol.get("schema")
        != "orion.stage2l_v11_identifiable_smoke_protocol.v1"
        or preflight.get("schema")
        != "orion.stage2l_v11_identifiable_smoke_preflight.v1"
        or preflight.get("passed") is not True
        or preflight.get("training_started") is not False
        or preflight.get("validated_inputs") != actual
        or amendment.get("schema") != "orion.scenario_factory.amendment.v1"
        or amendment.get("status")
        != "immutable_v11_identifiability_smoke_authorization"
        or amendment.get("validated_inputs") != actual
        or amendment.get("protocol_sha256")
        != sha256_file(args.protocol.resolve())
        or amendment.get("preflight_sha256")
        != sha256_file(args.preflight.resolve())
        or run.get("output_root") != str(args.output_root.resolve())
        or run.get("maximum_submissions") != 1
        or run.get("automatic_retry") is not False
        or run.get("optimizer_steps") != 40
        or locks.get("stage2l_v11_bounded_smoke_allowed") is not True
        or locks.get("formal_stage2l_allowed") is not False
        or locks.get("stage2p_allowed") is not False
        or locks.get("closed_loop_allowed") is not False
    ):
        raise ValueError("v11 submission inputs or locks differ from lineage")
    value = {
        "schema": SCHEMA,
        "status": "single_v11_identifiability_job_submitted_and_attested",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_id": str(args.job_id),
        "job_name": "s2l_v11_ident",
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
        "amendment": {
            "path": str(args.amendment.resolve()),
            "sha256": sha256_file(args.amendment.resolve()),
        },
        "claim_boundary": (
            "Submission lineage only. It is not a training result, semantic "
            "pass, planning result or safety claim."
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
