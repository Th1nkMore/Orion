#!/usr/bin/env python3
"""Write the immutable submission attestation for one v10 Phase-A replay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2l_v10_phase_a_replay_submission_attestation.v1"


def _read(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--orion-config", type=Path, required=True)
    parser.add_argument("--orion-checkpoint", type=Path, required=True)
    parser.add_argument("--u-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--phase-a-checkpoint", type=Path, required=True)
    parser.add_argument("--v10-report", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--remote-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite replay submission attestation")
    amendment = _read(args.amendment.resolve())
    actual = {
        "evaluator_sha256": sha256_file(args.evaluator.resolve()),
        "dataset_manifest_sha256": sha256_file(args.dataset_manifest.resolve()),
        "orion_config_sha256": sha256_file(args.orion_config.resolve()),
        "orion_checkpoint_sha256": sha256_file(args.orion_checkpoint.resolve()),
        "u_tokenizer_checkpoint_sha256": sha256_file(
            args.u_tokenizer_checkpoint.resolve()
        ),
        "phase_a_checkpoint_sha256": sha256_file(
            args.phase_a_checkpoint.resolve()
        ),
        "v10_report_sha256": sha256_file(args.v10_report.resolve()),
    }
    run = amendment.get("authorized_run", {})
    if (
        amendment.get("schema") != "orion.scenario_factory.amendment.v1"
        or amendment.get("status") != "immutable_evaluation_only_authorization"
        or amendment.get("validated_inputs") != actual
        or amendment.get("protocol_sha256") != sha256_file(args.protocol.resolve())
        or amendment.get("preflight_sha256") != sha256_file(args.preflight.resolve())
        or run.get("maximum_submissions") != 1
        or run.get("optimizer_steps") != 0
        or run.get("automatic_retry") is not False
    ):
        raise ValueError("replay launch inputs or locks differ from amendment")
    if not str(args.job_id).isdigit():
        raise ValueError("Slurm job id must be numeric")
    result = {
        "schema": SCHEMA,
        "status": "single_evaluation_only_submission_attested",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_id": str(args.job_id),
        "job_name": "s2l_v10_replay",
        "remote_log": str(args.remote_log.resolve()),
        "authorized_output_root": run["output_root"],
        "optimizer_steps": 0,
        "checkpoint_update": False,
        "automatic_retry": False,
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
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
