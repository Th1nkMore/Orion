#!/usr/bin/env python3
"""Write the immutable attestation for the single Stage2-L v10 smoke."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2l_v10_submission_attestation.v1"


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
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--u-tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--orion-config", type=Path, required=True)
    parser.add_argument("--orion-checkpoint", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--remote-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite v10 submission attestation")
    amendment = _read(args.amendment.resolve())
    validated = amendment.get("validated_inputs", {})
    actual = {
        "trainer_sha256": sha256_file(args.trainer.resolve()),
        "protocol_sha256": sha256_file(args.protocol.resolve()),
        "preflight_sha256": sha256_file(args.preflight.resolve()),
        "dataset_manifest_sha256": sha256_file(args.dataset_manifest.resolve()),
        "u_tokenizer_checkpoint_sha256": sha256_file(
            args.u_tokenizer_checkpoint.resolve()
        ),
        "orion_config_sha256": sha256_file(args.orion_config.resolve()),
        "orion_checkpoint_sha256": sha256_file(args.orion_checkpoint.resolve()),
    }
    run = amendment.get("authorized_run", {})
    locks = amendment.get("launch_locks", {})
    if (
        amendment.get("schema") != "orion.scenario_factory.amendment.v1"
        or amendment.get("status") != "immutable_single_run_authorization"
        or validated != actual
        or run.get("maximum_submissions") != 1
        or run.get("automatic_retry") is not False
        or locks.get("stage2l_v10_bounded_smoke_allowed") is not True
        or locks.get("formal_stage2l_allowed") is not False
        or locks.get("stage2p_allowed") is not False
        or locks.get("closed_loop_allowed") is not False
    ):
        raise ValueError("v10 launch inputs or locks differ from amendment")
    if not str(args.job_id).isdigit():
        raise ValueError("Slurm job id must be numeric")
    result = {
        "schema": SCHEMA,
        "status": "single_gate_staged_submission_attested",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_id": str(args.job_id),
        "job_name": "s2l_v10_17e",
        "remote_log": str(args.remote_log.resolve()),
        "authorized_output_root": run["output_root"],
        "maximum_submissions": 1,
        "automatic_retry": False,
        "formal_stage2l": False,
        "stage2p": False,
        "closed_loop": False,
        "validated_inputs": actual,
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
