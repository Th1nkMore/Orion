#!/usr/bin/env python3
"""Write the immutable attestation for the single MR2 coverage submission."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2l_mr2_submission_attestation.v1"


def _read(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--trainer-preflight", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--remote-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite MR2 attestation")
    amendment = _read(args.amendment.resolve())
    validated = amendment.get("validated_inputs", {})
    run = amendment.get("authorized_run", {})
    if (
        amendment.get("schema") != "orion.scenario_factory.amendment.v1"
        or amendment.get("launch_locks", {}).get(
            "stage2l_mr2_bounded_preexperiment_allowed"
        )
        is not True
        or amendment.get("launch_locks", {}).get(
            "formal_stage2l_training_allowed"
        )
        is not False
        or amendment.get("launch_locks", {}).get("stage2p_allowed") is not False
        or int(run.get("maximum_submissions", 0)) != 1
        or int(run.get("maximum_optimizer_steps", 0)) != 40
        or int(run.get("dataset_event_count", 0)) != 17
        or run.get("automatic_retry_or_extension") is not False
        or run.get("formal_training") is not False
        or validated.get("trainer_sha256")
        != sha256_file(args.trainer.resolve())
        or validated.get("training_protocol_sha256")
        != sha256_file(args.training_protocol.resolve())
        or validated.get("dataset_manifest_sha256")
        != sha256_file(args.dataset_manifest.resolve())
        or validated.get("trainer_preflight_sha256")
        != sha256_file(args.trainer_preflight.resolve())
    ):
        raise ValueError("MR2 launch inputs differ from the amendment")
    if not str(args.job_id).isdigit():
        raise ValueError("Slurm job id must be numeric")
    result = {
        "schema": SCHEMA,
        "status": "single_bounded_submission_attested",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_id": str(args.job_id),
        "job_name": "s2l_mr2_17event",
        "remote_log": str(args.remote_log.resolve()),
        "authorized_output_root": run["output_root"],
        "maximum_optimizer_steps": 40,
        "automatic_retry_or_extension": False,
        "formal_training": False,
        "inputs": {
            "amendment": {
                "path": str(args.amendment.resolve()),
                "sha256": sha256_file(args.amendment.resolve()),
            },
            "trainer": {
                "path": str(args.trainer.resolve()),
                "sha256": sha256_file(args.trainer.resolve()),
            },
            "training_protocol": {
                "path": str(args.training_protocol.resolve()),
                "sha256": sha256_file(args.training_protocol.resolve()),
            },
            "dataset_manifest": {
                "path": str(args.dataset_manifest.resolve()),
                "sha256": sha256_file(args.dataset_manifest.resolve()),
            },
            "trainer_preflight": {
                "path": str(args.trainer_preflight.resolve()),
                "sha256": sha256_file(args.trainer_preflight.resolve()),
            },
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
