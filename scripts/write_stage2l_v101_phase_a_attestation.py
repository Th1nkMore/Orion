#!/usr/bin/env python3
"""Write the hash-bound submission record for one v10.1 Phase-A-only job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2l_v101_phase_a_submission.v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--query-module", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--view-feature-cache", type=Path, required=True)
    parser.add_argument("--orion-config", type=Path, required=True)
    parser.add_argument("--orion-checkpoint", type=Path, required=True)
    parser.add_argument("--v10-phase-a-checkpoint", type=Path, required=True)
    parser.add_argument("--v10-report", type=Path, required=True)
    parser.add_argument("--remote-log", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not str(args.job_id).isdigit():
        raise ValueError("v10.1 submission attestation target/job id is invalid")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    amendment = json.loads(args.amendment.read_text(encoding="utf-8"))
    validated = {
        "trainer_sha256": sha256_file(args.trainer.resolve()),
        "query_module_sha256": sha256_file(args.query_module.resolve()),
        "dataset_manifest_sha256": sha256_file(args.dataset_manifest.resolve()),
        "view_feature_cache_sha256": sha256_file(args.view_feature_cache.resolve()),
        "orion_config_sha256": sha256_file(args.orion_config.resolve()),
        "orion_checkpoint_sha256": sha256_file(args.orion_checkpoint.resolve()),
        "v10_phase_a_checkpoint_sha256": sha256_file(
            args.v10_phase_a_checkpoint.resolve()
        ),
        "v10_report_sha256": sha256_file(args.v10_report.resolve()),
    }
    if (
        protocol.get("validated_inputs") != validated
        or preflight.get("validated_inputs") != validated
        or amendment.get("validated_inputs") != validated
        or preflight.get("passed") is not True
        or amendment.get("protocol_sha256") != sha256_file(args.protocol.resolve())
        or amendment.get("preflight_sha256") != sha256_file(args.preflight.resolve())
        or amendment.get("authorized_run", {}).get("output_root")
        != str(args.output_root.resolve())
    ):
        raise ValueError("v10.1 submission inputs differ from frozen lineage")
    value = {
        "schema": SCHEMA,
        "status": "submitted_phase_a_only_job_attested",
        "job_id": str(args.job_id),
        "remote_log": str(args.remote_log),
        "output_root": str(args.output_root.resolve()),
        "validated_inputs": validated,
        "protocol_sha256": sha256_file(args.protocol.resolve()),
        "preflight_sha256": sha256_file(args.preflight.resolve()),
        "amendment_sha256": sha256_file(args.amendment.resolve()),
        "maximum_optimizer_steps": 120,
        "stage1_uq_loaded": False,
        "phase_b": False,
        "phase_c": False,
        "formal_stage2l": False,
        "stage2p": False,
        "closed_loop": False,
        "route203_native_glare_submission": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
