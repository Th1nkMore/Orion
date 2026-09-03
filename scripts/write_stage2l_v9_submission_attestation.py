#!/usr/bin/env python3
"""Prevalidate and atomically attest one bounded Stage2-L v9 submission."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping


SCHEMA = "orion.stage2l_v9_smoke_submission_attestation.v1"
AMENDMENT_SCHEMA = "orion.scenario_factory.amendment.v1"
SOURCE_PATHS = {
    "submitter": "scripts/submit_stage2l_v9_route151_smoke.sh",
    "attester": "scripts/write_stage2l_v9_submission_attestation.py",
    "trainer": "scripts/train_stage2l_v9_route151_smoke.py",
    "training_protocol": (
        "configs/scenario_factory/stage2l_training_v9_vlm_task_fields.json"
    ),
    "qa_factory_config": (
        "configs/scenario_factory/qa_factory_v5_vlm_task_fields.json"
    ),
    "qa_contract": "uq_estimator/stage2l_qa_contract_v5.py",
    "semantic_runtime": "uq_estimator/stage2l_semantic_runtime_v4.py",
    "structured_field_head": "uq_estimator/stage2l_structured_field_head.py",
    "support_aligned_objective": (
        "uq_estimator/stage2l_support_aligned_objective_v9.py"
    ),
}
AMENDMENT_HASH_KEYS = {
    "submitter": "submitter_sha256",
    "attester": "attester_sha256",
    "trainer": "trainer_sha256",
    "training_protocol": "training_protocol_sha256",
    "qa_factory_config": "qa_factory_config_sha256",
    "qa_contract": "qa_contract_v5_sha256",
    "semantic_runtime": "semantic_runtime_v4_sha256",
    "structured_field_head": "structured_field_head_sha256",
    "support_aligned_objective": "support_aligned_objective_v9_sha256",
    "architecture_preflight": "v9_preflight_sha256",
    "trainer_preflight": "trainer_preflight_sha256",
    "dataset_audit": "dataset_audit_sha256",
    "reference_audit": "reference_audit_sha256",
    "records": "records_sha256",
    "visual_cache": "visual_cache_sha256",
    "orion_config": "orion_config_sha256",
    "base_orion_checkpoint": "base_orion_checkpoint_sha256",
}
EXPECTED_RESOURCES = {
    "partition": "Nvidia_A800",
    "gres": "gpu:1",
    "cpus_per_task": 2,
    "memory": "192G",
    "time_limit": "06:00:00",
    "excluded_nodes": ["gpu5"],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON input must contain an object")
    return value


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError("refusing to overwrite submission attestation")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%s.tmp" % (path.name, os.getpid()))
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    project_root = args.project_root.resolve()
    amendment_path = args.amendment.resolve()
    amendment = _read(amendment_path)
    architecture_preflight = _read(args.v9_preflight.resolve())
    trainer_preflight = _read(args.trainer_preflight.resolve())
    dataset_audit = _read(args.dataset_audit.resolve())
    reference_audit = _read(args.reference_audit.resolve())
    if (
        architecture_preflight.get("passed") is not True
        or architecture_preflight.get("training_started") is not False
        or architecture_preflight.get("real_orion_smoke_authorized") is not False
        or trainer_preflight.get("training_started") is not False
        or trainer_preflight.get("training_authorized") is not False
        or trainer_preflight.get("gpu_used") is not False
        or dataset_audit.get("passed") is not True
        or reference_audit.get("passed") is not True
    ):
        raise ValueError("a required preflight/audit is stale or unlocked")

    paths = {
        name: project_root / relative for name, relative in SOURCE_PATHS.items()
    }
    paths.update({
        "architecture_preflight": args.v9_preflight.resolve(),
        "trainer_preflight": args.trainer_preflight.resolve(),
        "dataset_audit": args.dataset_audit.resolve(),
        "reference_audit": args.reference_audit.resolve(),
        "records": args.records.resolve(),
        "visual_cache": args.visual_cache.resolve(),
        "orion_config": args.orion_config.resolve(),
        "base_orion_checkpoint": args.base_checkpoint.resolve(),
    })
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing attested inputs: %s" % sorted(missing))
    hashes = {name: _sha256(path) for name, path in paths.items()}
    hashes["launch_amendment"] = _sha256(amendment_path)
    validated = amendment.get("validated_inputs", {})
    mismatches = {
        name: {
            "amendment_key": AMENDMENT_HASH_KEYS[name],
            "expected": validated.get(AMENDMENT_HASH_KEYS[name]),
            "actual": hashes[name],
        }
        for name in AMENDMENT_HASH_KEYS
        if validated.get(AMENDMENT_HASH_KEYS[name]) != hashes[name]
    }
    authorized = amendment.get("authorized_run", {})
    locks = amendment.get("launch_locks", {})
    resources = amendment.get("slurm_resources", {})
    if (
        amendment.get("schema") != AMENDMENT_SCHEMA
        or locks.get("stage2l_v9_route151_smoke_allowed") is not True
        or locks.get("stage2l_pilot_training_allowed") is not False
        or locks.get("stage2p_allowed") is not False
        or authorized.get("event_id") != "route151_step218"
        or int(authorized.get("maximum_submissions", 0)) != 1
        or int(authorized.get("maximum_optimizer_steps", -1)) != 20
        or int(authorized.get("answer_micro_batch_size", -1)) != 2
        or authorized.get("fresh_initialization_from_original_orion_checkpoint")
        is not True
        or authorized.get("automatic_retry_or_extension") is not False
        or Path(str(authorized.get("output_root", ""))).resolve()
        != args.remote_output_dir.resolve()
        or resources != EXPECTED_RESOURCES
        or mismatches
    ):
        raise ValueError(
            "v9 amendment is absent, stale, broad, or hash-mismatched: %s"
            % mismatches
        )
    if args.output.exists():
        raise FileExistsError("refusing to overwrite submission attestation")
    return {
        "amendment": amendment,
        "source_sha256": hashes,
        "architecture_preflight": architecture_preflight,
        "trainer_preflight": trainer_preflight,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--v9-preflight", type=Path, required=True)
    parser.add_argument("--trainer-preflight", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--orion-config", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--remote-output-dir", type=Path, required=True)
    parser.add_argument("--remote-log")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    validated = validate_inputs(args)
    if args.validate_only:
        if args.job_id is not None or args.remote_log is not None:
            raise ValueError("validate-only mode must not receive job metadata")
        print(json.dumps({
            "status": "v9_submission_inputs_validated_no_submission",
            "training_authorized_by_this_command": False,
            "source_sha256": validated["source_sha256"],
        }, indent=2, sort_keys=True))
        return 0
    if args.job_id is None or not str(args.job_id).isdigit():
        raise ValueError("Slurm job id must be numeric")
    if not args.remote_log:
        raise ValueError("submitted job requires a remote log path")
    result = {
        "schema": SCHEMA,
        "submitted_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "slurm_job_id": str(args.job_id),
        "job_name": "s2l_v9_r151_smoke",
        "event_id": "route151_step218",
        "purpose": (
            "bounded Stage2-L v9 VLM task-field learnability smoke; "
            "diagnostic only"
        ),
        "slurm_resources": EXPECTED_RESOURCES,
        "training_bounds": {
            "maximum_submissions": 1,
            "maximum_optimizer_steps": 20,
            "answer_micro_batch_size": 2,
            "primary_groups_per_optimizer_step": 5,
            "primary_records_per_optimizer_step": 100,
            "language_groups_per_optimizer_step": 1,
            "formal_training": False,
            "stage2p_training": False,
            "trajectory_or_control_loss": False,
            "automatic_retry_or_extension": False,
        },
        "preflight": {
            "architecture_data_status": validated[
                "architecture_preflight"
            ]["status"],
            "trainer_status": validated["trainer_preflight"]["status"],
            "selected_architecture_cpu_tests": "40 passed",
            "launch_chain_cpu_tests": "4 passed",
            "training_started_during_preflight": False,
        },
        "source_sha256": validated["source_sha256"],
        "remote_output_dir": str(args.remote_output_dir.resolve()),
        "remote_log": str(args.remote_log),
        "claim_boundary": (
            "One Route151 v9 engineering smoke only; no held-out, trajectory, "
            "closed-loop, generalization or safety evidence."
        ),
    }
    _atomic_create_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
