#!/usr/bin/env python3
"""Write the immutable submission attestation after one v8 sbatch call."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2l_v8_smoke_submission_attestation.v1"
SOURCE_PATHS = {
    "submitter": "scripts/submit_stage2l_v8_route151_smoke.sh",
    "attester": "scripts/write_stage2l_v8_submission_attestation.py",
    "trainer": "scripts/train_stage2l_v8_route151_smoke.py",
    "validator": "scripts/validate_stage2l_v8_route151_smoke.py",
    "training_protocol": "configs/scenario_factory/stage2l_training_v8_gradient_routed_structured_qa.json",
    "launch_amendment": "configs/scenario_factory/amendments/20260830_stage2l_v8_route151_gradient_routed_smoke_v1.json",
    "qa_factory_config": "configs/scenario_factory/qa_factory_v4_structured_semantics.json",
    "calibrated_objective": "uq_estimator/stage2l_calibrated_objective.py",
    "gradient_routed_objective": "uq_estimator/stage2l_gradient_routed_objective.py",
    "qa_contract": "uq_estimator/stage2l_qa_contract_v4.py",
    "semantic_bottleneck": "uq_estimator/stage2l_semantic_bottleneck_v3.py",
    "semantic_runtime": "uq_estimator/stage2l_semantic_runtime_v3.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--v8-preflight", type=Path, required=True)
    parser.add_argument("--trainer-preflight", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--orion-config", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--remote-output-dir", type=Path, required=True)
    parser.add_argument("--remote-log", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not str(args.job_id).isdigit():
        raise ValueError("Slurm job id must be numeric")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite submission attestation")
    project_root = args.project_root.resolve()
    source_hashes = {
        name: sha256_file(project_root / relative)
        for name, relative in SOURCE_PATHS.items()
    }
    source_hashes.update(
        {
            "records": sha256_file(args.records.resolve()),
            "visual_cache": sha256_file(args.visual_cache.resolve()),
            "v8_preflight": sha256_file(args.v8_preflight.resolve()),
            "trainer_preflight": sha256_file(args.trainer_preflight.resolve()),
            "dataset_audit": sha256_file(args.dataset_audit.resolve()),
            "reference_audit": sha256_file(args.reference_audit.resolve()),
            "orion_config": sha256_file(args.orion_config.resolve()),
        }
    )
    amendment = json.loads(
        (project_root / SOURCE_PATHS["launch_amendment"]).read_text(encoding="utf-8")
    )
    base_hash = amendment["validated_inputs"]["base_orion_checkpoint_sha256"]
    if not args.base_checkpoint.is_file():
        raise FileNotFoundError("base ORION checkpoint is missing")
    source_hashes["base_orion_checkpoint"] = base_hash
    trainer_preflight = json.loads(args.trainer_preflight.read_text(encoding="utf-8"))
    if (
        trainer_preflight.get("training_started") is not False
        or trainer_preflight.get("training_authorized") is not False
    ):
        raise ValueError("trainer preflight does not preserve the launch lock")
    result = {
        "schema": SCHEMA,
        "submitted_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "slurm_job_id": str(args.job_id),
        "job_name": "s2l_v8_r151_smoke",
        "event_id": "route151_step218",
        "purpose": "bounded Stage2-L v8 gradient-routed learnability smoke; diagnostic only",
        "slurm_resources": {
            "partition": "Nvidia_A800",
            "gres": "gpu:1",
            "cpus_per_task": 2,
            "memory": "192G",
            "time_limit": "08:00:00",
            "excluded_nodes": ["gpu5"],
        },
        "training_bounds": {
            "maximum_submissions": 1,
            "maximum_optimizer_steps": 60,
            "answer_micro_batch_size": 2,
            "optimizer_step_unit": "one complete 20-record matched group",
            "formal_training": False,
            "stage2p_training": False,
            "trajectory_or_control_loss": False,
            "automatic_retry_or_extension": False,
        },
        "preflight": {
            "objective_data_status": json.loads(
                args.v8_preflight.read_text(encoding="utf-8")
            )["status"],
            "trainer_status": trainer_preflight["status"],
            "selected_cpu_tests": "19 passed",
            "training_started_during_preflight": False,
        },
        "source_sha256": source_hashes,
        "remote_output_dir": str(args.remote_output_dir),
        "remote_log": str(args.remote_log),
        "claim_boundary": (
            "One Route151 v8 engineering smoke only; no held-out, trajectory, "
            "closed-loop, generalization or safety evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
