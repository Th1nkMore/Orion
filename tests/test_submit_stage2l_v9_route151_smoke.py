import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


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
HASH_KEYS = {
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


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _attester_command(paths, *extra):
    return [
        sys.executable,
        str(ROOT / "scripts/write_stage2l_v9_submission_attestation.py"),
        "--project-root",
        str(paths["project"]),
        "--amendment",
        str(paths["amendment"]),
        "--records",
        str(paths["records"]),
        "--visual-cache",
        str(paths["visual_cache"]),
        "--v9-preflight",
        str(paths["architecture_preflight"]),
        "--trainer-preflight",
        str(paths["trainer_preflight"]),
        "--dataset-audit",
        str(paths["dataset_audit"]),
        "--reference-audit",
        str(paths["reference_audit"]),
        "--orion-config",
        str(paths["orion_config"]),
        "--base-checkpoint",
        str(paths["base_orion_checkpoint"]),
        "--remote-output-dir",
        str(paths["remote_output_dir"]),
        "--output",
        str(paths["output"]),
        *extra,
    ]


def _make_valid_attester_fixture(tmp_path):
    project = tmp_path / "project"
    paths = {"project": project}
    for name, relative in SOURCE_PATHS.items():
        paths[name] = _write(project / relative, "%s\n" % name)
    paths.update({
        "architecture_preflight": _write(
            tmp_path / "architecture_preflight.json",
            json.dumps({
                "passed": True,
                "training_started": False,
                "real_orion_smoke_authorized": False,
                "status": "passed_preflight_only",
            }),
        ),
        "trainer_preflight": _write(
            tmp_path / "trainer_preflight.json",
            json.dumps({
                "training_started": False,
                "training_authorized": False,
                "gpu_used": False,
                "status": "passed_preflight_only",
            }),
        ),
        "dataset_audit": _write(
            tmp_path / "audit.json", json.dumps({"passed": True})
        ),
        "reference_audit": _write(
            tmp_path / "reference_audit.json", json.dumps({"passed": True})
        ),
        "records": _write(tmp_path / "records.jsonl", "{}\n"),
        "visual_cache": _write(tmp_path / "visual_cache.pt", "cache\n"),
        "orion_config": _write(tmp_path / "orion_config.py", "config\n"),
        "base_orion_checkpoint": _write(
            tmp_path / "Orion.pth", "checkpoint\n"
        ),
        "remote_output_dir": tmp_path / "training",
        "output": tmp_path / "submission_attestation.json",
    })
    hashes = {name: _sha256(path) for name, path in paths.items()
              if name in HASH_KEYS}
    amendment = {
        "schema": "orion.scenario_factory.amendment.v1",
        "launch_locks": {
            "stage2l_v9_route151_smoke_allowed": True,
            "stage2l_pilot_training_allowed": False,
            "stage2p_allowed": False,
        },
        "authorized_run": {
            "event_id": "route151_step218",
            "maximum_submissions": 1,
            "maximum_optimizer_steps": 20,
            "answer_micro_batch_size": 2,
            "fresh_initialization_from_original_orion_checkpoint": True,
            "automatic_retry_or_extension": False,
            "output_root": str(paths["remote_output_dir"]),
        },
        "slurm_resources": {
            "partition": "Nvidia_A800",
            "gres": "gpu:1",
            "cpus_per_task": 2,
            "memory": "192G",
            "time_limit": "06:00:00",
            "excluded_nodes": ["gpu5"],
        },
        "validated_inputs": {
            HASH_KEYS[name]: value for name, value in hashes.items()
        },
    }
    paths["amendment"] = _write(
        tmp_path / "amendment.json", json.dumps(amendment)
    )
    return paths


def test_v9_submitter_is_fixed_bounded_prevalidated_and_locked():
    source = (
        ROOT / "scripts/submit_stage2l_v9_route151_smoke.sh"
    ).read_text()
    assert "immutable launch amendment is absent" in source
    assert source.index("--validate-only") < source.index("sbatch --parsable")
    assert 'env "PYTHONPATH=${runtime_pythonpath}"' in source
    assert "--max-optimizer-steps 20" in source
    assert "--answer-batch-size 2" in source
    assert "--gres=gpu:1" in source
    assert "--cpus-per-task=2" in source
    assert "--mem=192G" in source
    assert "--time=06:00:00" in source
    assert "--exclude=gpu5" in source
    assert "refusing duplicate active v9 smoke submission" in source
    assert source.index("sbatch --parsable") < source.index(
        "trap cancel_unattested_submission EXIT INT TERM"
    )
    assert 'scancel "${unattested_job_id}" || true' in source
    assert source.index('unattested_job_id=""') < source.index(
        "trap - EXIT INT TERM"
    )
    assert "route151_v9_architecture_data_preflight_v4/preflight.json" in source
    assert "route151_v9_trainer_preflight_v3/preflight.json" in source


def test_v9_submitter_exits_before_sbatch_without_amendment():
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/submit_stage2l_v9_route151_smoke.sh")],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "immutable launch amendment is absent" in result.stderr
    assert result.stdout == ""


def test_v9_attester_has_atomic_no_overwrite_and_no_retry_contract():
    source = (
        ROOT / "scripts/write_stage2l_v9_submission_attestation.py"
    ).read_text()
    assert "os.link(temporary, path)" in source
    assert '"maximum_submissions": 1' in source
    assert '"maximum_optimizer_steps": 20' in source
    assert '"automatic_retry_or_extension": False' in source
    assert '"formal_training": False' in source
    assert '"stage2p_training": False' in source
    assert "validate-only mode must not receive job metadata" in source


def test_v9_attester_validate_only_then_atomic_create_and_refuse_overwrite(
    tmp_path,
):
    paths = _make_valid_attester_fixture(tmp_path)
    validation = subprocess.run(
        _attester_command(paths, "--validate-only"),
        text=True,
        capture_output=True,
        check=False,
    )
    assert validation.returncode == 0, validation.stderr
    assert "v9_submission_inputs_validated_no_submission" in validation.stdout
    assert not paths["output"].exists()

    first = subprocess.run(
        _attester_command(
            paths,
            "--job-id",
            "12345",
            "--remote-log",
            "/tmp/v9-12345.out",
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    attestation = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert attestation["slurm_job_id"] == "12345"
    assert attestation["training_bounds"]["maximum_submissions"] == 1
    assert not list(tmp_path.glob(".*.tmp"))

    second = subprocess.run(
        _attester_command(
            paths,
            "--job-id",
            "67890",
            "--remote-log",
            "/tmp/v9-67890.out",
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode != 0
    assert "refusing to overwrite submission attestation" in second.stderr
    unchanged = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert unchanged["slurm_job_id"] == "12345"
