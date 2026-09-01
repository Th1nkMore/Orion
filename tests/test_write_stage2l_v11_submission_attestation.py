import argparse
import json
import sys

from scripts.scenario_factory_lib import sha256_file
from scripts.write_stage2l_v11_submission_attestation import (
    _validated_inputs,
    main,
)


def test_submission_attestation_binds_exact_v11_lineage(tmp_path, monkeypatch):
    input_names = (
        "trainer",
        "runtime",
        "identifiability_audit",
        "parent_contract",
        "dataset_manifest",
        "v11_records",
        "dataset_audit_report",
        "view_feature_cache",
        "u_tokenizer_checkpoint",
        "v101_checkpoint",
        "v101_report",
        "orion_config",
        "orion_checkpoint",
    )
    paths = {}
    for name in input_names:
        path = tmp_path / (name + ".bin")
        path.write_bytes(name.encode())
        paths[name] = path
    actual = _validated_inputs(argparse.Namespace(**paths))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "schema": "orion.stage2l_v11_identifiable_smoke_protocol.v1"
    }))
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({
        "schema": "orion.stage2l_v11_identifiable_smoke_preflight.v1",
        "passed": True,
        "training_started": False,
        "validated_inputs": actual,
    }))
    output_root = tmp_path / "training"
    amendment = tmp_path / "amendment.json"
    amendment.write_text(json.dumps({
        "schema": "orion.scenario_factory.amendment.v1",
        "status": "immutable_v11_identifiability_smoke_authorization",
        "validated_inputs": actual,
        "protocol_sha256": sha256_file(protocol),
        "preflight_sha256": sha256_file(preflight),
        "authorized_run": {
            "output_root": str(output_root.resolve()),
            "maximum_submissions": 1,
            "automatic_retry": False,
            "optimizer_steps": 40,
        },
        "launch_locks": {
            "stage2l_v11_bounded_smoke_allowed": True,
            "formal_stage2l_allowed": False,
            "stage2p_allowed": False,
            "closed_loop_allowed": False,
        },
    }))
    remote_log = tmp_path / "job.out"
    output = tmp_path / "submission.json"
    argv = ["attester", "--job-id", "12345"]
    for name, path in paths.items():
        argv.extend(("--" + name.replace("_", "-"), str(path)))
    argv.extend((
        "--protocol", str(protocol),
        "--preflight", str(preflight),
        "--amendment", str(amendment),
        "--remote-log", str(remote_log),
        "--output-root", str(output_root),
        "--output", str(output),
    ))
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 0
    value = json.loads(output.read_text())
    assert value["job_id"] == "12345"
    assert value["validated_inputs"] == actual
    assert value["only_trainable_module"] == "TaskRiskLanguageBridge"
