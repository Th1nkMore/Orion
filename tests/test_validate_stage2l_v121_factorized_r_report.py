import json
from pathlib import Path

import pytest
import torch

from scripts.scenario_factory_lib import sha256_file
from scripts.validate_stage2l_v121_factorized_r_report import (
    REPORT_SCHEMA,
    recompute_factorized_checks,
    validate_report,
)


def _json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _bytes(path: Path, value: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _gates():
    return {
        "train_min_supported_macro_recall": 0.8,
        "dev_min_route_front_recall": 0.62,
        "dev_min_actor_front_recall": 0.4,
        "dev_min_actor_nonfront_macro_recall": 0.35,
        "dev_min_each_actor_nonfront_recall": 0.05,
        "dev_max_mean_background_fpr": 0.1,
        "baseline_dev_actor_nonfront_macro_recall": 0.09806547619047619,
        "minimum_actor_nonfront_absolute_improvement": 0.25,
        "background_fpr_cells": [
            "route/CAM_FRONT",
            "actor/CAM_FRONT",
            "actor/CAM_FRONT_LEFT",
            "actor/CAM_FRONT_RIGHT",
            "actor/CAM_BACK",
            "actor/CAM_BACK_LEFT",
        ],
    }


def _split_metrics(prefix, count, *, macro, nonfront, route=0.7, actor=0.5):
    views = (
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    )
    cells = {}
    for component in ("route", "actor"):
        cells[component] = {}
        for view in views:
            if component == "route" and view == "CAM_FRONT":
                recall = route
            elif component == "actor" and view == "CAM_FRONT":
                recall = actor
            elif component == "actor" and view in (
                "CAM_FRONT_LEFT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
            ):
                recall = nonfront
            else:
                recall = None
            cells[component][view] = {
                "mean_group_foreground_recall": recall,
                "mean_group_background_false_positive_rate": 0.01,
            }
    return {
        "group_count": count,
        "supported_component_view_macro_recall": macro,
        "actor_nonfront_macro_recall": nonfront,
        "per_component_view": cells,
        "per_group": {"%s_%02d" % (prefix, index): {} for index in range(count)},
    }


def _metrics(*, passed):
    nonfront = 0.5 if passed else 0.0
    return {
        "train": _split_metrics("train", 60, macro=0.9, nonfront=0.6),
        "dev": _split_metrics("dev", 20, macro=0.5, nonfront=nonfront),
    }


def _fixture(tmp_path, *, passed):
    project = tmp_path / "project"
    validator = _bytes(
        project / "scripts/validate_stage2l_v121_factorized_r_report.py",
        b"validator\n",
    )
    trainer = _bytes(
        project / "scripts/train_stage2l_v121_factorized_r_smoke.py",
        b"trainer\n",
    )
    module = _bytes(
        project / "uq_estimator/stage2l_factorized_relevance_v121.py",
        b"factorized\n",
    )
    input_files = {
        "dataset_manifest_sha256": _bytes(tmp_path / "data/manifest.json", b"manifest"),
        "view_feature_cache_sha256": _bytes(tmp_path / "data/features.pt", b"features"),
        "v101_checkpoint_sha256": _bytes(tmp_path / "data/v101.pt", b"v101"),
        "v101_report_sha256": _bytes(tmp_path / "data/v101.json", b"report"),
        "factorized_cpu_report_sha256": _bytes(tmp_path / "data/cpu.json", b"cpu"),
        "orion_config_sha256": _bytes(tmp_path / "data/orion.py", b"config"),
        "orion_checkpoint_sha256": _bytes(tmp_path / "data/orion.pt", b"checkpoint"),
        "trainer_sha256": trainer,
        "factorized_module_sha256": module,
    }
    validated = {
        name: sha256_file(path) for name, path in input_files.items()
    }
    output_root = tmp_path / "run/training"
    output_root.mkdir(parents=True)
    protocol = _json(
        project / "configs/protocol.json",
        {
            "schema": "orion.stage2l_v12_1_factorized_r_smoke_protocol.v1",
            "validated_inputs": validated,
            "output_root": str(output_root.resolve()),
            "input_paths": {
                "dataset_manifest": str(input_files["dataset_manifest_sha256"]),
                "view_feature_cache": str(input_files["view_feature_cache_sha256"]),
                "v101_checkpoint": str(input_files["v101_checkpoint_sha256"]),
                "v101_report": str(input_files["v101_report_sha256"]),
                "factorized_cpu_report": str(input_files["factorized_cpu_report_sha256"]),
                "orion_config": str(input_files["orion_config_sha256"]),
                "orion_checkpoint": str(input_files["orion_checkpoint_sha256"]),
            },
            "engineering_gates": _gates(),
        },
    )
    preflight = _json(
        tmp_path / "run/trainer_preflight.json",
        {
            "schema": "orion.stage2l_v12_1_factorized_r_smoke_preflight.v1",
            "validated_inputs": validated,
            "protocol_sha256": sha256_file(protocol),
        },
    )
    launch = _json(
        project / "configs/launch.json",
        {
            "schema": "orion.stage2l_v12_1_factorized_r_smoke_launch.v1",
            "validated_inputs": validated,
            "protocol_sha256": sha256_file(protocol),
            "preflight_sha256": sha256_file(preflight),
            "authorized_run": {"output_root": str(output_root.resolve())},
        },
    )
    attestation = _json(
        tmp_path / "run/submission_attestation.json",
        {
            "schema": "orion.stage2l_v12_1_factorized_r_submission_attestation.v1",
            "validated_inputs": validated,
            "authorized_output_root": str(output_root.resolve()),
            "protocol": {"sha256": sha256_file(protocol)},
            "preflight": {"sha256": sha256_file(preflight)},
            "launch_amendment": {"sha256": sha256_file(launch)},
        },
    )
    audit = _json(
        project / "configs/audit.json",
        {
            "schema": "orion.stage2l_v12_1_factorized_r_terminal_audit.v1",
            "status": "frozen_outcome_blind_terminal_audit",
            "validated_lineage": {
                "training_protocol_sha256": sha256_file(protocol),
                "trainer_preflight_sha256": sha256_file(preflight),
                "launch_amendment_sha256": sha256_file(launch),
                "submission_attestation_sha256": sha256_file(attestation),
            },
            "implementation_hashes": {
                "validator_sha256": sha256_file(validator),
                "trainer_sha256": sha256_file(trainer),
                "factorized_module_sha256": sha256_file(module),
            },
        },
    )
    metrics = _metrics(passed=passed)
    checks = recompute_factorized_checks(metrics, _gates())
    assert all(checks.values()) is passed
    steps = 20 if passed else 40
    evaluation_steps = [20] if passed else [20, 40]
    warm = {"source": "v101"}
    provenance = {
        "validated_inputs": validated,
        "protocol_sha256": sha256_file(protocol),
        "preflight_sha256": sha256_file(preflight),
        "launch_amendment_sha256": sha256_file(launch),
        "warm_start": warm,
    }
    evaluations = []
    group_ids = {
        "train_%02d" % index for index in range(60)
    } | {"dev_%02d" % index for index in range(20)}
    component_probability = torch.full((1, 2, 6, 10, 10), 0.2)
    component_target = torch.zeros((1, 2, 6, 10, 10))
    for step in evaluation_steps:
        evaluations.append(
            {
                "optimizer_step": step,
                "metrics": metrics,
                "checks": checks,
                "passed": passed,
            }
        )
        torch.save(
            {
                "schema": REPORT_SCHEMA,
                "status": (
                    "factorized_r_gate_pass"
                    if passed
                    else "factorized_r_gate_failed"
                ),
                "optimizer_steps": step,
                "lora": {"lora_%03d" % i: torch.zeros(1) for i in range(256)},
                "view_aligned_relevance_queries": {
                    "query_%02d" % i: torch.zeros(1) for i in range(14)
                },
                "factorized_relevance_head": {
                    "head_%02d" % i: torch.zeros(1) for i in range(8)
                },
                "stage1_uq_loaded": False,
                "u_tokenizer_loaded": False,
                "language_training_used": False,
                "trajectory_or_control_loss_used": False,
                "provenance": provenance,
            },
            output_root / ("factorized_r_step%03d.pt" % step),
        )
        derived = torch.maximum(
            component_probability[:, 0], component_probability[:, 1]
        )
        maximum_target = torch.maximum(
            component_target[:, 0], component_target[:, 1]
        )
        spatial = {
            group_id: {
                "component_probability": component_probability,
                "component_target": component_target,
                "derived_union_probability": derived,
                "pooled_raw_union_target": maximum_target,
                "max_pooled_component_target": maximum_target,
            }
            for group_id in group_ids
        }
        torch.save(spatial, output_root / ("spatial_maps_step%03d.pt" % step))
    report = _json(
        output_root / "report.json",
        {
            "schema": REPORT_SCHEMA,
            "status": (
                "factorized_r_engineering_gate_passed"
                if passed
                else "factorized_r_stopped_without_gate_pass"
            ),
            "optimizer_steps": steps,
            "stop_reason": (
                "factorized_r_engineering_gate_passed_early"
                if passed
                else "maximum_steps_reached"
            ),
            "before": {"train": metrics["train"], "dev": metrics["dev"]},
            "history": [
                {
                    "optimizer_step": step,
                    "primary_group_ids": ["train_%02d" % i for i in range(13)],
                    "primary_event_ids": ["event_%02d" % i for i in range(13)],
                    "loss": 0.1,
                    "gradient_norm_before_clip": 0.2,
                    "finite": True,
                }
                for step in range(1, steps + 1)
            ],
            "evaluations": evaluations,
            "final_metrics": metrics,
            "final_checks": checks,
            "warm_start": warm,
            "provenance": provenance,
            "locks": {
                "stage1_uq_loaded": False,
                "u_tokenizer_loaded": False,
                "language_training_used": False,
                "trajectory_or_control_loss_used": False,
                "locked_test_read": False,
                "formal_stage2l_ready": False,
                "stage2p_ready": False,
            },
        },
    )
    return {
        "report": report,
        "attestation": attestation,
        "protocol": protocol,
        "preflight": preflight,
        "launch": launch,
        "audit": audit,
        "output_root": output_root,
        "project": project,
    }


@pytest.mark.parametrize("passed", [True, False])
def test_independently_validates_honest_pass_and_failure(tmp_path, passed):
    fixture = _fixture(tmp_path, passed=passed)
    result = validate_report(
        report_path=fixture["report"],
        submission_attestation_path=fixture["attestation"],
        training_protocol_path=fixture["protocol"],
        trainer_preflight_path=fixture["preflight"],
        launch_amendment_path=fixture["launch"],
        audit_protocol_path=fixture["audit"],
        output_root=fixture["output_root"],
        project_root=fixture["project"],
    )
    assert result["integrity_valid"] is True
    assert result["smoke_passed"] is passed
    assert result["status"] == ("validated_pass" if passed else "validated_failed_gate")
    assert result["optimizer_steps"] == (20 if passed else 40)


def test_rejects_trainer_check_disagreement(tmp_path):
    fixture = _fixture(tmp_path, passed=True)
    value = json.loads(fixture["report"].read_text())
    value["evaluations"][0]["checks"]["dev_actor_front"] = False
    fixture["report"].write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="independent recomputation"):
        validate_report(
            report_path=fixture["report"],
            submission_attestation_path=fixture["attestation"],
            training_protocol_path=fixture["protocol"],
            trainer_preflight_path=fixture["preflight"],
            launch_amendment_path=fixture["launch"],
            audit_protocol_path=fixture["audit"],
            output_root=fixture["output_root"],
            project_root=fixture["project"],
        )


def test_rejects_unexpected_terminal_artifact(tmp_path):
    fixture = _fixture(tmp_path, passed=True)
    (fixture["output_root"] / "unregistered.pt").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="unexpected files"):
        validate_report(
            report_path=fixture["report"],
            submission_attestation_path=fixture["attestation"],
            training_protocol_path=fixture["protocol"],
            trainer_preflight_path=fixture["preflight"],
            launch_amendment_path=fixture["launch"],
            audit_protocol_path=fixture["audit"],
            output_root=fixture["output_root"],
            project_root=fixture["project"],
        )
