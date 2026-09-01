#!/usr/bin/env python3
"""Independently validate one terminal Stage2-L v12.1 factorized-R smoke."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

try:
    from scripts.scenario_factory_lib import sha256_file
except ModuleNotFoundError:
    from scenario_factory_lib import sha256_file


REPORT_SCHEMA = "orion.stage2l_v12_1_factorized_r_smoke.v1"
ATTESTATION_SCHEMA = (
    "orion.stage2l_v12_1_factorized_r_submission_attestation.v1"
)
PROTOCOL_SCHEMA = "orion.stage2l_v12_1_factorized_r_smoke_protocol.v1"
PREFLIGHT_SCHEMA = "orion.stage2l_v12_1_factorized_r_smoke_preflight.v1"
LAUNCH_SCHEMA = "orion.stage2l_v12_1_factorized_r_smoke_launch.v1"
AUDIT_PROTOCOL_SCHEMA = "orion.stage2l_v12_1_factorized_r_terminal_audit.v1"
VALIDATION_SCHEMA = "orion.stage2l_v12_1_factorized_r_validation.v1"
CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
COMPONENT_ORDER = ("route", "actor")
EXPECTED_SUPPORTED = (
    "route/CAM_FRONT",
    "actor/CAM_FRONT",
    "actor/CAM_FRONT_LEFT",
    "actor/CAM_FRONT_RIGHT",
    "actor/CAM_BACK",
    "actor/CAM_BACK_LEFT",
)
EXPECTED_LOCKS = {
    "stage1_uq_loaded": False,
    "u_tokenizer_loaded": False,
    "language_training_used": False,
    "trajectory_or_control_loss_used": False,
    "locked_test_read": False,
    "formal_stage2l_ready": False,
    "stage2p_ready": False,
}


def _read(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty terminal metric")
    return float(sum(values) / len(values))


def recompute_factorized_checks(
    metrics: Mapping[str, Any], gates: Mapping[str, Any]
) -> Dict[str, bool]:
    """Recompute the frozen gates without calling the training implementation."""

    train = metrics["train"]
    dev = metrics["dev"]
    actor = dev["per_component_view"]["actor"]
    nonfront_views = (
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
    )
    nonfront = [
        float(actor[view]["mean_group_foreground_recall"])
        for view in nonfront_views
    ]
    background = []
    for name in gates["background_fpr_cells"]:
        component, view = str(name).split("/", 1)
        background.append(
            float(
                dev["per_component_view"][component][view][
                    "mean_group_background_false_positive_rate"
                ]
            )
        )
    result = {
        "train_supported_macro_recall": float(
            train["supported_component_view_macro_recall"]
        )
        >= float(gates["train_min_supported_macro_recall"]),
        "dev_route_front_retained": float(
            dev["per_component_view"]["route"]["CAM_FRONT"][
                "mean_group_foreground_recall"
            ]
        )
        >= float(gates["dev_min_route_front_recall"]),
        "dev_actor_front": float(
            actor["CAM_FRONT"]["mean_group_foreground_recall"]
        )
        >= float(gates["dev_min_actor_front_recall"]),
        "dev_actor_nonfront_macro": float(
            dev["actor_nonfront_macro_recall"]
        )
        >= float(gates["dev_min_actor_nonfront_macro_recall"]),
        "dev_actor_nonfront_each_positive": min(nonfront)
        >= float(gates["dev_min_each_actor_nonfront_recall"]),
        "dev_background_fpr": max(background)
        <= float(gates["dev_max_mean_background_fpr"]),
        "dev_actor_nonfront_absolute_improvement": float(
            dev["actor_nonfront_macro_recall"]
        )
        - float(gates["baseline_dev_actor_nonfront_macro_recall"])
        >= float(gates["minimum_actor_nonfront_absolute_improvement"]),
    }
    if set(result) != {
        "train_supported_macro_recall",
        "dev_route_front_retained",
        "dev_actor_front",
        "dev_actor_nonfront_macro",
        "dev_actor_nonfront_each_positive",
        "dev_background_fpr",
        "dev_actor_nonfront_absolute_improvement",
    }:
        raise AssertionError("terminal gate implementation changed unexpectedly")
    return result


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError("factorized checkpoint is not a dictionary")
    return value


def _validate_spatial_maps(
    path: Path, expected_groups: set[str]
) -> Dict[str, Any]:
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict) or set(value) != expected_groups:
        raise ValueError("factorized spatial-map group set differs")
    maximum_union_error = 0.0
    maximum_component_target_error = 0.0
    for group_id, row in value.items():
        expected_keys = {
            "component_probability",
            "component_target",
            "derived_union_probability",
            "pooled_raw_union_target",
            "max_pooled_component_target",
        }
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise ValueError("factorized spatial-map fields differ: %s" % group_id)
        component_probability = row["component_probability"]
        component_target = row["component_target"]
        derived_union = row["derived_union_probability"]
        pooled_union = row["pooled_raw_union_target"]
        maximum_target = row["max_pooled_component_target"]
        if (
            tuple(component_probability.shape) != (1, 2, 6, 10, 10)
            or tuple(component_target.shape) != (1, 2, 6, 10, 10)
            or tuple(derived_union.shape) != (1, 6, 10, 10)
            or tuple(pooled_union.shape) != (1, 6, 10, 10)
            or tuple(maximum_target.shape) != (1, 6, 10, 10)
        ):
            raise ValueError("factorized spatial-map shape differs: %s" % group_id)
        tensors = (
            component_probability,
            component_target,
            derived_union,
            pooled_union,
            maximum_target,
        )
        if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise ValueError("factorized spatial-map tensor is non-finite")
        union_error = float(
            torch.max(
                torch.abs(
                    derived_union
                    - torch.maximum(
                        component_probability[:, 0], component_probability[:, 1]
                    )
                )
            ).item()
        )
        target_error = float(
            torch.max(
                torch.abs(
                    maximum_target
                    - torch.maximum(component_target[:, 0], component_target[:, 1])
                )
            ).item()
        )
        maximum_union_error = max(maximum_union_error, union_error)
        maximum_component_target_error = max(
            maximum_component_target_error, target_error
        )
    if maximum_union_error != 0.0 or maximum_component_target_error != 0.0:
        raise ValueError("factorized spatial-map derivation differs")
    return {
        "group_count": len(value),
        "maximum_derived_union_error": maximum_union_error,
        "maximum_component_target_error": maximum_component_target_error,
    }


def validate_report(
    *,
    report_path: Path,
    submission_attestation_path: Path,
    training_protocol_path: Path,
    trainer_preflight_path: Path,
    launch_amendment_path: Path,
    audit_protocol_path: Path,
    output_root: Path,
    project_root: Path,
) -> Dict[str, Any]:
    paths = [
        report_path,
        submission_attestation_path,
        training_protocol_path,
        trainer_preflight_path,
        launch_amendment_path,
        audit_protocol_path,
    ]
    if not all(path.resolve().is_file() for path in paths):
        raise FileNotFoundError("terminal factorized-R lineage is incomplete")
    report_path = report_path.resolve()
    submission_attestation_path = submission_attestation_path.resolve()
    training_protocol_path = training_protocol_path.resolve()
    trainer_preflight_path = trainer_preflight_path.resolve()
    launch_amendment_path = launch_amendment_path.resolve()
    audit_protocol_path = audit_protocol_path.resolve()
    output_root = output_root.resolve()
    project_root = project_root.resolve()

    report = _read(report_path)
    attestation = _read(submission_attestation_path)
    protocol = _read(training_protocol_path)
    preflight = _read(trainer_preflight_path)
    launch = _read(launch_amendment_path)
    audit = _read(audit_protocol_path)
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("factorized-R report schema differs")
    if attestation.get("schema") != ATTESTATION_SCHEMA:
        raise ValueError("factorized-R attestation schema differs")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("factorized-R protocol schema differs")
    if preflight.get("schema") != PREFLIGHT_SCHEMA:
        raise ValueError("factorized-R preflight schema differs")
    if launch.get("schema") != LAUNCH_SCHEMA:
        raise ValueError("factorized-R launch schema differs")
    if (
        audit.get("schema") != AUDIT_PROTOCOL_SCHEMA
        or audit.get("status") != "frozen_outcome_blind_terminal_audit"
    ):
        raise ValueError("factorized-R terminal audit protocol differs")

    lineage_paths = {
        "training_protocol_sha256": training_protocol_path,
        "trainer_preflight_sha256": trainer_preflight_path,
        "launch_amendment_sha256": launch_amendment_path,
        "submission_attestation_sha256": submission_attestation_path,
    }
    for name, path in lineage_paths.items():
        if audit["validated_lineage"].get(name) != sha256_file(path):
            raise ValueError("terminal audit lineage hash differs: %s" % name)
    implementation_paths = {
        "validator_sha256": project_root
        / "scripts/validate_stage2l_v121_factorized_r_report.py",
        "trainer_sha256": project_root
        / "scripts/train_stage2l_v121_factorized_r_smoke.py",
        "factorized_module_sha256": project_root
        / "uq_estimator/stage2l_factorized_relevance_v121.py",
    }
    for name, path in implementation_paths.items():
        if (
            not path.is_file()
            or audit["implementation_hashes"].get(name) != sha256_file(path)
        ):
            raise ValueError("terminal audit implementation differs: %s" % name)

    validated_inputs = protocol["validated_inputs"]
    if (
        preflight.get("validated_inputs") != validated_inputs
        or launch.get("validated_inputs") != validated_inputs
        or attestation.get("validated_inputs") != validated_inputs
        or report.get("provenance", {}).get("validated_inputs")
        != validated_inputs
    ):
        raise ValueError("terminal validated inputs differ")
    if (
        preflight.get("protocol_sha256") != sha256_file(training_protocol_path)
        or launch.get("protocol_sha256") != sha256_file(training_protocol_path)
        or launch.get("preflight_sha256") != sha256_file(trainer_preflight_path)
        or attestation.get("protocol", {}).get("sha256")
        != sha256_file(training_protocol_path)
        or attestation.get("preflight", {}).get("sha256")
        != sha256_file(trainer_preflight_path)
        or attestation.get("launch_amendment", {}).get("sha256")
        != sha256_file(launch_amendment_path)
    ):
        raise ValueError("terminal protocol/preflight/launch linkage differs")
    provenance = report["provenance"]
    if (
        provenance.get("protocol_sha256") != sha256_file(training_protocol_path)
        or provenance.get("preflight_sha256") != sha256_file(trainer_preflight_path)
        or provenance.get("launch_amendment_sha256")
        != sha256_file(launch_amendment_path)
        or report.get("warm_start") != provenance.get("warm_start")
    ):
        raise ValueError("terminal report provenance differs")

    input_paths = {
        "dataset_manifest_sha256": Path(protocol["input_paths"]["dataset_manifest"]),
        "view_feature_cache_sha256": Path(
            protocol["input_paths"]["view_feature_cache"]
        ),
        "v101_checkpoint_sha256": Path(protocol["input_paths"]["v101_checkpoint"]),
        "v101_report_sha256": Path(protocol["input_paths"]["v101_report"]),
        "factorized_cpu_report_sha256": Path(
            protocol["input_paths"]["factorized_cpu_report"]
        ),
        "orion_config_sha256": Path(protocol["input_paths"]["orion_config"]),
        "orion_checkpoint_sha256": Path(
            protocol["input_paths"]["orion_checkpoint"]
        ),
        "trainer_sha256": implementation_paths["trainer_sha256"],
        "factorized_module_sha256": implementation_paths[
            "factorized_module_sha256"
        ],
    }
    for name, path in input_paths.items():
        if not path.is_file() or sha256_file(path) != validated_inputs.get(name):
            raise ValueError("terminal frozen input hash differs: %s" % name)

    if (
        str(output_root) != protocol.get("output_root")
        or str(output_root) != launch.get("authorized_run", {}).get("output_root")
        or str(output_root)
        != attestation.get("authorized_output_root")
        or output_root != report_path.parent
    ):
        raise ValueError("terminal output root differs")
    history = report.get("history", [])
    steps = int(report.get("optimizer_steps", -1))
    if steps not in (20, 40) or len(history) != steps:
        raise ValueError("terminal optimizer-step count differs")
    if [int(row.get("optimizer_step", -1)) for row in history] != list(
        range(1, steps + 1)
    ):
        raise ValueError("terminal optimization history is incomplete")
    for row in history:
        if (
            row.get("finite") is not True
            or not math.isfinite(float(row.get("loss", math.nan)))
            or not math.isfinite(
                float(row.get("gradient_norm_before_clip", math.nan))
            )
            or len(row.get("primary_event_ids", [])) != 13
            or len(set(row.get("primary_event_ids", []))) != 13
        ):
            raise ValueError("terminal optimization history violates protocol")

    evaluations = report.get("evaluations", [])
    evaluation_steps = [int(row.get("optimizer_step", -1)) for row in evaluations]
    if evaluation_steps not in ([20], [20, 40]) or evaluation_steps[-1] != steps:
        raise ValueError("terminal evaluation milestones differ")
    gates = protocol["engineering_gates"]
    for evaluation in evaluations:
        recomputed = recompute_factorized_checks(evaluation["metrics"], gates)
        if (
            evaluation.get("checks") != recomputed
            or evaluation.get("passed") is not all(recomputed.values())
        ):
            raise ValueError("trainer gate differs from independent recomputation")
    final = evaluations[-1]
    passed = bool(final["passed"])
    checks = recompute_factorized_checks(final["metrics"], gates)
    if (
        report.get("final_metrics") != final.get("metrics")
        or report.get("final_checks") != checks
        or report.get("locks") != EXPECTED_LOCKS
    ):
        raise ValueError("terminal report summary or locks differ")
    expected_status = (
        "factorized_r_engineering_gate_passed"
        if passed
        else "factorized_r_stopped_without_gate_pass"
    )
    if report.get("status") != expected_status:
        raise ValueError("terminal report status differs")
    if passed:
        if report.get("stop_reason") != "factorized_r_engineering_gate_passed_early":
            raise ValueError("passing terminal stop reason differs")
    elif steps != 40 or report.get("stop_reason") not in (
        "maximum_steps_reached",
        "clear_factorized_train_dev_overfit_early_stop",
    ):
        raise ValueError("failed terminal stop reason differs")

    train_groups = set(final["metrics"]["train"]["per_group"])
    dev_groups = set(final["metrics"]["dev"]["per_group"])
    if len(train_groups) != 60 or len(dev_groups) != 20 or train_groups & dev_groups:
        raise ValueError("terminal train/dev group boundary differs")
    all_groups = train_groups | dev_groups
    expected_files = {"report.json"}
    artifact_hashes: Dict[str, str] = {"report.json": sha256_file(report_path)}
    spatial_checks = {}
    for evaluation in evaluations:
        step = int(evaluation["optimizer_step"])
        checkpoint_name = "factorized_r_step%03d.pt" % step
        spatial_name = "spatial_maps_step%03d.pt" % step
        expected_files.update((checkpoint_name, spatial_name))
        checkpoint_path = output_root / checkpoint_name
        spatial_path = output_root / spatial_name
        if not checkpoint_path.is_file() or not spatial_path.is_file():
            raise ValueError("terminal checkpoint or spatial map is absent")
        checkpoint = _load_checkpoint(checkpoint_path)
        expected_checkpoint_status = (
            "factorized_r_gate_pass"
            if evaluation["passed"]
            else "factorized_r_gate_failed"
        )
        if (
            checkpoint.get("schema") != REPORT_SCHEMA
            or checkpoint.get("status") != expected_checkpoint_status
            or int(checkpoint.get("optimizer_steps", -1)) != step
            or checkpoint.get("provenance") != provenance
            or checkpoint.get("stage1_uq_loaded") is not False
            or checkpoint.get("u_tokenizer_loaded") is not False
            or checkpoint.get("language_training_used") is not False
            or checkpoint.get("trajectory_or_control_loss_used") is not False
            or len(checkpoint.get("lora", {})) != 256
            or len(checkpoint.get("view_aligned_relevance_queries", {})) != 14
            or len(checkpoint.get("factorized_relevance_head", {})) != 8
        ):
            raise ValueError("terminal checkpoint contract differs")
        spatial_checks[spatial_name] = _validate_spatial_maps(
            spatial_path, all_groups
        )
        artifact_hashes[checkpoint_name] = sha256_file(checkpoint_path)
        artifact_hashes[spatial_name] = sha256_file(spatial_path)
    actual_files = {path.name for path in output_root.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("terminal output contains missing or unexpected files")

    failed_checks = sorted(name for name, value in checks.items() if not value)
    if passed:
        decision = "engineering_factorized_r_candidate_passed"
        next_action = (
            "preserve the engineering checkpoint and expand independent-event "
            "coverage before any formal or language run"
        )
    elif checks.get("train_supported_macro_recall") and any(
        not value for name, value in checks.items() if name.startswith("dev_")
    ):
        decision = "held_out_factorized_r_transfer_failed"
        next_action = (
            "stop this interface path and inspect actor supervision or contextual "
            "representation; do not add epochs automatically"
        )
    else:
        decision = "factorized_r_learnability_or_retention_failed"
        next_action = (
            "stop and inspect component objective or warm-start retention; do not "
            "start language, Stage2-P or closed loop"
        )
    return {
        "schema": VALIDATION_SCHEMA,
        "status": "validated_pass" if passed else "validated_failed_gate",
        "integrity_valid": True,
        "smoke_passed": passed,
        "optimizer_steps": steps,
        "evaluation_steps": evaluation_steps,
        "checks": checks,
        "failed_checks": failed_checks,
        "decision": decision,
        "next_action": next_action,
        "artifact_hashes": artifact_hashes,
        "spatial_map_checks": spatial_checks,
        "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "submission_attestation": {
            "path": str(submission_attestation_path),
            "sha256": sha256_file(submission_attestation_path),
        },
        "locks": dict(EXPECTED_LOCKS),
        "claim_boundary": (
            "Independent integrity and frozen engineering-gate validation for one "
            "17-event factorized-R-only smoke. Passing does not validate semantic "
            "R, learned U, language, planning, closed loop or safety."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--submission-attestation", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--trainer-preflight", type=Path, required=True)
    parser.add_argument("--launch-amendment", type=Path, required=True)
    parser.add_argument("--audit-protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite factorized-R validation")
    value = validate_report(
        report_path=args.report,
        submission_attestation_path=args.submission_attestation,
        training_protocol_path=args.training_protocol,
        trainer_preflight_path=args.trainer_preflight,
        launch_amendment_path=args.launch_amendment,
        audit_protocol_path=args.audit_protocol,
        output_root=args.output_root,
        project_root=args.project_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
