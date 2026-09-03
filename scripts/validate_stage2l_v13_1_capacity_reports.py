#!/usr/bin/env python3
"""Independently validate both terminal Stage2-L v13.1 capacity arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Mapping

import torch


REPORT_SCHEMA = "orion.stage2l_v13_process_qa_capacity_smoke.v1"
ATTESTATION_SCHEMA = "orion.stage2l_v13_process_qa_submission_attestation.v1"
PROTOCOL_SCHEMA = "orion.stage2l_v13_process_qa_capacity_protocol.v1"
PREFLIGHT_SCHEMA = "orion.stage2l_v13_process_qa_preflight.v1"
LAUNCH_SCHEMA = "orion.stage2l_v13_process_qa_launch.v1"
VALIDATION_SCHEMA = "orion.stage2l_v13_1_capacity_validation.v1"
ARMS = ("lora", "partial_unfreeze")
VARIANTS = ("zero_uq", "on_path_uq", "off_path_uq", "view_shuffled_uq")
PROCESS_FAMILIES = (
    "observation_semantics",
    "epistemic_limitation",
    "task_relevance",
    "driving_implication",
)
FORBIDDEN_LOG_PATTERNS = (
    "Traceback",
    "CUDA out of memory",
    "optimization became non-finite",
    "gradient escaped into the U tokenizer",
    "direct-token language loss has no gradient graph",
    "disconnected trainable group",
    "disconnected language group",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def recompute_quality(report: Mapping[str, Any], protocol: Mapping[str, Any]) -> Dict[str, bool]:
    before = report["language_before"]
    after = report["language_after"]
    gates = protocol["language_diagnostics"]
    return {
        "train_target_nll_improved": (
            float(after["train"]["mean_target_nll"])
            < float(before["train"]["mean_target_nll"])
        ),
        "dev_target_nll_improved": (
            float(after["dev"]["mean_target_nll"])
            < float(before["dev"]["mean_target_nll"])
        ),
        "dev_full_preference_above_no_u": (
            float(after["dev"]["full_minus_no_u_preference_fraction"])
            >= float(gates["minimum_full_minus_no_u_fraction"])
        ),
        "dev_on_path_preference": (
            float(after["dev"]["full_preference_fraction_by_variant"]["on_path_uq"])
            >= float(gates["minimum_on_path_preference_fraction"])
        ),
        "dev_zero_u_preference": (
            float(after["dev"]["full_preference_fraction_by_variant"]["zero_uq"])
            >= float(gates["minimum_zero_u_preference_fraction"])
        ),
    }


def compare_capacity(reports: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    lora = reports["lora"]
    partial = reports["partial_unfreeze"]
    lora_dev = lora["language_after"]["dev"]
    partial_dev = partial["language_after"]["dev"]
    preference_keys = (
        "full_minus_no_u_preference_fraction",
        "full_overall_preference_fraction",
        "no_u_overall_preference_fraction",
        "full_preference_fraction_by_variant",
        "no_u_preference_fraction_by_variant",
    )
    preferences_identical = all(
        lora_dev[key] == partial_dev[key] for key in preference_keys
    )
    partial_step_improvement = {
        family: (
            float(partial_dev["mean_step_nll"][family])
            < float(lora_dev["mean_step_nll"][family])
        )
        for family in PROCESS_FAMILIES
    }
    result = {
        "before_metrics_identical": (
            lora["language_before"] == partial["language_before"]
        ),
        "partial_dev_target_nll_lower": (
            float(partial_dev["mean_target_nll"])
            < float(lora_dev["mean_target_nll"])
        ),
        "partial_dev_target_nll_ratio_to_lora": (
            float(partial_dev["mean_target_nll"])
            / float(lora_dev["mean_target_nll"])
        ),
        "partial_dev_step_nll_lower": partial_step_improvement,
        "dev_preference_diagnostics_identical": preferences_identical,
        "lora_dev_full_minus_no_u": float(
            lora_dev["full_minus_no_u_preference_fraction"]
        ),
        "partial_dev_full_minus_no_u": float(
            partial_dev["full_minus_no_u_preference_fraction"]
        ),
        "lora_dev_on_path_preference": float(
            lora_dev["full_preference_fraction_by_variant"]["on_path_uq"]
        ),
        "partial_dev_on_path_preference": float(
            partial_dev["full_preference_fraction_by_variant"]["on_path_uq"]
        ),
    }
    result["decision"] = (
        "capacity_increases_likelihood_fit_but_not_counterfactual_u_semantics"
        if (
            result["before_metrics_identical"]
            and result["partial_dev_target_nll_lower"]
            and all(partial_step_improvement.values())
            and preferences_identical
            and result["partial_dev_on_path_preference"] == 0.0
        )
        else "capacity_result_requires_manual_interpretation"
    )
    return result


def _validate_checkpoint(path: Path, arm: str) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("v13.1 checkpoint is not a dictionary")
    expected_keys = {
        "schema",
        "status",
        "training_arm",
        "optimizer_steps",
        "orion_lora",
        "partial_decoder",
        "view_aligned_relevance_queries",
        "factorized_relevance_head",
        "task_risk_language_bridge_present",
        "k_used_as_model_input",
        "formal_stage2l_ready",
        "stage2p_ready",
        "closed_loop_eligible",
    }
    if set(checkpoint) != expected_keys:
        raise ValueError("v13.1 checkpoint fields differ")
    if (
        checkpoint["schema"] != REPORT_SCHEMA
        or checkpoint["status"] != "bounded_capacity_smoke_complete"
        or checkpoint["training_arm"] != arm
        or checkpoint["optimizer_steps"] != 200
        or checkpoint["task_risk_language_bridge_present"] is not False
        or checkpoint["k_used_as_model_input"] is not False
        or checkpoint["formal_stage2l_ready"] is not False
        or checkpoint["stage2p_ready"] is not False
        or checkpoint["closed_loop_eligible"] is not False
    ):
        raise ValueError("v13.1 checkpoint hard invariant differs")
    state = {
        key: checkpoint[key]
        for key in (
            "orion_lora",
            "partial_decoder",
            "view_aligned_relevance_queries",
            "factorized_relevance_head",
        )
    }
    if not all(
        isinstance(values, dict)
        and all(torch.isfinite(tensor).all().item() for tensor in values.values())
        for values in state.values()
    ):
        raise ValueError("v13.1 checkpoint contains a non-finite tensor")
    counts = {
        key: {
            "tensor_count": len(values),
            "element_count": sum(tensor.numel() for tensor in values.values()),
        }
        for key, values in state.items()
    }
    if (
        counts["orion_lora"] != {"tensor_count": 256, "element_count": 16777216}
        or counts["view_aligned_relevance_queries"]
        != {"tensor_count": 14, "element_count": 2407424}
        or counts["factorized_relevance_head"]
        != {"tensor_count": 8, "element_count": 1057538}
    ):
        raise ValueError("v13.1 checkpoint trainable-state size differs")
    if arm == "lora":
        if counts["partial_decoder"] != {"tensor_count": 0, "element_count": 0}:
            raise ValueError("LoRA arm unexpectedly stores partial decoder state")
    elif counts["partial_decoder"] != {
        "tensor_count": 40,
        "element_count": 809533696,
    }:
        raise ValueError("partial arm decoder state size differs")
    del checkpoint
    return counts


def _validate_arm(
    *,
    arm: str,
    report_path: Path,
    checkpoint_path: Path,
    runtime_probe_path: Path,
    attestation_path: Path,
    preflight_path: Path,
    log_path: Path,
    output_root: Path,
    protocol: Mapping[str, Any],
    launch: Mapping[str, Any],
    protocol_path: Path,
    launch_path: Path,
    project_root: Path,
) -> Dict[str, Any]:
    report = _read(report_path)
    probe = _read(runtime_probe_path)
    attestation = _read(attestation_path)
    preflight = _read(preflight_path)
    validated = preflight.get("validated_inputs", {})
    protocol_inputs = dict(validated)
    protocol_inputs.pop("trainer_sha256", None)
    expected_output = str(output_root.resolve())
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("training_arm") != arm
        or report.get("optimizer_steps") != 200
        or preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("training_arm") != arm
        or preflight.get("passed") is not True
        or preflight.get("gpu_used") is not False
        or preflight.get("training_started") is not False
        or attestation.get("schema") != ATTESTATION_SCHEMA
        or attestation.get("training_arm") != arm
        or protocol.get("input_sha256") != protocol_inputs
        or launch.get("validated_inputs") != validated
        or attestation.get("validated_inputs") != validated
        or report.get("provenance", {}).get("validated_inputs") != validated
    ):
        raise ValueError("v13.1 arm schema, identity, or input lineage differs")
    preflight_hash = sha256_file(preflight_path)
    protocol_hash = sha256_file(protocol_path)
    launch_hash = sha256_file(launch_path)
    authorized = launch["authorized_arms"][arm]
    if (
        preflight.get("protocol_sha256") != protocol_hash
        or preflight.get("output_root") != expected_output
        or protocol["capacity_arms"][arm]["output_root"] != expected_output
        or authorized.get("preflight_sha256") != preflight_hash
        or authorized.get("output_root") != expected_output
        or authorized.get("maximum_submissions") != 1
        or authorized.get("optimizer_steps") != 200
        or attestation.get("authorized_output_root") != expected_output
        or attestation.get("protocol", {}).get("sha256") != protocol_hash
        or attestation.get("preflight", {}).get("sha256") != preflight_hash
        or attestation.get("launch", {}).get("sha256") != launch_hash
        or report["provenance"].get("protocol_sha256") != protocol_hash
        or report["provenance"].get("preflight_sha256") != preflight_hash
        or report["provenance"].get("launch_amendment_sha256") != launch_hash
    ):
        raise ValueError("v13.1 arm protocol/preflight/launch linkage differs")
    implementation = {
        "trainer_sha256": project_root / "scripts/train_stage2l_v13_process_qa_smoke.py",
        "process_module_sha256": project_root / "uq_estimator/stage2l_process_qa_v13.py",
        "v122_lineage_helper_sha256": project_root / "scripts/train_stage2l_v122_vertical_slice_semantic_smoke.py",
        "factorized_relevance_sha256": project_root / "uq_estimator/stage2l_factorized_relevance_v121.py",
    }
    if any(
        not path.is_file() or validated.get(name) != sha256_file(path)
        for name, path in implementation.items()
    ):
        raise ValueError("v13.1 implementation hash differs")
    history = report.get("history", [])
    if (
        len(history) != 200
        or [row.get("optimizer_step") for row in history] != list(range(1, 201))
        or any(row.get("training_arm") != arm for row in history)
        or any(row.get("finite") is not True for row in history)
        or any(row.get("variant") not in VARIANTS for row in history)
        or any(row.get("process_family") not in PROCESS_FAMILIES for row in history)
        or not _all_finite(history)
    ):
        raise ValueError("v13.1 optimization history differs or is non-finite")
    architecture = report.get("architecture_invariants", {})
    locks = report.get("locks", {})
    if (
        architecture.get("direct_stage1_u_tokens_enter_orion") is not True
        or architecture.get("direct_r_hidden_tokens_enter_orion") is not True
        or architecture.get("task_risk_language_bridge_present") is not False
        or architecture.get("k_used_as_model_input") is not False
        or architecture.get("stage1_frozen") is not True
        or architecture.get("u_tokenizer_frozen") is not True
        or architecture.get("trajectory_or_control_loss") is not False
        or locks != {
            "automatic_extension": False,
            "closed_loop_eligible": False,
            "formal_stage2l_ready": False,
            "locked_test_read": False,
            "stage2p_ready": False,
        }
    ):
        raise ValueError("v13.1 architecture or downstream lock differs")
    if probe != report.get("runtime_gradient_probe"):
        raise ValueError("v13.1 runtime probe differs from report")
    required_gradient_groups = {"orion_lora": 256}
    if arm == "partial_unfreeze":
        required_gradient_groups["partial_decoder"] = 36
    if (
        probe.get("status") != "direct_token_backward_connected"
        or probe.get("training_arm") != arm
        or probe.get("finite") is not True
        or probe.get("conditioning_is_detached_leaf") is not True
        or probe.get("optimizer_step_taken") is not False
        or probe.get("u_tokenizer_gradient_parameter_count") != 0
        or probe.get("language_gradient_parameter_counts") != required_gradient_groups
        or probe.get("r_gradient_parameter_counts")
        != {"relevance_head": 8, "relevance_queries": 14}
    ):
        raise ValueError("v13.1 runtime gradient probe differs")
    quality = recompute_quality(report, protocol)
    if (
        quality != report.get("quality_diagnostics")
        or bool(all(quality.values())) is not report.get("quality_diagnostics_passed")
        or report.get("status")
        != (
            "bounded_capacity_smoke_quality_diagnostics_passed"
            if all(quality.values())
            else "bounded_capacity_smoke_completed_with_soft_quality_failures"
        )
    ):
        raise ValueError("v13.1 soft quality diagnostics differ")
    if (
        report.get("process_dataset_audit", {}).get("passed") is not True
        or report["process_dataset_audit"].get("failed_group_ids") != []
        or report["process_dataset_audit"].get("group_count") != 80
        or report["process_dataset_audit"].get("chain_count") != 320
        or report["process_dataset_audit"].get("step_target_count") != 1280
        or not _all_finite(report)
    ):
        raise ValueError("v13.1 report audit or finite check differs")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if any(pattern in log_text for pattern in FORBIDDEN_LOG_PATTERNS):
        raise ValueError("v13.1 complete log contains a fail-closed error")
    logged_steps = [
        int(value) for value in re.findall(r'"optimizer_step":\s*(\d+)', log_text)
    ]
    if not logged_steps or logged_steps[-1] != 200:
        raise ValueError("v13.1 complete log lacks terminal step 200")
    checkpoint_counts = _validate_checkpoint(checkpoint_path, arm)
    r_losses = [float(row["factorized_r_loss"]) for row in history]
    return {
        "status": "hard_invariants_validated",
        "training_arm": arm,
        "quality_diagnostics": quality,
        "quality_diagnostics_passed": all(quality.values()),
        "train_target_nll_before": float(report["language_before"]["train"]["mean_target_nll"]),
        "train_target_nll_after": float(report["language_after"]["train"]["mean_target_nll"]),
        "dev_target_nll_before": float(report["language_before"]["dev"]["mean_target_nll"]),
        "dev_target_nll_after": float(report["language_after"]["dev"]["mean_target_nll"]),
        "dev_step_nll_after": report["language_after"]["dev"]["mean_step_nll"],
        "dev_preference_after": {
            key: report["language_after"]["dev"][key]
            for key in (
                "full_minus_no_u_preference_fraction",
                "full_preference_fraction_by_variant",
                "no_u_preference_fraction_by_variant",
            )
        },
        "factorized_r_training_diagnostic": {
            "mean_loss": sum(r_losses) / len(r_losses),
            "final_loss": r_losses[-1],
            "all_finite": True,
            "post_training_spatial_metric_available": False,
        },
        "trainable_scope": report["trainable_scope"],
        "checkpoint_state": checkpoint_counts,
        "runtime_gradient_probe": probe,
        "artifact_sha256": {
            "report": sha256_file(report_path),
            "checkpoint": sha256_file(checkpoint_path),
            "runtime_gradient_probe": sha256_file(runtime_probe_path),
            "submission_attestation": sha256_file(attestation_path),
            "trainer_preflight": preflight_hash,
            "complete_log": sha256_file(log_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--lora-root", type=Path, required=True)
    parser.add_argument("--lora-log", type=Path, required=True)
    parser.add_argument("--partial-root", type=Path, required=True)
    parser.add_argument("--partial-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite v13.1 terminal validation")
    protocol = _read(args.protocol.resolve())
    launch = _read(args.launch.resolve())
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or launch.get("schema") != LAUNCH_SCHEMA
        or launch.get("status") != "immutable_two_arm_process_qa_smoke_authorization"
    ):
        raise ValueError("v13.1 protocol or launch schema differs")
    roots = {
        "lora": args.lora_root.resolve(),
        "partial_unfreeze": args.partial_root.resolve(),
    }
    logs = {
        "lora": args.lora_log.resolve(),
        "partial_unfreeze": args.partial_log.resolve(),
    }
    reports = {
        arm: _read(roots[arm] / "training/report.json") for arm in ARMS
    }
    arm_results = {
        arm: _validate_arm(
            arm=arm,
            report_path=roots[arm] / "training/report.json",
            checkpoint_path=roots[arm] / "training/stage2l_v13_process_qa.pt",
            runtime_probe_path=roots[arm] / "training/runtime_gradient_probe.json",
            attestation_path=roots[arm] / "submission_attestation.json",
            preflight_path=roots[arm] / "trainer_preflight.json",
            log_path=logs[arm],
            output_root=roots[arm] / "training",
            protocol=protocol,
            launch=launch,
            protocol_path=args.protocol.resolve(),
            launch_path=args.launch.resolve(),
            project_root=args.project_root.resolve(),
        )
        for arm in ARMS
    }
    comparison = compare_capacity(reports)
    value = {
        "schema": VALIDATION_SCHEMA,
        "status": "hard_valid_capacity_smoke_soft_quality_failed",
        "hard_invariants_validated": True,
        "quality_diagnostics_passed": False,
        "arms": arm_results,
        "capacity_comparison": comparison,
        "locks": {
            "formal_stage2l_allowed": False,
            "stage2p_allowed": False,
            "closed_loop_allowed": False,
            "locked_test_allowed": False,
            "extra_epochs_allowed": False,
            "automatic_retry_allowed": False,
        },
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(args.protocol.resolve()),
        "launch_sha256": sha256_file(args.launch.resolve()),
        "claim_boundary": (
            "Both direct-token capacity arms are runtime-valid bounded smokes. "
            "Lower NLL is not evidence that ORION learned counterfactual U "
            "semantics, planning behavior, closed-loop safety, or formal "
            "generalization."
        ),
    }
    if any(result["quality_diagnostics_passed"] for result in arm_results.values()):
        raise ValueError("v13.1 terminal quality status changed unexpectedly")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": value["status"],
        "decision": comparison["decision"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
