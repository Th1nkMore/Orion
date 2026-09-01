#!/usr/bin/env python3
"""Independently validate the terminal v12.2 semantic-slice artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import torch


REPORT_SCHEMA = "orion.stage2l_v12_2_vertical_slice_semantic_smoke.v1"
SPATIAL_SCHEMA = "orion.stage2l_v12_2_spatial_diagnostic.v1"
ATTESTATION_SCHEMA = "orion.stage2l_v12_2_submission_attestation.v1"
REQUIRED_VARIANTS = (
    "zero_uq",
    "on_path_uq",
    "off_path_uq",
    "view_shuffled_uq",
)


def _sha256(path: Path) -> str:
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


def _finite_tensor(
    value: Any, shape: tuple[int, ...], name: str
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError("%s shape/type differs" % name)
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("%s is non-finite or non-floating" % name)
    return value


def _validate_spatial(value: Mapping[str, Any]) -> Dict[str, Any]:
    if (
        value.get("schema") != SPATIAL_SCHEMA
        or value.get("component_order") != ["route", "actor"]
        or value.get("camera_order")
        != [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
        or value.get("union_definition")
        != "max(sigmoid(route_logit), sigmoid(actor_logit))"
        or len(value.get("events", {})) != 17
    ):
        raise ValueError("v12.2 spatial artifact contract differs")
    maximum_probability_error = 0.0
    maximum_union_error = 0.0
    maximum_k_error = 0.0
    zero_u_exact = True
    zero_k_exact = True
    split_counts: Dict[str, int] = {}
    for event_id, event in sorted(value["events"].items()):
        split = str(event.get("split"))
        if split not in ("train", "dev"):
            raise ValueError("spatial event split differs: %s" % event_id)
        split_counts[split] = split_counts.get(split, 0) + 1
        logits = _finite_tensor(
            event.get("r_component_logits"),
            (1, 2, 6, 10, 10),
            "%s component logits" % event_id,
        )
        probabilities = _finite_tensor(
            event.get("r_component_probability"),
            (1, 2, 6, 10, 10),
            "%s component probabilities" % event_id,
        )
        union = _finite_tensor(
            event.get("r_union_probability"),
            (1, 6, 10, 10),
            "%s union" % event_id,
        )
        _finite_tensor(
            event.get("r_union_target"),
            (1, 6, 10, 10),
            "%s target" % event_id,
        )
        expected_probabilities = logits.sigmoid()
        expected_union = expected_probabilities.amax(dim=1)
        maximum_probability_error = max(
            maximum_probability_error,
            float((probabilities - expected_probabilities).abs().max()),
        )
        maximum_union_error = max(
            maximum_union_error,
            float((union - expected_union).abs().max()),
        )
        u_by_variant = event.get("u_by_variant", {})
        k_by_variant = event.get("k_by_variant", {})
        fields = event.get("structured_fields", {})
        if (
            set(u_by_variant) != set(REQUIRED_VARIANTS)
            or set(k_by_variant) != set(REQUIRED_VARIANTS)
            or set(fields) != set(REQUIRED_VARIANTS)
        ):
            raise ValueError("spatial variants differ: %s" % event_id)
        for variant in REQUIRED_VARIANTS:
            uq = _finite_tensor(
                u_by_variant[variant],
                (1, 6, 10, 10),
                "%s/%s U" % (event_id, variant),
            )
            risk = _finite_tensor(
                k_by_variant[variant],
                (1, 6, 10, 10),
                "%s/%s K" % (event_id, variant),
            )
            if bool((uq < 0.0).any()) or bool((uq > 1.0).any()):
                raise ValueError("U lies outside [0,1]")
            maximum_k_error = max(
                maximum_k_error,
                float((risk - uq * union).abs().max()),
            )
            if not isinstance(fields[variant], dict):
                raise ValueError("structured fields are malformed")
        zero_u_exact = zero_u_exact and int(
            torch.count_nonzero(u_by_variant["zero_uq"])
        ) == 0
        zero_k_exact = zero_k_exact and int(
            torch.count_nonzero(k_by_variant["zero_uq"])
        ) == 0
    cross_device_probability_tolerance = 1e-7
    checks = {
        "event_count_17": len(value["events"]) == 17,
        "split_counts_13_4": split_counts == {"train": 13, "dev": 4},
        "component_probability_within_cross_device_tolerance": (
            maximum_probability_error <= cross_device_probability_tolerance
        ),
        "derived_union_within_cross_device_tolerance": (
            maximum_union_error <= cross_device_probability_tolerance
        ),
        "task_risk_product_exact": maximum_k_error == 0.0,
        "zero_u_exact": zero_u_exact,
        "zero_k_exact": zero_k_exact,
    }
    if not all(checks.values()):
        raise ValueError("v12.2 spatial checks failed: %s" % checks)
    return {
        "checks": checks,
        "split_counts": split_counts,
        "maximum_component_probability_error": maximum_probability_error,
        "maximum_derived_union_error": maximum_union_error,
        "maximum_task_risk_product_error": maximum_k_error,
        "cross_device_probability_tolerance": (
            cross_device_probability_tolerance
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--spatial", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--submission-attestation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = (
        args.report,
        args.bridge,
        args.spatial,
        args.protocol,
        args.preflight,
        args.launch,
        args.submission_attestation,
    )
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("v12.2 terminal validation input is missing")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite v12.2 terminal validation")

    report = _read(args.report)
    protocol = _read(args.protocol)
    preflight = _read(args.preflight)
    launch = _read(args.launch)
    attestation = _read(args.submission_attestation)
    bridge = torch.load(args.bridge, map_location="cpu")
    spatial = torch.load(args.spatial, map_location="cpu")
    expected_status = "vertical_slice_semantic_completed_with_soft_quality_failures"
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("status") != expected_status
        or report.get("optimizer_steps") != 40
        or report.get("hard_integrity_passed") is not True
        or report.get("soft_quality_passed") is not False
        or report.get("formal_stage2l_ready") is not False
        or report.get("stage2p_ready") is not False
        or len(report.get("history", [])) != 40
        or any(row.get("finite") is not True for row in report["history"])
        or any(row.get("trainable_scope") != "TaskRiskLanguageBridge_only" for row in report["history"])
    ):
        raise ValueError("v12.2 terminal report contract differs")
    if not all(report.get("hard_integrity_checks", {}).values()):
        raise ValueError("v12.2 hard integrity check failed")
    locks = report.get("locks", {})
    if (
        locks.get("only_task_risk_language_bridge_trained") is not True
        or locks.get("stage1_or_u_tokenizer_trained") is not False
        or locks.get("factorized_relevance_or_orion_lora_trained") is not False
        or locks.get("trajectory_or_control_loss_used") is not False
        or locks.get("locked_test_read") is not False
    ):
        raise ValueError("v12.2 report locks differ")
    provenance = report.get("provenance", {})
    if (
        provenance.get("validated_inputs") != preflight.get("validated_inputs")
        or provenance.get("protocol_sha256") != _sha256(args.protocol)
        or provenance.get("preflight_sha256") != _sha256(args.preflight)
        or provenance.get("launch_amendment_sha256") != _sha256(args.launch)
    ):
        raise ValueError("v12.2 report provenance differs")
    if (
        bridge.get("schema") != REPORT_SCHEMA
        or bridge.get("status") != expected_status
        or bridge.get("optimizer_steps") != 40
        or bridge.get("formal_stage2l_ready") is not False
        or bridge.get("stage2p_ready") is not False
        or not isinstance(bridge.get("task_risk_language_bridge"), dict)
        or not bridge["task_risk_language_bridge"]
        or any(
            not isinstance(value, torch.Tensor)
            or not bool(torch.isfinite(value).all())
            for value in bridge["task_risk_language_bridge"].values()
        )
        or bridge.get("factorized_relevance_checkpoint_sha256")
        != protocol["input_sha256"]["v121_checkpoint_sha256"]
        or bridge.get("u_tokenizer_checkpoint_sha256")
        != protocol["input_sha256"]["u_tokenizer_checkpoint_sha256"]
    ):
        raise ValueError("v12.2 bridge checkpoint contract differs")
    if (
        attestation.get("schema") != ATTESTATION_SCHEMA
        or attestation.get("status")
        != "single_v12_2_semantic_slice_submitted_and_attested"
        or attestation.get("job_id") != "1122494"
        or attestation.get("validated_inputs") != preflight.get("validated_inputs")
        or attestation.get("protocol", {}).get("sha256")
        != _sha256(args.protocol)
        or attestation.get("preflight", {}).get("sha256")
        != _sha256(args.preflight)
        or attestation.get("launch", {}).get("sha256") != _sha256(args.launch)
    ):
        raise ValueError("v12.2 submission attestation differs")
    spatial_checks = _validate_spatial(spatial)
    soft = report["soft_quality_checks"]
    if soft.get("dev_target_nll_improved") is not True:
        raise ValueError("expected recorded dev NLL improvement is absent")
    if (
        soft.get("dev_every_variant_prefers_target") is not False
        or soft.get("dev_full_improves_over_no_u") is not False
    ):
        raise ValueError("expected semantic-insensitivity diagnosis differs")
    value = {
        "schema": "orion.stage2l_v12_2_vertical_slice_semantic_validation.v1",
        "status": "validated_integrity_pass_soft_semantic_failure",
        "integrity_valid": True,
        "optimizer_steps": 40,
        "job_id": "1122494",
        "decision": "carry_labeled_semantic_insensitive_bridge_to_bounded_stage2p_interface",
        "artifact_hashes": {
            "report.json": _sha256(args.report),
            "v122_semantic_bridge.pt": _sha256(args.bridge),
            "spatial_u_r_k_maps.pt": _sha256(args.spatial),
        },
        "spatial_checks": spatial_checks,
        "semantic_diagnosis": {
            "train_target_nll_before": report["language_before"]["train"]["mean_target_nll"],
            "train_target_nll_after": report["language_after"]["train"]["mean_target_nll"],
            "dev_target_nll_before": report["language_before"]["dev"]["mean_target_nll"],
            "dev_target_nll_after": report["language_after"]["dev"]["mean_target_nll"],
            "dev_preference_before": report["language_before"]["dev"]["full_conditioning"]["overall_preference_fraction"],
            "dev_preference_after": report["language_after"]["dev"]["full_conditioning"]["overall_preference_fraction"],
            "dev_no_u_preference_after": report["language_after"]["dev"]["no_u_ablation"]["overall_preference_fraction"],
            "dev_full_minus_no_u_after": report["language_after"]["dev"]["full_minus_no_u_preference_fraction"],
            "interpretation": "The bridge reduced generic target NLL but did not learn U-dependent answer preference; full conditioning equals the no-U ablation on dev."
        },
        "controlled_u_diagnosis": {
            "train_on_over_off_fraction": report["factorization_before"]["train"]["controlled_u_fractions"]["on_over_off"],
            "dev_on_over_off_fraction": report["factorization_before"]["dev"]["controlled_u_fractions"]["on_over_off"],
        },
        "locks": {
            "formal_stage2l_ready": False,
            "stage2p_quality_ready": False,
            "closed_loop_ready": False,
            "locked_test_read": False,
            "automatic_retry": False,
        },
        "claim_boundary": "Independent terminal integrity and diagnosis for one controlled-U semantic engineering slice; not a language, planning, closed-loop or safety claim."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
