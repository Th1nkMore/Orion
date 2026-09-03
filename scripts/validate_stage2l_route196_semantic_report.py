#!/usr/bin/env python3
"""Independently validate the bounded Route196 semantic-smoke report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    from scripts.scenario_factory_lib import sha256_file
except ModuleNotFoundError:
    from scenario_factory_lib import sha256_file


REPORT_SCHEMA = "orion.stage2l_route196_structured_semantic_smoke.v1"
ATTESTATION_SCHEMA = "orion.stage2l_submission_attestation.v1"
PROTOCOL_SCHEMA = "orion.stage2l_uq_language_grounding_protocol.v1"
EXPECTED_CHECKS = {
    "optimization_reduces_language_nll",
    "optimization_reduces_relevance_bce",
    "optimization_reduces_structured_stance_ce",
    "all_structured_stances_match_targets",
    "structured_target_probability_floor",
    "on_path_risk_exceeds_off_path_by_margin",
    "zero_uq_prefers_maintain",
    "off_path_prefers_maintain",
    "on_path_prefers_conservative",
    "all_generated_stances_match_targets",
}


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _reference_hash(reference: Mapping[str, Any], name: str) -> str:
    value = str(reference.get("sha256", ""))
    if len(value) != 64:
        raise ValueError("%s lacks a SHA-256" % name)
    return value


def _recompute_checks(report: Mapping[str, Any]) -> Dict[str, bool]:
    before = report["before"]
    after = report["after"]
    expected = after["expected_answer_nll"]
    alternative = after["counterfactual_answer_nll"]
    return {
        "optimization_reduces_language_nll": (
            float(after["mean_language_nll"]) < float(before["mean_language_nll"])
        ),
        "optimization_reduces_relevance_bce": (
            float(after["relevance_bce"]) < float(before["relevance_bce"])
        ),
        "optimization_reduces_structured_stance_ce": (
            float(after["mean_structured_stance_ce"])
            < float(before["mean_structured_stance_ce"])
        ),
        "all_structured_stances_match_targets": (
            float(after["structured_stance_accuracy"]) == 1.0
        ),
        "structured_target_probability_floor": (
            float(after["minimum_structured_target_probability"]) >= 0.55
        ),
        "on_path_risk_exceeds_off_path_by_margin": (
            float(after["on_minus_off_peak_task_risk"]) >= 0.2
        ),
        "zero_uq_prefers_maintain": (
            float(expected["zero_uq"]) < float(alternative["zero_uq"])
        ),
        "off_path_prefers_maintain": (
            float(expected["off_path_uq"]) < float(alternative["off_path_uq"])
        ),
        "on_path_prefers_conservative": (
            float(expected["on_path_uq"]) < float(alternative["on_path_uq"])
        ),
        "all_generated_stances_match_targets": (
            float(after["generated_stance_accuracy"]) == 1.0
        ),
    }


def validate_report(
    *,
    report_path: Path,
    attestation_path: Path,
    protocol_path: Path,
    project_root: Path,
) -> Dict[str, Any]:
    report_path = report_path.resolve()
    attestation_path = attestation_path.resolve()
    protocol_path = protocol_path.resolve()
    project_root = project_root.resolve()
    report = _load(report_path)
    attestation = _load(attestation_path)
    protocol = _load(protocol_path)
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("unsupported semantic-smoke report schema")
    if attestation.get("schema") != ATTESTATION_SCHEMA:
        raise ValueError("unsupported submission attestation schema")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported training protocol schema")
    if int(report.get("steps", -1)) != 240 or int(
        attestation.get("authorized_run", {}).get("steps", -1)
    ) != 240:
        raise ValueError("semantic smoke did not use exactly 240 steps")

    attested_sources = attestation["source_hashes"]
    source_paths = {
        "training_protocol_sha256": protocol_path,
        "trainer_sha256": project_root / "scripts/train_stage2l_route196_semantic_smoke.py",
        "bridge_trainer_helpers_sha256": project_root / "scripts/train_stage2l_route196_bridge_smoke.py",
        "uq_relevance_bridge_sha256": project_root / "uq_estimator/uq_relevance_tokenizer.py",
        "two_pass_runtime_sha256": project_root / "uq_estimator/stage2l_bridge_runtime.py",
        "semantic_bottleneck_sha256": project_root / "uq_estimator/stage2l_semantic_bottleneck.py",
        "semantic_runtime_sha256": project_root / "uq_estimator/stage2l_semantic_runtime.py",
        "submit_wrapper_sha256": project_root / "scripts/submit_stage2l_route196_semantic_smoke.sh",
    }
    for name, path in source_paths.items():
        if not path.is_file() or sha256_file(path) != attested_sources.get(name):
            raise ValueError("attested source hash mismatch: %s" % name)

    report_inputs = report["inputs"]
    if _reference_hash(report_inputs["training_protocol"], "report protocol") != sha256_file(protocol_path):
        raise ValueError("report training protocol differs from attestation")
    if _reference_hash(report_inputs["launch_amendment"], "report amendment") != attested_sources.get("launch_amendment_sha256"):
        raise ValueError("report launch amendment differs from attestation")
    for report_name, attestation_name in (
        ("records", "records_sha256"),
        ("visual_cache", "visual_cache_sha256"),
        ("base_orion_checkpoint", "base_orion_checkpoint_sha256"),
    ):
        if _reference_hash(report_inputs[report_name], report_name) != attestation["data_hashes"].get(attestation_name):
            raise ValueError("report data hash differs from attestation: %s" % report_name)
    checkpoint_path = Path(str(report["checkpoint"]["path"]))
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != report["checkpoint"].get("sha256"):
        raise ValueError("semantic checkpoint is absent or hash-mismatched")

    architecture = report.get("architecture", {})
    if (
        architecture.get("ground_truth_stance_enters_forward") is not False
        or architecture.get("semantic_token_uses_predicted_distribution") is not True
        or architecture.get("trajectory") is not False
        or architecture.get("direct_control") is not False
        or architecture.get("legacy_density_uq_used") is not False
        or architecture.get("hard_governor_used") is not False
    ):
        raise ValueError("report architecture violates the Stage2-L boundary")

    recorded_checks = report.get("checks", {})
    if set(recorded_checks) != EXPECTED_CHECKS:
        raise ValueError("report check set differs from preregistration")
    recomputed = _recompute_checks(report)
    if recorded_checks != recomputed:
        raise ValueError("trainer-recorded checks differ from independent recomputation")
    passed = all(recomputed.values())
    expected_status = (
        "engineering_structured_semantic_overfit_pass"
        if passed
        else "engineering_structured_semantic_overfit_failed_gate"
    )
    if report.get("status") != expected_status:
        raise ValueError("report status differs from independently recomputed outcome")
    if report.get("stage2l_pilot_training_ready") is not False:
        raise ValueError("one-event report improperly unlocks pilot training")
    if report.get("stage2l_pilot_migration_review_ready") is not passed:
        raise ValueError("migration-review flag differs from smoke outcome")
    failed = sorted(name for name, value in recomputed.items() if not value)
    return {
        "schema": "orion.stage2l_route196_semantic_validation.v1",
        "status": "validated_pass" if passed else "validated_failed_gate",
        "integrity_valid": True,
        "smoke_passed": passed,
        "failed_checks": failed,
        "checks": recomputed,
        "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "submission_attestation": {
            "path": str(attestation_path),
            "sha256": sha256_file(attestation_path),
        },
        "next_action": (
            "review migration to the frozen eight-event pilot; training remains launch-locked"
            if passed
            else "pause Route196 iterations and review the failed discrete-language contract"
        ),
        "claim_boundary": "Independent one-event report integrity and gate validation only; no held-out, trajectory, closed-loop, or safety evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--submission-attestation", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite semantic validation report")
    result = validate_report(
        report_path=args.report,
        attestation_path=args.submission_attestation,
        protocol_path=args.training_protocol,
        project_root=args.project_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
