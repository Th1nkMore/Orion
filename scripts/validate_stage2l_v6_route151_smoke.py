#!/usr/bin/env python3
"""Independently validate the bounded Route151 Stage2-L v6 smoke report."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    from scripts.scenario_factory_lib import sha256_file
except ModuleNotFoundError:
    from scenario_factory_lib import sha256_file


REPORT_SCHEMA = "orion.stage2l_v6_matched_group_smoke.v1"
ATTESTATION_SCHEMA = "orion.stage2l_smoke_submission_attestation.v1"
PROTOCOL_SCHEMA = "orion.stage2l_uq_language_grounding_protocol.v2"
AMENDMENT_SCHEMA = "orion.scenario_factory.amendment.v1"
EXPECTED_EVENT_ID = "route151_step218"
EXPECTED_STEPS = 20
EXPECTED_CHECKS = {
    "exactly_one_optimizer_step_per_complete_group",
    "language_nll_decreases",
    "relevance_bce_decreases",
    "on_path_exceeds_off_path",
    "hard_stance_accuracy_gte_two_thirds",
    "cross_family_margin_fraction_improves",
    "diagnostic_driving_targets_excluded",
    "trajectory_and_control_remain_disabled",
}


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _require_finite(value: Any, path: str = "report") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("non-finite numeric value at %s" % path)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require_finite(child, "%s.%s" % (path, key))
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite(child, "%s[%d]" % (path, index))


def _recompute_checks(report: Mapping[str, Any]) -> Dict[str, bool]:
    before = report["before"]
    after = report["after"]
    audit = report["matched_record_audit"]
    history = report["history"]
    return {
        "exactly_one_optimizer_step_per_complete_group": (
            int(report["optimizer_steps"]) == EXPECTED_STEPS
            and len(history) == EXPECTED_STEPS
            and all(
                int(row["records_in_optimizer_unit"]) == 20
                and int(row["optimizer_steps_inside_group"]) == 0
                for row in history
            )
        ),
        "language_nll_decreases": (
            float(after["first_group_mean_hard_language_nll"])
            < float(before["first_group_mean_hard_language_nll"])
        ),
        "relevance_bce_decreases": (
            float(after["mean_relevance_bce"])
            < float(before["mean_relevance_bce"])
        ),
        "on_path_exceeds_off_path": (
            float(after["mean_on_minus_off_peak_task_risk"]) >= 0.2
        ),
        "hard_stance_accuracy_gte_two_thirds": (
            float(after["hard_stance_accuracy"]) >= (2.0 / 3.0)
        ),
        "cross_family_margin_fraction_improves": (
            float(after["first_group_cross_family_margin_pass_fraction"])
            > float(before["first_group_cross_family_margin_pass_fraction"])
        ),
        "diagnostic_driving_targets_excluded": (
            int(audit["diagnostic_only_driving_record_count"])
            == 2 * int(audit["matched_group_count"])
        ),
        "trajectory_and_control_remain_disabled": True,
    }


def _validate_source_integrity(
    *, attestation: Mapping[str, Any], project_root: Path
) -> None:
    paths = {
        "submitter": "scripts/submit_stage2l_v6_route151_smoke.sh",
        "trainer": "scripts/train_stage2l_v6_route151_smoke.py",
        "semantic_bottleneck": "uq_estimator/stage2l_semantic_bottleneck_v2.py",
        "semantic_runtime": "uq_estimator/stage2l_semantic_runtime_v2.py",
        "matched_objective": "uq_estimator/stage2l_matched_objective.py",
        "training_protocol": "configs/scenario_factory/stage2l_training_v6_matched_magnitude_cross_family.json",
        "launch_amendment": "configs/scenario_factory/amendments/20260830_stage2l_v6_route151_matched_smoke_v1.json",
        "trainer_test": "tests/test_train_stage2l_v6_route151_smoke.py",
    }
    expected = attestation["source_sha256"]
    for name, relative in paths.items():
        path = project_root / relative
        if not path.is_file() or sha256_file(path) != expected.get(name):
            raise ValueError("attested source hash mismatch: %s" % name)


def validate_report(
    *, report_path: Path, checkpoint_path: Path, attestation_path: Path,
    protocol_path: Path, amendment_path: Path, project_root: Path,
) -> Dict[str, Any]:
    report_path = report_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    attestation_path = attestation_path.resolve()
    protocol_path = protocol_path.resolve()
    amendment_path = amendment_path.resolve()
    project_root = project_root.resolve()
    report = _load(report_path)
    attestation = _load(attestation_path)
    protocol = _load(protocol_path)
    amendment = _load(amendment_path)
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("unsupported Route151 v6 smoke report schema")
    if attestation.get("schema") != ATTESTATION_SCHEMA:
        raise ValueError("unsupported Route151 submission attestation schema")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported Stage2-L v6 protocol schema")
    if amendment.get("schema") != AMENDMENT_SCHEMA:
        raise ValueError("unsupported smoke launch amendment schema")
    if protocol.get("launch_locks", {}).get("stage2l_pilot_training_allowed") is not False:
        raise ValueError("v6 smoke protocol unexpectedly unlocks pilot training")
    _validate_source_integrity(attestation=attestation, project_root=project_root)
    _require_finite(report)

    bounds = attestation["training_bounds"]
    if (
        int(bounds.get("maximum_submissions", -1)) != 1
        or int(bounds.get("maximum_optimizer_steps", -1)) != EXPECTED_STEPS
        or bounds.get("formal_training") is not False
        or bounds.get("stage2p_training") is not False
        or bounds.get("trajectory_or_control_loss") is not False
    ):
        raise ValueError("submission attestation violates the bounded smoke contract")
    if (
        report.get("event_id") != EXPECTED_EVENT_ID
        or int(report.get("optimizer_steps", -1)) != EXPECTED_STEPS
        or int(report.get("record_equivalent_presentations", -1)) != 400
        or int(report.get("language_anchor_presentations", -1)) != 360
    ):
        raise ValueError("report violates fixed event or presentation bounds")

    audit = report["matched_record_audit"]
    expected_audit = {
        "record_count": 100,
        "matched_group_count": 5,
        "optimizer_step_count_per_epoch": 5,
        "optimizer_steps_inside_group": 0,
        "hard_language_record_count": 90,
        "hard_stance_record_count": 15,
        "diagnostic_only_driving_record_count": 10,
        "cross_family_pairwise_comparison_count": 270,
    }
    if audit != expected_audit:
        raise ValueError("matched-record audit differs from the frozen Route151 unit")
    history = report["history"]
    if [int(row["optimizer_step"]) for row in history] != list(
        range(1, EXPECTED_STEPS + 1)
    ):
        raise ValueError("optimizer-step history is incomplete or reordered")
    group_counts = Counter(str(row["group_id"]) for row in history)
    if len(group_counts) != 5 or set(group_counts.values()) != {4}:
        raise ValueError("the four smoke epochs do not each cover all five groups")
    if any(
        int(row["hard_language_anchors"]) != 18
        or float(row["gradient_norm_before_clip"]) <= 0.0
        for row in history
    ):
        raise ValueError("history violates anchor or gradient invariants")

    architecture = report["architecture"]
    expected_architecture = {
        "raw_k_magnitude_side_channel": True,
        "magnitude_layer_norm_before_classifier": False,
        "one_optimizer_step_per_20_record_group": True,
        "observed_driving_is_diagnostic_only": True,
        "view_shuffled_driving_is_diagnostic_only": True,
        "ground_truth_stance_enters_forward": False,
        "legacy_density_uq_used": False,
        "hard_governor_used": False,
    }
    if architecture != expected_architecture:
        raise ValueError("report architecture differs from the frozen v6 boundary")
    losses = report["loss_weights"]
    if float(losses.get("trajectory", -1.0)) != 0.0 or float(
        losses.get("direct_control", -1.0)
    ) != 0.0:
        raise ValueError("trajectory or direct-control loss was enabled")
    if any(
        report.get(name) is not False
        for name in ("formal_training_ready", "stage2l_pilot_training_ready", "stage2p_ready")
    ) or report.get("engineering_smoke_only") is not True:
        raise ValueError("one-event smoke improperly expands its claim or launch status")

    attested_hashes = attestation["source_sha256"]
    provenance = report["provenance"]
    expected_provenance = {
        "records": attested_hashes["records"],
        "visual_cache": attested_hashes["visual_cache"],
        "training_protocol": attested_hashes["training_protocol"],
        "launch_amendment": attested_hashes["launch_amendment"],
        "trainer": attested_hashes["trainer"],
        "base_orion_checkpoint": attested_hashes["base_orion_checkpoint"],
    }
    for name, expected_hash in expected_provenance.items():
        if provenance.get(name, {}).get("sha256") != expected_hash:
            raise ValueError("report provenance differs from attestation: %s" % name)
    if sha256_file(protocol_path) != attested_hashes["training_protocol"]:
        raise ValueError("supplied v6 protocol differs from attestation")
    if sha256_file(amendment_path) != attested_hashes["launch_amendment"]:
        raise ValueError("supplied launch amendment differs from attestation")
    checkpoint_hash = sha256_file(checkpoint_path)
    if provenance.get("checkpoint", {}).get("sha256") != checkpoint_hash:
        raise ValueError("downloaded smoke checkpoint is absent or hash-mismatched")

    recorded_checks = report.get("checks", {})
    if set(recorded_checks) != EXPECTED_CHECKS:
        raise ValueError("report check set differs from the frozen smoke gates")
    recomputed = _recompute_checks(report)
    if recorded_checks != recomputed:
        raise ValueError("trainer checks differ from independent recomputation")
    passed = all(recomputed.values())
    expected_status = (
        "engineering_v6_matched_smoke_pass"
        if passed
        else "engineering_v6_matched_smoke_failed_gate"
    )
    if report.get("status") != expected_status:
        raise ValueError("report status differs from independent outcome")
    failed = sorted(name for name, value in recomputed.items() if not value)
    return {
        "schema": "orion.stage2l_v6_route151_independent_validation.v1",
        "status": "validated_pass" if passed else "validated_failed_gate",
        "integrity_valid": True,
        "smoke_passed": passed,
        "failed_checks": failed,
        "checks": recomputed,
        "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_hash},
        "submission_attestation": {
            "path": str(attestation_path),
            "sha256": sha256_file(attestation_path),
        },
        "next_action": (
            "complete human QA geometry review, then prepare a separately authorized held-out v6 pilot"
            if passed
            else "keep pilot locked and diagnose the failed Route151 semantic gates"
        ),
        "claim_boundary": "Independent Route151 engineering-smoke integrity and gate validation only; no held-out, trajectory, closed-loop, generalization, or safety evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--submission-attestation", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--launch-amendment", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite independent validation")
    result = validate_report(
        report_path=args.report,
        checkpoint_path=args.checkpoint,
        attestation_path=args.submission_attestation,
        protocol_path=args.training_protocol,
        amendment_path=args.launch_amendment,
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
