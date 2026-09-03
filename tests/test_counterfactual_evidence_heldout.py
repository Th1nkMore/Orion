import pytest

from uq_estimator.counterfactual_evidence_heldout import (
    CounterfactualHeldoutError,
    amended_training_gate,
    heldout_family_transfer_gate,
)


def _historical_gate():
    checks = [
        {"metric": "metric_%d" % index, "passed": True, "value": 1.0, "threshold": 0.0}
        for index in range(7)
    ]
    checks.append(
        {
            "metric": "reference_prediction_p95",
            "passed": False,
            "value": 0.35,
            "threshold": 0.2,
        }
    )
    return {"passed": False, "checks": checks}


def _evaluation(high=0.5):
    return {
        "combined_patch_spearman": 0.2,
        "combined_target_top20_auroc": 0.7,
        "median_record_within_intervened_view_target_top20_auroc": 0.65,
        "reference_prediction_mean": 0.1,
        "by_family": {"local_glare": {"score_mean": 0.3}},
        "by_family_severity": {
            "local_glare/severity_1": {"score_mean": 0.2},
            "local_glare/severity_3": {"score_mean": high},
        },
    }


def _thresholds():
    return {
        "combined_patch_spearman_min": 0.1,
        "combined_target_top20_auroc_min": 0.6,
        "median_record_within_intervened_view_target_top20_auroc_min": 0.55,
        "severity_1_uplift_over_reference_min": 0.0,
        "severity_3_uplift_over_reference_min": 0.0,
        "severity_3_minus_severity_1_min": 0.0,
    }


def test_amended_gate_preserves_failed_reference_as_diagnostic():
    result = amended_training_gate(_historical_gate())
    assert result["relative_core_passed"] is True
    assert result["relative_core_check_count"] == 7
    assert result["reference_tail_diagnostic"]["passed"] is False
    assert result["thresholds_retroactively_changed"] is False


def test_amended_gate_rejects_failed_relative_core():
    gate = _historical_gate()
    gate["checks"][2]["passed"] = False
    assert amended_training_gate(gate)["relative_core_passed"] is False


def test_heldout_transfer_gate_requires_monotonic_relative_response():
    assert heldout_family_transfer_gate(_evaluation(), _thresholds())["passed"] is True
    failed = heldout_family_transfer_gate(_evaluation(high=0.15), _thresholds())
    assert failed["passed"] is False
    assert next(
        row for row in failed["checks"] if row["metric"] == "severity_3_minus_severity_1"
    )["passed"] is False


def test_heldout_transfer_gate_rejects_non_glare_population():
    evaluation = _evaluation()
    evaluation["by_family"] = {"local_blur": {"score_mean": 0.3}}
    with pytest.raises(CounterfactualHeldoutError):
        heldout_family_transfer_gate(evaluation, _thresholds())
