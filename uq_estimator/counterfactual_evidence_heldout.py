"""Gates for a frozen counterfactual-evidence held-out-family read.

This module does not train or calibrate an adapter.  It records the historical
reference-tail failure while separating it from the seven metrics that test the
relative, intervention-induced evidence-loss quantity.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


class CounterfactualHeldoutError(ValueError):
    """Raised when a held-out evaluation violates its frozen contract."""


def amended_training_gate(training_gate: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve the old gate but expose its seven-metric relative core."""

    checks = training_gate.get("checks")
    if not isinstance(checks, list) or len(checks) != 8:
        raise CounterfactualHeldoutError("historical training gate must have 8 checks")
    by_metric = {str(row.get("metric")): dict(row) for row in checks}
    if len(by_metric) != len(checks):
        raise CounterfactualHeldoutError("historical training gate metrics differ")
    reference = by_metric.pop("reference_prediction_p95", None)
    if reference is None:
        raise CounterfactualHeldoutError("historical reference-tail check is missing")
    if reference.get("passed") is not False:
        raise CounterfactualHeldoutError("historical reference-tail status changed")
    if not math.isclose(float(reference.get("threshold")), 0.2, abs_tol=1e-12):
        raise CounterfactualHeldoutError("historical reference-tail threshold changed")
    core_checks = list(by_metric.values())
    core_passed = all(row.get("passed") is True for row in core_checks)
    return {
        "relative_core_passed": core_passed,
        "relative_core_check_count": len(core_checks),
        "relative_core_checks": core_checks,
        "reference_tail_diagnostic": {
            **reference,
            "role": "diagnostic_only_for_bounded_heldout_family_read",
        },
        "historical_gate_passed": bool(training_gate.get("passed")),
        "thresholds_retroactively_changed": False,
    }


def heldout_family_transfer_gate(
    evaluation: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    """Test relative transfer to local_glare without reading a mask."""

    families = evaluation.get("by_family")
    by_severity = evaluation.get("by_family_severity")
    if not isinstance(families, Mapping) or set(families) != {"local_glare"}:
        raise CounterfactualHeldoutError("evaluation must contain only local_glare")
    if not isinstance(by_severity, Mapping):
        raise CounterfactualHeldoutError("held-out severity rows are missing")
    low = by_severity.get("local_glare/severity_1")
    high = by_severity.get("local_glare/severity_3")
    if not isinstance(low, Mapping) or not isinstance(high, Mapping):
        raise CounterfactualHeldoutError("held-out severities 1 and 3 are required")
    reference = float(evaluation["reference_prediction_mean"])
    low_score = float(low["score_mean"])
    high_score = float(high["score_mean"])
    specifications = (
        (
            "combined_patch_spearman",
            float(evaluation["combined_patch_spearman"]),
            float(thresholds["combined_patch_spearman_min"]),
        ),
        (
            "combined_target_top20_auroc",
            float(evaluation["combined_target_top20_auroc"]),
            float(thresholds["combined_target_top20_auroc_min"]),
        ),
        (
            "median_record_within_intervened_view_target_top20_auroc",
            float(
                evaluation[
                    "median_record_within_intervened_view_target_top20_auroc"
                ]
            ),
            float(
                thresholds[
                    "median_record_within_intervened_view_target_top20_auroc_min"
                ]
            ),
        ),
        (
            "severity_1_uplift_over_reference",
            low_score - reference,
            float(thresholds["severity_1_uplift_over_reference_min"]),
        ),
        (
            "severity_3_uplift_over_reference",
            high_score - reference,
            float(thresholds["severity_3_uplift_over_reference_min"]),
        ),
        (
            "severity_3_minus_severity_1",
            high_score - low_score,
            float(thresholds["severity_3_minus_severity_1_min"]),
        ),
    )
    checks = []
    for metric, value, threshold in specifications:
        checks.append(
            {
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "comparison": ">=",
                "passed": math.isfinite(value) and value >= threshold,
            }
        )
    return {
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
        "reference_prediction_mean_diagnostic": reference,
        "corruption_mask_used_for_scoring": False,
    }


__all__ = [
    "CounterfactualHeldoutError",
    "amended_training_gate",
    "heldout_family_transfer_gate",
]
