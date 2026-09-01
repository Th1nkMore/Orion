from scripts.validate_stage2l_v13_1_capacity_reports import (
    compare_capacity,
    recompute_quality,
)


def _split(target_nll, *, full_minus, on_path, zero, step):
    return {
        "mean_target_nll": target_nll,
        "mean_step_nll": {
            "observation_semantics": step,
            "epistemic_limitation": step,
            "task_relevance": step,
            "driving_implication": step,
        },
        "full_minus_no_u_preference_fraction": full_minus,
        "full_overall_preference_fraction": 0.3125,
        "no_u_overall_preference_fraction": 0.25,
        "full_preference_fraction_by_variant": {
            "zero_uq": zero,
            "on_path_uq": on_path,
            "off_path_uq": 0.75,
            "view_shuffled_uq": 0.25,
        },
        "no_u_preference_fraction_by_variant": {
            "zero_uq": zero,
            "on_path_uq": 0.0,
            "off_path_uq": 0.75,
            "view_shuffled_uq": 0.0,
        },
    }


def _report(after_nll, step_nll):
    before = {
        "train": _split(13.0, full_minus=0.0, on_path=0.0, zero=0.0, step=14.0),
        "dev": _split(14.0, full_minus=0.0, on_path=0.0, zero=0.0, step=15.0),
    }
    after = {
        "train": _split(after_nll, full_minus=0.0625, on_path=0.0, zero=0.25, step=step_nll),
        "dev": _split(after_nll, full_minus=0.0625, on_path=0.0, zero=0.25, step=step_nll),
    }
    return {"language_before": before, "language_after": after}


def test_recompute_quality_keeps_preference_failures_soft() -> None:
    protocol = {
        "language_diagnostics": {
            "minimum_full_minus_no_u_fraction": 0.25,
            "minimum_on_path_preference_fraction": 0.75,
            "minimum_zero_u_preference_fraction": 0.75,
        }
    }
    quality = recompute_quality(_report(1.0, 2.0), protocol)
    assert quality == {
        "train_target_nll_improved": True,
        "dev_target_nll_improved": True,
        "dev_full_preference_above_no_u": False,
        "dev_on_path_preference": False,
        "dev_zero_u_preference": False,
    }


def test_capacity_comparison_separates_fit_from_u_semantics() -> None:
    reports = {
        "lora": _report(4.0, 5.0),
        "partial_unfreeze": _report(0.4, 1.0),
    }
    result = compare_capacity(reports)
    assert result["before_metrics_identical"] is True
    assert result["partial_dev_target_nll_lower"] is True
    assert all(result["partial_dev_step_nll_lower"].values())
    assert result["dev_preference_diagnostics_identical"] is True
    assert result["decision"] == (
        "capacity_increases_likelihood_fit_but_not_counterfactual_u_semantics"
    )
