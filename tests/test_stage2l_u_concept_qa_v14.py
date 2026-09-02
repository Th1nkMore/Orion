import torch

from uq_estimator.stage2l_u_concept_qa_v14 import (
    TAG_ORDER,
    U_VARIANTS,
    audit_u_variant_group,
    build_distribution_preserving_u_variants,
    build_u_qa_row,
    render_u_answer,
    summarize_u_components,
)


def _observed() -> torch.Tensor:
    value = torch.zeros(1, 4, 6, 40, 40, 3)
    value[0, 0, 0, 4, 5] = torch.tensor((0.2, 0.1, 0.1))
    value[0, 1, 0, 4, 5] = torch.tensor((0.4, 0.1, 0.1))
    value[0, 2, 0, 4, 5] = torch.tensor((0.7, 0.1, 0.1))
    value[0, 3, 0, 4, 5] = torch.tensor((1.0, 0.1, 0.1))
    return value


def test_zero_summary_and_exact_answer_contract():
    summary = summarize_u_components(torch.zeros(1, 4, 6, 40, 40, 3))
    assert summary.present == "no"
    assert summary.view == "none"
    assert summary.region == "none"
    answer = render_u_answer(summary)
    assert answer.splitlines() == [
        "<U_PRESENT> no",
        "<U_VIEW> none",
        "<U_REGION> none",
        "<U_LEVEL> none",
        "<U_TREND> stable",
        "<U_COMPONENT> none",
    ]
    row = build_u_qa_row(summary)
    assert row["conversation"][1]["value"] == answer
    assert "route" not in row["conversation"][0]["value"].lower()


def test_observed_summary_reports_spatial_temporal_component_semantics():
    summary = summarize_u_components(_observed())
    assert summary.present == "yes"
    assert summary.view == "front"
    assert summary.region == "upper_left"
    assert summary.level == "medium"
    assert summary.trend == "rising"
    assert summary.component == "persistent_direction"
    for tag in TAG_ORDER:
        row = build_u_qa_row(summary, tag)
        assert row["conversation"][1]["value"].startswith("<%s> " % tag)


def test_counterfactuals_preserve_observed_value_distribution():
    observed = _observed()
    variants = build_distribution_preserving_u_variants(observed, "group-1")
    assert tuple(variants) == U_VARIANTS
    reference = torch.sort(observed.flatten()).values
    for name in U_VARIANTS[2:]:
        assert torch.equal(torch.sort(variants[name].flatten()).values, reference)
    audit = audit_u_variant_group(variants)
    assert audit["passed"]
    assert audit["distinct_answer_count"] >= 4


def test_invalid_component_range_fails_closed():
    invalid = _observed()
    invalid[0, 0, 0, 0, 0, 0] = 1.1
    try:
        summarize_u_components(invalid)
    except ValueError as error:
        assert "[0,1]" in str(error)
    else:
        raise AssertionError("out-of-range U did not fail closed")
