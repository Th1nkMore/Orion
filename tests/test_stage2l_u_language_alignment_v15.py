from collections import Counter

import torch

from uq_estimator.stage2l_u_concept_explicit_schema_v14_2 import (
    FIELD_VOCABULARIES,
)
from uq_estimator.stage2l_u_concept_qa_v14 import (
    TAG_ORDER,
    U_VARIANTS,
    UConceptSummary,
)
from uq_estimator.stage2l_u_language_alignment_v15 import (
    all_candidate_cross_entropy,
    build_balanced_field_schedule,
    exact_nll_gradient_coefficients,
    field_qa_and_candidates,
    target_margin,
)


def _summary_for(tag: str, value: str) -> UConceptSummary:
    fields = {
        "U_PRESENT": "yes",
        "U_VIEW": "front",
        "U_REGION": "middle_center",
        "U_LEVEL": "medium",
        "U_TREND": "stable",
        "U_COMPONENT": "mixed",
    }
    fields[tag] = value
    if fields["U_PRESENT"] == "no":
        fields.update(
            U_VIEW="none",
            U_REGION="none",
            U_LEVEL="none",
            U_TREND="stable",
            U_COMPONENT="none",
        )
    return UConceptSummary(
        present=fields["U_PRESENT"],
        view=fields["U_VIEW"],
        region=fields["U_REGION"],
        level=fields["U_LEVEL"],
        trend=fields["U_TREND"],
        component=fields["U_COMPONENT"],
        peak=0.5,
        first_peak=0.5,
        latest_peak=0.5,
    )


def _complete_summaries():
    summaries = {}
    groups = []
    index = 0
    for tag in TAG_ORDER:
        for value in FIELD_VOCABULARIES[tag]:
            group = "group-%03d" % index
            groups.append(group)
            for variant in U_VARIANTS:
                summaries[(group, variant)] = _summary_for(tag, value)
            index += 1
    return tuple(groups), summaries


def test_balanced_schedule_covers_fields_and_each_canonical_value():
    groups, summaries = _complete_summaries()
    steps = len(TAG_ORDER) * 70
    schedule = build_balanced_field_schedule(
        group_ids=groups,
        summaries=summaries,
        optimizer_steps=steps,
        seed=7,
    )
    assert len(schedule) == steps
    field_counts = Counter(item.tag for item in schedule)
    assert set(field_counts.values()) == {70}
    for tag in TAG_ORDER:
        observed = {item.target for item in schedule if item.tag == tag}
        assert observed == set(FIELD_VOCABULARIES[tag])
    assert schedule == build_balanced_field_schedule(
        group_ids=groups,
        summaries=summaries,
        optimizer_steps=steps,
        seed=7,
    )


def test_field_prompt_candidates_and_target_are_exactly_aligned():
    summary = _summary_for("U_VIEW", "rear_left")
    row, answers, target_index = field_qa_and_candidates(summary, "U_VIEW")
    assert "exactly this one-line schema" in row["conversation"][0]["value"]
    assert answers[target_index] == "<U_VIEW> rear_left"
    assert len(answers) == len(FIELD_VOCABULARIES["U_VIEW"])


def test_all_candidate_loss_and_replayed_coefficients_have_exact_gradient():
    nlls = torch.tensor([1.4, 0.8, 2.1, 1.2], requires_grad=True)
    loss = all_candidate_cross_entropy(nlls, 1)
    expected_gradient = torch.autograd.grad(loss, nlls)[0]
    coefficients = exact_nll_gradient_coefficients(nlls.detach(), 1)
    assert torch.allclose(coefficients, expected_gradient)
    assert float(loss) > 0.0
    assert abs(float(coefficients.sum())) < 1e-6


def test_target_margin_is_positive_only_when_target_is_best():
    assert target_margin([2.0, 0.5, 1.0], 1) == 0.5
    assert target_margin([0.2, 0.5, 1.0], 1) == -0.3
