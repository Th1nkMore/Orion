import pytest

from uq_estimator.stage2l_u_concept_explicit_schema_v14_2 import (
    FIELD_VOCABULARIES,
)
from uq_estimator.stage2l_u_concept_qa_v14 import TAG_ORDER, UConceptSummary
from uq_estimator.stage2l_u_text_oracle_v15_2 import (
    decode_candidate_nlls,
    parse_text_oracle_answer,
    render_text_oracle_answer,
    render_text_oracle_summary,
    text_oracle_candidates,
    text_oracle_field_row,
    text_oracle_full_row,
)


def _summary(present: str = "yes") -> UConceptSummary:
    if present == "no":
        return UConceptSummary(
            present="no",
            view="none",
            region="none",
            level="none",
            trend="stable",
            component="none",
            peak=0.0,
            first_peak=0.0,
            latest_peak=0.0,
        )
    return UConceptSummary(
        present="yes",
        view="front_left",
        region="lower_right",
        level="high",
        trend="rising",
        component="transient_inconsistency",
        peak=0.9,
        first_peak=0.2,
        latest_peak=0.9,
    )


def test_text_oracle_summary_is_literal_natural_language_not_u_tokens():
    value = render_text_oracle_summary(_summary())
    assert "front-left camera view" in value
    assert "lower-right image region" in value
    assert "level is high" in value
    assert "temporal trend is rising" in value
    assert "transient-inconsistency" in value
    assert "<U_" not in value


def test_zero_u_summary_explicitly_states_every_nonapplicable_field():
    value = render_text_oracle_summary(_summary("no"))
    assert "No observation uncertainty is present" in value
    assert "no camera view" in value
    assert "no image region" in value
    assert "temporal trend is stable" in value


def test_every_field_row_exposes_all_candidates_and_only_plain_value_target():
    summary = _summary()
    for tag in TAG_ORDER:
        row = text_oracle_field_row(summary, tag)
        prompt = row["conversation"][0]["value"]
        answer = row["conversation"][1]["value"]
        assert render_text_oracle_summary(summary) in prompt
        assert answer == summary.fields()[tag]
        assert text_oracle_candidates(tag) == FIELD_VOCABULARIES[tag]
        assert all(value in prompt for value in FIELD_VOCABULARIES[tag])
        assert "task relevance" in prompt
        assert "Output only that value" in prompt


def test_full_answer_round_trip_is_strict_and_free_generation_is_secondary():
    summary = _summary()
    answer = render_text_oracle_answer(summary)
    assert parse_text_oracle_answer(answer) == summary.fields()
    assert text_oracle_full_row(summary)["conversation"][1]["value"] == answer
    with pytest.raises(ValueError):
        parse_text_oracle_answer(answer + "\n")
    with pytest.raises(ValueError):
        parse_text_oracle_answer(answer.replace("view=front_left", "view=left"))


def test_all_candidate_decode_selects_minimum_finite_nll():
    candidates = text_oracle_candidates("U_LEVEL")
    assert candidates == ("low", "medium", "high", "none")
    assert decode_candidate_nlls("U_LEVEL", [2.0, 0.8, 0.1, 4.0]) == "high"
    with pytest.raises(ValueError):
        decode_candidate_nlls("U_LEVEL", [0.0, float("nan"), 1.0, 2.0])
