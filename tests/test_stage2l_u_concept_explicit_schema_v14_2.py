import pytest

from uq_estimator.stage2l_u_concept_explicit_schema_v14_2 import (
    FIELD_VOCABULARIES,
    build_explicit_u_qa_row,
    candidate_answers,
    decode_candidate_nlls,
    explicit_u_request,
    parse_strict_u_answer,
)
from uq_estimator.stage2l_u_concept_qa_v14 import TAG_ORDER, UConceptSummary


def _summary() -> UConceptSummary:
    return UConceptSummary(
        present="yes",
        view="front",
        region="middle_center",
        level="high",
        trend="rising",
        component="persistent_direction",
        peak=0.9,
        first_peak=0.4,
        latest_peak=0.9,
    )


def test_full_prompt_exposes_literal_order_values_and_no_extra_text_rule():
    prompt = explicit_u_request()
    positions = [prompt.index("<%s>" % tag) for tag in TAG_ORDER]
    assert positions == sorted(positions)
    for tag, values in FIELD_VOCABULARIES.items():
        assert "<%s> [%s]" % (tag, "|".join(values)) in prompt
    assert "Output no explanation" in prompt
    row = build_explicit_u_qa_row(_summary())
    assert row["conversation"][0]["value"] == prompt
    assert parse_strict_u_answer(row["conversation"][1]["value"]) == dict(
        _summary().fields()
    )


def test_single_field_prompt_and_candidates_are_exactly_aligned():
    row = build_explicit_u_qa_row(_summary(), "U_VIEW")
    prompt = row["conversation"][0]["value"]
    assert "exactly this one-line schema" in prompt
    assert "<U_VIEW> [front|front_left|front_right|rear|rear_left|rear_right|none]" in prompt
    assert candidate_answers("U_VIEW") == tuple(
        "<U_VIEW> %s" % value for value in FIELD_VOCABULARIES["U_VIEW"]
    )
    assert parse_strict_u_answer(row["conversation"][1]["value"], "U_VIEW") == {
        "U_VIEW": "front"
    }


@pytest.mark.parametrize(
    "text",
    (
        "<U_PRESENT> yes\n<U_VIEW> front",
        "prefix\n<U_PRESENT> yes\n<U_VIEW> front\n<U_REGION> middle_center\n<U_LEVEL> high\n<U_TREND> rising\n<U_COMPONENT> persistent_direction",
        "<U_PRESENT> yes\n<U_VIEW> front\n<U_REGION> middle_center\n<U_LEVEL> high\n<U_TREND> rising\n<U_COMPONENT> persistent_direction\n",
        "<U_PRESENT> maybe\n<U_VIEW> front\n<U_REGION> middle_center\n<U_LEVEL> high\n<U_TREND> rising\n<U_COMPONENT> persistent_direction",
        "<U_PRESENT> yes\n<U_VW> front\n<U_REGION> middle_center\n<U_LEVEL> high\n<U_TREND> rising\n<U_COMPONENT> persistent_direction",
    ),
)
def test_strict_parser_rejects_partial_repaired_or_extra_output(text):
    with pytest.raises(ValueError):
        parse_strict_u_answer(text)


def test_candidate_decoder_is_finite_and_vocabulary_bound():
    nlls = [3.0] * len(FIELD_VOCABULARIES["U_LEVEL"])
    nlls[2] = 0.1
    assert decode_candidate_nlls("U_LEVEL", nlls) == "high"
    with pytest.raises(ValueError):
        decode_candidate_nlls("U_LEVEL", [0.1])
    with pytest.raises(ValueError):
        decode_candidate_nlls(
            "U_LEVEL", [float("nan")] * len(FIELD_VOCABULARIES["U_LEVEL"])
        )
