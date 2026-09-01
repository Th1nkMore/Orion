import copy
import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "uq_estimator"
    / "stage2l_qa_contract_v4.py"
)
_SPEC = importlib.util.spec_from_file_location("stage2l_qa_contract_v4", _MODULE_PATH)
contract = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(contract)
QUESTION_FAMILIES = contract.QUESTION_FAMILIES
expected_semantic_fields = contract.expected_semantic_fields
generation_semantic_metrics = contract.generation_semantic_metrics
parse_semantic_fields = contract.parse_semantic_fields
render_structured_answer = contract.render_structured_answer
same_family_unique_structured_answers = contract.same_family_unique_structured_answers


def _summary(level="high", stance="prepare_to_yield", region="lower_center"):
    return {
        "observation_uncertainty": {
            "level": level,
            "peak_view": "CAM_FRONT",
            "peak_region": region,
            "temporal_trend": "rising",
        },
        "relevance_at_most_uncertain_region": {"level": "high"},
        "task_risk": {
            "level": "high",
            "peak_view": "CAM_FRONT",
            "peak_region": region,
        },
        "planning_implication": {
            "stance": stance,
            "is_direct_control_command": False,
        },
    }


def test_structured_answers_round_trip_all_semantic_fields():
    summary = _summary()
    for family in QUESTION_FAMILIES:
        answer = render_structured_answer(family, summary)
        assert parse_semantic_fields(answer, family) == expected_semantic_fields(
            family, summary
        )
        assert "<" not in answer and ">" not in answer


def test_generation_metrics_score_semantics_separately_from_prefix():
    summaries = {
        "zero_uq": _summary(level="low", stance="maintain", region="upper_left"),
        "off_path_uq": _summary(level="high", stance="maintain", region="lower_left"),
        "on_path_uq": _summary(),
    }
    generated = {
        variant: {
            family: render_structured_answer(family, summary)
            for family in QUESTION_FAMILIES
        }
        for variant, summary in summaries.items()
    }
    # Remove one human-readable prefix while keeping all semantic fields.
    generated["zero_uq"]["driving_implication"] = generated["zero_uq"][
        "driving_implication"
    ].split(":", 1)[1]
    metrics = generation_semantic_metrics(generated, summaries)
    assert metrics["semantic_answer_exact_match"] == 1.0
    assert metrics["semantic_field_accuracy"] == 1.0
    assert metrics["format_prefix_accuracy_diagnostic"] == 11 / 12


def test_parser_rejects_missing_or_duplicate_fields():
    answer = render_structured_answer("driving_implication", _summary())
    with pytest.raises(ValueError, match="do not match"):
        parse_semantic_fields(answer.replace("; direct_control=no", ""), "driving_implication")
    with pytest.raises(ValueError, match="duplicate"):
        parse_semantic_fields(answer + " stance=maintain", "driving_implication")


def test_same_family_candidates_exclude_exact_duplicates():
    base = {
        "sample_id": "a",
        "counterfactual": {"group_id": "g", "variant": "zero_uq"},
        "question_family": "driving_implication",
        "target": {"structured_summary": _summary(stance="maintain")},
    }
    duplicate = copy.deepcopy(base)
    duplicate["sample_id"] = "b"
    different = copy.deepcopy(base)
    different["sample_id"] = "c"
    different["target"]["structured_summary"] = _summary(stance="caution")
    answers = same_family_unique_structured_answers(
        [base, duplicate, different], base
    )
    assert len(answers) == 2
    assert "stance=maintain" in answers[0]
    assert "stance=caution" in answers[1]
