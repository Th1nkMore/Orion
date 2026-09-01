import copy

import pytest

from uq_estimator.stage2l_qa_contract_v5 import (
    QUESTION_FAMILIES,
    deterministic_render_metrics,
    expected_semantic_fields,
    expected_task_field_targets,
    parse_semantic_fields,
    render_structured_answer,
)


def _summary(score=0.0):
    return {
        "observation_uncertainty": {
            "level": "low" if score == 0.0 else "high",
            "peak_score": score,
            "peak_view": "CAM_FRONT",
            "peak_region": "upper_left",
            "temporal_trend": "stable" if score == 0.0 else "rising",
        },
        "relevance_at_most_uncertain_region": {
            "level": "low" if score == 0.0 else "high",
            "score": score,
        },
        "task_risk": {
            "level": "low" if score == 0.0 else "medium",
            "peak_score": score,
            "peak_view": "CAM_FRONT",
            "peak_region": "lower_center",
        },
        "planning_implication": {
            "stance": "maintain" if score == 0.0 else "caution",
            "is_direct_control_command": False,
        },
    }


def test_zero_uq_uses_explicit_no_signal_sentinels():
    summary = _summary(0.0)
    observation = expected_semantic_fields("observation_semantics", summary)
    epistemic = expected_semantic_fields("epistemic_limitation", summary)
    relevance = expected_semantic_fields("task_relevance", summary)
    assert observation["uq_view"] == "none"
    assert observation["uq_region"] == "none"
    assert epistemic == {
        "evidence": "not_flagged",
        "evidence_view": "none",
        "evidence_region": "none",
        "hidden_content": "not_applicable",
        "task_relevance": "separate",
    }
    assert relevance == {
        "relevance_level": "not_applicable",
        "risk_level": "none",
        "risk_view": "none",
        "risk_region": "none",
    }


def test_nonzero_uq_preserves_location_and_epistemic_limit():
    summary = _summary(0.8)
    observation = expected_semantic_fields("observation_semantics", summary)
    epistemic = expected_semantic_fields("epistemic_limitation", summary)
    assert observation["uq_view"] == "CAM_FRONT"
    assert observation["uq_region"] == "upper_left"
    assert epistemic["evidence"] == "unreliable"
    assert epistemic["hidden_content"] == "unknown"


def test_nonzero_off_path_uq_does_not_invent_task_risk_location():
    summary = _summary(0.8)
    summary["relevance_at_most_uncertain_region"] = {
        "level": "low",
        "score": 0.0,
    }
    summary["task_risk"] = {
        "level": "low",
        "peak_score": 0.0,
        "peak_view": "CAM_FRONT",
        "peak_region": "upper_left",
    }
    fields = expected_semantic_fields("task_relevance", summary)
    assert fields == {
        "relevance_level": "low",
        "risk_level": "none",
        "risk_view": "none",
        "risk_region": "none",
    }
    assert expected_task_field_targets(summary) == {
        **fields,
        "stance": "caution",
    }


def test_deterministic_render_round_trips_all_families():
    summary = _summary(0.8)
    for family in QUESTION_FAMILIES:
        text = render_structured_answer(family, summary)
        assert parse_semantic_fields(text, family) == expected_semantic_fields(
            family, summary
        )


def test_field_metrics_score_predictions_not_free_generation_format():
    target = _summary(0.8)
    predictions = {
        "on_path_uq": {
            family: expected_semantic_fields(family, target)
            for family in QUESTION_FAMILIES
        }
    }
    metrics = deterministic_render_metrics(
        predictions, {"on_path_uq": target}
    )
    assert metrics["semantic_parse_rate"] == 1.0
    assert metrics["semantic_field_accuracy"] == 1.0
    assert metrics["semantic_answer_exact_match"] == 1.0


def test_renderer_rejects_unrecognized_field_values():
    summary = _summary(0.8)
    fields = expected_semantic_fields("task_relevance", summary)
    fields = copy.deepcopy(fields)
    fields["risk_level"] = "catastrophic"
    with pytest.raises(ValueError, match="unsupported semantic field value"):
        from uq_estimator.stage2l_qa_contract_v5 import render_semantic_fields

        render_semantic_fields("task_relevance", fields)
