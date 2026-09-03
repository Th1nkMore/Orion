import pytest

from uq_estimator.stage2l_qa_contract_v3 import (
    FAMILY_RESPONSE_TAGS,
    HARD_STANCE_VARIANTS,
    QUESTION_FAMILIES,
    generation_contract_metrics,
)


def _answer(family, stance="maintain"):
    body = {
        "observation_semantics": "Observation uncertainty is low and stable.",
        "epistemic_limitation": "The signal marks unreliable evidence but not hidden facts.",
        "task_relevance": "The uncertain region has low task relevance.",
        "driving_implication": (
            "The uncertainty-aware planning stance is %s. This is a planning "
            "implication, not a direct brake or steering command." % stance
        ),
    }[family]
    return "%s %s" % (FAMILY_RESPONSE_TAGS[family], body)


def test_generation_contract_requires_family_tags_and_structured_stance_agreement():
    targets = {
        "zero_uq": "maintain",
        "off_path_uq": "maintain",
        "on_path_uq": "prepare_to_yield",
    }
    generated = {
        variant: {
            family: _answer(family, targets[variant])
            for family in QUESTION_FAMILIES
        }
        for variant in HARD_STANCE_VARIANTS
    }
    metrics = generation_contract_metrics(generated, targets)
    assert metrics["family_tag_parse_and_accuracy"] == 1.0
    assert metrics["nonrepeating_text_fraction"] == 1.0
    assert metrics["hard_driving_stance_parse_rate"] == 1.0
    assert metrics["hard_driving_stance_agreement"] == 1.0

    generated["on_path_uq"]["driving_implication"] = (
        "<planning_stance> The uncertainty-aware planning stance is maintain."
    )
    metrics = generation_contract_metrics(generated, targets)
    assert metrics["hard_driving_stance_agreement"] < 1.0

    generated["on_path_uq"]["driving_implication"] = (
        "<planning_stance> go more go more go more go more"
    )
    metrics = generation_contract_metrics(generated, targets)
    assert metrics["nonrepeating_text_fraction"] < 1.0


def test_generation_contract_fails_closed_on_wrong_or_missing_tag():
    targets = {variant: "maintain" for variant in HARD_STANCE_VARIANTS}
    generated = {
        variant: {
            family: _answer(family, "maintain")
            for family in QUESTION_FAMILIES
        }
        for variant in HARD_STANCE_VARIANTS
    }
    generated["zero_uq"]["task_relevance"] = "No structured family tag here."
    metrics = generation_contract_metrics(generated, targets)
    assert metrics["family_tag_parse_and_accuracy"] == pytest.approx(11 / 12)
