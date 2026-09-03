from scripts.upgrade_stage2l_v8_qa_records import upgrade_records


VARIANTS = ("observed", "zero_uq", "on_path_uq", "off_path_uq", "view_shuffled_uq")
FAMILIES = ("observation_semantics", "epistemic_limitation", "task_relevance", "driving_implication")


def _summary(index, variant):
    stance = "maintain"
    if variant == "on_path_uq":
        stance = "caution" if index < 2 else "prepare_to_yield"
    return {
        "observation_uncertainty": {
            "level": "high" if variant != "zero_uq" else "low",
            "peak_view": "CAM_FRONT",
            "peak_region": "cell_%s" % variant,
            "temporal_trend": "rising",
        },
        "relevance_at_most_uncertain_region": {"level": "high" if variant == "on_path_uq" else "low"},
        "task_risk": {
            "level": "high" if variant == "on_path_uq" else "low",
            "peak_view": "CAM_FRONT",
            "peak_region": "risk_%s" % variant,
        },
        "planning_implication": {"stance": stance, "is_direct_control_command": False},
    }


def _records():
    rows = []
    for index in range(5):
        for variant in VARIANTS:
            for family in FAMILIES:
                rows.append({
                    "schema": "old",
                    "sample_id": "g%s/%s/%s" % (index, variant, family),
                    "counterfactual": {"group_id": "g%s" % index, "variant": variant},
                    "question_family": family,
                    "conversation": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "old"}],
                    "target": {"structured_summary": _summary(index, variant)},
                })
    return rows


def test_upgrade_builds_complete_structured_contract():
    upgraded, audit = upgrade_records(_records())
    assert len(upgraded) == 100
    assert audit["passed"] is True
    assert audit["hard_language_record_count"] == 90
    assert audit["hard_stance_record_count"] == 15
    assert audit["stance_class_counts"] == {"maintain": 10, "caution": 2, "prepare_to_yield": 3}
    assert all("<" not in row["conversation"][1]["value"] for row in upgraded)
