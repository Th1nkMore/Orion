import copy

from scripts.upgrade_stage2l_v9_qa_records import (
    HARD_STANCE_VARIANTS,
    MATCHED_VARIANTS,
    QUESTION_FAMILIES,
    audit_records,
    normalize_structured_summary,
    upgrade_records,
)


def _summary(variant):
    zero = variant == "zero_uq"
    off = variant == "off_path_uq"
    return {
        "observation_uncertainty": {
            "level": "low" if zero else "high",
            "peak_score": 0.0 if zero else 0.9,
            "peak_view": "CAM_FRONT",
            "peak_region": "upper_left" if zero else "lower_center",
            "temporal_trend": "stable" if zero else "rising",
            "temporal_peak_region_delta": 0.0 if zero else 0.5,
        },
        "relevance_at_most_uncertain_region": {
            "level": "low" if zero or off else "high",
            "score": 0.0 if zero or off else 0.7,
        },
        "task_risk": {
            "level": "low" if zero or off else "medium",
            "peak_score": 0.0 if zero or off else 0.6,
            "peak_view": "CAM_FRONT",
            "peak_region": "upper_left" if zero or off else "lower_center",
        },
        "planning_implication": {
            "stance": "prepare_to_yield" if variant == "on_path_uq" else "maintain",
            "risk_bearing": "forward_or_crossing",
            "is_direct_control_command": False,
        },
    }


def _records(group_count=1):
    rows = []
    for group in range(group_count):
        for variant in MATCHED_VARIANTS:
            for family in QUESTION_FAMILIES:
                rows.append({
                    "schema": "orion.uq_relevance_qa_record.v4",
                    "sample_id": f"g{group}/{variant}/{family}",
                    "question_family": family,
                    "counterfactual": {
                        "group_id": f"g{group}",
                        "variant": variant,
                    },
                    "conversation": [
                        {"from": "human", "value": "q"},
                        {"from": "gpt", "value": "old"},
                    ],
                    "target": {"structured_summary": _summary(variant)},
                    "model_input": {"unchanged": True},
                })
    return rows


def test_normalize_removes_zero_signal_and_zero_risk_argmax_artifacts():
    zero = normalize_structured_summary(_summary("zero_uq"))
    assert zero["observation_uncertainty"]["peak_view"] == "none"
    assert zero["relevance_at_most_uncertain_region"]["level"] == "not_applicable"
    assert zero["task_risk"] == {
        "level": "none",
        "peak_score": 0.0,
        "peak_view": "none",
        "peak_region": "none",
    }
    off = normalize_structured_summary(_summary("off_path_uq"))
    assert off["observation_uncertainty"]["peak_view"] == "CAM_FRONT"
    assert off["task_risk"]["peak_view"] == "none"
    assert off["task_risk"]["peak_region"] == "none"


def test_upgrade_preserves_inputs_and_assigns_only_owned_field_targets():
    source = _records()
    original = copy.deepcopy(source)
    upgraded, audit = upgrade_records(source)
    assert audit["passed"]
    assert source == original
    assert all(
        row["model_input"] == {"unchanged": True} for row in upgraded
    )
    for row in upgraded:
        family = row["question_family"]
        variant = row["counterfactual"]["variant"]
        fields = row["target"]["vlm_task_field_targets"]
        if family == "task_relevance":
            assert set(fields) == {
                "relevance_level",
                "risk_level",
                "risk_view",
                "risk_region",
            }
        elif family == "driving_implication" and variant in HARD_STANCE_VARIANTS:
            assert set(fields) == {"stance"}
        else:
            assert fields == {}


def test_audit_rejects_reintroduced_zero_risk_location():
    upgraded, _ = upgrade_records(_records())
    target = next(
        row
        for row in upgraded
        if row["counterfactual"]["variant"] == "off_path_uq"
        and row["question_family"] == "task_relevance"
    )
    target["target"]["structured_summary"]["task_risk"]["peak_view"] = "CAM_FRONT"
    audit = audit_records(upgraded)
    assert not audit["passed"]
    assert any("arbitrary location" in row["error"] for row in audit["errors"])
