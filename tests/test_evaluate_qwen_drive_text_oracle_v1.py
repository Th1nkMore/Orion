from scripts.evaluate_qwen_drive_text_oracle_v1 import (
    FIELD_VOCABULARIES,
    TAG_ORDER,
    U_VARIANTS,
    aggregate_records,
    field_prompt,
    load_frozen_states,
    render_text_oracle_summary,
)


def _fields(index=0):
    return {
        "U_PRESENT": "yes",
        "U_VIEW": ("front", "front_left")[index % 2],
        "U_REGION": "upper_right",
        "U_LEVEL": "medium",
        "U_TREND": "rising",
        "U_COMPONENT": "persistent_magnitude",
    }


def test_prompt_is_exactly_the_v15_2_prompt():
    fields = _fields()
    description = render_text_oracle_summary(fields)
    assert description == (
        "Observation uncertainty is present. Its strongest location is in the "
        "front camera view and the upper-right image region. Its overall "
        "uncertainty level is medium. From the first frame to the latest frame, "
        "its temporal trend is rising. Its dominant uncertainty component is "
        "persistent-magnitude."
    )
    assert field_prompt(fields, "U_VIEW") == (
        "The following is an exact, authoritative, task-free description of "
        "observation uncertainty. Read the description literally. Do not infer "
        "task relevance, driving risk, an action, a trajectory, or control.\n"
        f"Observation-uncertainty description: {description}\n"
        "Question: Which camera view contains the strongest observation uncertainty?\n"
        "Answer with exactly one canonical value from this list: front, front_left, "
        "front_right, rear, rear_left, rear_right, none. Output only that value, "
        "with no explanation or punctuation."
    )


def test_extracts_the_frozen_120_state_contract():
    records = {}
    for group_index in range(20):
        group_id = "group_%02d" % group_index
        for variant_index, variant in enumerate(U_VARIANTS):
            key = "%s::%s" % (group_id, variant)
            fields = _fields(variant_index)
            records[key] = {
                "group_id": group_id,
                "variant": variant,
                "fields": {
                    tag: {"expected": value} for tag, value in fields.items()
                },
            }
    report = {
        "schema": "orion.stage2l-v15-2-text-oracle-localization/v1",
        "status": "text_oracle_localization_complete",
        "training_performed": False,
        "continuous_u_tokens_present": False,
        "model_controls": {
            "original_orion": {"records": records},
            "v15_lora": {"records": records},
        },
    }
    states = load_frozen_states(report)
    assert len(states) == 120
    assert {value["group_id"] for value in states.values()} == {
        "group_%02d" % index for index in range(20)
    }


def test_aggregate_reports_perfect_counterfactual_response():
    records = {}
    for group_index in range(20):
        group_id = "group_%02d" % group_index
        for variant_index, variant in enumerate(U_VARIANTS):
            fields = _fields(variant_index)
            records["%s::%s" % (group_id, variant)] = {
                "group_id": group_id,
                "variant": variant,
                "fields": {
                    tag: {
                        "expected": value,
                        "predicted": value,
                        "correct": True,
                        "candidate_nlls": {
                            candidate: float(candidate != value)
                            for candidate in FIELD_VOCABULARIES[tag]
                        },
                    }
                    for tag, value in fields.items()
                },
            }
    result = aggregate_records(records)
    assert result["field_decision_count"] == 720
    assert result["nonzero_accuracy_excluding_presence"] == 1.0
    assert result["counterfactual"]["changed_field_exact_response_fraction"] == 1.0
