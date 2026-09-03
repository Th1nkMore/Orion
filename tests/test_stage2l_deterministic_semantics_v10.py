import torch

from uq_estimator.stage2l_deterministic_semantics_v10 import (
    deterministic_task_semantics,
)


def test_zero_uq_decodes_explicit_absence_without_learned_classifier():
    u = torch.zeros(1, 6, 9, 9)
    r = torch.zeros_like(u)
    output = deterministic_task_semantics(u, r)
    assert output.learned_structured_field_classifier_used is False
    assert output.structured_fields[0] == {
        "relevance_level": "not_applicable",
        "risk_level": "none",
        "risk_view": "none",
        "risk_region": "none",
        "stance": "maintain",
        "direct_control": "no",
        "response_basis": "observation_uncertainty",
    }


def test_predicted_front_task_risk_decodes_geometry_and_stance():
    u = torch.zeros(1, 6, 9, 9)
    r = torch.full_like(u, -20.0)
    u[0, 0, 8, 4] = 0.9
    r[0, 0, 8, 4] = 20.0
    output = deterministic_task_semantics(u, r)
    assert output.structured_fields[0]["risk_view"] == "CAM_FRONT"
    assert output.structured_fields[0]["risk_region"] == "lower_center"
    assert output.structured_fields[0]["risk_level"] == "high"
    assert output.structured_fields[0]["stance"] == "prepare_to_yield"


def test_predicted_rear_task_risk_is_capped_at_caution():
    u = torch.zeros(1, 6, 9, 9)
    r = torch.full_like(u, -20.0)
    u[0, 4, 4, 0] = 0.9
    r[0, 4, 4, 0] = 20.0
    output = deterministic_task_semantics(u, r)
    assert output.structured_fields[0]["risk_view"] == "CAM_BACK_LEFT"
    assert output.structured_fields[0]["risk_region"] == "middle_left"
    assert output.structured_fields[0]["stance"] == "caution"
