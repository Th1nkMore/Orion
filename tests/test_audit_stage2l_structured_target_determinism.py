import numpy as np

from scripts.audit_stage2l_structured_target_determinism import (
    reconstruct_summary,
)


CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
LEVELS = {"medium": 0.33, "high": 0.66}
STANCE = {"caution": 0.25, "prepare_to_yield": 0.55}


def _decode(u, r):
    return reconstruct_summary(
        u,
        r,
        camera_order=CAMERAS,
        level_thresholds=LEVELS,
        stance_thresholds=STANCE,
        rearward_high_risk_stance_cap="caution",
    )


def test_zero_uq_has_absence_semantics_without_arbitrary_argmax_location():
    u = np.zeros((4, 6, 9, 9), dtype=np.float32)
    r = np.ones((6, 9, 9), dtype=np.float32)
    result = _decode(u, r)
    assert result["observation_uncertainty"]["peak_view"] == "none"
    assert result["relevance_at_most_uncertain_region"]["level"] == "not_applicable"
    assert result["task_risk"] == {
        "level": "none",
        "peak_score": 0.0,
        "peak_view": "none",
        "peak_region": "none",
    }
    assert result["planning_implication"]["stance"] == "maintain"


def test_on_path_front_signal_decodes_view_region_level_and_yield():
    u = np.zeros((4, 6, 9, 9), dtype=np.float32)
    r = np.zeros((6, 9, 9), dtype=np.float32)
    u[:, 0, 8, 4] = [0.1, 0.3, 0.6, 0.9]
    r[0, 8, 4] = 0.8
    result = _decode(u, r)
    assert result["observation_uncertainty"]["temporal_trend"] == "rising"
    assert result["relevance_at_most_uncertain_region"]["level"] == "high"
    assert result["task_risk"]["level"] == "high"
    assert result["task_risk"]["peak_view"] == "CAM_FRONT"
    assert result["task_risk"]["peak_region"] == "lower_center"
    assert result["planning_implication"]["stance"] == "prepare_to_yield"


def test_rearward_high_risk_is_capped_at_caution():
    u = np.zeros((4, 6, 9, 9), dtype=np.float32)
    r = np.zeros((6, 9, 9), dtype=np.float32)
    u[:, 4, 4, 0] = 0.9
    r[4, 4, 0] = 0.9
    result = _decode(u, r)
    assert result["task_risk"]["peak_view"] == "CAM_BACK_LEFT"
    assert result["task_risk"]["peak_region"] == "middle_left"
    assert result["planning_implication"] == {
        "stance": "caution",
        "risk_bearing": "rearward",
        "is_direct_control_command": False,
    }
