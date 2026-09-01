import numpy as np

from scripts.audit_stage2l_v121_factorized_r_preflight import (
    component_support_statistics,
    identifiable_component_views,
)
from uq_estimator.task_relevance_geometry import CAMERA_ORDER


def test_component_statistics_keep_empty_view_out_of_positive_support():
    target = np.zeros((6, 10, 10), dtype=np.float32)
    probability = np.zeros_like(target)
    target[0, 4, 5] = 0.75
    probability[0, 4, 5] = 0.8
    result = component_support_statistics(
        target, probability, support_fraction=0.1
    )
    assert result["CAM_FRONT"]["positive"] is True
    assert result["CAM_FRONT"]["foreground_recall"] == 1.0
    assert result["CAM_FRONT_LEFT"]["positive"] is False
    assert result["CAM_FRONT_LEFT"]["foreground_recall"] is None
    assert result["CAM_FRONT_LEFT"]["foreground_cell_share_within_component"] == 0.0


def test_empty_component_uses_background_threshold_without_positive_cells():
    target = np.zeros((6, 10, 10), dtype=np.float32)
    probability = np.full_like(target, 0.6)
    result = component_support_statistics(
        target, probability, support_fraction=0.1
    )
    assert all(not result[view]["positive"] for view in CAMERA_ORDER)
    assert all(
        result[view]["background_false_positive_rate"] == 1.0
        for view in CAMERA_ORDER
    )


def test_identifiable_rule_requires_train_and_dev_independent_events():
    aggregate = {
        split: {
            component: {
                view: {"positive_event_count": 0} for view in CAMERA_ORDER
            }
            for component in ("route", "actor")
        }
        for split in ("train", "dev")
    }
    aggregate["train"]["actor"]["CAM_FRONT_LEFT"]["positive_event_count"] = 2
    aggregate["dev"]["actor"]["CAM_FRONT_LEFT"]["positive_event_count"] = 1
    aggregate["train"]["actor"]["CAM_FRONT_RIGHT"]["positive_event_count"] = 1
    aggregate["dev"]["actor"]["CAM_FRONT_RIGHT"]["positive_event_count"] = 1
    result = identifiable_component_views(
        aggregate, minimum_train_events=2, minimum_dev_events=1
    )
    assert {
        (row["component"], row["view"]) for row in result["supported"]
    } == {("actor", "CAM_FRONT_LEFT")}
    right = next(
        row
        for row in result["unsupported"]
        if row["component"] == "actor" and row["view"] == "CAM_FRONT_RIGHT"
    )
    assert right["reason"] == "insufficient_train_events"
