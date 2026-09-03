import numpy as np

from scripts.audit_stage2l_v111_r_binding import (
    nearest_same_view_training_support,
    view_support_statistics,
)


def test_view_support_statistics_reports_exact_view_weight_and_recall():
    target = np.zeros((6, 10, 10), dtype=np.float32)
    probability = np.zeros_like(target)
    route = np.zeros_like(target)
    actor = np.zeros_like(target)
    target[0, 4, 5] = 1.0
    target[1, 2:4, 2:4] = 0.5
    route[0, 4, 5] = 1.0
    actor[1, 2:4, 2:4] = 1.0
    probability[0, 4, 5] = 0.9
    probability[1, 2, 2] = 0.2
    rows, support = view_support_statistics(
        target=target,
        probability=probability,
        route=route,
        actor=actor,
        support_fraction=0.1,
    )
    assert int(support.sum()) == 5
    assert rows["CAM_FRONT"]["foreground_cells"] == 1
    assert rows["CAM_FRONT_LEFT"]["foreground_cells"] == 4
    assert rows["CAM_FRONT"]["foreground_brier_weight_share"] == 0.2
    assert rows["CAM_FRONT_LEFT"]["foreground_brier_weight_share"] == 0.8
    assert rows["CAM_FRONT"]["foreground_recall"] == 1.0
    assert rows["CAM_FRONT_LEFT"]["foreground_recall"] == 0.25
    assert rows["CAM_FRONT_LEFT"]["actor_support_cells"] == 4


def test_nearest_same_view_training_support_is_deterministic():
    query = np.zeros(1024, dtype=np.float32)
    query[0] = 1.0
    same = query.copy()
    other = np.zeros_like(query)
    other[1] = 1.0
    result = nearest_same_view_training_support(
        query_feature=query,
        query_centroid_yx=[0.5, 0.5],
        training_rows=[
            {
                "group_id": "other",
                "event_id": "event_other",
                "feature": other,
                "centroid_yx": [0.5, 0.5],
            },
            {
                "group_id": "same",
                "event_id": "event_same",
                "feature": same,
                "centroid_yx": [0.25, 0.5],
            },
        ],
    )
    assert result["nearest_train_group"] == "same"
    assert result["nearest_train_event"] == "event_same"
    assert result["cosine_similarity"] == 1.0
    assert result["support_centroid_distance"] == 0.25


def test_nearest_same_view_training_support_returns_none_without_coverage():
    query = np.zeros(1024, dtype=np.float32)
    query[0] = 1.0
    assert (
        nearest_same_view_training_support(
            query_feature=query,
            query_centroid_yx=[0.5, 0.5],
            training_rows=[],
        )
        is None
    )
