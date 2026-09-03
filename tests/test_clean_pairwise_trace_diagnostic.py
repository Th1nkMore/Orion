from scripts.verify_clean_pairwise_trace_diagnostic import (
    CAMERA_ORDER,
    validate_observation_record,
)


def _observation():
    return {
        "checkpoint_sha256": "abc",
        "camera_order": list(CAMERA_ORDER),
        "pooled_grids": [
            [[float(view)] * 10 for _ in range(10)]
            for view in range(len(CAMERA_ORDER))
        ],
    }


def test_all_view_observation_contract_accepts_six_finite_grids():
    assert validate_observation_record(_observation(), "abc") == []


def test_all_view_observation_contract_rejects_wrong_hash_and_grid_shape():
    observation = _observation()
    observation["pooled_grids"][2] = [[1.0]]
    errors = validate_observation_record(observation, "different")
    assert "checkpoint_sha256" in errors
    assert "pooled_grid_2_height" in errors
