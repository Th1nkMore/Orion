import torch

from uq_estimator.native_manifold_audit import (
    MANIFOLD_CANDIDATE_TAILS,
    route_balanced_manifold_raw_maps,
)


def test_route_balanced_manifold_is_leave_route_out_and_spatial():
    clean = torch.tensor(
        [
            [[[[1.00, 0.00], [0.98, 0.02]]]],
            [[[[0.99, 0.01], [0.97, 0.03]]]],
            [[[[0.00, 1.00], [0.02, 0.98]]]],
            [[[[0.01, 0.99], [0.03, 0.97]]]],
            [[[[0.70, 0.70], [0.72, 0.68]]]],
            [[[[0.68, 0.72], [0.70, 0.70]]]],
        ],
        dtype=torch.float32,
    )
    routes = ["a", "a", "b", "b", "c", "c"]
    calibration = route_balanced_manifold_raw_maps(
        clean,
        clean,
        routes,
        nearest_route_count=1,
        position_chunk_size=1,
        query_route_ids=routes,
        leave_query_route_out=True,
    )
    assert set(calibration) == set(MANIFOLD_CANDIDATE_TAILS)
    assert all(value.shape == (6, 1, 1, 2) for value in calibration.values())
    assert calibration["appearance_route_knn_cosine_z"].min() > 0

    queries = torch.stack((clean[0], torch.tensor([[[[[-1.0, 0.0], [-1.0, 0.0]]]]])[0]))
    native = route_balanced_manifold_raw_maps(
        clean,
        queries,
        routes,
        nearest_route_count=1,
        position_chunk_size=1,
    )
    for value in native.values():
        assert value.shape == (2, 1, 1, 2)
        assert value[1].mean() > value[0].mean()
