import torch

from scripts.cache_density_descriptors import parse_feature_name
from scripts.fit_density_uq import split_routes
from uq_estimator.density import DensityUQEstimator, compute_view_moments


def test_compute_view_moments_shape_and_values():
    tokens = torch.tensor([[[[1.0, 2.0], [3.0, 6.0]]]])
    descriptor = compute_view_moments(tokens)
    expected = torch.tensor([[2.0, 4.0, 1.0, 2.0]])
    torch.testing.assert_close(descriptor, expected)


def test_route_split_has_no_leakage():
    routes = ["route_a", "route_a", "route_b", "route_c", "route_d", "route_e"]
    types = ["normal", "normal", "normal", "adverse", "adverse", "adverse"]
    assignment = split_routes(routes, types, seed=7, ratios=(0.6, 0.2, 0.2))
    assert set(assignment) == set(routes)
    assert set(assignment.values()).issubset({"train", "calibration", "test"})
    assert assignment["route_a"] == assignment["route_a"]


def test_parse_feature_name():
    route, weather = parse_feature_name(
        "Scenario_Town04_Route166_Weather10__00278.pt"
    )
    assert route == "Scenario_Town04_Route166_Weather10"
    assert weather == 10


def test_density_estimator_output_contract():
    descriptor_dim = 12
    embedding_dim = 4
    checkpoint = {
        "model_state": {
            "descriptor_mean": torch.zeros(descriptor_dim),
            "descriptor_scale": torch.ones(descriptor_dim),
            "pca_mean": torch.zeros(descriptor_dim),
            "pca_components": torch.eye(embedding_dim, descriptor_dim),
            "latent_mean": torch.zeros(embedding_dim),
            "whitening": torch.eye(embedding_dim),
            "calibration_distances": torch.tensor([0.5, 1.0, 2.0]),
            "output_projection": torch.cat(
                (torch.eye(embedding_dim), torch.zeros(2, embedding_dim)), dim=0
            ),
        }
    }
    model = DensityUQEstimator.from_checkpoint(checkpoint)
    tokens = torch.randn(2, 2, 5, 3)
    output = model(tokens)
    assert output.embedding.shape == (2, embedding_dim + 2)
    assert output.score.shape == (2, 1)
    assert output.active_embedding.shape == (2, embedding_dim)
    assert torch.all((output.score >= 0) & (output.score <= 1))
    torch.testing.assert_close(
        torch.linalg.vector_norm(output.embedding[:, :embedding_dim], dim=-1),
        torch.ones(2),
        atol=1e-5,
        rtol=1e-5,
    )

    half_output = model.half()(tokens.half())
    assert half_output.embedding.dtype == torch.float32
    assert torch.isfinite(half_output.embedding).all()
    assert torch.isfinite(half_output.score).all()
