import torch

from uq_estimator.lens_waterdrop import apply_lens_waterdrop


def _images():
    values = torch.linspace(-1.0, 1.0, 1 * 6 * 3 * 48 * 64)
    return values.reshape(1, 6, 3, 48, 64)


def test_waterdrop_is_deterministic_actor_independent_and_view_local():
    images = _images()
    first = apply_lens_waterdrop(
        images, severity=2, view_indices=[0], seed=17, elapsed_seconds=0.5
    )
    second = apply_lens_waterdrop(
        images, severity=2, view_indices=[0], seed=17, elapsed_seconds=0.5
    )
    assert torch.equal(first.images, second.images)
    assert torch.equal(first.mask, second.mask)
    assert not torch.equal(first.images[:, 0], images[:, 0])
    assert torch.equal(first.images[:, 1:], images[:, 1:])
    assert first.metadata["placement_policy"].endswith("actor_independent")
    assert first.metadata["mask_fraction"] > 0.0


def test_waterdrop_changes_smoothly_with_elapsed_time_and_stays_finite():
    images = _images()
    at_zero = apply_lens_waterdrop(
        images, severity=3, view_indices=[0], seed=3, elapsed_seconds=0.0
    )
    nearby = apply_lens_waterdrop(
        images, severity=3, view_indices=[0], seed=3, elapsed_seconds=0.05
    )
    later = apply_lens_waterdrop(
        images, severity=3, view_indices=[0], seed=3, elapsed_seconds=2.0
    )
    near_delta = (nearby.images - at_zero.images).abs().mean()
    far_delta = (later.images - at_zero.images).abs().mean()
    assert 0.0 < float(near_delta) < float(far_delta)
    assert torch.isfinite(later.images).all()


def test_waterdrop_metadata_replays_geometry():
    result = apply_lens_waterdrop(
        _images(), severity=1, view_indices=[0, 2], seed=9, elapsed_seconds=1.25
    )
    assert result.metadata["schema_version"] == "orion.lens_waterdrop.v1"
    assert result.metadata["view_indices"] == [0, 2]
    assert set(result.metadata["droplets_by_view"]) == {"0", "2"}
    assert all(len(rows) == 4 for rows in result.metadata["droplets_by_view"].values())
