import json

import pytest
import torch

from uq_estimator.corruptions import (
    CORRUPTION_METADATA_SCHEMA_V1,
    corrupt_batch_images,
    corrupt_batch_images_with_metadata,
    corrupt_multiview_images,
    corrupt_multiview_images_with_metadata,
    normalized_front_tensor_to_bgr,
)


@pytest.mark.parametrize("name", ("blur", "dark", "camera_dropout"))
def test_corruption_preserves_shape_and_is_deterministic(name):
    images = torch.randn(1, 6, 3, 16, 16)
    first = corrupt_multiview_images(images, name, severity=2)
    second = corrupt_multiview_images(images, name, severity=2)
    assert first.shape == images.shape
    assert torch.equal(first, second)
    assert not torch.equal(first, images)


def test_corruption_rejects_invalid_severity():
    with pytest.raises(ValueError):
        corrupt_multiview_images(
            torch.randn(6, 3, 16, 16), "blur", severity=4
        )


def test_camera_dropout_can_target_a_specific_view():
    images = torch.randn(1, 6, 3, 8, 8)
    corrupted = corrupt_multiview_images(
        images,
        "camera_dropout",
        severity=1,
        view_indices=[3],
    )
    assert torch.equal(corrupted[:, :3], images[:, :3])
    assert not torch.equal(corrupted[:, 3], images[:, 3])
    assert torch.equal(corrupted[:, 4:], images[:, 4:])


def test_camera_dropout_rejects_invalid_view_indices():
    images = torch.randn(1, 6, 3, 8, 8)
    with pytest.raises(ValueError):
        corrupt_multiview_images(
            images,
            "camera_dropout",
            severity=1,
            view_indices=[6],
        )


def test_exact_front_tensor_preview_reverses_normalization_and_rgb_order():
    mean = torch.tensor((123.675, 116.28, 103.53)).view(1, 1, 3, 1, 1)
    std = torch.tensor((58.395, 57.12, 57.375)).view(1, 1, 3, 1, 1)
    rgb = torch.tensor((10.0, 20.0, 30.0)).view(1, 1, 3, 1, 1)
    normalized = ((rgb - mean) / std).expand(1, 6, 3, 2, 3).clone()

    batched = normalized_front_tensor_to_bgr(normalized)
    unbatched = normalized_front_tensor_to_bgr(normalized[0])

    assert batched.shape == (2, 3, 3)
    assert batched.dtype == torch.uint8
    assert torch.equal(batched, unbatched)
    assert torch.equal(batched[0, 0], torch.tensor([30, 20, 10], dtype=torch.uint8))


def test_exact_front_tensor_preview_rejects_ambiguous_or_unnormalized_inputs():
    with pytest.raises(ValueError, match="single-item batch"):
        normalized_front_tensor_to_bgr(torch.zeros(2, 6, 3, 4, 4))
    with pytest.raises(TypeError, match="floating-point"):
        normalized_front_tensor_to_bgr(torch.zeros(6, 3, 4, 4, dtype=torch.uint8))


@pytest.mark.parametrize(
    "name", ("local_blur", "local_dark", "local_glare", "local_occlusion")
)
def test_spatial_corruption_returns_exact_mask_and_metadata(name):
    images = torch.linspace(-2.0, 2.0, 2 * 6 * 3 * 32 * 48).reshape(
        2, 6, 3, 32, 48
    )
    result = corrupt_multiview_images_with_metadata(
        images,
        name,
        severity=2,
        view_indices=[0, 2],
        seed=17,
        region=(0.25, 0.25, 0.75, 0.75),
    )

    assert result.images.shape == images.shape
    assert result.mask.shape == (2, 6, 1, 32, 48)
    assert result.mask.dtype == torch.bool
    assert result.metadata.schema_version == CORRUPTION_METADATA_SCHEMA_V1
    assert result.metadata.corruption == name
    assert result.metadata.seed == 17
    assert result.metadata.severity == 2
    assert result.metadata.view_indices == (0, 2)
    assert result.metadata.normalized_region == (0.25, 0.25, 0.75, 0.75)
    assert result.mask[:, 0, :, 8:24, 12:36].all()
    assert result.mask[:, 2, :, 8:24, 12:36].all()
    assert not result.mask[:, 1].any()

    changed = (result.images != images).any(dim=2, keepdim=True)
    assert not changed.logical_and(~result.mask).any()
    assert changed.logical_and(result.mask).any()
    json.dumps(result.metadata.to_dict())


@pytest.mark.parametrize(
    "name", ("local_blur", "local_dark", "local_glare", "local_occlusion")
)
def test_spatial_corruption_seed_is_replayable_and_changes_location(name):
    images = torch.randn(1, 3, 3, 64, 64)
    first = corrupt_multiview_images_with_metadata(
        images, name, severity=1, view_indices=[0], seed=123
    )
    replay = corrupt_multiview_images_with_metadata(
        images, name, severity=1, view_indices=[0], seed=123
    )
    other = corrupt_multiview_images_with_metadata(
        images, name, severity=1, view_indices=[0], seed=124
    )

    assert torch.equal(first.images, replay.images)
    assert torch.equal(first.mask, replay.mask)
    assert first.metadata == replay.metadata
    assert first.metadata.normalized_region != other.metadata.normalized_region
    assert not torch.equal(first.mask, other.mask)


def test_explicit_region_overrides_random_location_for_on_off_path_pairs():
    images = torch.randn(1, 6, 3, 40, 80)
    on_path = corrupt_multiview_images_with_metadata(
        images,
        "local_occlusion",
        severity=2,
        view_indices=[0],
        seed=1,
        region=(0.5, 0.375, 1.0, 0.625),
    )
    replay = corrupt_multiview_images_with_metadata(
        images,
        "local_occlusion",
        severity=2,
        view_indices=[0],
        seed=999,
        region=(0.5, 0.375, 1.0, 0.625),
    )
    off_path = corrupt_multiview_images_with_metadata(
        images,
        "local_occlusion",
        severity=2,
        view_indices=[0],
        seed=1,
        region=(0.0, 0.0, 0.25, 0.25),
    )

    assert torch.equal(on_path.images, replay.images)
    assert torch.equal(on_path.mask, replay.mask)
    assert not torch.equal(on_path.mask, off_path.mask)
    assert on_path.metadata.seed != replay.metadata.seed


def test_metadata_api_preserves_rank_but_mask_is_always_batched():
    images = torch.randn(6, 3, 16, 20)
    result = corrupt_multiview_images_with_metadata(
        images,
        "local_dark",
        severity=1,
        view_indices=[4],
        seed=4,
        region=(0.0, 0.0, 0.5, 0.5),
    )
    assert result.images.shape == images.shape
    assert result.mask.shape == (1, 6, 1, 16, 20)


def test_metadata_camera_dropout_is_full_view_and_diagnostic():
    images = torch.randn(1, 6, 3, 12, 18)
    result = corrupt_multiview_images_with_metadata(
        images,
        "camera_dropout",
        severity=2,
        view_indices=[1, 3, 5],
        seed=9,
    )

    assert result.metadata.view_indices == (1, 3)
    assert result.metadata.normalized_region == (0.0, 0.0, 1.0, 1.0)
    assert result.metadata.parameters["diagnostic_only"] is True
    assert result.metadata.parameters["candidate_view_indices"] == [1, 3, 5]
    assert result.mask[:, 1].all()
    assert result.mask[:, 3].all()
    assert not result.mask[:, 0].any()
    assert not result.mask[:, 5].any()
    assert torch.equal(result.images[:, 0], images[:, 0])


def test_batch_metadata_api_deep_copies_and_legacy_api_stays_dict_only():
    images = torch.randn(1, 6, 3, 16, 16)
    batch = {"img": [images], "tag": {"value": 1}}
    result = corrupt_batch_images_with_metadata(
        batch,
        "local_glare",
        severity=3,
        view_indices=[0],
        seed=5,
        region=(0.25, 0.25, 0.75, 0.75),
    )
    legacy = corrupt_batch_images(
        batch,
        "camera_dropout",
        severity=1,
        view_indices=[0],
    )

    assert isinstance(result.batch, dict)
    assert isinstance(legacy, dict)
    assert "mask" not in legacy
    assert result.batch is not batch
    assert result.batch["img"] is not batch["img"]
    assert torch.equal(batch["img"][0], images)
    assert not torch.equal(result.batch["img"][0], images)
    assert result.mask.shape == (1, 6, 1, 16, 16)


@pytest.mark.parametrize(
    "region",
    (
        (),
        (0.0, 0.0, 1.0),
        (-0.1, 0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5, 0.5),
        (0.0, 0.7, 0.5, 0.6),
        (0.0, 0.0, 1.1, 1.0),
        (0.0, 0.0, float("nan"), 1.0),
    ),
)
def test_metadata_corruption_rejects_invalid_or_empty_region(region):
    images = torch.randn(1, 2, 3, 16, 16)
    with pytest.raises(ValueError):
        corrupt_multiview_images_with_metadata(
            images,
            "local_blur",
            severity=1,
            view_indices=[0],
            region=region,
        )


def test_metadata_corruption_rejects_empty_views_and_partial_dropout_region():
    images = torch.randn(1, 2, 3, 16, 16)
    with pytest.raises(ValueError, match="must not be empty"):
        corrupt_multiview_images_with_metadata(
            images, "local_dark", view_indices=[]
        )
    with pytest.raises(ValueError, match="complete views"):
        corrupt_multiview_images_with_metadata(
            images,
            "camera_dropout",
            view_indices=[0],
            region=(0.0, 0.0, 0.5, 0.5),
        )
