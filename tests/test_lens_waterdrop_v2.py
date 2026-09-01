from pathlib import Path

import numpy as np
import pytest

from uq_estimator.lens_waterdrop_v2 import (
    PROFILES,
    REQUIRED_RESOLUTION,
    apply_lens_waterdrop_v2,
)


ROOT = Path(__file__).resolve().parents[1]
MASK_ROOT = ROOT / "assets/waterdrop_patterns/evocargo_ccby4_v1"
MASK = MASK_ROOT / "D1__D1_0095.png"
METADATA = MASK_ROOT / "metadata.json"


def _image() -> np.ndarray:
    width, height = REQUIRED_RESOLUTION
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    return np.stack(
        (
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(y, (height, width)),
            ((x.astype(np.uint16) + y.astype(np.uint16)) % 256).astype(np.uint8),
        ),
        axis=2,
    )


def test_v2_is_deterministic_full_resolution_and_exact_outside_mask():
    image = _image()
    first = apply_lens_waterdrop_v2(
        image, mask_path=MASK, metadata_path=METADATA, profile="medium"
    )
    second = apply_lens_waterdrop_v2(
        image, mask_path=MASK, metadata_path=METADATA, profile="medium"
    )
    assert first.image.shape == (900, 1600, 3)
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.displacement_px, second.displacement_px)
    assert np.array_equal(first.image[~first.silhouette], image[~first.silhouette])
    assert not np.array_equal(first.image[first.silhouette], image[first.silhouette])


def test_v2_exposes_bounded_nontrivial_optical_fields_and_provenance():
    result = apply_lens_waterdrop_v2(
        _image(), mask_path=MASK, metadata_path=METADATA, profile="heavy"
    )
    assert result.alpha.min() >= 0.0
    assert result.alpha.max() <= 1.0
    assert np.linalg.norm(result.displacement_px, axis=2).max() > 20.0
    assert result.edge_contribution.min() < 0.0
    assert result.edge_contribution.max() <= 0.0
    assert result.highlight_contribution.min() >= 0.0
    assert result.highlight_contribution.max() <= PROFILES["heavy"]["highlight"]
    assert result.metadata["real_data_scope"] == "binary_silhouette_only"
    assert result.metadata["source_silhouette"]["source_license"] == "CC-BY-4.0"
    assert "actor" in result.metadata["placement_policy"]


def test_v2_rejects_review_downsample_and_unfrozen_mask(tmp_path):
    small = np.zeros((270, 480, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="1600x900"):
        apply_lens_waterdrop_v2(
            small, mask_path=MASK, metadata_path=METADATA, profile="light"
        )
    altered = tmp_path / MASK.name
    altered.write_bytes(MASK.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="SHA-256"):
        apply_lens_waterdrop_v2(
            _image(), mask_path=altered, metadata_path=METADATA, profile="light"
        )


def test_v2_api_has_no_actor_bbox_or_semantic_input():
    import inspect

    parameters = set(inspect.signature(apply_lens_waterdrop_v2).parameters)
    assert parameters == {
        "image",
        "mask_path",
        "metadata_path",
        "profile",
        "require_resolution",
    }
