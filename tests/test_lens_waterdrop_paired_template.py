from pathlib import Path

import numpy as np
import pytest

from uq_estimator.lens_waterdrop_paired_template import (
    REQUIRED_RESOLUTION,
    apply_paired_waterdrop_template,
    extract_paired_waterdrop_template,
)


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "assets/waterdrop_patterns/icra2023_paired_template_v1"
CLEAN = BANK / "test__syn__clean_vid__0003__000075.png"
RAINY = BANK / "test__syn__rainy_vid__0003__000075.png"
METADATA = BANK / "metadata.json"


@pytest.fixture(scope="module")
def template():
    return extract_paired_waterdrop_template(
        clean_path=CLEAN, rainy_path=RAINY, metadata_path=METADATA
    )


def _target() -> np.ndarray:
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


def test_extraction_is_auditable_and_does_not_retain_source_rgb(template):
    assert 0.04 < template.metadata["support_fraction"] < 0.20
    assert template.metadata["displacement_px_max"] > 2.0
    assert template.metadata["source_pair_reconstruction_mae_rgb"] < 4.0
    assert template.metadata["component_weighted_luminance_bias_after_max_abs"] < 0.02
    assert template.metadata["source_rgb_retained_in_template"] is False
    assert template.metadata["source_chromatic_residual_retained"] is False
    assert template.metadata["real_reference_frames_used_in_extraction"] is False
    assert template.luminance_residual.ndim == 2
    assert template.displacement_px.shape[2] == 2


def test_application_is_deterministic_full_resolution_and_exact_outside_support(template):
    image = _target()
    first = apply_paired_waterdrop_template(image, template=template, profile="medium")
    second = apply_paired_waterdrop_template(image, template=template, profile="medium")
    assert first.image.shape == (900, 1600, 3)
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.image[~first.support], image[~first.support])
    assert not np.array_equal(first.image[first.support], image[first.support])
    assert first.metadata["source_rgb_copied_to_target"] is False


def test_profile_strength_is_monotonic_without_changing_support(template):
    image = _target()
    rows = [
        apply_paired_waterdrop_template(image, template=template, profile=profile)
        for profile in ("light", "medium", "heavy")
    ]
    assert np.array_equal(rows[0].support, rows[1].support)
    assert np.array_equal(rows[1].support, rows[2].support)
    changes = [np.abs(row.image.astype(float) - image).mean() for row in rows]
    assert changes[0] < changes[1] < changes[2]


def test_rejects_review_downsample_and_api_has_no_task_inputs(template):
    import inspect

    with pytest.raises(ValueError, match="1600x900"):
        apply_paired_waterdrop_template(
            np.zeros((270, 480, 3), dtype=np.uint8),
            template=template,
            profile="light",
        )
    assert set(inspect.signature(apply_paired_waterdrop_template).parameters) == {
        "image",
        "template",
        "profile",
        "require_resolution",
    }
