import pytest
import torch

from uq_estimator.corruptions import corrupt_multiview_images


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
