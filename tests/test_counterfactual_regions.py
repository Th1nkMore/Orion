import pytest
import torch

from uq_estimator.corruptions import corrupt_multiview_images_with_metadata
from uq_estimator.counterfactual_regions import (
    select_matched_counterfactual_regions,
)


def test_selects_equal_area_on_and_zero_overlap_off_path_regions():
    corridor = torch.zeros(12, 20)
    corridor[:, 9:11] = 1.0
    regions = select_matched_counterfactual_regions(corridor, region_hw=(4, 5))
    assert regions.on_path_overlap > 0
    assert regions.off_path_overlap == 0
    on = regions.on_path_region
    off = regions.off_path_region
    assert (on[2] - on[0]) == pytest.approx(off[2] - off[0])
    assert (on[3] - on[1]) == pytest.approx(off[3] - off[1])


def test_regions_drive_replayable_matched_corruptions():
    corridor = torch.zeros(8, 16)
    corridor[2:8, 7:9] = 1.0
    regions = select_matched_counterfactual_regions(corridor, region_hw=(3, 4))
    images = torch.randn(1, 6, 3, 80, 160)
    on = corrupt_multiview_images_with_metadata(
        images,
        "local_blur",
        severity=2,
        view_indices=[0],
        seed=10,
        region=regions.on_path_region,
    )
    off = corrupt_multiview_images_with_metadata(
        images,
        "local_blur",
        severity=2,
        view_indices=[0],
        seed=10,
        region=regions.off_path_region,
    )
    assert on.mask.sum() == off.mask.sum()
    assert not torch.equal(on.mask, off.mask)
    assert on.metadata.parameters["kernel_size"] == off.metadata.parameters["kernel_size"]


def test_selection_is_deterministic():
    corridor = torch.zeros(10, 10)
    corridor[:, 4:6] = 1.0
    first = select_matched_counterfactual_regions(corridor, (2, 2))
    second = select_matched_counterfactual_regions(corridor, (2, 2))
    assert first == second


def test_rejects_scene_with_no_strict_off_path_region():
    corridor = torch.ones(5, 5)
    with pytest.raises(ValueError, match="off-path"):
        select_matched_counterfactual_regions(corridor, (3, 3))

