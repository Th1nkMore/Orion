import pytest
import torch

from uq_estimator.path_projection import project_path_corridor_to_patches
from uq_estimator.spatial_uq import cvar_path_risk


def _centered_projection(batch=1, views=1):
    matrix = torch.eye(4).repeat(batch, views, 1, 1)
    matrix[..., 0, 3] = 49.5
    matrix[..., 1, 3] = 49.5
    return matrix


def test_center_path_projects_to_center_patch_and_reports_coverage():
    path = torch.tensor([[[[0.0, 0.0, 1.0]]]])
    output = project_path_corridor_to_patches(
        path,
        _centered_projection(),
        image_hw=(100, 100),
        patch_hw=(10, 10),
        corridor_radius_patches=0.8,
        soft=False,
    )
    assert output.mask.shape == (1, 1, 1, 10, 10)
    assert output.point_visible.shape == (1, 1, 1, 1)
    assert output.point_visible.item()
    assert output.coverage.item() == 1.0
    peak = torch.nonzero(output.mask[0, 0, 0], as_tuple=False)
    assert peak.tolist() in ([[4, 4]], [[5, 5]], [[4, 4], [4, 5], [5, 4], [5, 5]])


def test_off_image_and_behind_camera_points_have_zero_mask():
    path = torch.tensor(
        [[[[1000.0, 0.0, 1.0]], [[0.0, 0.0, -1.0]]]]
    )
    output = project_path_corridor_to_patches(
        path,
        _centered_projection(),
        image_hw=(100, 100),
        patch_hw=(10, 10),
    )
    assert output.mask.shape == (1, 2, 1, 10, 10)
    assert torch.count_nonzero(output.mask) == 0
    assert torch.equal(output.coverage, torch.zeros(1, 2))


def test_on_path_failure_has_higher_fixed_risk_than_off_path_failure():
    path = torch.tensor([[[[0.0, 0.0, 1.0]]]])
    projection = project_path_corridor_to_patches(
        path,
        _centered_projection(),
        image_hw=(100, 100),
        patch_hw=(10, 10),
        corridor_radius_patches=1.0,
        soft=False,
    )
    on_path_failure = torch.zeros(1, 1, 10, 10)
    on_path_failure[..., 4:6, 4:6] = 1.0
    off_path_failure = torch.zeros_like(on_path_failure)
    off_path_failure[..., :2, :2] = 1.0

    on = cvar_path_risk(
        on_path_failure[:, None], projection.mask, top_q=1.0, spatial_ndim=3
    )
    off = cvar_path_risk(
        off_path_failure[:, None], projection.mask, top_q=1.0, spatial_ndim=3
    )
    assert on.risk.item() > off.risk.item()
    assert off.risk.item() == 0.0


def test_multiple_cameras_and_dynamic_patch_grid():
    path = torch.tensor([[[[0.0, 0.0, 1.0], [5.0, 0.0, 1.0]]]])
    matrices = _centered_projection(1, 3)
    matrices[:, 1, 0, 3] = 10.0
    matrices[:, 2, 0, 3] = 200.0
    output = project_path_corridor_to_patches(
        path,
        matrices,
        image_hw=(100, 200),
        patch_hw=(7, 13),
    )
    assert output.mask.shape == (1, 1, 3, 7, 13)
    assert output.patch_xy.shape == (1, 1, 2, 3, 2)
    assert output.depth.shape == (1, 1, 2, 3)
    assert output.coverage.shape == (1, 1)


@pytest.mark.parametrize(
    "path,matrices,error",
    (
        (torch.zeros(1, 3, 2), torch.eye(4)[None, None], "path_xyz"),
        (torch.zeros(1, 1, 1, 2), torch.eye(3)[None, None], "lidar2img"),
        (torch.zeros(2, 1, 1, 2), torch.eye(4)[None, None], "batch"),
    ),
)
def test_projection_rejects_invalid_shapes(path, matrices, error):
    with pytest.raises(ValueError, match=error):
        project_path_corridor_to_patches(
            path.float(), matrices.float(), image_hw=(10, 10), patch_hw=(2, 2)
        )
