import json

import pytest
import torch

from uq_estimator.object_failure_targets import (
    PATCH_ATTRIBUTION_CLAIM,
    POST_AUGMENTATION_PROJECTION,
    TargetProvenanceV1,
    aggregate_projected_visible_support,
)
from uq_estimator.projected_visible_support import (
    ORION_CAMERA_ORDER,
    SupportRefinementResultV1,
    VisibleSupportProjectionError,
    lidar_boxes_to_corners,
    make_projection_overlay_data,
    project_boxes_to_visible_patch_support,
    render_projection_overlay_image,
)


def _six_view_matrices(*, focal: float = 50.0) -> torch.Tensor:
    """Synthetic camera: only CAM_FRONT sees positive synthetic z-depth."""

    matrices = []
    for view_index in range(6):
        depth_sign = 1.0 if view_index == 0 else -1.0
        matrices.append(
            torch.tensor(
                [
                    [focal, 0.0, 50.0, 0.0],
                    [0.0, focal, 50.0, 0.0],
                    [0.0, 0.0, depth_sign, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
    return torch.stack(matrices)


def _project(
    boxes: torch.Tensor,
    *,
    matrices: torch.Tensor | None = None,
    patch_hw=(2, 2),
    expected_patch_hw=(2, 2),
    **overrides,
):
    values = {
        "camera_order": ORION_CAMERA_ORDER,
        "matrix_camera_order": ORION_CAMERA_ORDER,
        "image_shape_camera_order": ORION_CAMERA_ORDER,
        "image_transform_id": "post-aug-test-transform-v1",
        "box_z_origin": "center",
        "patch_hw": patch_hw,
        "expected_patch_hw": expected_patch_hw,
    }
    values.update(overrides)
    return project_boxes_to_visible_patch_support(
        boxes,
        _six_view_matrices() if matrices is None else matrices,
        [(100, 100)] * 6,
        **values,
    )


def _target_provenance() -> TargetProvenanceV1:
    return TargetProvenanceV1(
        base_checkpoint_sha256="a" * 64,
        inference_config_sha256="b" * 64,
        git_revision="test-revision",
        route_id="route-test",
        town="Town01",
        frame_idx=0,
        observation_branch="observed",
        temporal_history_id="observed-history",
        paired_history_protocol_id="paired-replay",
        class_mapping_id="class-map-v1",
        decoder_policy_id="decoder-policy-v1",
        camera_order=ORION_CAMERA_ORDER,
        image_transform_id="post-aug-test-transform-v1",
    )


def test_front_visible_support_is_fractional_and_directly_consumable():
    result = _project(
        torch.tensor([[0.0, 0.0, 5.0, 2.0, 2.0, 2.0, 0.0]])
    )
    assert result.support.shape == (6, 4, 1)
    assert result.valid_patch_mask.shape == (6, 4)
    assert result.object_visible_mask[:, 0].tolist() == [True] + [False] * 5
    torch.testing.assert_close(
        result.support[0, :, 0], torch.full((4,), 0.0625)
    )
    assert result.support[1:].count_nonzero() == 0
    assert result.projection_provenance.projection_matrix_kind == (
        POST_AUGMENTATION_PROJECTION
    )
    assert result.projection_provenance.attribution == PATCH_ATTRIBUTION_CLAIM
    assert result.projection_provenance.attribution_is_causal is False
    assert result.projection_provenance.refinement_applied is False

    # The output axes and provenance are already the exact contract consumed
    # by the target aggregator/exporter; no transpose or inferred metadata is
    # necessary.
    target = aggregate_projected_visible_support(
        result.support,
        torch.tensor([0.8]),
        torch.tensor([True]),
        result.valid_patch_mask,
        support_provenance=result.support_provenance,
        target_provenance=_target_provenance(),
    )
    assert target.error.shape == (6, 4)
    assert target.error[0].sum().item() == pytest.approx(0.2)
    assert target.attribution_is_causal is False


def test_box_fully_behind_camera_has_zero_support_and_no_fake_visibility():
    result = _project(
        torch.tensor([[0.0, 0.0, -5.0, 2.0, 2.0, 2.0, 0.0]])
    )
    assert result.support.count_nonzero() == 0
    assert not result.object_visible_mask.any()
    assert torch.isnan(result.projected_depth_range).all()
    assert all(
        polygon == tuple()
        for view_polygons in result.projected_polygons_xy
        for polygon in view_polygons
    )


def test_near_plane_and_image_edge_are_clipped_before_area_pooling():
    edge = _project(
        torch.tensor([[-4.0, 0.0, 5.0, 4.0, 2.0, 2.0, 0.0]]),
        patch_hw=(1, 1),
        expected_patch_hw=(1, 1),
    )
    polygon = edge.projected_polygons_xy[0][0]
    assert min(point[0] for point in polygon) == pytest.approx(0.0)
    assert max(point[0] for point in polygon) <= 100.0
    assert 0.0 < edge.support[0, 0, 0].item() < 1.0

    # Four lower corners are behind near=0.5; the four vertical box edges are
    # clipped to that plane rather than discarding the partially visible box.
    near_clipped = _project(
        torch.tensor([[0.0, 0.0, 0.75, 0.4, 0.4, 1.0, 0.0]]),
        patch_hw=(1, 1),
        expected_patch_hw=(1, 1),
        near_plane_depth=0.5,
    )
    assert near_clipped.object_visible_mask[0, 0]
    assert near_clipped.support[0, 0, 0] > 0
    assert near_clipped.projected_depth_range[0, 0, 0].item() >= 0.5


def test_polygon_crosses_patch_boundaries_and_small_object_keeps_point_one_support():
    crossing = _project(
        torch.tensor([[0.0, 0.0, 5.0, 2.0, 2.0, 2.0, 0.0]])
    )
    assert (crossing.support[0, :, 0] > 0).tolist() == [True] * 4

    # With f=50, z~=10 and an almost depth-flat 6.324 x 6.324 box, the
    # projected convex hull covers approximately 10% of a 100 x 100 patch.
    small = _project(
        torch.tensor([[0.0, 0.0, 10.0, 6.324, 6.324, 0.001, 0.0]]),
        patch_hw=(1, 1),
        expected_patch_hw=(1, 1),
    )
    assert small.support[0, 0, 0].item() == pytest.approx(0.10, abs=2e-4)
    assert small.object_visible_mask[0, 0]


@pytest.mark.parametrize(
    "field,bad_order,error",
    (
        (
            "camera_order",
            ORION_CAMERA_ORDER[:1] + ORION_CAMERA_ORDER[2:3] + ORION_CAMERA_ORDER[1:2] + ORION_CAMERA_ORDER[3:],
            "camera_order",
        ),
        (
            "matrix_camera_order",
            ORION_CAMERA_ORDER[:1] + ORION_CAMERA_ORDER[2:3] + ORION_CAMERA_ORDER[1:2] + ORION_CAMERA_ORDER[3:],
            "matrix_camera_order",
        ),
        (
            "image_shape_camera_order",
            ORION_CAMERA_ORDER[:1] + ORION_CAMERA_ORDER[2:3] + ORION_CAMERA_ORDER[1:2] + ORION_CAMERA_ORDER[3:],
            "image_shape_camera_order",
        ),
    ),
)
def test_camera_matrix_and_image_shape_order_mismatches_fail_closed(
    field, bad_order, error
):
    with pytest.raises(VisibleSupportProjectionError, match=error):
        _project(
            torch.tensor([[0.0, 0.0, 5.0, 2.0, 2.0, 2.0, 0.0]]),
            **{field: bad_order},
        )


def test_production_40x40_grid_and_valid_mask_alignment_fail_closed():
    box = torch.tensor([[0.0, 0.0, 5.0, 2.0, 2.0, 2.0, 0.0]])
    production = _project(
        box,
        patch_hw=(40, 40),
        expected_patch_hw=(40, 40),
    )
    assert production.support.shape == (6, 1600, 1)
    assert production.support_provenance.patch_hw == (40, 40)

    with pytest.raises(VisibleSupportProjectionError, match="40x40 alignment"):
        _project(
            box,
            patch_hw=(39, 40),
            expected_patch_hw=(40, 40),
        )
    with pytest.raises(VisibleSupportProjectionError, match="input_patch_valid_mask"):
        _project(box, input_patch_valid_mask=torch.ones(6, 3, dtype=torch.bool))
    with pytest.raises(VisibleSupportProjectionError, match="uniform processed"):
        project_boxes_to_visible_patch_support(
            box,
            _six_view_matrices(),
            [(100, 100)] * 5 + [(90, 100)],
            camera_order=ORION_CAMERA_ORDER,
            matrix_camera_order=ORION_CAMERA_ORDER,
            image_shape_camera_order=ORION_CAMERA_ORDER,
            image_transform_id="post-aug-test-transform-v1",
            box_z_origin="center",
        )


def test_box_geometry_requires_explicit_valid_z_origin_and_full_dimensions():
    box = torch.tensor([[0.0, 0.0, 5.0, 2.0, 2.0, 2.0, 0.0]])
    center = lidar_boxes_to_corners(box, box_z_origin="center")
    bottom = lidar_boxes_to_corners(box, box_z_origin="bottom")
    assert center[:, :, 2].min().item() == pytest.approx(4.0)
    assert bottom[:, :, 2].min().item() == pytest.approx(5.0)
    with pytest.raises(VisibleSupportProjectionError, match="explicitly"):
        lidar_boxes_to_corners(box, box_z_origin="inferred")
    with pytest.raises(VisibleSupportProjectionError, match="positive"):
        lidar_boxes_to_corners(
            torch.tensor([[0.0, 0.0, 5.0, 0.0, 2.0, 2.0, 0.0]]),
            box_z_origin="center",
        )


class _SemanticRefiner:
    refinement_id = "cpu-test-semantic-refiner/v1"
    required_modalities = ("semantic",)

    def refine(self, request, modality_payloads):
        assert modality_payloads["semantic"] == "explicit-semantic-data"
        return SupportRefinementResultV1(
            patch_support=request.base_patch_support * 0.5,
            patch_valid_mask=torch.ones_like(
                request.base_patch_support, dtype=torch.bool
            ),
            audit_note="class-compatible pixels intersected",
        )


class _ExpandingRefiner(_SemanticRefiner):
    refinement_id = "invalid-expanding-refiner/v1"

    def refine(self, request, modality_payloads):
        return SupportRefinementResultV1(
            patch_support=torch.ones_like(request.base_patch_support),
            patch_valid_mask=torch.ones_like(
                request.base_patch_support, dtype=torch.bool
            ),
            audit_note="invalid expansion",
        )


def test_semantic_depth_refinement_is_explicit_and_never_implied_without_data():
    box = torch.tensor([[0.0, 0.0, 5.0, 2.0, 2.0, 2.0, 0.0]])
    base = _project(box)
    assert base.projection_provenance.refinement_id is None
    assert base.projection_provenance.refinement_modalities == ()
    assert base.projection_provenance.refinement_call_count == 0
    assert base.projection_provenance.refinement_applied is False

    with pytest.raises(VisibleSupportProjectionError, match="exactly match"):
        _project(box, refiner=_SemanticRefiner())
    with pytest.raises(VisibleSupportProjectionError, match="without an explicit refiner"):
        _project(box, refinement_payloads={"semantic": "data"})

    refined = _project(
        box,
        refiner=_SemanticRefiner(),
        refinement_payloads={"semantic": "explicit-semantic-data"},
    )
    torch.testing.assert_close(refined.support, base.support * 0.5)
    assert refined.projection_provenance.refinement_applied is True
    assert refined.projection_provenance.refinement_modalities == ("semantic",)
    assert refined.projection_provenance.refinement_call_count == 1

    with pytest.raises(VisibleSupportProjectionError, match="must not expand"):
        _project(
            box,
            refiner=_ExpandingRefiner(),
            refinement_payloads={"semantic": "explicit-semantic-data"},
        )


def test_overlay_data_and_image_interfaces_need_no_real_camera_frame():
    result = _project(
        torch.tensor([[0.0, 0.0, 5.0, 2.0, 2.0, 2.0, 0.0]])
    )
    data = make_projection_overlay_data(result, 0)
    assert data["camera_name"] == "CAM_FRONT"
    assert data["objects"][0]["visible"] is True
    assert len(data["objects"][0]["polygon_xy"]) == 4
    assert len(data["objects"][0]["patches"]) == 4
    assert data["claim_boundary"]["attribution_is_causal"] is False
    json.dumps(data)

    pytest.importorskip("PIL")
    image = render_projection_overlay_image(result, 0)
    assert image.shape == (100, 100, 3)
    assert image.dtype == torch.uint8
    assert image.device.type == "cpu"
