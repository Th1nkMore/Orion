"""CPU end-to-end tests for the concrete production branch-target builder."""

import pytest
import torch

from uq_estimator.bev_target_rasterizer import (
    GT_RASTERIZER_ID,
    PAIRWISE_BEV_IOU_POLICY_ID,
    SELECTED_MODE_RASTERIZER_ID,
    rasterize_orion_selected_mode_v1,
)
from uq_estimator.decoded_actual_target_export import DecodedORIONFrameV1
from uq_estimator.orion_actual_target_builder import (
    OrionActualTargetBuilderError,
    ProductionActualTargetBranchBuilderV1,
    ProductionBranchTargetConfigV1,
)
from uq_estimator.orion_actual_target_runner import (
    BOX_Z_ORIGIN_POLICY_ID,
    PILOT_MATCH_POLICY_ID,
    PILOT_MAXIMUM_CENTER_DISTANCE_M,
    PILOT_MINIMUM_PREDICTION_SCORE,
    TRAFFIC_SEMANTICS_ID,
    build_production_runtime_hooks_v1,
    canonicalize_orion_test_batch,
    filter_v1_gt_target_eligibility,
    pilot_failure_event_policy,
)
from uq_estimator.orion_decode_adapter import ORIONDecodeAdapterConfigV1
from uq_estimator.projected_visible_support import (
    ORION_CAMERA_ORDER,
    VISIBLE_SUPPORT_PROJECTION_VERSION,
)


ROUTE = "v1/OppositeVehicleTakingPriority_Town04_Route214_Weather6"
CLASS_MAP = "b2d-nine-class/v1"
DECODER_POLICY = "custom-nms-free-coder-exact-v1"


def _six_view_matrices():
    matrices = []
    for index in range(6):
        depth_sign = 1.0 if index == 0 else -1.0
        matrices.append(
            torch.tensor(
                [
                    [50.0, 0.0, 50.0, 0.0],
                    [0.0, 50.0, 50.0, 0.0],
                    [0.0, 0.0, depth_sign, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
    return torch.stack(matrices)


def _batch(*, bad_attr_category=False):
    # Canonical B2D wrapper tensors are bottom-origin.
    boxes = torch.tensor(
        [
            [0.0, 0.0, 5.0, 2.0, 2.0, 2.0, 0.0],
            [3.0, 0.0, 5.0, 1.0, 1.0, 2.0, 0.0],
            [-3.0, 0.0, 5.0, 1.0, 1.0, 2.0, 0.0],
        ]
    )
    classes = torch.tensor([0, 6, 4])
    attr = torch.zeros(3, 34)
    attr[:, 12:18] = 1.0
    attr[:, 27] = classes.float()
    if bad_attr_category:
        attr[0, 27] = 7.0
    return {
        "img": torch.zeros(1, 6, 3, 100, 100),
        "img_metas": [[{
            "frame_idx": 0,
            "scene_token": ROUTE,
            "camera_order": ORION_CAMERA_ORDER,
            "pad_shape": [(100, 100, 3)] * 6,
        }]],
        "traffic_state": torch.tensor([[-1, -1], [2, 1], [-1, -1]]),
        "traffic_state_mask": torch.tensor([False, True, False]),
        "gt_actor_ids": torch.tensor([101, 202, 303]),
        "lidar2img": _six_view_matrices().unsqueeze(0),
        "cam_intrinsic": torch.eye(4).repeat(1, 6, 1, 1),
        "timestamp": torch.tensor([0.0]),
        "ego_pose": torch.eye(4).unsqueeze(0),
        "ego_pose_inv": torch.eye(4).unsqueeze(0),
        "command": torch.tensor([1.0]),
        "can_bus": torch.zeros(1, 18),
        "gt_bboxes_3d": boxes,
        "gt_labels_3d": classes,
        "gt_attr_labels": attr,
    }


def _decoded():
    # Decoded boxes remain center-origin at this adapter boundary.
    boxes = torch.tensor(
        [
            [0.2, 0.0, 6.0, 2.0, 2.0, 2.0, 0.0],
            [3.0, 0.0, 6.0, 1.0, 1.0, 2.0, 0.0],
        ]
    )
    selected_deltas = torch.zeros(2, 6, 2)
    occupancy = rasterize_orion_selected_mode_v1(
        boxes, selected_deltas
    ).per_object
    probabilities = torch.full((2, 9), 0.01)
    probabilities[0, 0] = 0.90
    probabilities[1, 6] = 0.85
    modes = torch.zeros(2, 2, 6, 2)
    mode_scores = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    return DecodedORIONFrameV1(
        centers_lidar=boxes[:, :3],
        boxes_lidar=boxes,
        classes=torch.tensor([0, 6]),
        scores=torch.tensor([0.90, 0.85]),
        source_query_index=torch.tensor([11, 17]),
        class_probabilities=probabilities,
        selected_motion_occupancy=occupancy,
        traffic_state_logits=torch.tensor(
            [[0.0, 0.0, 0.0, -2.0], [-2.0, -2.0, 2.0, 2.0]]
        ),
        all_trajectory_modes=modes,
        trajectory_mode_scores=mode_scores,
        selected_motion_mode_index=mode_scores.argmax(dim=1),
        occupancy_rasterizer_id=SELECTED_MODE_RASTERIZER_ID,
        decoder_layer=5,
        decoder_policy_id=DECODER_POLICY,
        class_mapping_id=CLASS_MAP,
        with_light_state=True,
    )


def _builder():
    return ProductionActualTargetBranchBuilderV1(
        ProductionBranchTargetConfigV1(
            base_checkpoint_sha256="a" * 64,
            inference_config_sha256="b" * 64,
            git_revision="fixture-revision",
            route_id=ROUTE,
            town="Town04",
            class_mapping_id=CLASS_MAP,
            decoder_policy_id=DECODER_POLICY,
            image_transform_id="fixture-post-augmentation/v1",
            observed_corruption_family="local_occlusion",
            observed_severity=2.0,
            observed_seed=20260826,
            event_window_frames=(0, 63),
        )
    )


def _hooks(builder):
    decode_config = ORIONDecodeAdapterConfigV1(
        num_classes=9,
        max_num=18,
        post_center_range=(-61.2, -61.2, -10.0, 61.2, 61.2, 10.0),
        class_mapping_id=CLASS_MAP,
        occupancy_rasterizer_id=SELECTED_MODE_RASTERIZER_ID,
        with_light_state=True,
    )
    return build_production_runtime_hooks_v1(
        decode_config=decode_config,
        branch_target_builder=builder,
        corruption_transform=lambda batch, context: batch,
        record_sink=lambda record, bundle, context: None,
        decoder_parity_check=lambda: True,
        selected_motion_mode_check=lambda: True,
        projection_overlay_check=lambda: True,
        gt_axis_alignment_check=lambda: True,
        gt_box_z_origin="bottom",
        decoded_box_z_origin="center",
    )


def _context(hooks, *, branch="clean"):
    return {
        "plan_id": "fixture-plan",
        "paired_replay_id": "fixture-paired-replay",
        "branch_history_id": "%s-history" % branch,
        "branch": branch,
        "frame_idx": 0,
        "previous_frame_idx": None,
        "failure_event_policy": pilot_failure_event_policy(),
        "object_matching_policy_id": PILOT_MATCH_POLICY_ID,
        "minimum_prediction_score": PILOT_MINIMUM_PREDICTION_SCORE,
        "maximum_center_distance_m": PILOT_MAXIMUM_CENTER_DISTANCE_M,
        "traffic_semantics_id": TRAFFIC_SEMANTICS_ID,
        "gt_occupancy_rasterizer": hooks.gt_occupancy_rasterizer,
        "pairwise_bev_iou": hooks.pairwise_bev_iou,
        "project_visible_support": hooks.project_visible_support,
        "gt_occupancy_rasterizer_id": hooks.gt_occupancy_rasterizer_id,
        "predicted_occupancy_rasterizer_id": hooks.occupancy_rasterizer_id,
        "pairwise_bev_iou_policy_id": hooks.pairwise_bev_iou_policy_id,
        "support_projector_id": hooks.support_projector_id,
        "gt_box_z_origin": hooks.gt_box_z_origin,
        "decoded_box_z_origin": hooks.decoded_box_z_origin,
        "box_z_origin_policy_id": BOX_Z_ORIGIN_POLICY_ID,
        "post_augmentation_lidar2img_cpu": _six_view_matrices(),
        "processed_image_hw_by_view": [(100, 100)] * 6,
        "gt_target_eligibility_policy": (
            "safety-actors-plus-affecting-traffic-light/v1"
        ),
        "gt_target_eligibility_filter": filter_v1_gt_target_eligibility,
    }


def test_concrete_builder_end_to_end_retains_dual_ids_axes_and_traffic_masks():
    builder = _builder()
    hooks = _hooks(builder)
    canonical = canonicalize_orion_test_batch(_batch())
    result = builder(_decoded(), canonical, _context(hooks, branch="clean"))
    bundle = result.bundle

    assert result.eligibility_audit["pre_filter_count"] == 3
    assert result.eligibility_audit["post_filter_count"] == 2
    assert result.eligibility_audit["selected_indices"] == [0, 1]
    assert result.eligibility_audit["gt_and_support_actor_ids_exactly_equal"] is True
    assert result.eligibility_audit["planningmetric_category_axis_matches_classes"] is True
    assert result.eligibility_audit["gt_projection_box_z_origin"] == "bottom"
    assert result.eligibility_audit["pred_projection_box_z_origin"] == "center"

    assert bundle.gt_classes.tolist() == [0, 6]
    assert bundle.gt_traffic_states.tolist() == [0, 2]
    assert bundle.gt_traffic_state_valid.tolist() == [False, True]
    assert bundle.occupancy_rasterizer_id == SELECTED_MODE_RASTERIZER_ID
    assert bundle.gt_occupancy_rasterizer_id == GT_RASTERIZER_ID
    assert bundle.bev_iou_policy_id == PAIRWISE_BEV_IOU_POLICY_ID
    assert bundle.minimum_prediction_score == 0.5
    torch.testing.assert_close(
        bundle.max_center_distance,
        torch.full((2,), 4.0),
    )
    assert bundle.gt_projected_support.shape == (6, 1600, 2)
    assert bundle.pred_projected_support.shape == (6, 1600, 2)
    assert bundle.error_severity_target.shape == (6, 1600)
    assert bundle.target_provenance.observation_branch == "clean"
    assert bundle.observation.corruption_family == "none"


def test_observed_branch_records_fixed_corruption_and_chronology():
    builder = _builder()
    hooks = _hooks(builder)
    canonical = canonicalize_orion_test_batch(_batch())
    result = builder(_decoded(), canonical, _context(hooks, branch="observed"))
    assert result.bundle.observation.corruption_family == "local_occlusion"
    assert result.bundle.observation.event_window_frames == (0, 63)
    assert result.bundle.observation.active_at_frame is True
    assert result.bundle.chronology.branch_history_id == "observed-history"
    assert result.bundle.chronology.paired_replay_id == "fixture-paired-replay"


def test_builder_fails_closed_when_planningmetric_category_axis_diverges():
    builder = _builder()
    hooks = _hooks(builder)
    canonical = canonicalize_orion_test_batch(_batch(bad_attr_category=True))
    with pytest.raises(OrionActualTargetBuilderError, match="gt_attr"):
        builder(_decoded(), canonical, _context(hooks))


def test_builder_rejects_context_policy_or_matrix_drift():
    builder = _builder()
    hooks = _hooks(builder)
    canonical = canonicalize_orion_test_batch(_batch())
    context = _context(hooks)
    context["minimum_prediction_score"] = 0.0
    with pytest.raises(OrionActualTargetBuilderError, match="minimum_prediction_score"):
        builder(_decoded(), canonical, context)

    context = _context(hooks)
    context["post_augmentation_lidar2img_cpu"] = torch.eye(4).repeat(6, 1, 1)
    with pytest.raises(OrionActualTargetBuilderError, match="matrices differ"):
        builder(_decoded(), canonical, context)
