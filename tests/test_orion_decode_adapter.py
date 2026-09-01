import math

import pytest
import torch

from uq_estimator.orion_decode_adapter import (
    ORIONDecodeAdapterConfigV1,
    ORIONDecodeAdapterError,
    SelectedMotionRasterInputV1,
    adapt_orion_head_outputs_v1,
    denormalize_orion_bbox_exact,
)


def _fixture_head_outputs():
    # Final-layer class probabilities:
    # q0 = [.95, .90], q1 = [.80, .10], q2 = [.70, .60].
    # Global top-5 therefore contains two entries for q0 and two for q2.
    probabilities = torch.tensor(
        [[[[0.2, 0.1], [0.3, 0.2], [0.4, 0.3]]],
         [[[0.95, 0.90], [0.80, 0.10], [0.70, 0.60]]]],
        dtype=torch.float32,
    )
    cls = torch.logit(probabilities)

    bbox_final = torch.tensor(
        [
            [0.0, 0.0, math.log(2.0), math.log(3.0), 1.0, math.log(4.0), 0.0, 1.0, 0.1, 0.2],
            [20.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [2.0, -1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.3, 0.4],
        ],
        dtype=torch.float32,
    )
    bbox = torch.stack((torch.zeros_like(bbox_final), bbox_final), dim=0).unsqueeze(1)

    # [L=2,B=1,Q=3,M=2,2T=4]
    traj = torch.zeros(2, 1, 3, 2, 4)
    traj[-1, 0, 0, 0] = torch.tensor([1.0, 0.0, 2.0, 0.0])
    traj[-1, 0, 0, 1] = torch.tensor([3.0, 0.0, 4.0, 0.0])
    traj[-1, 0, 1, 0] = torch.tensor([5.0, 0.0, 6.0, 0.0])
    traj[-1, 0, 1, 1] = torch.tensor([7.0, 0.0, 8.0, 0.0])
    traj[-1, 0, 2, 0] = torch.tensor([9.0, 0.0, 10.0, 0.0])
    traj[-1, 0, 2, 1] = torch.tensor([11.0, 0.0, 12.0, 0.0])
    traj_cls = torch.zeros(2, 1, 3, 2)
    traj_cls[-1, 0] = torch.tensor([[1.0, 2.0], [0.0, 1.0], [3.0, 1.0]])

    traffic = torch.zeros(2, 1, 3, 4)
    traffic[-1, 0, 0] = torch.tensor([10.0, 11.0, 12.0, 13.0])
    traffic[-1, 0, 1] = torch.tensor([20.0, 21.0, 22.0, 23.0])
    traffic[-1, 0, 2] = torch.tensor([30.0, 31.0, 32.0, 33.0])
    return {
        "all_cls_scores": cls,
        "all_bbox_preds": bbox,
        "all_traj_preds": traj,
        "all_traj_cls_scores": traj_cls,
        "all_traffic_states": traffic,
    }


def _config(**overrides):
    values = {
        "num_classes": 2,
        "max_num": 5,
        "post_center_range": (-10.0, -10.0, -5.0, 10.0, 10.0, 5.0),
        "class_mapping_id": "fixture-two-class-v1",
        "occupancy_rasterizer_id": "fixture-selected-mode-v1",
        "with_light_state": True,
    }
    values.update(overrides)
    return ORIONDecodeAdapterConfigV1(**values)


def _audited_fixture_rasterizer(value: SelectedMotionRasterInputV1):
    assert value.trajectories_are_step_deltas is True
    assert value.batch_index == 0
    assert value.selected_deltas.shape[-2:] == (2, 2)
    # Encode the first selected delta into an otherwise synthetic occupancy.
    occupancy = torch.zeros(
        value.selected_deltas.shape[0], 2, 2, 2,
        device=value.selected_deltas.device,
    )
    occupancy[:, :, 0, 0] = value.selected_deltas[:, :1, 0] / 20.0
    return occupancy


def test_matches_query_class_topk_range_mask_and_retains_query_tensors():
    result = adapt_orion_head_outputs_v1(
        _fixture_head_outputs(),
        config=_config(),
        occupancy_rasterizer=_audited_fixture_rasterizer,
    )
    assert len(result.frames) == len(result.audits) == 1
    frame = result.frames[0]
    audit = result.audits[0]

    # q1 is third in top-k but its x=20 center fails post_center_range.
    assert audit.topk_source_query_index.tolist() == [0, 0, 1, 2, 2]
    assert audit.topk_class_index.tolist() == [0, 1, 0, 0, 1]
    assert audit.post_center_mask.tolist() == [True, True, False, True, True]
    assert audit.final_mask.tolist() == [True, True, False, True, True]
    assert frame.source_query_index.tolist() == [0, 0, 2, 2]
    assert frame.classes.tolist() == [0, 1, 0, 1]
    assert frame.duplicate_source_queries_present is True
    assert audit.duplicate_source_queries_present is True

    torch.testing.assert_close(
        frame.class_probabilities,
        torch.tensor(
            [[0.95, 0.90], [0.95, 0.90], [0.70, 0.60], [0.70, 0.60]]
        ),
    )
    torch.testing.assert_close(
        frame.traffic_state_logits,
        torch.tensor(
            [
                [10.0, 11.0, 12.0, 13.0],
                [10.0, 11.0, 12.0, 13.0],
                [30.0, 31.0, 32.0, 33.0],
                [30.0, 31.0, 32.0, 33.0],
            ]
        ),
    )
    assert frame.selected_motion_mode_index.tolist() == [1, 1, 0, 0]
    # Raw head mode logits are preserved; no implicit sigmoid is applied.
    torch.testing.assert_close(
        frame.trajectory_mode_scores,
        torch.tensor([[1.0, 2.0], [1.0, 2.0], [3.0, 1.0], [3.0, 1.0]]),
    )
    torch.testing.assert_close(
        frame.all_trajectory_modes[0, 1],
        torch.tensor([[3.0, 0.0], [4.0, 0.0]]),
    )
    assert frame.decoder_layer == 1
    assert frame.decoder_topk == 5
    assert frame.traffic_probability_transform == "sigmoid"


def test_denormalization_is_the_repository_field_order():
    source = torch.tensor(
        [[1.0, 2.0, math.log(3.0), math.log(4.0), 5.0, math.log(6.0), 1.0, 0.0, 7.0, 8.0]]
    )
    decoded = denormalize_orion_bbox_exact(source)
    torch.testing.assert_close(
        decoded,
        torch.tensor([[1.0, 2.0, 5.0, 3.0, 4.0, 6.0, math.pi / 2, 7.0, 8.0]]),
    )


def test_score_threshold_preserves_strict_first_then_adaptive_fallback():
    preds = _fixture_head_outputs()
    # Make the only top-k value exactly 0.5. The coder's first strict > 0.5
    # attempt is empty, then its >= 0.45 fallback retains it.
    preds["all_cls_scores"][-1, 0] = torch.logit(
        torch.tensor([[0.5, 0.1], [0.2, 0.1], [0.3, 0.1]])
    )
    result = adapt_orion_head_outputs_v1(
        preds,
        config=_config(max_num=1, score_threshold=0.5),
        occupancy_rasterizer=_audited_fixture_rasterizer,
    )
    audit = result.audits[0]
    assert audit.effective_score_threshold == pytest.approx(0.45)
    assert audit.score_threshold_mask.tolist() == [True]
    assert result.frames[0].scores.item() == pytest.approx(0.5)


def test_requires_traffic_state_and_light_state_provenance():
    with pytest.raises(ORIONDecodeAdapterError, match="with_light_state=True"):
        _config(with_light_state=False)
    with pytest.raises(ORIONDecodeAdapterError, match="sigmoid focal-loss"):
        _config(traffic_probability_transform="softmax")

    preds = _fixture_head_outputs()
    del preds["all_traffic_states"]
    with pytest.raises(ORIONDecodeAdapterError, match="all_traffic_states"):
        adapt_orion_head_outputs_v1(
            preds,
            config=_config(),
            occupancy_rasterizer=_audited_fixture_rasterizer,
        )


@pytest.mark.parametrize(
    "field, replacement, message",
    [
        ("all_cls_scores", torch.zeros(2, 1, 3, 3), "num_classes"),
        ("all_bbox_preds", torch.zeros(2, 1, 4, 10), "all_bbox_preds"),
        ("all_bbox_preds", torch.zeros(2, 1, 3, 9), "ORION shape"),
        ("all_traj_preds", torch.zeros(2, 1, 3, 2, 5), "positive even"),
        ("all_traj_cls_scores", torch.zeros(2, 1, 3, 3), "all_traj_cls_scores"),
        ("all_traffic_states", torch.zeros(2, 1, 4, 4), "all_traffic_states"),
    ],
)
def test_rejects_final_layer_shape_mismatch(field, replacement, message):
    preds = _fixture_head_outputs()
    preds[field] = replacement
    with pytest.raises(ORIONDecodeAdapterError, match=message):
        adapt_orion_head_outputs_v1(
            preds,
            config=_config(),
            occupancy_rasterizer=_audited_fixture_rasterizer,
        )


def test_rasterizer_must_return_selected_mode_occupancy_contract():
    def bad_rasterizer(value):
        return torch.full((value.selected_deltas.shape[0], 2, 2, 2), 2.0)

    with pytest.raises(ORIONDecodeAdapterError, match=r"\[0, 1\]"):
        adapt_orion_head_outputs_v1(
            _fixture_head_outputs(),
            config=_config(),
            occupancy_rasterizer=bad_rasterizer,
        )
