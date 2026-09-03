from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from uq_estimator.decoded_actual_target_export import (
    ActualTargetExportError,
    DecodedORIONFrameV1,
    FailureEventPolicyV1,
    bridge_actual_target_bundle_to_v2_record,
    build_cpu_mock_actual_target_bundle,
    load_paired_actual_target_bundle,
    pair_actual_target_branches,
    save_paired_actual_target_bundle,
)
from uq_estimator.spatial_training import (
    TARGET_ACTUAL_FAILURE,
    load_paired_feature_records,
    save_paired_feature_records,
)


def _minimal_decoded(**overrides) -> DecodedORIONFrameV1:
    mode_scores = torch.tensor([[0.1, 0.9]])
    values = {
        "centers_lidar": torch.tensor([[0.0, 0.0]]),
        "boxes_lidar": torch.tensor(
            [[0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]]
        ),
        "classes": torch.tensor([0]),
        "scores": torch.tensor([0.8]),
        "source_query_index": torch.tensor([4]),
        "class_probabilities": torch.tensor([[0.8, 0.2]]),
        "selected_motion_occupancy": torch.zeros(1, 2, 2, 2),
        "traffic_state_logits": torch.tensor([[0.0, 1.0]]),
        "all_trajectory_modes": torch.zeros(1, 2, 2, 2),
        "trajectory_mode_scores": mode_scores,
        "selected_motion_mode_index": mode_scores.argmax(dim=1),
        "occupancy_rasterizer_id": "planningmetric-compatible-test-v1",
        "decoder_layer": 5,
        "decoder_policy_id": "custom-nms-free-topk-v1",
        "class_mapping_id": "b2d-class-map-v1",
        "with_light_state": True,
    }
    values.update(overrides)
    return DecodedORIONFrameV1(**values)


def test_cpu_mock_builds_actual_separate_targets_and_keeps_decoder_audit():
    bundle, observed_features, clean_features, corruption_mask = (
        build_cpu_mock_actual_target_bundle()
    )
    assert bundle.real_orion_hook_executed is False
    assert bundle.patch_attribution_is_causal is False
    assert bundle.observed.error_severity_target.shape == (2, 6)
    assert bundle.observed.component_errors.shape == (2, 6, 6)
    assert bundle.observed.component_error_names[-1] == "false_positive"
    assert bundle.observed.object_component_values.shape == (5, 6)  # G=2, N=3
    assert bundle.observed.object_component_valid.shape == (5, 6)
    assert bundle.delta_error[bundle.paired_valid_mask].mean() > 0
    assert bundle.observed.failure_event_target.dtype == torch.bool
    assert not torch.equal(
        bundle.observed.error_severity_target,
        bundle.observed.failure_event_target.float(),
    )
    assert bundle.observed.duplicate_source_queries_present is True
    assert bundle.observed.source_query_index.tolist() == [7, 11, 11]
    assert bundle.observed.full_class_sigmoid.shape == (3, 3)
    assert bundle.observed.traffic_state_logits.shape == (3, 2)
    assert bundle.observed.all_trajectory_modes.shape == (3, 2, 2, 2)
    assert bundle.observed.selected_motion_mode_index.tolist() == [0, 1, 1]
    # A 0.10-support hard FP and a 0.10-support hard miss remain binary events
    # even though their continuous projected component errors are below 0.5.
    assert bundle.observed.component_errors[0, 3, -1].item() == pytest.approx(0.08)
    assert bundle.observed.component_errors[0, 4, 0].item() == pytest.approx(0.10)
    assert bundle.observed.failure_event_target[0, 3].item() is True
    assert bundle.observed.failure_event_target[0, 4].item() is True
    assert bundle.observed.failure_event_policy.minimum_patch_support == 0.01
    assert bundle.observed.selected_motion_occupancy.shape == (3, 2, 2, 2)
    assert bundle.observed.gt_motion_occupancy.shape == (2, 2, 2, 2)
    assert bundle.observed.gt_motion_valid_mask.all()
    assert observed_features.shape == clean_features.shape == (2, 6, 8)
    assert corruption_mask.shape == (2, 6)


def test_v2_bridge_round_trip_keeps_actual_contract(tmp_path):
    bundle, observed_features, clean_features, corruption_mask = (
        build_cpu_mock_actual_target_bundle(feature_dim=4)
    )
    record = bridge_actual_target_bundle_to_v2_record(
        bundle,
        observed_features,
        clean_features,
        record_id="record-42",
        pair_id="pair-42",
        corruption_mask=corruption_mask,
    )
    assert record.target_provenance == TARGET_ACTUAL_FAILURE
    assert record.error_severity_target.shape == (2, 6)
    assert record.failure_event_target.dtype == torch.bool
    assert record.clean_error_severity_target is not None
    assert record.component_error_names[-1] == "false_positive"
    assert record.metadata["claim_boundary"]["real_orion_hook_completed"] is False

    path = tmp_path / "record.pt"
    save_paired_feature_records(path, [record])
    restored = load_paired_feature_records(path)[0]
    torch.testing.assert_close(restored.component_errors, record.component_errors)
    assert restored.metadata == record.metadata


def test_bundle_tensor_primitive_round_trip_includes_bev_sidecars(tmp_path):
    bundle, *_ = build_cpu_mock_actual_target_bundle()
    path = tmp_path / "bundle.pt"
    save_paired_actual_target_bundle(path, bundle)
    restored = load_paired_actual_target_bundle(path)
    assert restored.bundle_id == bundle.bundle_id
    torch.testing.assert_close(restored.delta_error, bundle.delta_error)
    assert restored.observed.bev_occupancy_sidecar is not None
    torch.testing.assert_close(
        restored.observed.bev_occupancy_sidecar.absolute_error,
        bundle.observed.bev_occupancy_sidecar.absolute_error,
    )
    assert restored.observed.duplicate_source_queries_present
    torch.testing.assert_close(
        restored.observed.selected_motion_occupancy,
        bundle.observed.selected_motion_occupancy,
    )
    torch.testing.assert_close(
        restored.observed.decoded_boxes_lidar,
        bundle.observed.decoded_boxes_lidar,
    )
    torch.testing.assert_close(
        restored.observed.gt_boxes_lidar,
        bundle.observed.gt_boxes_lidar,
    )
    torch.testing.assert_close(
        restored.observed.gt_motion_occupancy,
        bundle.observed.gt_motion_occupancy,
    )
    assert (
        restored.observed.occupancy_rasterizer_id
        == "cpu-mock-planningmetric-compatible-v1"
    )
    with pytest.raises(ActualTargetExportError, match="overwrite"):
        save_paired_actual_target_bundle(path, bundle)


def test_light_state_and_decoder_parity_fail_closed():
    with pytest.raises(ActualTargetExportError, match="with_light_state=True"):
        _minimal_decoded(with_light_state=False)
    with pytest.raises(ActualTargetExportError, match="selected entry"):
        _minimal_decoded(scores=torch.tensor([0.7]))
    with pytest.raises(ActualTargetExportError, match="mode-score argmax"):
        _minimal_decoded(selected_motion_mode_index=torch.tensor([0]))
    with pytest.raises(ActualTargetExportError, match="query×class top-k"):
        _minimal_decoded(decoder_flatten_policy="classwise_nms")

    duplicate_modes = torch.zeros(2, 2, 2, 2)
    duplicate_scores = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
    with pytest.raises(ActualTargetExportError, match="duplicate source query"):
        _minimal_decoded(
            centers_lidar=torch.zeros(2, 2),
            boxes_lidar=torch.tensor(
                [
                    [0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],
                    [0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],
                ]
            ),
            classes=torch.tensor([0, 1]),
            scores=torch.tensor([0.8, 0.2]),
            source_query_index=torch.tensor([4, 4]),
            class_probabilities=torch.tensor([[0.8, 0.2], [0.8, 0.2]]),
            selected_motion_occupancy=torch.zeros(2, 2, 2, 2),
            traffic_state_logits=torch.zeros(2, 2),
            all_trajectory_modes=duplicate_modes,
            trajectory_mode_scores=duplicate_scores,
            selected_motion_mode_index=torch.tensor([1, 1]),
        )

    with pytest.raises(ActualTargetExportError, match="thresholds must lie"):
        FailureEventPolicyV1(
            component_thresholds=(True, 0.5, 0.5, 0.5, 0.5, 0.5)
        )
    with pytest.raises(ActualTargetExportError, match="minimum_patch_support"):
        FailureEventPolicyV1(minimum_patch_support=True)


def test_branch_specific_histories_pair_but_shared_protocol_mismatch_fails():
    bundle, *_ = build_cpu_mock_actual_target_bundle()
    assert (
        bundle.observed.chronology.branch_history_id
        != bundle.clean.chronology.branch_history_id
    )
    assert (
        bundle.observed.chronology.paired_replay_id
        == bundle.clean.chronology.paired_replay_id
    )
    changed_clean = replace(
        bundle.clean,
        target_provenance=replace(
            bundle.clean.target_provenance,
            paired_history_protocol_id="incompatible-protocol",
        ),
        chronology=replace(
            bundle.clean.chronology,
            paired_replay_id="incompatible-protocol",
        ),
    )
    with pytest.raises(ActualTargetExportError, match="paired_history_protocol_id"):
        pair_actual_target_branches(
            bundle.observed,
            changed_clean,
            bundle_id="bad-pair",
        )


def test_pairing_rejects_changed_gt_support_or_frame():
    bundle, *_ = build_cpu_mock_actual_target_bundle()
    changed_support = bundle.clean.gt_projected_support.clone()
    changed_support[0, 0, 0] = 0.2
    with pytest.raises(ActualTargetExportError, match="projected support"):
        pair_actual_target_branches(
            bundle.observed,
            replace(bundle.clean, gt_projected_support=changed_support),
            bundle_id="changed-support",
        )
    changed_frame = replace(
        bundle.clean,
        target_provenance=replace(bundle.clean.target_provenance, frame_idx=43),
        chronology=replace(bundle.clean.chronology, frame_idx=43),
    )
    with pytest.raises(ActualTargetExportError, match="frame_idx"):
        pair_actual_target_branches(
            bundle.observed,
            changed_frame,
            bundle_id="changed-frame",
        )


def test_branch_bundle_rejects_tampered_severity_component_or_event_maps():
    bundle, *_ = build_cpu_mock_actual_target_bundle()
    branch = bundle.observed

    changed_severity = branch.error_severity_target.clone()
    changed_severity[0, 0] = (changed_severity[0, 0] + 0.1).clamp(max=1.0)
    with pytest.raises(ActualTargetExportError, match="error_severity_target is inconsistent"):
        replace(branch, error_severity_target=changed_severity)

    changed_component = branch.component_errors.clone()
    changed_component[0, 0, 0] = (changed_component[0, 0, 0] + 0.1).clamp(max=1.0)
    with pytest.raises(ActualTargetExportError, match="component_errors are inconsistent"):
        replace(branch, component_errors=changed_component)

    changed_event = branch.failure_event_target.clone()
    changed_event[0, 4] = ~changed_event[0, 4]
    with pytest.raises(ActualTargetExportError, match="failure_event_target is inconsistent"):
        replace(branch, failure_event_target=changed_event)


def test_cli_mock_dry_run_and_write(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_decoded_actual_targets.py"
    bundle_path = tmp_path / "bundle.pt"
    record_path = tmp_path / "record.pt"
    dry = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mock",
            "--dry-run",
            "--output",
            str(bundle_path),
            "--record-output",
            str(record_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    summary = json.loads(dry.stdout)
    assert summary["writes_performed"] is False
    assert summary["real_orion_hook_executed"] is False
    assert summary["duplicate_source_queries_preserved"] is True
    assert not bundle_path.exists()
    assert not record_path.exists()

    written = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mock",
            "--output",
            str(bundle_path),
            "--record-output",
            str(record_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    summary = json.loads(written.stdout)
    assert summary["writes_performed"] is True
    assert bundle_path.exists() and record_path.exists()
