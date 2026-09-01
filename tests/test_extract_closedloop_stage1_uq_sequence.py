import numpy as np
import pytest
import json

from scripts.extract_closedloop_stage1_uq_sequence import (
    _keyframe_targets,
    _write_sequence,
    robust_component_calibration,
)
from scripts.scenario_factory_lib import sha256_file


def test_robust_component_calibration_preserves_shape_and_bounds():
    raw = np.full((6, 2, 3, 4, 3), 0.1, dtype=np.float32)
    raw[5, 0, 1, 2, 0] = 1.0
    normalized, metadata = robust_component_calibration(
        raw, baseline_indices=(0, 1, 2, 3)
    )
    assert normalized.shape == raw.shape
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0
    assert normalized[5, 0, 1, 2, 0] > normalized[0, 0, 1, 2, 0]
    assert metadata["uses_route_or_actor_inputs"] is False


def test_component_mean_is_valid_scalar_uq_contract():
    components = np.zeros((4, 6, 10, 10, 3), dtype=np.float32)
    components[-1, 0, 5, 5] = (0.3, 0.6, 0.9)
    scalar = components.mean(axis=-1)
    assert scalar.shape == (4, 6, 10, 10)
    assert scalar[-1, 0, 5, 5] == pytest.approx(0.6)


def test_keyframe_targets_verify_provenance_and_model_independent_policy(tmp_path):
    event = tmp_path / "event.json"
    event.write_text("{}")
    keyframes = tmp_path / "keyframes.json"
    keyframes.write_text(json.dumps({
        "schema": "orion.scenario_event_keyframes.v1",
        "selection_policy": {
            "uses_learned_uq": False,
            "uses_stage2_outputs": False,
            "uses_qa_answers": False,
            "uses_closed_loop_improvement": False,
        },
        "keyframes": [
            {"selected_saved_frame_index": value}
            for value in (10, 12, 14, 16, 18)
        ],
        "provenance": {
            "event_package": {"path": str(event), "sha256": sha256_file(event)}
        },
    }))
    targets, rows, _ = _keyframe_targets(
        keyframes, event, list(range(20))
    )
    assert targets == [10, 12, 14, 16, 18]
    assert rows[14]["selected_saved_frame_index"] == 14


def test_write_sequence_emits_existing_single_frame_contract(tmp_path):
    event = tmp_path / "event.json"
    event.write_text("{}")
    adapter = tmp_path / "adapter.pt"
    adapter.write_bytes(b"adapter")
    process_frames = [0, 1, 2, 3, 4, 5]
    raw = np.zeros((6, 2, 3, 4, 3), dtype=np.float32)
    normalized = np.full_like(raw, 0.2)
    inventory = [
        {"saved_frame_index": frame, "camera_files": []}
        for frame in process_frames
    ]
    manifest = _write_sequence(
        output_dir=tmp_path / "frame_0005",
        event_package_path=event,
        adapter_checkpoint=adapter,
        adapter_metadata={"sha256": "a" * 64, "schema_version": "adapter/v1"},
        calibration={"schema": "calibration/v1"},
        backbone_record={"type": "mock"},
        process_frames=process_frames,
        frame_inventory=inventory,
        normalized_components=normalized,
        raw_components=raw,
        target_frame=5,
        context_frames=4,
    )
    assert manifest["latest_frame_index"] == 5
    assert manifest["context_saved_frame_indices"] == [2, 3, 4, 5]
    assert manifest["uncertainty"]["shape"] == [4, 2, 3, 4]
