import json
from pathlib import Path

import pytest

from scripts.scenario_factory_lib import sha256_file
from scripts.validate_stage2l_formal_stage1_reuse import validate_reuse


CHECKPOINT = "a" * 64


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(tmp_path: Path, *, native_glare: bool = False, checkpoint: str = CHECKPOINT):
    run = _write(
        tmp_path / "run.json",
        {
            "pilot_condition": "clean_off",
            "orion_closedloop_uq_mode": "none",
            "orion_closedloop_risk_mode": "off",
            "orion_planning_response_mode": "off",
            "orion_enable_legacy_density_uq": "0",
            "orion_closedloop_corruption": None,
            "render_condition": {
                "schema": "orion.closedloop_render_condition.v1",
                "kind": "carla_native_low_sun_glare" if native_glare else "standard_carla_rgb",
                "native_glare_profile": "medium" if native_glare else "none",
                "camera_postprocess_override": native_glare,
            },
        },
    )
    event = _write(
        tmp_path / "event.json",
        {
            "schema": "orion.scenario_event_package.v1",
            "runtime": {"valid": True},
            "qa_input_ready": True,
            "route": {"route_index": 7},
            "critical_event": {"step": 42},
            "source_files": {
                "run_manifest": {"path": str(run), "sha256": sha256_file(run)}
            },
        },
    )
    keyframes = _write(
        tmp_path / "keyframes.json",
        {
            "event_id": "route7_step42",
            "keyframes": [
                {"selected_saved_frame_index": frame} for frame in (2, 4, 6)
            ],
        },
    )
    sequences = []
    for frame in (2, 4, 6):
        sequence = _write(
            tmp_path / ("sequence_%d.json" % frame),
            {
                "schema": "orion.stage1_observation_uq_sequence.v1",
                "status": "offline_frozen_stage1_output",
                "control_influence": False,
                "event_package_sha256": sha256_file(event),
                "checkpoint_sha256": checkpoint,
                "forbidden_inputs": {
                    "route": False,
                    "actor_geometry": False,
                    "ttc": False,
                    "collision_outcome": False,
                    "corruption_metadata": False,
                },
            },
        )
        sequences.append(
            {
                "selected_saved_frame_index": frame,
                "manifest": {"path": str(sequence), "sha256": sha256_file(sequence)},
            }
        )
    stage1 = _write(
        tmp_path / "stage1.json",
        {
            "schema": "orion.stage1_observation_uq_multiframe.v1",
            "status": "offline_frozen_stage1_multiframe_output",
            "control_influence": False,
            "event_package": {"path": str(event), "sha256": sha256_file(event)},
            "keyframe_manifest": {
                "path": str(keyframes),
                "sha256": sha256_file(keyframes),
            },
            "sequences": sequences,
        },
    )
    plan = _write(
        tmp_path / "plan.json",
        {
            "schema": "orion.stage2_l.formal_route_plan.v1",
            "events": [
                {
                    "route_index": 7,
                    "event_id": "route7_step42",
                    "formal_split": "train",
                }
            ],
        },
    )
    return plan, event, stage1


def test_accepts_hash_bound_clean_stage1_reuse(tmp_path: Path) -> None:
    plan, event, stage1 = _fixture(tmp_path)
    result = validate_reuse(
        formal_route_plan=plan,
        event_package=event,
        stage1_multiframe_manifest=stage1,
        expected_checkpoint_sha256=CHECKPOINT,
    )
    assert result["eligible"] is True
    assert result["event_id"] == "route7_step42"
    assert result["formal_split"] == "train"
    assert result["keyframe_count"] == 3


def test_rejects_native_glare_source(tmp_path: Path) -> None:
    plan, event, stage1 = _fixture(tmp_path, native_glare=True)
    with pytest.raises(ValueError, match="clean render condition"):
        validate_reuse(
            formal_route_plan=plan,
            event_package=event,
            stage1_multiframe_manifest=stage1,
            expected_checkpoint_sha256=CHECKPOINT,
        )


def test_rejects_checkpoint_drift(tmp_path: Path) -> None:
    plan, event, stage1 = _fixture(tmp_path, checkpoint="b" * 64)
    with pytest.raises(ValueError, match="checkpoint"):
        validate_reuse(
            formal_route_plan=plan,
            event_package=event,
            stage1_multiframe_manifest=stage1,
            expected_checkpoint_sha256=CHECKPOINT,
        )
