import json
from pathlib import Path

from scripts.build_event_keyframe_manifest import select_event_keyframes
from scripts.scenario_factory_lib import CAMERA_DIRECTORIES, sha256_file


def _fixture(tmp_path: Path):
    scenario = tmp_path / "scenario"
    for directory in CAMERA_DIRECTORIES:
        root = scenario / directory
        root.mkdir(parents=True)
        for frame in range(8):
            (root / ("%04d.png" % frame)).write_bytes(b"png")
    meta = scenario / "meta"
    meta.mkdir()
    for frame in range(8):
        (meta / ("%04d.json" % frame)).write_text("{}")
    trace = tmp_path / "control_trace.jsonl"
    rows = []
    for step in range(80):
        rows.append({
            "step": step,
            "sim_time_seconds": step / 10.0,
            "route_progress": step / 80.0,
            "speed": 4.0,
            "closedloop_safety": {
                "actors": [{
                    "actor_id": 7,
                    "obb_collision_ttc_seconds": max(0.0, 4.0 - step / 10.0),
                    "obb_separating_axis_gap_m": 1.0,
                }]
            },
        })
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    package = tmp_path / "event_package.json"
    package.write_text(json.dumps({
        "schema": "orion.scenario_event_package.v1",
        "qa_input_ready": True,
        "runtime": {"valid": True},
        "route": {"route_index": 42},
        "critical_event": {
            "step": 40,
            "sim_time_seconds": 4.0,
            "actor": {"actor_id": 7},
        },
        "camera_inventory": {
            directory: {"path": str(scenario / directory)}
            for directory in CAMERA_DIRECTORIES
        },
        "source_files": {
            "control_trace": {"path": str(trace), "sha256": sha256_file(trace)}
        },
    }))
    return package


def test_fixed_offsets_select_distinct_aligned_frames(tmp_path):
    package = _fixture(tmp_path)
    report = select_event_keyframes(
        event_package_path=package,
        offsets_seconds=(-2.0, -1.0, 0.0, 1.0, 2.0),
    )
    assert report["event_id"] == "route42_step40"
    assert [row["selected_saved_frame_index"] for row in report["keyframes"]] == [2, 3, 4, 5, 6]
    assert report["selection_policy"]["uses_learned_uq"] is False
    assert report["selection_policy"]["uses_stage2_outputs"] is False


def test_keyframes_reject_missing_zero_offset(tmp_path):
    package = _fixture(tmp_path)
    try:
        select_event_keyframes(
            event_package_path=package,
            offsets_seconds=(-1.0, 1.0),
        )
    except ValueError as error:
        assert "include the event center" in str(error)
    else:
        raise AssertionError("missing event-center keyframe was accepted")
