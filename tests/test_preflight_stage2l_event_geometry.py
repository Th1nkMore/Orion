import json

from scripts import preflight_stage2l_event_geometry as module
from scripts.scenario_factory_lib import sha256_file
from uq_estimator.task_relevance_geometry import TaskRelevanceGeometryError


class _Geometry:
    route_point_coverage = 0.75


def _inputs(tmp_path, visible_count):
    scenario = tmp_path / "scenario"
    front = scenario / "rgb_front"
    meta = scenario / "meta"
    front.mkdir(parents=True)
    meta.mkdir()
    frames = (10, 12, 14, 16, 18)
    for index, frame in enumerate(frames):
        (meta / ("%04d.json" % frame)).write_text(json.dumps({
            "plan": [[1.0 if index < visible_count else -1.0, 2.0]] * 6,
            "closedloop_safety": {"available": True},
        }))
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({
        "schema": "orion.scenario_event_package.v1",
        "route": {"route_index": 1},
        "critical_event": {"step": 100},
        "camera_inventory": {"rgb_front": {"path": str(front)}},
    }))
    keyframe_path = tmp_path / "keyframes.json"
    keyframe_path.write_text(json.dumps({
        "schema": "orion.scenario_event_keyframes.v1",
        "event_id": "route1_step100",
        "keyframes": [
            {"selected_saved_frame_index": frame} for frame in frames
        ],
        "provenance": {
            "event_package": {
                "path": str(event_path),
                "sha256": sha256_file(event_path),
            }
        },
    }))
    return event_path, keyframe_path


def _fake_geometry(plan, _safety, patch_hw):
    assert patch_hw == (40, 40)
    if plan[0][0] < 0:
        raise TaskRelevanceGeometryError(
            "the ORION route has no visible camera support"
        )
    return _Geometry()


def test_preflight_keeps_three_visible_fixed_keyframes(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "build_task_relevance_map", _fake_geometry)
    event, keyframes = _inputs(tmp_path, visible_count=3)
    report = module.preflight_geometry(
        event_package_path=event, keyframe_manifest_path=keyframes
    )
    assert report["eligible"]
    assert report["retained_keyframe_count"] == 3
    assert len(report["excluded"]) == 2
    assert not report["selection_and_label_inputs"]["uses_observation_uq"]


def test_preflight_rejects_before_stage1_when_only_two_are_visible(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(module, "build_task_relevance_map", _fake_geometry)
    event, keyframes = _inputs(tmp_path, visible_count=2)
    report = module.preflight_geometry(
        event_package_path=event, keyframe_manifest_path=keyframes
    )
    assert not report["eligible"]
    assert report["status"] == "ineligible_before_stage1_extraction"
