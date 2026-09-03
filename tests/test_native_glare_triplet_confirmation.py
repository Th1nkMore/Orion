import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_native_glare_confirmation_route import prepare_route
from validate_native_glare_triplet_capture import validate_capture


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_confirmation_route_changes_weather_only(tmp_path):
    source = tmp_path / "route.xml"
    source.write_text(
        '<routes><route id="1" town="Town04"><waypoints>'
        '<position x="0" y="0" z="0"/><position x="-2" y="0" z="0"/>'
        '</waypoints><scenarios><scenario name="PedestrianCrossing_1" '
        'type="PedestrianCrossing"><trigger_point x="-1" y="0" z="0"/>'
        '</scenario></scenarios><weathers><weather route_percentage="0" '
        'sun_altitude_angle="45"/><weather route_percentage="100" '
        'sun_altitude_angle="45"/></weathers></route></routes>',
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "route": {
            "route_index": 203,
            "town": "Town04",
            "scenario_type": "PedestrianCrossing",
            "source_sha256": _sha256(source),
        },
        "weather": {"sun_altitude_angle": 8.0, "sun_azimuth_angle": 180.0},
    }), encoding="utf-8")
    output = tmp_path / "derived.xml"
    manifest = tmp_path / "manifest.json"

    result = prepare_route(source, protocol, output, manifest)

    assert result["only_weather_changed"] is True
    assert result["structure_without_weather_sha256_before"] == result["structure_without_weather_sha256_after"]
    assert result["town"] == "Town04"
    assert result["scenario_type"] == "PedestrianCrossing"


def test_project_actor_box_and_equal_area_control_are_valid():
    pytest.importorskip("cv2")
    from analyze_native_glare_triplet_capture import _control_roi, _project_actor_bbox

    actor = {
        "bbox_world_vertices": [
            [10.0, y, z]
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
            for _ in (0, 1)
        ]
    }
    bbox = _project_actor_bbox(actor, np.eye(4).reshape(-1).tolist(), 1600, 900, 70.0)
    assert bbox is not None
    assert bbox[0] < 800 < bbox[2]
    assert bbox[1] < 450 < bbox[3]
    control = _control_roi(bbox, 1600, 900)
    assert control[2] - control[0] == bbox[2] - bbox[0]
    assert control[3] - control[1] == bbox[3] - bbox[1]


def test_triplet_validator_requires_exact_sync_and_readback(tmp_path):
    root = tmp_path / "capture"
    records = root / "records"
    records.mkdir(parents=True)
    files = {}
    for name in ("clean", "medium", "heavy", "bev"):
        path = records / (name + ".png")
        path.write_bytes(b"png")
        files[name] = str(path)
    profiles = {
        "clean": {"sensor_id": "CAM_FRONT_CLEAN", "lens_flare_intensity": 0.0, "bloom_intensity": 0.0},
        "medium": {"sensor_id": "CAM_FRONT_MEDIUM", "lens_flare_intensity": 0.75, "bloom_intensity": 1.5},
        "heavy": {"sensor_id": "CAM_FRONT_HEAVY", "lens_flare_intensity": 1.5, "bloom_intensity": 3.0},
    }
    weather = {"sun_altitude_angle": 8.0, "sun_azimuth_angle": 180.0}
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "capture": {
            "minimum_saved_frames": 3,
            "maximum_saved_frames": 4,
            "stride_simulator_ticks": 2,
            "route_progress_window": [0.0, 0.35],
            "minimum_visible_pedestrian_frames": 1,
        },
        "camera_profiles": profiles,
        "weather": weather,
    }), encoding="utf-8")
    readback = {
        profile["sensor_id"]: {
            "attributes": {
                "enable_postprocess_effects": "true",
                "lens_flare_intensity": str(profile["lens_flare_intensity"]),
                "bloom_intensity": str(profile["bloom_intensity"]),
            }
        }
        for profile in profiles.values()
    }
    readback["bev"] = {"attributes": {"enable_postprocess_effects": "false"}}
    rows = []
    for index in range(3):
        sensor_frames = {profile["sensor_id"]: 100 + index * 2 for profile in profiles.values()}
        rows.append({
            "capture_index": index,
            "step": 10 + index * 2,
            "route_progress": 0.1 * index,
            "orion_loaded": False,
            "adapter_loaded": False,
            "same_tick": True,
            "sensor_frames": sensor_frames,
            "camera_profiles_requested": profiles,
            "weather_requested": weather,
            "weather_readback": weather,
            "sensor_readback": readback,
            "front": {name: files[name] for name in ("clean", "medium", "heavy")},
            "bev": files["bev"],
            "nearby_actors": ([{"category": "walker"}] if index == 1 else []),
        })
    (records / "capture_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = validate_capture(root, protocol)

    assert report["valid"] is True
    assert report["all_triplets_same_tick"] is True
    assert report["frame_count"] == 3


def test_triplet_agent_is_orion_free_and_uses_three_colocated_rgb_cameras():
    source = (ROOT / "team_code" / "glare_triplet_capture_agent.py").read_text(encoding="utf-8")
    assert "import orion" not in source.lower()
    assert '"orion_loaded": False' in source
    assert '"adapter_loaded": False' in source
    for sensor_id in ("CAM_FRONT_CLEAN", "CAM_FRONT_MEDIUM", "CAM_FRONT_HEAVY"):
        assert sensor_id in (ROOT / "configs" / "native_glare_independent_confirmation_route203_v1.json").read_text()
    assert "len(set(sensor_frames.values())) != 1" in source
    assert "bbox_world_vertices" in source
    assert "sensor_readback" in source
    assert 'split("+", 1)[0]' in source
    assert 'self.sensor_interface, "_sensors_objects"' in source
    assert '"registry": "AutonomousAgent.sensor_interface._sensors_objects"' in source
