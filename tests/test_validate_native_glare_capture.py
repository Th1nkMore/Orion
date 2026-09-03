import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_native_glare_capture import validate_capture


def test_validate_capture_requires_walker_and_orion_free(tmp_path):
    capture = tmp_path / "capture"
    records = capture / "records"
    front = records / "rgb_front"
    bev = records / "bev"
    front.mkdir(parents=True)
    bev.mkdir()
    protocol = tmp_path / "protocol.json"
    camera = {"lens_flare_intensity": 0.0, "bloom_intensity": 0.0}
    weather = {"sun_altitude_angle": 8.0}
    protocol.write_text(json.dumps({"methods": {"carla_native_low_sun": {
        "camera_profiles": {"clean": camera},
        "weather_shared_by_all_profiles": weather,
    }}}))
    rows = []
    for index in range(3):
        front_path = front / ("%04d.png" % index)
        bev_path = bev / ("%04d.png" % index)
        front_path.write_bytes(b"png")
        bev_path.write_bytes(b"png")
        rows.append({
            "profile": "clean",
            "camera_postprocess": camera,
            "weather": weather,
            "capture_index": index,
            "front": str(front_path),
            "bev": str(bev_path),
            "orion_loaded": False,
            "route_progress": 0.1 * index,
            "nearby_actors": ([{"type_id": "walker.pedestrian.0001"}] if index == 1 else []),
        })
    (records / "capture_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    report = validate_capture(capture, protocol, "clean", minimum_frames=3)

    assert report["frame_count"] == 3
    assert report["walker_frame_count"] == 1
    assert report["orion_loaded"] is False
