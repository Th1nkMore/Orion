import importlib.util
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/scenario_factory/corruption_hardcase_clean_render_diagnostic_route151_v1.json"
SCRIPT = ROOT / "scripts/render_clean_camera_diagnostic.py"


def _module():
    spec = importlib.util.spec_from_file_location("clean_camera_diagnostic", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_is_no_orion_and_has_single_attribute_ablation():
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["schema"] == "orion.clean_camera_render_diagnostic.v1"
    assert [row["profile"] for row in protocol["conditions"]] == [
        "none",
        "intensity_zero_only",
        "clean",
    ]
    assert protocol["conditions"][0]["front_camera_override"] is None
    assert protocol["conditions"][1]["front_camera_override"] == {
        "motion_blur_intensity": 0.0
    }
    assert protocol["execution_locks"]["orion_loaded"] is False
    assert protocol["execution_locks"]["orion_hardcase_screen"] is False


def test_renderer_preserves_raw_resolution_before_review_resize(tmp_path):
    module = _module()
    root = tmp_path / "run"
    for profile_index, profile in enumerate(module.PROFILES):
        record = root / "captures" / profile / "records"
        frame_root = record / "rgb_front"
        frame_root.mkdir(parents=True)
        rows = []
        for index, progress in enumerate((0.38, 0.40)):
            frame = frame_root / ("%04d.png" % index)
            Image.new("RGB", (64, 36), (profile_index * 40, index * 20, 10)).save(frame)
            rows.append(
                {
                    "route_progress": progress,
                    "front": str(frame.resolve()),
                    "camera_postprocess_readback": {"profile": profile},
                    "weather": {"cloudiness": 5.0},
                }
            )
        (record / "capture_trace.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    output = root / "visual_review"
    result_path = module.render(root, PROTOCOL, output)
    result = json.loads(result_path.read_text())
    assert result["status"] == "pending_human_visual_review"
    assert result["orion_loaded"] is False
    assert result["raw_evidence"]["none"]["size"] == [64, 36]
    assert (output / "front_contact.png").is_file()
    assert (output / "front_diagnostic.gif").is_file()
