import importlib.util
import json
from pathlib import Path
import sys


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_qwen_navsim_dropout_manifest.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_qwen_navsim_dropout_manifest_test", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_selects_only_current_front_frame_by_default(tmp_path):
    scene_path = tmp_path / "scenes.jsonl"
    content = [{"text": "<FRONT VIEW>"}]
    content += [{"image": f"CAM_F0/{index}.jpg"} for index in range(4)]
    content += [{"image": f"CAM_L0/{index}.jpg"} for index in range(4)]
    content += [{"image": f"CAM_R0/{index}.jpg"} for index in range(4)]
    scene_path.write_text(
        json.dumps({"messages": [{"content": content}]}) + "\n",
        encoding="utf-8",
    )

    assert MODULE.selected_paths(scene_path, camera_index=0, scope="current") == [
        "CAM_F0/3.jpg"
    ]
    assert MODULE.selected_paths(scene_path, camera_index=0, scope="all-history") == [
        f"CAM_F0/{index}.jpg" for index in range(4)
    ]
