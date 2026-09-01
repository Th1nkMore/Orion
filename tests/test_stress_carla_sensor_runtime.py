import ast
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "stress_carla_sensor_runtime.py"
)


def test_runtime_gate_script_keeps_exact_orion_camera_contract():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "CAMERAS"
    }
    cameras = assignments["CAMERAS"]

    assert [camera[0] for camera in cameras] == [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
        "bev",
    ]
    assert all(camera[7:9] == (1600, 900) for camera in cameras[:6])
    assert cameras[-1][7:10] == (512, 512, 50)
