import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_PATH = PROJECT_ROOT / "uq_estimator" / "qwen_drive_bridge.py"
CONFIG_PATH = PROJECT_ROOT / "configs" / "qwen_drive_b2d_agent_v1.json"
REASONING_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "qwen_drive_b2d_agent_reasoning_sft_v1.json"
)
ORACLE_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "qwen_drive_b2d_agent_oracle_visibility_sft_v1.json"
)
AGENT_PATH = PROJECT_ROOT / "team_code" / "qwen_drive_b2d_agent.py"
SMOKE_PATH = PROJECT_ROOT / "scripts" / "smoke_qwen_drive_b2d_bridge.py"
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_qwen_drive_b2d_smoke.sh"
SUBMIT_PATH = PROJECT_ROOT / "scripts" / "submit_qwen_drive_b2d_bridge_smoke.sh"
CUDA_BUILD_PATH = PROJECT_ROOT / "scripts" / "build_qwen_drive_cuda_extensions.sh"
CUDA_SUBMIT_PATH = PROJECT_ROOT / "scripts" / "submit_qwen_drive_cuda_extensions.sh"
CLOSED_LOOP_SUBMIT_PATH = (
    PROJECT_ROOT / "scripts" / "submit_qwen_drive_b2d_closedloop_smoke.sh"
)


def _load_bridge():
    spec = importlib.util.spec_from_file_location("_qwen_drive_bridge_test", BRIDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bridge = _load_bridge()


def test_standalone_bridge_import_does_not_load_torch():
    code = """
import importlib.util, pathlib, sys
path = pathlib.Path(r'%s')
spec = importlib.util.spec_from_file_location('_bridge_isolation', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
print(module.CONFIG_SCHEMA)
""" % BRIDGE_PATH
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == bridge.CONFIG_SCHEMA


def test_config_locks_direct_single_sample_planning(tmp_path):
    config = bridge.load_bridge_config(CONFIG_PATH)
    assert list(config["sensors"]) == list(bridge.QWEN_VIEW_BY_SENSOR)
    assert config["planning"]["mode"] == "direct_planning"
    assert config["planning"]["num_samples"] == 1
    assert config["runtime"]["attention_implementation"] == "flash_attention_2"
    assert config["images"]["history_size"] is None
    assert config["images"]["current_size"] is None
    assert config["images"]["transport_format"] == "png"

    config["planning"]["num_samples"] = 6
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="num_samples=1"):
        bridge.load_bridge_config(invalid)


def test_reasoning_config_uses_the_official_sft_reasoning_path():
    config = bridge.load_bridge_config(REASONING_CONFIG_PATH)
    assert config["planning"]["mode"] == "reasoning_planning"
    assert config["planning"]["num_samples"] == 1
    assert config["runtime"]["planner"].endswith("/planner-sft")


def test_oracle_visibility_config_is_explicit_and_does_not_replace_rgb_views():
    config = bridge.load_bridge_config(ORACLE_CONFIG_PATH)
    assert list(config["sensors"]) == list(bridge.QWEN_VIEW_BY_SENSOR)
    assert all(
        sensor["type"] == "sensor.camera.rgb"
        for sensor in config["sensors"].values()
    )
    oracle = config["oracle_visibility"]
    assert oracle["enabled"] is True
    assert list(oracle["depth_sensor_by_rgb"].values()) == list(
        bridge.QWEN_VIEW_BY_SENSOR
    )
    assert set(oracle["depth_sensor_by_rgb"]).isdisjoint(config["sensors"])
    assert oracle["audit_snapshot_steps"] == [0, 200, 260, 280, 300]
    assert oracle["audit_depth_max_m"] == 60.0
    assert oracle["temporal_memory"] == {
        "enabled": True,
        "max_age_seconds": 10.0,
        "observed_ratio_threshold": 0.5,
    }
    assert oracle["exposure"]["enabled"] is True
    assert oracle["exposure"]["safe_deceleration_mps2"] == 4.0
    assert config["planning"]["mode"] == "reasoning_planning"
    assert config["runtime"]["planner"].endswith("/planner-sft")


def test_oracle_temporal_and_exposure_config_fail_closed(tmp_path):
    config = bridge.load_bridge_config(ORACLE_CONFIG_PATH)
    config["oracle_visibility"]["temporal_memory"]["max_age_seconds"] = 0.0
    invalid_age = tmp_path / "invalid_age.json"
    invalid_age.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="max_age_seconds"):
        bridge.load_bridge_config(invalid_age)

    config = bridge.load_bridge_config(ORACLE_CONFIG_PATH)
    config["oracle_visibility"]["exposure"]["safe_deceleration_mps2"] = 0.0
    invalid_deceleration = tmp_path / "invalid_deceleration.json"
    invalid_deceleration.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="safe_deceleration_mps2"):
        bridge.load_bridge_config(invalid_deceleration)


def test_carla_to_qwen_coordinate_convention():
    np.testing.assert_allclose(
        bridge.world_vector_to_qwen([3.0, -2.0], 0.0), [3.0, 2.0]
    )
    np.testing.assert_allclose(
        bridge.world_vector_to_qwen([2.0, 3.0], 90.0), [3.0, 2.0], atol=1e-6
    )
    pose = bridge.world_pose_to_qwen([-1.0, -2.0, -10.0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(pose[:2], [-1.0, 2.0])
    assert pose[2] == pytest.approx(np.deg2rad(10.0))


def test_ego_history_is_padded_and_current_pose_is_zero():
    history = bridge.EgoHistoryBuffer(num_points=16)
    history.append([0.0, 0.0, 0.0], [2.0, 0.0], [1.0, 0.0])
    history.append([1.0, 0.0, 0.0], [3.0, 0.0], [0.0, 0.0])
    result = history.build()
    assert result["history"].shape == (16, 3)
    assert result["history_velocity"].shape == (16, 2)
    assert result["history_acceleration"].shape == (16, 2)
    np.testing.assert_allclose(result["history"][-1], 0.0)
    np.testing.assert_allclose(result["history"][-2], [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(result["ego_velocity"], [3.0, 0.0])


@pytest.mark.parametrize(
    "value, expected_one_hot, expected_nav",
    [
        (1, [1, 0, 0, 0], 1),
        (5, [1, 0, 0, 0], 1),
        (2, [0, 0, 1, 0], 2),
        (6, [0, 0, 1, 0], 2),
        (3, [0, 1, 0, 0], 0),
        (4, [0, 1, 0, 0], 0),
        (-1, [0, 0, 0, 1], 0),
    ],
)
def test_bench2drive_command_mapping(value, expected_one_hot, expected_nav):
    one_hot, nav = bridge.bench2drive_command_to_qwen(value)
    np.testing.assert_array_equal(one_hot, expected_one_hot)
    assert nav == expected_nav


def test_image_buffer_can_transport_lossless_sensor_resolution():
    cv2 = pytest.importorskip("cv2")
    buffer = bridge.ImageHistoryBuffer(
        tuple(bridge.QWEN_VIEW_BY_SENSOR),
        num_frames=4,
        history_size=None,
        current_size=None,
        jpeg_quality=100,
        transport_format="png",
    )
    images = {
        sensor_id: np.full((90, 160, 3), index * 30, dtype=np.uint8)
        for index, sensor_id in enumerate(bridge.QWEN_VIEW_BY_SENSOR, start=1)
    }
    views = buffer.capture(images)
    assert list(views) == list(bridge.QWEN_VIEW_BY_SENSOR.values())
    assert all(len(frames) == 4 for frames in views.values())
    history_frame = cv2.imdecode(
        np.frombuffer(next(iter(views.values()))[0], dtype=np.uint8), cv2.IMREAD_COLOR
    )
    current_frame = cv2.imdecode(
        np.frombuffer(next(iter(views.values()))[-1], dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert history_frame.shape[:2] == (90, 160)
    assert current_frame.shape[:2] == (90, 160)
    np.testing.assert_array_equal(history_frame, images["CAM_FRONT"])
    np.testing.assert_array_equal(current_frame, images["CAM_FRONT"])
    assert buffer.retained_bytes > 0
    assert buffer.retained_bytes < sum(image.nbytes for image in images.values())


def test_request_shapes_and_trajectory_conversion():
    frames = {view: [b"jpeg"] * 4 for view in bridge.QWEN_VIEW_BY_SENSOR.values()}
    ego = {
        "history": np.zeros((16, 3), dtype=np.float32),
        "history_velocity": np.zeros((16, 2), dtype=np.float32),
        "history_acceleration": np.zeros((16, 2), dtype=np.float32),
        "ego_velocity": np.zeros(2, dtype=np.float32),
        "ego_acceleration": np.zeros(2, dtype=np.float32),
    }
    request = bridge.make_inference_request(
        frames, ego, [0, 1, 0, 0], nav_command=0, token="test"
    )
    assert request["schema"] == bridge.RPC_SCHEMA
    for key in (
        "history",
        "history_velocity",
        "history_acceleration",
        "ego_velocity",
        "ego_acceleration",
        "driving_command",
    ):
        assert isinstance(request[key], list)
        assert not isinstance(request[key], np.ndarray)

    trajectory = np.zeros((50, 3), dtype=np.float32)
    trajectory[:, 0] = np.arange(1, 51, dtype=np.float32) / 10.0
    world = bridge.qwen_trajectory_to_world(trajectory, [10.0, 20.0, 0.0])
    pid = bridge.world_trajectory_to_pid(
        world,
        [10.0, 20.0, 0.0],
        elapsed_seconds=0.0,
        source_hz=10,
        target_hz=2,
        horizon_points=6,
    )
    np.testing.assert_allclose(pid[:, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(pid[:, 1], [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    delayed = bridge.world_trajectory_to_pid(
        world,
        [10.0, 20.0, 0.0],
        elapsed_seconds=0.4,
        source_hz=10,
        target_hz=2,
        horizon_points=6,
    )
    np.testing.assert_allclose(
        delayed[:, 1], [0.9, 1.4, 1.9, 2.4, 2.9, 3.4], atol=1e-6
    )


def test_left_qwen_trajectory_maps_to_negative_pid_lateral():
    trajectory = np.zeros((50, 3), dtype=np.float32)
    trajectory[:, 0] = np.linspace(0.1, 5.0, 50)
    trajectory[:, 1] = np.linspace(0.05, 2.5, 50)
    world = bridge.qwen_trajectory_to_world(trajectory, [0.0, 0.0, 0.0])
    pid = bridge.world_trajectory_to_pid(
        world, [0.0, 0.0, 0.0], 0.0, 10, 2, 6
    )
    assert np.all(pid[:, 0] < 0.0)
    assert np.all(pid[:, 1] > 0.0)

    controller_spec = importlib.util.spec_from_file_location(
        "_qwen_pid_test", PROJECT_ROOT / "team_code" / "pid_controller.py"
    )
    controller_module = importlib.util.module_from_spec(controller_spec)
    controller_spec.loader.exec_module(controller_module)
    controller = controller_module.PIDController()
    steer, _, _, _ = controller.control_pid(
        pid, np.float32(1.0), np.asarray([-1.0, 5.0], dtype=np.float32)
    )
    assert steer < 0.0


def test_agent_source_has_no_orion_or_torch_runtime_imports():
    source = AGENT_PATH.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import mmcv" not in source
    assert "orion_b2d_agent" not in source
    assert "CAM_BACK" not in source
    assert "sensor.other.gnss" not in source
    assert "sensor.other.imu" not in source


def test_real_bridge_smoke_cli_help_is_lightweight():
    result = subprocess.run(
        [sys.executable, str(SMOKE_PATH), "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--front-left" in result.stdout
    assert "--output" in result.stdout
    assert "--repeat" in result.stdout


def test_closed_loop_runner_shell_syntax():
    subprocess.run(["bash", "-n", str(RUNNER_PATH)], cwd=PROJECT_ROOT, check=True)
    subprocess.run(["bash", "-n", str(SUBMIT_PATH)], cwd=PROJECT_ROOT, check=True)
    subprocess.run(["bash", "-n", str(CUDA_BUILD_PATH)], cwd=PROJECT_ROOT, check=True)
    subprocess.run(["bash", "-n", str(CUDA_SUBMIT_PATH)], cwd=PROJECT_ROOT, check=True)
    subprocess.run(
        ["bash", "-n", str(CLOSED_LOOP_SUBMIT_PATH)], cwd=PROJECT_ROOT, check=True
    )
