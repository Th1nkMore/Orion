import importlib.util
import json
import queue
import sys
import time
import types
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "uq_estimator"
    / "closedloop_sensor_diagnostics.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "_closedloop_sensor_diagnostics_test", MODULE_PATH
)
diagnostics = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = diagnostics
MODULE_SPEC.loader.exec_module(diagnostics)

install_exact_frame_speedometer = diagnostics.install_exact_frame_speedometer
install_oracle_depth_camera_support = (
    diagnostics.install_oracle_depth_camera_support
)
install_sensor_queue_diagnostics = diagnostics.install_sensor_queue_diagnostics


def make_sensor_module():
    class SensorReceivedNoData(Exception):
        pass

    class SensorInterface:
        def __init__(self, tags=("front", "imu"), opendrive_tag=None):
            self._sensors_objects = {tag: object() for tag in tags}
            self._data_buffers = queue.Queue()
            self._opendrive_tag = opendrive_tag

        def update_sensor(self, tag, data, frame):
            if tag not in self._sensors_objects:
                raise RuntimeError(tag)
            self._data_buffers.put((tag, frame, data))

        def get_data(self, frame):
            return {"original": (frame, None)}

    return types.SimpleNamespace(
        SensorInterface=SensorInterface,
        SensorReceivedNoData=SensorReceivedNoData,
        Empty=queue.Empty,
    )


def test_disabled_install_preserves_original_method():
    module = make_sensor_module()
    original = module.SensorInterface.get_data
    assert not install_sensor_queue_diagnostics(module, enabled=False)
    assert module.SensorInterface.get_data is original


def test_complete_current_frame_bundle_is_unchanged():
    module = make_sensor_module()
    messages = []
    assert install_sensor_queue_diagnostics(
        module, enabled=True, timeout_seconds=0.1, emit=messages.append
    )
    interface = module.SensorInterface()
    interface.update_sensor("front", "image", 7)
    interface.update_sensor("imu", "motion", 7)

    assert interface.get_data(7) == {
        "front": (7, "image"),
        "imu": (7, "motion"),
    }
    assert interface._orion_last_callback_frame == {"front": 7, "imu": 7}
    assert any("enabled strict current-frame" in item for item in messages)


def test_timeout_reports_exact_missing_tag_and_never_reuses_stale_data(tmp_path):
    module = make_sensor_module()
    output = tmp_path / "sensor_queue.jsonl"
    messages = []
    install_sensor_queue_diagnostics(
        module,
        enabled=True,
        timeout_seconds=0.01,
        output_path=output,
        emit=messages.append,
    )
    interface = module.SensorInterface()
    interface.update_sensor("front", "stale", 8)
    interface.update_sensor("front", "current", 9)

    with pytest.raises(module.SensorReceivedNoData, match="missing: imu"):
        interface.get_data(9)

    payload = json.loads(output.read_text().strip())
    assert payload["target_frame"] == 9
    assert payload["received_tags"] == ["front"]
    assert payload["missing_tags"] == ["imu"]
    assert payload["last_callback_frame_by_tag"] == {"front": 9}
    assert payload["off_target_frame_items_discarded_by_tag"] == {"front": 1}
    assert payload["strict_current_frame_semantics"] is True
    assert payload["stale_frame_reuse"] is False


def test_opendrive_is_optional_only_when_it_is_the_last_missing_tag():
    module = make_sensor_module()
    install_sensor_queue_diagnostics(
        module, enabled=True, timeout_seconds=0.1, emit=lambda _: None
    )
    interface = module.SensorInterface(
        tags=("front", "opendrive"), opendrive_tag="opendrive"
    )
    interface.update_sensor("front", "image", 3)
    assert interface.get_data(3) == {"front": (3, "image")}


def test_install_is_idempotent():
    module = make_sensor_module()
    assert install_sensor_queue_diagnostics(
        module, enabled=True, timeout_seconds=0.1, emit=lambda _: None
    )
    patched = module.SensorInterface.get_data
    assert install_sensor_queue_diagnostics(
        module, enabled=True, timeout_seconds=0.2, emit=lambda _: None
    )
    assert module.SensorInterface.get_data is patched


def make_speedometer_module(*, advance_during_first_read=False):
    class GameTime:
        frame = 0

        @classmethod
        def get_frame(cls):
            return cls.frame

    class GenericMeasurement:
        def __init__(self, data, frame):
            self.data = data
            self.frame = frame

    class SpeedometerReader:
        def __init__(self):
            self._run_ps = True
            self._callback = None
            self._read_count = 0
            self.run()

        def __call__(self):
            self._read_count += 1
            sampled_frame = GameTime.get_frame()
            if advance_during_first_read and self._read_count == 1:
                GameTime.frame += 1
            return {"speed": sampled_frame / 10.0}

        def run(self):
            raise AssertionError("stock floating-time reader should be replaced")

        def listen(self, callback):
            self._callback = callback

        def stop(self):
            self._run_ps = False

    return types.SimpleNamespace(
        GameTime=GameTime,
        GenericMeasurement=GenericMeasurement,
        SpeedometerReader=SpeedometerReader,
    )


def wait_for_measurements(items, count):
    deadline = time.monotonic() + 0.5
    while len(items) < count and time.monotonic() < deadline:
        time.sleep(0.001)
    assert len(items) >= count


def test_exact_frame_speedometer_emits_once_for_each_nonzero_frame():
    module = make_speedometer_module()
    messages = []
    assert install_exact_frame_speedometer(
        module, enabled=True, poll_seconds=0.0001, emit=messages.append
    )
    reader = module.SpeedometerReader()
    measurements = []
    reader.listen(measurements.append)

    module.GameTime.frame = 41
    wait_for_measurements(measurements, 1)
    time.sleep(0.005)
    assert [(item.frame, item.data) for item in measurements] == [
        (41, {"speed": 4.1})
    ]

    module.GameTime.frame = 42
    wait_for_measurements(measurements, 2)
    reader.stop()
    assert [(item.frame, item.data) for item in measurements] == [
        (41, {"speed": 4.1}),
        (42, {"speed": 4.2}),
    ]
    assert any("stale_reuse=false" in item for item in messages)


def test_exact_frame_speedometer_discards_cross_tick_read():
    module = make_speedometer_module(advance_during_first_read=True)
    install_exact_frame_speedometer(
        module, enabled=True, poll_seconds=0.0001, emit=lambda _: None
    )
    reader = module.SpeedometerReader()
    measurements = []
    reader.listen(measurements.append)

    module.GameTime.frame = 51
    wait_for_measurements(measurements, 1)
    reader.stop()
    assert measurements[0].frame == 52
    assert measurements[0].data == {"speed": 5.2}
    assert reader._read_count == 2


def test_exact_frame_speedometer_disabled_and_idempotent():
    module = make_speedometer_module()
    stock_run = module.SpeedometerReader.run
    assert not install_exact_frame_speedometer(module, enabled=False)
    assert module.SpeedometerReader.run is stock_run
    assert install_exact_frame_speedometer(
        module, enabled=True, poll_seconds=0.0001, emit=lambda _: None
    )
    patched_run = module.SpeedometerReader.run
    assert install_exact_frame_speedometer(
        module, enabled=True, poll_seconds=0.1, emit=lambda _: None
    )
    assert module.SpeedometerReader.run is patched_run


def make_agent_wrapper_module():
    class AgentWrapper:
        def _preprocess_sensor_spec(self, sensor_spec):
            return ("original", sensor_spec["id"])

    return types.SimpleNamespace(
        AgentWrapper=AgentWrapper,
        ALLOWED_SENSORS=("sensor.camera.rgb",),
        SENSORS_LIMITS={"sensor.camera.rgb": 8},
        QUALIFIER_SENSORS_LIMITS={"sensor.camera.rgb": 4},
    )


def make_carla_module():
    class Value:
        def __init__(self, **values):
            self.values = values

    class Transform:
        def __init__(self, location, rotation):
            self.location = location
            self.rotation = rotation

    return types.SimpleNamespace(Location=Value, Rotation=Value, Transform=Transform)


def test_oracle_depth_camera_support_is_disabled_by_default_contract():
    module = make_agent_wrapper_module()
    original = module.AgentWrapper._preprocess_sensor_spec
    assert not install_oracle_depth_camera_support(
        module, make_carla_module(), enabled=False
    )
    assert module.AgentWrapper._preprocess_sensor_spec is original
    assert "sensor.camera.depth" not in module.ALLOWED_SENSORS


def test_oracle_depth_camera_support_extends_only_depth_camera_handling():
    module = make_agent_wrapper_module()
    messages = []
    sensor_icons = {"sensor.camera.rgb": "carla_camera"}
    assert install_oracle_depth_camera_support(
        module,
        make_carla_module(),
        sensor_icons=sensor_icons,
        enabled=True,
        emit=messages.append,
    )
    assert "sensor.camera.depth" in module.ALLOWED_SENSORS
    assert module.SENSORS_LIMITS["sensor.camera.depth"] == 8
    assert sensor_icons["sensor.camera.depth"] == "carla_camera"
    wrapper = module.AgentWrapper()
    assert wrapper._preprocess_sensor_spec(
        {"type": "sensor.camera.rgb", "id": "CAM_FRONT"}
    ) == ("original", "CAM_FRONT")
    sensor_type, sensor_id, transform, attributes = wrapper._preprocess_sensor_spec(
        {
            "type": "sensor.camera.depth",
            "id": "DEPTH_FRONT",
            "x": 0.8,
            "y": 0.0,
            "z": 1.6,
            "pitch": 0.0,
            "roll": 0.0,
            "yaw": 0.0,
            "width": 1600,
            "height": 900,
            "fov": 70,
        }
    )
    assert (sensor_type, sensor_id) == ("sensor.camera.depth", "DEPTH_FRONT")
    assert transform.location.values == {"x": 0.8, "y": 0.0, "z": 1.6}
    assert attributes == {
        "image_size_x": "1600",
        "image_size_y": "900",
        "fov": "70",
    }
    assert any("official_sensor_track_eligible=false" in item for item in messages)
    patched = module.AgentWrapper._preprocess_sensor_spec
    assert install_oracle_depth_camera_support(
        module, make_carla_module(), enabled=True, emit=messages.append
    )
    assert module.AgentWrapper._preprocess_sensor_spec is patched
