"""Opt-in diagnostics for Bench2Drive's synchronous sensor queue.

The patch deliberately preserves the evaluator's strict current-frame contract:
it never substitutes stale data, drops a required sensor, or advances CARLA after
an incomplete sensor bundle.  It only makes a missing-frame failure observable
and bounds the time spent waiting for a bundle that can no longer complete.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional


_TRUE_VALUES = {"1", "true", "yes", "on"}


def _enabled(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in _TRUE_VALUES


def _positive_timeout(value: Any) -> float:
    timeout = float(value)
    if timeout <= 0:
        raise ValueError("sensor queue diagnostic timeout must be positive")
    return timeout


def _emit_diagnostic(
    payload: dict[str, Any],
    *,
    output_path: Optional[Path],
    emit: Callable[[str], None],
) -> None:
    line = json.dumps(payload, sort_keys=True, allow_nan=False)
    emit(f"[SensorQueueDiagnostic] {line}")
    if output_path is None:
        return
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception as exc:  # diagnostics must not mask the evaluator error
        emit(f"[SensorQueueDiagnosticWriteError] {exc!r}")


def install_sensor_queue_diagnostics(
    sensor_interface_module: Any,
    *,
    enabled: Optional[bool] = None,
    timeout_seconds: Optional[float] = None,
    output_path: Optional[Path] = None,
    emit: Callable[[str], None] = lambda message: print(message, flush=True),
) -> bool:
    """Instrument ``SensorInterface`` before any agent instance is created.

    Returns ``True`` when the patch is active. Repeated installation is
    idempotent. When disabled, no methods or evaluator semantics are changed.
    """

    if enabled is None:
        enabled = _enabled(os.environ.get("ORION_SENSOR_QUEUE_DIAGNOSTICS"))
    if not enabled:
        return False

    if timeout_seconds is None:
        timeout_seconds = _positive_timeout(
            os.environ.get("ORION_SENSOR_QUEUE_TIMEOUT_SECONDS", "60")
        )
    else:
        timeout_seconds = _positive_timeout(timeout_seconds)

    if output_path is None:
        raw_path = os.environ.get("ORION_SENSOR_DIAGNOSTIC_PATH", "").strip()
        output_path = Path(raw_path) if raw_path else None

    interface_cls = sensor_interface_module.SensorInterface
    if getattr(interface_cls, "_orion_sensor_diagnostics_installed", False):
        return True

    original_update_sensor = interface_cls.update_sensor
    empty_exception = sensor_interface_module.Empty
    no_data_exception = sensor_interface_module.SensorReceivedNoData

    def update_sensor(self: Any, tag: str, data: Any, frame: int) -> Any:
        callback_frames = getattr(self, "_orion_last_callback_frame", None)
        if callback_frames is None:
            callback_frames = {}
            self._orion_last_callback_frame = callback_frames
        callback_frames[str(tag)] = int(frame)
        return original_update_sensor(self, tag, data, frame)

    def get_data(self: Any, frame: int) -> dict[str, tuple[int, Any]]:
        expected_tags = list(self._sensors_objects.keys())
        expected_set = set(expected_tags)
        data_dict: dict[str, tuple[int, Any]] = {}
        observed_off_frame: dict[str, int] = {}
        deadline = time.monotonic() + timeout_seconds

        try:
            while len(data_dict) < len(expected_tags):
                if (
                    self._opendrive_tag
                    and self._opendrive_tag not in data_dict
                    and len(expected_tags) == len(data_dict) + 1
                ):
                    break

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise empty_exception
                tag, sensor_frame, data = self._data_buffers.get(True, remaining)
                tag = str(tag)
                sensor_frame = int(sensor_frame)
                if sensor_frame != int(frame):
                    observed_off_frame[tag] = observed_off_frame.get(tag, 0) + 1
                    continue
                data_dict[tag] = (sensor_frame, data)

        except empty_exception:
            optional_tags = {self._opendrive_tag} if self._opendrive_tag else set()
            missing_tags = sorted(expected_set - set(data_dict) - optional_tags)
            callback_frames = dict(
                sorted(getattr(self, "_orion_last_callback_frame", {}).items())
            )
            payload = {
                "schema": "orion.closedloop_sensor_queue_diagnostic.v1",
                "event": "sensor_bundle_timeout",
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "target_frame": int(frame),
                "timeout_seconds": timeout_seconds,
                "expected_tags": expected_tags,
                "received_tags": sorted(data_dict),
                "missing_tags": missing_tags,
                "last_callback_frame_by_tag": callback_frames,
                "off_target_frame_items_discarded_by_tag": dict(
                    sorted(observed_off_frame.items())
                ),
                "queue_size_at_timeout": self._data_buffers.qsize(),
                "strict_current_frame_semantics": True,
                "stale_frame_reuse": False,
            }
            _emit_diagnostic(payload, output_path=output_path, emit=emit)
            raise no_data_exception(
                "A sensor took too long to send data for frame {} (missing: {})".format(
                    int(frame), ",".join(missing_tags) or "unknown"
                )
            )

        return data_dict

    interface_cls.update_sensor = update_sensor
    interface_cls.get_data = get_data
    interface_cls._orion_sensor_diagnostics_installed = True
    interface_cls._orion_sensor_diagnostics_timeout_seconds = timeout_seconds
    emit(
        "[SensorQueueDiagnostic] enabled strict current-frame timeout={}s path={}".format(
            timeout_seconds, str(output_path) if output_path else "stdout-only"
        )
    )
    return True


def install_exact_frame_speedometer(
    sensor_interface_module: Any,
    *,
    enabled: Optional[bool] = None,
    poll_seconds: float = 0.001,
    emit: Callable[[str], None] = lambda message: print(message, flush=True),
) -> bool:
    """Make Bench2Drive's speedometer emit once per ``GameTime`` frame.

    Bench2Drive's stock ``BaseReader`` schedules pseudo sensors from a floating
    point time difference in a background thread.  At a reading frequency equal
    to the simulator tick rate, the strict ``> 1 / frequency`` boundary can
    leave ``SPEED`` one frame behind every other CARLA sensor.  The evaluator's
    current-frame barrier then has no way to complete.

    This opt-in repair changes only the speedometer producer.  It still reads
    the ego vehicle's current velocity and transform through the stock
    ``SpeedometerReader.__call__`` method, labels the sample with the exact
    current ``GameTime`` frame, and emits at most once for that frame.  It never
    reuses a stale measurement and does not weaken ``SensorInterface.get_data``.
    """

    if enabled is None:
        enabled = _enabled(os.environ.get("ORION_EXACT_FRAME_SPEEDOMETER"))
    if not enabled:
        return False

    poll_seconds = float(poll_seconds)
    if poll_seconds <= 0:
        raise ValueError("exact-frame speedometer poll interval must be positive")

    speedometer_cls = sensor_interface_module.SpeedometerReader
    if getattr(speedometer_cls, "_orion_exact_frame_installed", False):
        return True

    game_time = sensor_interface_module.GameTime
    measurement_cls = sensor_interface_module.GenericMeasurement
    original_run = speedometer_cls.run

    def run_exact_frame(self: Any) -> threading.Thread:
        def produce() -> None:
            last_emitted_frame: Optional[int] = None
            while self._run_ps:
                callback = self._callback
                frame_before = int(game_time.get_frame())
                if (
                    callback is None
                    or frame_before == 0
                    or frame_before == last_emitted_frame
                ):
                    time.sleep(poll_seconds)
                    continue

                data = self.__call__()
                frame_after = int(game_time.get_frame())
                if frame_after != frame_before:
                    # A startup/world-transition tick crossed the vehicle RPC
                    # read. Discard it instead of attaching ambiguous data to
                    # either frame; the next loop reads the new frame afresh.
                    continue

                callback(measurement_cls(data, frame_before))
                last_emitted_frame = frame_before

        thread = threading.Thread(
            target=produce,
            name="orion-exact-frame-speedometer",
            daemon=True,
        )
        thread.start()
        return thread

    speedometer_cls.run = run_exact_frame
    speedometer_cls._orion_exact_frame_installed = True
    speedometer_cls._orion_original_run = original_run
    speedometer_cls._orion_exact_frame_poll_seconds = poll_seconds
    emit(
        "[ExactFrameSpeedometer] enabled frame-driven current-state reader "
        "poll={}s stale_reuse=false".format(poll_seconds)
    )
    return True


def install_oracle_depth_camera_support(
    agent_wrapper_module: Any,
    carla_module: Any,
    *,
    sensor_icons: Optional[dict[str, str]] = None,
    enabled: Optional[bool] = None,
    max_instances: int = 8,
    emit: Callable[[str], None] = lambda message: print(message, flush=True),
) -> bool:
    """Opt in to CARLA depth cameras for privileged oracle-U experiments.

    Bench2Drive's official SENSORS track rejects depth cameras and its wrapper
    initializes intrinsics/extrinsics only for RGB cameras. This narrow patch
    extends both pieces without weakening validation for any other sensor type.
    It is deliberately disabled by default because a run using it is an
    oracle-only, non-official evaluation and must not be reported as an
    eligible Bench2Drive sensor-track result.
    """

    if enabled is None:
        enabled = _enabled(os.environ.get("ORION_ALLOW_ORACLE_DEPTH_SENSOR"))
    if not enabled:
        return False
    max_instances = int(max_instances)
    if max_instances <= 0:
        raise ValueError("oracle depth camera limit must be positive")

    sensor_type = "sensor.camera.depth"
    wrapper_cls = agent_wrapper_module.AgentWrapper
    if sensor_icons is not None:
        sensor_icons[sensor_type] = "carla_camera"
    if getattr(wrapper_cls, "_orion_oracle_depth_installed", False):
        return True

    allowed = tuple(agent_wrapper_module.ALLOWED_SENSORS)
    if sensor_type not in allowed:
        agent_wrapper_module.ALLOWED_SENSORS = allowed + (sensor_type,)
    agent_wrapper_module.SENSORS_LIMITS[sensor_type] = max_instances
    agent_wrapper_module.QUALIFIER_SENSORS_LIMITS[sensor_type] = max_instances

    original_preprocess = wrapper_cls._preprocess_sensor_spec

    def preprocess_sensor_spec(self: Any, sensor_spec: dict[str, Any]):
        if sensor_spec.get("type") != sensor_type:
            return original_preprocess(self, sensor_spec)
        attributes = {
            "image_size_x": str(sensor_spec["width"]),
            "image_size_y": str(sensor_spec["height"]),
            "fov": str(sensor_spec["fov"]),
        }
        sensor_location = carla_module.Location(
            x=sensor_spec["x"], y=sensor_spec["y"], z=sensor_spec["z"]
        )
        sensor_rotation = carla_module.Rotation(
            pitch=sensor_spec["pitch"],
            roll=sensor_spec["roll"],
            yaw=sensor_spec["yaw"],
        )
        return (
            sensor_type,
            sensor_spec["id"],
            carla_module.Transform(sensor_location, sensor_rotation),
            attributes,
        )

    wrapper_cls._preprocess_sensor_spec = preprocess_sensor_spec
    wrapper_cls._orion_oracle_depth_installed = True
    wrapper_cls._orion_original_preprocess_sensor_spec = original_preprocess
    emit(
        "[OracleDepthHarness] enabled sensor.camera.depth max_instances={} "
        "official_sensor_track_eligible=false".format(max_instances)
    )
    return True
