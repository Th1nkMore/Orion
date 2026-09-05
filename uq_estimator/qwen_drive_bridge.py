"""Lightweight Bench2Drive-to-Qwen-Drive bridge.

The CARLA agent imports this module without importing PyTorch, Transformers, or
Qwen-Drive.  The heavyweight model is loaded only by the ``serve`` subcommand
in a separate Python environment.  A local Unix socket carries encoded camera
histories and NumPy ego-state arrays between the two processes.  The active
configuration transports sensor-resolution frames losslessly and leaves the
history/current pixel-budget resize to the official Qwen processor.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import tempfile
import time
import uuid
from collections import deque
from dataclasses import dataclass
from multiprocessing.connection import Client, Connection, Listener
from pathlib import Path
from typing import Deque, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


CONFIG_SCHEMA = "orion.qwen-drive-bench2drive-bridge/v1"
RPC_SCHEMA = "orion.qwen-drive-local-rpc/v1"
PLANNING_MODES = {"direct_planning", "reasoning_planning"}
QWEN_VIEW_BY_SENSOR = {
    "CAM_FRONT": "<FRONT VIEW>",
    "CAM_FRONT_LEFT": "<FRONT LEFT VIEW>",
    "CAM_FRONT_RIGHT": "<FRONT RIGHT VIEW>",
}
_AUTHKEY = b"orion-qwen-drive-local-v1"


def load_bridge_config(path: Union[str, Path]) -> dict:
    """Load and validate the bridge JSON configuration."""

    config_path = Path(str(path).split("+", 1)[0]).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected Qwen-Drive bridge config schema")
    for section in ("sensors", "sampling", "images", "planning", "runtime", "control"):
        if not isinstance(payload.get(section), dict):
            raise ValueError("bridge config is missing section %s" % section)
    sensors = payload["sensors"]
    if tuple(sensors) != tuple(QWEN_VIEW_BY_SENSOR):
        raise ValueError(
            "sensors must be ordered CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT"
        )
    sampling = payload["sampling"]
    if int(sampling["image_history_frames"]) != 4:
        raise ValueError("Qwen-Drive requires exactly four frames per camera")
    if int(sampling["ego_history_points"]) != 16:
        raise ValueError("Qwen-Drive requires exactly sixteen ego history points")
    if int(sampling["simulator_hz"]) % int(sampling["ego_history_hz"]):
        raise ValueError("simulator_hz must be divisible by ego_history_hz")
    if int(sampling["inference_stride_steps"]) <= 0:
        raise ValueError("inference_stride_steps must be positive")
    images = payload["images"]
    if images.get("transport_format") not in {"jpeg", "png"}:
        raise ValueError("images.transport_format must be jpeg or png")
    for key in ("history_size", "current_size"):
        size = images.get(key)
        if size is not None and (
            not isinstance(size, list)
            or len(size) != 2
            or any(int(value) <= 0 for value in size)
        ):
            raise ValueError("images.%s must be null or [width,height]" % key)
    if images["transport_format"] == "jpeg":
        quality = int(images.get("jpeg_quality", 0))
        if not 1 <= quality <= 100:
            raise ValueError("images.jpeg_quality must be in [1,100]")
    if payload["planning"].get("mode") not in PLANNING_MODES:
        raise ValueError(
            "closed-loop bridge planning.mode must be one of %s"
            % sorted(PLANNING_MODES)
        )
    if int(payload["planning"].get("num_samples", 0)) != 1:
        raise ValueError("closed-loop bridge requires num_samples=1")
    return payload


def wrap_angle_radians(angle: float) -> float:
    """Wrap one angle into ``[-pi, pi)``."""

    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def world_vector_to_qwen(vector_xy: Sequence[float], yaw_degrees: float) -> np.ndarray:
    """Rotate a CARLA world vector into Qwen's ``(forward, left)`` frame."""

    # [2] -> [2]
    vector = np.asarray(vector_xy, dtype=np.float64)
    if vector.shape != (2,):
        raise ValueError("world vector must have shape [2]")
    yaw = math.radians(float(yaw_degrees))
    forward = math.cos(yaw) * vector[0] + math.sin(yaw) * vector[1]
    left = math.sin(yaw) * vector[0] - math.cos(yaw) * vector[1]
    return np.asarray([forward, left], dtype=np.float32)


def world_pose_to_qwen(
    pose_xy_yaw: Sequence[float], current_xy_yaw: Sequence[float]
) -> np.ndarray:
    """Express a CARLA pose in the current Qwen ego frame."""

    # [3], [3] -> [3]
    pose = np.asarray(pose_xy_yaw, dtype=np.float64)
    current = np.asarray(current_xy_yaw, dtype=np.float64)
    if pose.shape != (3,) or current.shape != (3,):
        raise ValueError("poses must have shape [3]")
    position = world_vector_to_qwen(pose[:2] - current[:2], current[2])
    heading = -wrap_angle_radians(math.radians(pose[2] - current[2]))
    return np.asarray([position[0], position[1], heading], dtype=np.float32)


@dataclass(frozen=True)
class EgoSample:
    """One 10 Hz ego sample kept by the lightweight CARLA process."""

    world_pose: np.ndarray
    body_velocity: np.ndarray
    body_acceleration: np.ndarray


class EgoHistoryBuffer:
    """Bounded 10 Hz state history with Qwen-compatible coordinate conversion."""

    def __init__(self, num_points: int):
        if int(num_points) < 2:
            raise ValueError("ego history needs at least two points")
        self.num_points = int(num_points)
        self._samples: Deque[EgoSample] = deque(maxlen=self.num_points)

    def append(
        self,
        world_pose: Sequence[float],
        world_velocity: Sequence[float],
        world_acceleration: Sequence[float],
    ) -> None:
        """Append world pose/kinematics, rotating vectors into that pose's body frame."""

        # [3], [2], [2] -> one bounded sample
        pose = np.asarray(world_pose, dtype=np.float64)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("world_pose must be finite with shape [3]")
        velocity = world_vector_to_qwen(world_velocity, pose[2])
        acceleration = world_vector_to_qwen(world_acceleration, pose[2])
        self._samples.append(EgoSample(pose.copy(), velocity, acceleration))

    def __len__(self) -> int:
        return len(self._samples)

    def build(self) -> Dict[str, np.ndarray]:
        """Return padded oldest-to-current arrays for a Qwen planning scene."""

        if not self._samples:
            raise RuntimeError("ego history is empty")
        samples = list(self._samples)
        if len(samples) < self.num_points:
            samples = [samples[0]] * (self.num_points - len(samples)) + samples
        current = samples[-1].world_pose
        # [H, 3], [H, 2], [H, 2]
        history = np.stack(
            [world_pose_to_qwen(sample.world_pose, current) for sample in samples]
        )
        velocity = np.stack([sample.body_velocity for sample in samples]).astype(np.float32)
        acceleration = np.stack(
            [sample.body_acceleration for sample in samples]
        ).astype(np.float32)
        history[-1] = 0.0
        return {
            "history": history,
            "history_velocity": velocity,
            "history_acceleration": acceleration,
            "ego_velocity": velocity[-1].copy(),
            "ego_acceleration": acceleration[-1].copy(),
        }


def bench2drive_command_to_qwen(command_value: int) -> Tuple[np.ndarray, int]:
    """Map Bench2Drive RoadOption values to Qwen command conditioning."""

    value = int(command_value)
    if value in (1, 5):
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 1
    if value in (2, 6):
        return np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32), 2
    if value in (3, 4):
        return np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32), 0
    return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), 0


def _encode_bgr_for_transport(
    image: np.ndarray,
    target_size: Optional[Sequence[int]],
    transport_format: str,
    jpeg_quality: int,
) -> bytes:
    """Optionally resize a CARLA BGR frame and encode it for local transport."""

    import cv2

    # [H, W, C] -> encoded bytes
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("camera image must have shape [H,W,C>=3]")
    encoded_image = array[:, :, :3]
    if target_size is not None:
        width, height = (int(target_size[0]), int(target_size[1]))
        if width <= 0 or height <= 0:
            raise ValueError("target image dimensions must be positive")
        encoded_image = cv2.resize(
            encoded_image, (width, height), interpolation=cv2.INTER_AREA
        )
    normalized_format = str(transport_format).lower()
    if normalized_format == "png":
        extension = ".png"
        parameters = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    elif normalized_format == "jpeg":
        extension = ".jpg"
        parameters = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    else:
        raise ValueError("transport_format must be jpeg or png")
    ok, encoded = cv2.imencode(extension, encoded_image, parameters)
    if not ok:
        raise RuntimeError("OpenCV failed to encode a camera frame")
    return encoded.tobytes()


class ImageHistoryBuffer:
    """Keep three encoded history frames per camera plus the current frame."""

    def __init__(
        self,
        sensor_ids: Sequence[str],
        num_frames: int,
        history_size: Optional[Sequence[int]],
        current_size: Optional[Sequence[int]],
        jpeg_quality: int,
        transport_format: str = "jpeg",
    ):
        if int(num_frames) != 4:
            raise ValueError("Qwen-Drive image history must contain four frames")
        if not 1 <= int(jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be in [1,100]")
        if str(transport_format).lower() not in {"jpeg", "png"}:
            raise ValueError("transport_format must be jpeg or png")
        self.sensor_ids = tuple(sensor_ids)
        if self.sensor_ids != tuple(QWEN_VIEW_BY_SENSOR):
            raise ValueError("unexpected Qwen camera order")
        self.num_frames = int(num_frames)
        self.history_size = (
            None if history_size is None else tuple(int(value) for value in history_size)
        )
        self.current_size = (
            None if current_size is None else tuple(int(value) for value in current_size)
        )
        self.jpeg_quality = int(jpeg_quality)
        self.transport_format = str(transport_format).lower()
        self._history = {
            sensor_id: deque(maxlen=self.num_frames - 1)
            for sensor_id in self.sensor_ids
        }

    def capture(self, bgr_images: Mapping[str, np.ndarray]) -> Dict[str, List[bytes]]:
        """Build one 12-frame request and retain only compressed future history."""

        missing = set(self.sensor_ids) - set(bgr_images)
        if missing:
            raise ValueError("camera input is missing %s" % sorted(missing))
        views: Dict[str, List[bytes]] = {}
        for sensor_id in self.sensor_ids:
            image = bgr_images[sensor_id]
            history_frame = _encode_bgr_for_transport(
                image,
                self.history_size,
                self.transport_format,
                self.jpeg_quality,
            )
            current = (
                history_frame
                if self.current_size == self.history_size
                else _encode_bgr_for_transport(
                    image,
                    self.current_size,
                    self.transport_format,
                    self.jpeg_quality,
                )
            )
            prior = list(self._history[sensor_id])
            if len(prior) < self.num_frames - 1:
                prior = [history_frame] * (self.num_frames - 1 - len(prior)) + prior
            views[QWEN_VIEW_BY_SENSOR[sensor_id]] = prior + [current]
            self._history[sensor_id].append(history_frame)
        return views

    @property
    def retained_bytes(self) -> int:
        """Number of compressed image bytes retained by the CARLA process."""

        return int(sum(len(frame) for frames in self._history.values() for frame in frames))


def validate_qwen_trajectory(trajectory: np.ndarray, num_future_points: int) -> np.ndarray:
    """Validate one Qwen trajectory and return a private float32 copy."""

    # [T, 3] -> [T, 3]
    array = np.asarray(trajectory, dtype=np.float32)
    expected = (int(num_future_points), 3)
    if array.shape != expected:
        raise ValueError("Qwen trajectory must have shape %r, got %r" % (expected, array.shape))
    if not np.isfinite(array).all():
        raise ValueError("Qwen trajectory contains non-finite values")
    return array.copy()


def qwen_trajectory_to_world(
    trajectory: np.ndarray, origin_xy_yaw: Sequence[float]
) -> np.ndarray:
    """Project Qwen ``(forward,left)`` trajectory positions into CARLA world XY."""

    # [T, 3], [3] -> [T, 2]
    array = np.asarray(trajectory, dtype=np.float64)
    origin = np.asarray(origin_xy_yaw, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2 or origin.shape != (3,):
        raise ValueError("invalid trajectory or origin shape")
    yaw = math.radians(float(origin[2]))
    forward_basis = np.asarray([math.cos(yaw), math.sin(yaw)])
    left_basis = np.asarray([math.sin(yaw), -math.cos(yaw)])
    return (
        origin[None, :2]
        + array[:, 0:1] * forward_basis[None, :]
        + array[:, 1:2] * left_basis[None, :]
    ).astype(np.float32)


def world_trajectory_to_pid(
    world_xy: np.ndarray,
    current_xy_yaw: Sequence[float],
    elapsed_seconds: float,
    source_hz: int,
    target_hz: int,
    horizon_points: int,
) -> np.ndarray:
    """Resample world points into Orion PID's ``(right,forward)`` convention."""

    # [T, 2], [3] -> [K, 2]
    points = np.asarray(world_xy, dtype=np.float64)
    current = np.asarray(current_xy_yaw, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or current.shape != (3,):
        raise ValueError("invalid world trajectory or current pose shape")
    if source_hz <= 0 or target_hz <= 0 or source_hz % target_hz:
        raise ValueError("source_hz must be a positive multiple of target_hz")
    if horizon_points < 4:
        raise ValueError("PID trajectory needs at least four waypoints")
    step = source_hz // target_hz
    offset = max(0, int(round(float(elapsed_seconds) * source_hz)))
    indices = [min(len(points) - 1, offset + (index + 1) * step - 1) for index in range(horizon_points)]
    selected = points[indices] - current[None, :2]
    yaw = math.radians(float(current[2]))
    forward = math.cos(yaw) * selected[:, 0] + math.sin(yaw) * selected[:, 1]
    left = math.sin(yaw) * selected[:, 0] - math.cos(yaw) * selected[:, 1]
    return np.stack([-left, forward], axis=-1).astype(np.float32)


def make_inference_request(
    views: Mapping[str, Sequence[bytes]],
    ego: Mapping[str, np.ndarray],
    driving_command: Sequence[float],
    nav_command: int,
    token: str,
) -> dict:
    """Build and validate one local-RPC inference payload."""

    if tuple(views) != tuple(QWEN_VIEW_BY_SENSOR.values()):
        raise ValueError("Qwen views are missing or out of order")
    if any(len(frames) != 4 for frames in views.values()):
        raise ValueError("each Qwen view must contain four frames")
    arrays = {
        "history": np.asarray(ego["history"], dtype=np.float32),
        "history_velocity": np.asarray(ego["history_velocity"], dtype=np.float32),
        "history_acceleration": np.asarray(
            ego["history_acceleration"], dtype=np.float32
        ),
        "ego_velocity": np.asarray(ego["ego_velocity"], dtype=np.float32),
        "ego_acceleration": np.asarray(ego["ego_acceleration"], dtype=np.float32),
        "driving_command": np.asarray(driving_command, dtype=np.float32),
    }
    request = {
        "schema": RPC_SCHEMA,
        "op": "infer",
        "views": {key: list(value) for key, value in views.items()},
        "nav_command": int(nav_command),
        "token": str(token),
    }
    expected_shapes = {
        "history": (16, 3),
        "history_velocity": (16, 2),
        "history_acceleration": (16, 2),
        "ego_velocity": (2,),
        "ego_acceleration": (2,),
        "driving_command": (4,),
    }
    for key, shape in expected_shapes.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError("%s must be finite with shape %r" % (key, shape))
        # NumPy 2 pickles arrays through numpy._core, which NumPy 1 cannot
        # import.  The CARLA and Qwen environments intentionally use different
        # NumPy generations, so keep the socket payload version-neutral.
        request[key] = arrays[key].tolist()
    if request["nav_command"] not in (0, 1, 2):
        raise ValueError("nav_command must be 0, 1 or 2")
    return request


class QwenDriveClient:
    """Own a persistent Qwen sidecar and issue blocking planning requests."""

    def __init__(self, config_path: Union[str, Path], config: Mapping[str, object]):
        self.config_path = Path(config_path).resolve()
        self.config = config
        self.connection: Optional[Connection] = None
        self.process: Optional[subprocess.Popen] = None
        self.socket_path: Optional[Path] = None
        self.log_stream = None
        self.last_metrics = None
        self.last_reasoning = None

    def start(self, log_path: Optional[Union[str, Path]] = None) -> None:
        """Launch the configured Qwen Python runtime and wait for model readiness."""

        if self.process is not None:
            raise RuntimeError("Qwen sidecar was already started")
        runtime = self.config["runtime"]
        token = uuid.uuid4().hex[:12]
        socket_root = Path(runtime.get("socket_directory", tempfile.gettempdir()))
        socket_root.mkdir(parents=True, exist_ok=True)
        self.socket_path = socket_root / ("orion-qwen-drive-%s-%s.sock" % (os.getpid(), token))
        command = []
        loader = str(runtime.get("glibc_loader", "")).strip()
        if loader:
            command.extend(
                [loader, "--library-path", str(runtime["runtime_library_path"])]
            )
        command.extend(
            [
                str(runtime["python"]),
                str(Path(__file__).resolve()),
                "serve",
                "--config",
                str(self.config_path),
                "--socket",
                str(self.socket_path),
            ]
        )
        environment = os.environ.copy()
        python_paths = [str(Path(runtime["qwen_source"]) / "src"), str(Path(__file__).resolve().parents[1])]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        compiler = str(runtime.get("triton_c_compiler", "")).strip()
        if compiler:
            environment["CC"] = compiler
        if log_path is None:
            log_path = self.socket_path.with_suffix(".log")
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_stream = log_file.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=self.log_stream,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        deadline = time.monotonic() + float(runtime["startup_timeout_seconds"])
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "Qwen sidecar exited during startup with code %s; see %s"
                    % (self.process.returncode, log_file)
                )
            if self.socket_path.exists():
                try:
                    self.connection = Client(
                        str(self.socket_path), family="AF_UNIX", authkey=_AUTHKEY
                    )
                    break
                except (ConnectionError, OSError) as error:
                    last_error = error
            time.sleep(0.2)
        if self.connection is None:
            raise TimeoutError(
                "Qwen sidecar did not open its socket before timeout: %r" % last_error
            )
        self.connection.send({"schema": RPC_SCHEMA, "op": "ping"})
        if not self.connection.poll(max(0.0, deadline - time.monotonic())):
            raise TimeoutError("Qwen sidecar did not finish loading before timeout")
        response = self.connection.recv()
        if response.get("status") != "ready":
            raise RuntimeError("unexpected Qwen startup response: %r" % response)

    def infer(self, request: Mapping[str, object]) -> Tuple[np.ndarray, float]:
        """Return one validated trajectory and measured sidecar inference latency."""

        if self.connection is None:
            raise RuntimeError("Qwen sidecar is not connected")
        self.connection.send(dict(request))
        timeout = float(self.config["runtime"]["inference_timeout_seconds"])
        if not self.connection.poll(timeout):
            raise TimeoutError("Qwen planning inference exceeded %.1f seconds" % timeout)
        response = self.connection.recv()
        if response.get("status") != "ok":
            raise RuntimeError("Qwen planning failed: %s" % response.get("error", response))
        self.last_metrics = dict(response.get("metrics", {}))
        self.last_reasoning = response.get("reasoning")
        trajectory = validate_qwen_trajectory(
            response["trajectory"], self.config["planning"]["num_future_points"]
        )
        return trajectory, float(response["inference_seconds"])

    def ping(self, timeout_seconds: float = 5.0) -> bool:
        """Return whether the already-loaded sidecar is responsive."""

        if self.connection is None or self.process is None or self.process.poll() is not None:
            return False
        try:
            self.connection.send({"schema": RPC_SCHEMA, "op": "ping"})
            if not self.connection.poll(float(timeout_seconds)):
                return False
            return self.connection.recv().get("status") == "ready"
        except (BrokenPipeError, EOFError, OSError):
            return False

    def close(self) -> None:
        """Ask the sidecar to exit, then release its process and socket resources."""

        connection, self.connection = self.connection, None
        if connection is not None:
            try:
                connection.send({"schema": RPC_SCHEMA, "op": "shutdown"})
                connection.poll(2.0)
            except (BrokenPipeError, EOFError, OSError):
                pass
            finally:
                connection.close()
        if self.process is not None:
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5.0)
            self.process = None
        if self.log_stream is not None:
            self.log_stream.close()
            self.log_stream = None
        if self.socket_path is not None:
            self.socket_path.unlink(missing_ok=True)
            self.socket_path = None


def _decode_scene(request: Mapping[str, object]):
    """Convert an RPC request into Qwen-Drive scene objects inside the sidecar."""

    from PIL import Image
    from qwen_drive import CameraFrame, DrivingScene

    views = {}
    image_owners = []
    for view_name in QWEN_VIEW_BY_SENSOR.values():
        frames = []
        for encoded in request["views"][view_name]:
            image = Image.open(io.BytesIO(encoded)).convert("RGB")
            image.load()
            image_owners.append(image)
            frames.append(CameraFrame(image=image))
        views[view_name] = frames
    scene = DrivingScene(
        views=views,
        history=np.asarray(request["history"], dtype=np.float32),
        history_velocity=np.asarray(request["history_velocity"], dtype=np.float32),
        history_acceleration=np.asarray(
            request["history_acceleration"], dtype=np.float32
        ),
        ego_velocity=np.asarray(request["ego_velocity"], dtype=np.float32),
        ego_acceleration=np.asarray(request["ego_acceleration"], dtype=np.float32),
        driving_command=np.asarray(request["driving_command"], dtype=np.float32),
        nav_command=request["nav_command"],
        token=request["token"],
    )
    return scene, image_owners


def serve(config_path: Union[str, Path], socket_path: Union[str, Path]) -> int:
    """Load Qwen-Drive once and serve planning requests on one local Unix socket."""

    import torch
    from qwen_drive import InferenceMode, QwenDriveForPlanning

    config = load_bridge_config(config_path)
    runtime = config["runtime"]
    planning = config["planning"]
    inference_mode = InferenceMode(planning["mode"])
    dtype = getattr(torch, str(runtime["dtype"]))
    model = QwenDriveForPlanning.from_pretrained(
        runtime["model"],
        planner=runtime["planner"],
        dtype=dtype,
        attn_implementation=runtime["attention_implementation"],
    )
    model = model.to(runtime["device"]).eval()
    socket = Path(socket_path)
    socket.unlink(missing_ok=True)
    listener = Listener(str(socket), family="AF_UNIX", authkey=_AUTHKEY)
    print("Qwen-Drive sidecar ready on %s" % socket, flush=True)
    connection = listener.accept()
    try:
        while True:
            request = connection.recv()
            if request.get("schema") != RPC_SCHEMA:
                connection.send({"status": "error", "error": "unexpected RPC schema"})
                continue
            operation = request.get("op")
            if operation == "ping":
                connection.send({"status": "ready"})
                continue
            if operation == "shutdown":
                connection.send({"status": "bye"})
                break
            if operation != "infer":
                connection.send({"status": "error", "error": "unknown operation"})
                continue
            started = time.monotonic()
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                scene, image_owners = _decode_scene(request)
                output = model.run(
                    inference_mode,
                    scene=scene,
                    num_samples=int(planning["num_samples"]),
                    seed=int(planning["seed"]),
                )
                trajectory = validate_qwen_trajectory(
                    output.trajectory, planning["num_future_points"]
                )
                metrics = {}
                if torch.cuda.is_available():
                    megabytes = 1024.0 * 1024.0
                    metrics = {
                        "device_name": torch.cuda.get_device_name(model.device),
                        "cuda_allocated_mb": torch.cuda.memory_allocated() / megabytes,
                        "cuda_reserved_mb": torch.cuda.memory_reserved() / megabytes,
                        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / megabytes,
                        "cuda_peak_reserved_mb": torch.cuda.max_memory_reserved() / megabytes,
                    }
                connection.send(
                    {
                        "status": "ok",
                        # The CARLA process uses NumPy 1 while this sidecar uses
                        # NumPy 2.  Lists avoid a numpy._core pickle dependency.
                        "trajectory": trajectory.tolist(),
                        "inference_seconds": time.monotonic() - started,
                        "reasoning": output.reasoning,
                        "metrics": metrics,
                    }
                )
                del output, trajectory, scene, image_owners
            except Exception as error:  # keep the route process alive for a safe stop
                connection.send(
                    {"status": "error", "error": "%s: %s" % (type(error).__name__, error)}
                )
    finally:
        connection.close()
        listener.close()
        socket.unlink(missing_ok=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    server = subparsers.add_parser("serve", help="run the Qwen planning sidecar")
    server.add_argument("--config", type=Path, required=True)
    server.add_argument("--socket", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "serve":
        return serve(args.config, args.socket)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
