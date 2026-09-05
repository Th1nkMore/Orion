#!/usr/bin/env python3
"""Bench2Drive closed-loop agent backed only by Qwen-Drive-1.0.

This agent intentionally does not import or instantiate Orion, MMCV, EVA, the
legacy language model, detection/map heads, BEV sensors, or UQ feature buffers.
Qwen runs in a persistent sidecar so its modern Python/PyTorch stack does not
conflict with the CARLA leaderboard environment.
"""

from __future__ import annotations

import atexit
import json
import importlib.util
import math
import os
import sys
from pathlib import Path

import carla
import numpy as np
from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

try:
    from pid_controller import PIDController
    from planner import RoutePlanner
except ModuleNotFoundError:
    from team_code.pid_controller import PIDController
    from team_code.planner import RoutePlanner

_BRIDGE_PATH = Path(__file__).resolve().parents[1] / "uq_estimator" / "qwen_drive_bridge.py"
_BRIDGE_MODULE_NAME = "_orion_qwen_drive_bridge"
_bridge_spec = importlib.util.spec_from_file_location(_BRIDGE_MODULE_NAME, _BRIDGE_PATH)
if _bridge_spec is None or _bridge_spec.loader is None:
    raise ImportError("cannot load Qwen-Drive bridge from %s" % _BRIDGE_PATH)
_bridge = importlib.util.module_from_spec(_bridge_spec)
sys.modules[_BRIDGE_MODULE_NAME] = _bridge
_bridge_spec.loader.exec_module(_bridge)

EgoHistoryBuffer = _bridge.EgoHistoryBuffer
ImageHistoryBuffer = _bridge.ImageHistoryBuffer
QWEN_VIEW_BY_SENSOR = _bridge.QWEN_VIEW_BY_SENSOR
QwenDriveClient = _bridge.QwenDriveClient
bench2drive_command_to_qwen = _bridge.bench2drive_command_to_qwen
load_bridge_config = _bridge.load_bridge_config
make_inference_request = _bridge.make_inference_request
qwen_trajectory_to_world = _bridge.qwen_trajectory_to_world
world_trajectory_to_pid = _bridge.world_trajectory_to_pid

_VISIBILITY_PATH = (
    Path(__file__).resolve().parents[1]
    / "uq_estimator"
    / "qwen_visibility_belief.py"
)
_visibility_spec = importlib.util.spec_from_file_location(
    "_orion_qwen_visibility_belief", _VISIBILITY_PATH
)
if _visibility_spec is None or _visibility_spec.loader is None:
    raise ImportError("cannot load Qwen visibility geometry from %s" % _VISIBILITY_PATH)
_visibility = importlib.util.module_from_spec(_visibility_spec)
sys.modules[_visibility_spec.name] = _visibility
_visibility_spec.loader.exec_module(_visibility)

belief_metadata = _visibility.belief_metadata
camera_from_carla_sensor = _visibility.camera_from_carla_sensor
compute_visibility_belief = _visibility.compute_visibility_belief
decode_carla_depth_bgra = _visibility.decode_carla_depth_bgra
make_colocated_depth_sensor_specs = _visibility.make_colocated_depth_sensor_specs
render_visibility_belief = _visibility.render_visibility_belief
visibility_grid_spec_from_mapping = _visibility.visibility_grid_spec_from_mapping

_SCHEDULE_PATH = (
    Path(__file__).resolve().parents[1] / "uq_estimator" / "corruption_schedule.py"
)
_schedule_spec = importlib.util.spec_from_file_location(
    "_orion_qwen_corruption_schedule", _SCHEDULE_PATH
)
if _schedule_spec is None or _schedule_spec.loader is None:
    raise ImportError("cannot load corruption schedule from %s" % _SCHEDULE_PATH)
_schedule = importlib.util.module_from_spec(_schedule_spec)
_schedule_spec.loader.exec_module(_schedule)
project_route_progress = _schedule.project_route_progress


_SHARED_CLIENT = None
_SHARED_CONFIG_PATH = None


def _close_shared_client():
    global _SHARED_CLIENT, _SHARED_CONFIG_PATH
    if _SHARED_CLIENT is not None:
        _SHARED_CLIENT.close()
    _SHARED_CLIENT = None
    _SHARED_CONFIG_PATH = None


atexit.register(_close_shared_client)


def get_entry_point():
    return "QwenDriveBench2DriveAgent"


def _config_path(path_to_conf_file):
    value = str(path_to_conf_file).split("+", 1)[0]
    if not value:
        raise ValueError("Qwen-Drive TEAM_CONFIG path is empty")
    return Path(value).resolve()


class QwenDriveBench2DriveAgent(autonomous_agent.AutonomousAgent):
    """Three-camera Qwen planner using the existing Orion PID control contract."""

    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.config_path = _config_path(path_to_conf_file)
        self.config = load_bridge_config(self.config_path)
        sampling = self.config["sampling"]
        images = self.config["images"]
        self.simulator_hz = int(sampling["simulator_hz"])
        self.inference_stride = int(sampling["inference_stride_steps"])
        self.ego_stride = self.simulator_hz // int(sampling["ego_history_hz"])
        self.ego_history = EgoHistoryBuffer(int(sampling["ego_history_points"]))
        self.image_history = ImageHistoryBuffer(
            tuple(QWEN_VIEW_BY_SENSOR),
            int(sampling["image_history_frames"]),
            images["history_size"],
            images["current_size"],
            int(images["jpeg_quality"]),
            images["transport_format"],
        )
        self.closedloop_corruption = os.environ.get(
            "ORION_CLOSEDLOOP_CORRUPTION", ""
        ).strip()
        if self.closedloop_corruption not in {"", "camera_dropout"}:
            raise ValueError(
                "Qwen screen supports only empty or camera_dropout corruption"
            )
        aliases = {
            "front": ("CAM_FRONT",),
            "front_group": tuple(QWEN_VIEW_BY_SENSOR),
            "all": tuple(QWEN_VIEW_BY_SENSOR),
        }
        view_spec = os.environ.get(
            "ORION_CLOSEDLOOP_CORRUPTION_VIEWS", "front"
        ).strip()
        self.corruption_sensors = aliases.get(
            view_spec,
            tuple(item.strip() for item in view_spec.split(",") if item.strip()),
        )
        unknown = set(self.corruption_sensors) - set(QWEN_VIEW_BY_SENSOR)
        if unknown or not self.corruption_sensors:
            raise ValueError("invalid Qwen corruption cameras: %s" % sorted(unknown))
        start_progress = os.environ.get(
            "ORION_CLOSEDLOOP_CORRUPTION_START_PROGRESS", ""
        ).strip()
        end_progress = os.environ.get(
            "ORION_CLOSEDLOOP_CORRUPTION_END_PROGRESS", ""
        ).strip()
        if bool(start_progress) != bool(end_progress):
            raise ValueError("route-progress corruption requires start and end")
        self.corruption_start_progress = (
            float(start_progress) if start_progress else None
        )
        self.corruption_end_progress = float(end_progress) if end_progress else None
        if self.corruption_start_progress is not None and not (
            0.0
            <= self.corruption_start_progress
            < self.corruption_end_progress
            <= 1.0
        ):
            raise ValueError("corruption progress window must lie inside [0,1]")
        self.corruption_start_seconds = float(
            os.environ.get("ORION_CLOSEDLOOP_CORRUPTION_START_SECONDS", "0")
        )
        end_seconds = os.environ.get(
            "ORION_CLOSEDLOOP_CORRUPTION_END_SECONDS", ""
        ).strip()
        self.corruption_end_seconds = (
            float(end_seconds) if end_seconds else float("inf")
        )
        if self.corruption_start_seconds < 0.0 or (
            self.corruption_end_seconds <= self.corruption_start_seconds
        ):
            raise ValueError("invalid corruption time window")
        pid_keys = (
            "turn_KP", "turn_KI", "turn_KD", "turn_n",
            "speed_KP", "speed_KI", "speed_KD", "speed_n",
            "max_throttle", "brake_speed", "brake_ratio", "clip_delta",
            "aim_dist", "angle_thresh", "dist_thresh",
        )
        self.pid = PIDController(
            **{key: self.config["control"][key] for key in pid_keys}
        )
        # PIDController has no notion of these two bridge-level safety limits.
        self.maximum_speed_mps = float(self.config["control"]["maximum_speed_mps"])
        self.maximum_plan_staleness = float(
            self.config["control"]["maximum_plan_staleness_seconds"]
        )
        self.step = -1
        self.initialized = False
        self.world_trajectory = None
        self.corruption_route_points = None
        self.last_plan_step = None
        self.last_inference_seconds = None
        self.last_oracle_visibility = None
        self.trace_stream = None
        save_path = os.environ.get("SAVE_PATH")
        if save_path:
            output_root = Path(save_path).resolve()
            output_root.mkdir(parents=True, exist_ok=True)
            self.trace_stream = (output_root / "qwen_drive_control_trace.jsonl").open(
                "a", encoding="utf-8"
            )
            service_log = output_root / "qwen_drive_sidecar.log"
        else:
            output_root = None
            service_log = Path("/tmp") / (
                "qwen_drive_sidecar_%d.log" % os.getpid()
            )
        oracle = self.config.get("oracle_visibility", {"enabled": False})
        self.oracle_visibility_enabled = bool(oracle.get("enabled", False))
        self.oracle_depth_sensor_specs = {}
        self.oracle_cameras = ()
        self.oracle_grid_spec = None
        self.oracle_far_plane_m = None
        self.oracle_frontier_threshold = None
        self.oracle_artifact_root = None
        if self.oracle_visibility_enabled:
            self.oracle_depth_sensor_specs = make_colocated_depth_sensor_specs(
                self.config["sensors"], oracle["depth_sensor_by_rgb"]
            )
            self.oracle_cameras = tuple(
                camera_from_carla_sensor(
                    depth_id,
                    self.config["sensors"][oracle["depth_sensor_by_rgb"][depth_id]],
                )
                for depth_id in self.oracle_depth_sensor_specs
            )
            self.oracle_grid_spec = visibility_grid_spec_from_mapping(oracle["grid"])
            self.oracle_far_plane_m = float(oracle["far_plane_m"])
            self.oracle_frontier_threshold = float(
                oracle["frontier_unknown_threshold"]
            )
            if bool(oracle["write_artifacts"]) and output_root is not None:
                self.oracle_artifact_root = output_root / "oracle_visibility"
                self.oracle_artifact_root.mkdir(parents=True, exist_ok=True)
        self.reuse_sidecar = bool(
            self.config["runtime"].get("reuse_sidecar_within_evaluator", True)
        )
        global _SHARED_CLIENT, _SHARED_CONFIG_PATH
        if self.reuse_sidecar and _SHARED_CLIENT is not None:
            if _SHARED_CONFIG_PATH != self.config_path or not _SHARED_CLIENT.ping():
                _close_shared_client()
        if self.reuse_sidecar and _SHARED_CLIENT is not None:
            self.client = _SHARED_CLIENT
            print("[QwenDrive] reusing loaded sidecar", flush=True)
        else:
            self.client = QwenDriveClient(self.config_path, self.config)
            try:
                self.client.start(service_log)
            except Exception:
                self.client.close()
                raise
            if self.reuse_sidecar:
                _SHARED_CLIENT = self.client
                _SHARED_CONFIG_PATH = self.config_path
        print(
            "[QwenDrive] sidecar ready; Orion and legacy feature branches are not loaded; "
            "mode=%s; log=%s" % (self.config["planning"]["mode"], service_log),
            flush=True,
        )

    def sensors(self):
        """Return Qwen RGB views and optional co-located oracle depth views."""

        result = []
        for sensor_id, configured in self.config["sensors"].items():
            sensor = dict(configured)
            sensor["id"] = sensor_id
            result.append(sensor)
        for sensor_id, configured in self.oracle_depth_sensor_specs.items():
            sensor = dict(configured)
            sensor["id"] = sensor_id
            result.append(sensor)
        return result

    def _capture_oracle_visibility(self, input_data):
        if not self.oracle_visibility_enabled:
            self.last_oracle_visibility = None
            return None
        depth_by_sensor = {
            sensor_id: decode_carla_depth_bgra(
                np.asarray(input_data[sensor_id][1]), self.oracle_far_plane_m
            )
            for sensor_id in self.oracle_depth_sensor_specs
        }
        belief = compute_visibility_belief(
            depth_by_sensor,
            self.oracle_cameras,
            self.oracle_grid_spec,
            frontier_unknown_threshold=self.oracle_frontier_threshold,
        )
        metadata = belief_metadata(belief)
        metadata.update(
            {
                "step": int(self.step),
                "oracle_depth": True,
                "used_by_qwen": False,
            }
        )
        artifact = None
        if self.oracle_artifact_root is not None:
            stem = "step_%06d" % self.step
            array_path = self.oracle_artifact_root / (stem + ".npz")
            image_path = self.oracle_artifact_root / (stem + ".png")
            np.savez_compressed(
                array_path,
                channels=belief.as_channels(),
                channel_names=np.asarray(belief.channel_names),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            import cv2

            rendered_rgb = render_visibility_belief(belief)
            if not cv2.imwrite(
                str(image_path), cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
            ):
                raise RuntimeError("failed to write oracle visibility PNG")
            artifact = {
                "npz": str(array_path),
                "png": str(image_path),
            }
        self.last_oracle_visibility = {
            "metadata": metadata,
            "artifact": artifact,
        }
        return belief

    def _initialize(self):
        hero = None
        get_hero = getattr(CarlaDataProvider, "get_hero_actor", None)
        if get_hero is not None:
            hero = get_hero()
        if hero is None:
            for actor in CarlaDataProvider.get_world().get_actors():
                if actor.attributes.get("role_name") == "hero":
                    hero = actor
                    break
        if hero is None:
            raise RuntimeError("Bench2Drive hero actor is unavailable")
        if not getattr(self, "_global_plan_world_coord", None):
            raise RuntimeError("Bench2Drive world route plan is unavailable")
        self.hero_actor = hero
        self.route_planner = RoutePlanner(4.0, 50.0)
        self.route_planner.set_route(self._global_plan_world_coord, gps=False)
        self.corruption_route_points = np.asarray(
            [
                [transform.location.x, transform.location.y]
                for transform, _ in self._global_plan_world_coord
            ],
            dtype=np.float64,
        )
        self.initialized = True

    def _ego_state(self):
        transform = self.hero_actor.get_transform()
        velocity = self.hero_actor.get_velocity()
        acceleration = self.hero_actor.get_acceleration()
        pose = np.asarray(
            [transform.location.x, transform.location.y, transform.rotation.yaw],
            dtype=np.float64,
        )
        world_velocity = np.asarray([velocity.x, velocity.y], dtype=np.float64)
        world_acceleration = np.asarray(
            [acceleration.x, acceleration.y], dtype=np.float64
        )
        speed = np.float32(math.hypot(velocity.x, velocity.y))
        return pose, world_velocity, world_acceleration, speed

    def _route_command(self, pose):
        (_, current_command), (near_xy, _) = self.route_planner.run_step(pose[:2])
        value = getattr(current_command, "value", current_command)
        driving_command, nav_command = bench2drive_command_to_qwen(int(value))
        return driving_command, nav_command, np.asarray(near_xy, dtype=np.float64)

    def _infer(
        self,
        input_data,
        pose,
        driving_command,
        nav_command,
        corruption_active,
    ):
        self._capture_oracle_visibility(input_data)
        bgr_images = {
            sensor_id: np.asarray(input_data[sensor_id][1])[:, :, :3]
            for sensor_id in QWEN_VIEW_BY_SENSOR
        }
        if corruption_active:
            for sensor_id in self.corruption_sensors:
                bgr_images[sensor_id] = np.zeros_like(bgr_images[sensor_id])
        views = self.image_history.capture(bgr_images)
        request = make_inference_request(
            views,
            self.ego_history.build(),
            driving_command,
            nav_command,
            token="bench2drive-step-%06d" % self.step,
        )
        trajectory, inference_seconds = self.client.infer(request)
        maximum = float(self.config["planning"]["max_abs_position_m"])
        if np.abs(trajectory[:, :2]).max() > maximum:
            raise ValueError("Qwen trajectory exceeds configured position bound")
        self.world_trajectory = qwen_trajectory_to_world(trajectory, pose)
        self.last_plan_step = self.step
        self.last_inference_seconds = inference_seconds
        return trajectory

    @staticmethod
    def _emergency_stop():
        return carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)

    def _write_trace(self, record):
        if self.trace_stream is None:
            return
        self.trace_stream.write(json.dumps(record, sort_keys=True) + "\n")
        self.trace_stream.flush()

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._initialize()
        self.step += 1
        pose, velocity, acceleration, speed = self._ego_state()
        if self.step % self.ego_stride == 0 or len(self.ego_history) == 0:
            self.ego_history.append(pose, velocity, acceleration)
        driving_command, nav_command, near_xy = self._route_command(pose)
        route_progress = project_route_progress(
            pose[:2], self.corruption_route_points
        )
        if self.corruption_start_progress is not None:
            within_corruption_window = bool(
                self.corruption_start_progress
                <= route_progress
                < self.corruption_end_progress
            )
            corruption_schedule = "route_progress"
        else:
            within_corruption_window = bool(
                self.corruption_start_seconds
                <= float(timestamp)
                < self.corruption_end_seconds
            )
            corruption_schedule = "simulation_time"
        corruption_active = bool(
            self.closedloop_corruption and within_corruption_window
        )
        inference_error = None
        raw_trajectory = None
        if self.step % self.inference_stride == 0:
            try:
                raw_trajectory = self._infer(
                    input_data,
                    pose,
                    driving_command,
                    nav_command,
                    corruption_active,
                )
            except Exception as error:
                inference_error = "%s: %s" % (type(error).__name__, error)
                print("[QwenDrive] inference failed: %s" % inference_error, flush=True)

        plan_age = (
            float("inf")
            if self.last_plan_step is None
            else (self.step - self.last_plan_step) / float(self.simulator_hz)
        )
        if self.world_trajectory is None or plan_age > self.maximum_plan_staleness:
            self._write_trace(
                {
                    "step": self.step,
                    "timestamp": float(timestamp),
                    "status": "emergency_stop_no_fresh_plan",
                    "ego_world_pose": pose.tolist(),
                    "speed_mps": float(speed),
                    "plan_age_seconds": plan_age,
                    "inference_error": inference_error,
                    "route_progress": route_progress,
                    "corruption": {
                        "family": self.closedloop_corruption or "none",
                        "active": corruption_active,
                        "sensors": list(self.corruption_sensors),
                        "schedule": corruption_schedule,
                    },
                }
            )
            return self._emergency_stop()

        planning = self.config["planning"]
        waypoints = world_trajectory_to_pid(
            self.world_trajectory,
            pose,
            elapsed_seconds=plan_age,
            source_hz=int(planning["source_hz"]),
            target_hz=int(planning["controller_hz"]),
            horizon_points=int(planning["controller_horizon_points"]),
        )
        # The Orion PID target branch is disabled, but pass a geometrically valid
        # route target to preserve the controller's diagnostic contract.
        route_delta = near_xy - pose[:2]
        yaw = math.radians(float(pose[2]))
        route_forward = math.cos(yaw) * route_delta[0] + math.sin(yaw) * route_delta[1]
        route_left = math.sin(yaw) * route_delta[0] - math.cos(yaw) * route_delta[1]
        local_route_target = np.asarray([-route_left, route_forward], dtype=np.float32)
        steer, throttle, brake, metadata = self.pid.control_pid(
            waypoints, speed, local_route_target
        )
        if float(brake) < 0.05:
            brake = 0.0
        if float(throttle) > float(brake):
            brake = 0.0
        if float(speed) > self.maximum_speed_mps:
            throttle = 0.0
        control = carla.VehicleControl(
            steer=float(steer), throttle=float(throttle), brake=float(brake)
        )
        self._write_trace(
            {
                "step": self.step,
                "timestamp": float(timestamp),
                "status": "control",
                "ego_world_pose": pose.tolist(),
                "nav_command": int(nav_command),
                "route_progress": route_progress,
                "corruption": {
                    "family": self.closedloop_corruption or "none",
                    "active": corruption_active,
                    "sensors": list(self.corruption_sensors),
                    "schedule": corruption_schedule,
                },
                "speed_mps": float(speed),
                "plan_age_seconds": plan_age,
                "inference_seconds": self.last_inference_seconds,
                "qwen_reasoning": (
                    self.client.last_reasoning if raw_trajectory is not None else None
                ),
                "qwen_runtime_metrics": self.client.last_metrics,
                "oracle_visibility": (
                    self.last_oracle_visibility
                    if raw_trajectory is not None
                    else None
                ),
                "inference_error": inference_error,
                "compressed_history_bytes": self.image_history.retained_bytes,
                "new_qwen_trajectory": (
                    None if raw_trajectory is None else raw_trajectory.tolist()
                ),
                "pid_waypoints": waypoints.tolist(),
                "control": {
                    "steer": float(control.steer),
                    "throttle": float(control.throttle),
                    "brake": float(control.brake),
                },
                "pid": metadata,
            }
        )
        return control

    def destroy(self):
        if getattr(self, "client", None) is not None:
            if not getattr(self, "reuse_sidecar", False):
                self.client.close()
            self.client = None
        if getattr(self, "trace_stream", None) is not None:
            self.trace_stream.close()
            self.trace_stream = None
        self.world_trajectory = None
        self.image_history = None
        self.ego_history = None
