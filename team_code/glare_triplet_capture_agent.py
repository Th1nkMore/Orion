#!/usr/bin/env python3
"""Same-tick CARLA-native glare capture without loading ORION.

Three co-located RGB cameras record clean, frozen-medium and frozen-heavy
post-processing on the same simulator tick.  A fourth camera supplies BEV
context with post-processing disabled.  The trace records sensor/weather
readback plus dynamic-actor 3D boxes for outcome-blind visual analysis.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import carla
import cv2

from agents.navigation.basic_agent import BasicAgent
from leaderboard.autoagents.agent_wrapper import AgentWrapper
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


SCHEMA = "orion.native_glare_same_tick_capture.v1"
RGB_GEOMETRY = {
    "x": 0.80,
    "y": 0.0,
    "z": 1.60,
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "width": 1600,
    "height": 900,
    "fov": 70,
}
READBACK_ATTRIBUTES = (
    "enable_postprocess_effects",
    "exposure_mode",
    "lens_flare_intensity",
    "bloom_intensity",
    "image_size_x",
    "image_size_y",
    "fov",
    "role_name",
)


def get_entry_point():
    return "NativeGlareTripletCaptureAgent"


def _finite_float(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % name)
    return value


def load_protocol(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "orion.native_glare_independent_confirmation.v1":
        raise ValueError("unexpected native-glare confirmation schema")
    profiles = payload.get("camera_profiles")
    if set(profiles or {}) != {"clean", "medium", "heavy"}:
        raise ValueError("camera profiles must be clean, medium and heavy")
    sensor_ids = [profiles[name].get("sensor_id") for name in ("clean", "medium", "heavy")]
    if len(set(sensor_ids)) != 3 or any(not value for value in sensor_ids):
        raise ValueError("camera profile sensor ids must be unique")
    for field in ("lens_flare_intensity", "bloom_intensity"):
        values = [_finite_float(profiles[name][field], field) for name in ("clean", "medium", "heavy")]
        if not values[0] == 0.0 or not values[0] < values[1] < values[2]:
            raise ValueError("%s profiles are not strictly ordered" % field)
    return payload


def protocol_path_from_agent_config(path_to_conf_file) -> Path:
    """Extract the first component from Bench2Drive's composite config string."""

    protocol = str(path_to_conf_file).split("+", 1)[0]
    if not protocol:
        raise ValueError("native-glare protocol path is empty")
    return Path(protocol).resolve()


def _install_sensor_attribute_patch(profile_by_sensor_id: Mapping[str, Mapping[str, float]]) -> None:
    if getattr(AgentWrapper, "_orion_native_glare_triplet_patch", False):
        raise RuntimeError("native-glare triplet sensor patch installed more than once")
    original = AgentWrapper._preprocess_sensor_spec

    def patched(wrapper, sensor_spec):
        type_, sensor_id, transform, attributes = original(wrapper, sensor_spec)
        if type_ == "sensor.camera.rgb" and sensor_id in profile_by_sensor_id:
            profile = profile_by_sensor_id[sensor_id]
            attributes = dict(attributes)
            attributes.update({
                "enable_postprocess_effects": "true",
                "exposure_mode": "histogram",
                "lens_flare_intensity": str(float(profile["lens_flare_intensity"])),
                "bloom_intensity": str(float(profile["bloom_intensity"])),
            })
        elif type_ == "sensor.camera.rgb" and sensor_id == "bev":
            attributes = dict(attributes)
            attributes.update({
                "enable_postprocess_effects": "false",
                "lens_flare_intensity": "0.0",
                "bloom_intensity": "0.0",
            })
        return type_, sensor_id, transform, attributes

    AgentWrapper._preprocess_sensor_spec = patched
    AgentWrapper._orion_native_glare_triplet_patch = True


def _polyline_progress(point: Tuple[float, float], route: Iterable[Tuple[float, float]]) -> float:
    route = tuple(route)
    if len(route) < 2:
        return 0.0
    lengths = [
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(route[:-1], route[1:])
    ]
    total = sum(lengths)
    if total <= 1e-9:
        return 0.0
    prefix = 0.0
    best_distance = float("inf")
    best_progress = 0.0
    for left, right, length in zip(route[:-1], route[1:], lengths):
        if length <= 1e-9:
            continue
        dx, dy = right[0] - left[0], right[1] - left[1]
        fraction = max(0.0, min(1.0, (
            (point[0] - left[0]) * dx + (point[1] - left[1]) * dy
        ) / (length * length)))
        projected = (left[0] + fraction * dx, left[1] + fraction * dy)
        distance = math.hypot(point[0] - projected[0], point[1] - projected[1])
        if distance < best_distance:
            best_distance = distance
            best_progress = (prefix + fraction * length) / total
        prefix += length
    return float(max(0.0, min(1.0, best_progress)))


def _extended_finish_xy(route: Iterable[Tuple[float, float]], extension_m: float) -> Tuple[float, float]:
    route = tuple(route)
    if len(route) < 2 or extension_m <= 0.0:
        raise ValueError("finish extension requires two route points and a positive distance")
    previous, finish = route[-2], route[-1]
    dx, dy = finish[0] - previous[0], finish[1] - previous[1]
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        raise ValueError("last two route points must differ")
    return finish[0] + extension_m * dx / norm, finish[1] + extension_m * dy / norm


def _transform_payload(transform):
    location = transform.location
    rotation = transform.rotation
    return {
        "location": [float(location.x), float(location.y), float(location.z)],
        "rotation": [float(rotation.roll), float(rotation.pitch), float(rotation.yaw)],
        "world_to_sensor": [float(value) for row in transform.get_inverse_matrix() for value in row],
    }


class NativeGlareTripletCaptureAgent(AutonomousAgent):
    """Map-following, ORION-free same-tick glare capture agent."""

    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        self.protocol_path = protocol_path_from_agent_config(path_to_conf_file)
        self.protocol = load_protocol(self.protocol_path)
        self.profiles = self.protocol["camera_profiles"]
        self.profile_by_sensor_id = {
            value["sensor_id"]: value for value in self.profiles.values()
        }
        _install_sensor_attribute_patch(self.profile_by_sensor_id)
        capture = self.protocol["capture"]
        self.progress_start, self.progress_end = [
            _finite_float(value, "route progress")
            for value in capture["route_progress_window"]
        ]
        if not 0.0 <= self.progress_start < self.progress_end <= 1.0:
            raise ValueError("invalid route progress capture window")
        self.capture_stride = int(capture["stride_simulator_ticks"])
        self.maximum_frames = int(capture["maximum_saved_frames"])
        if self.capture_stride < 1 or self.maximum_frames < 1:
            raise ValueError("invalid capture stride or maximum")
        self.weather_values = dict(self.protocol["weather"])
        self.output_root = Path(os.environ["SAVE_PATH"]).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.image_roots = {}
        for name in ("clean", "medium", "heavy"):
            root = self.output_root / ("rgb_front_%s" % name)
            root.mkdir(parents=True, exist_ok=True)
            self.image_roots[name] = root
        self.bev_root = self.output_root / "bev"
        self.bev_root.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.output_root / "capture_trace.jsonl"
        if self.trace_path.exists():
            raise FileExistsError("refusing to overwrite native glare triplet trace")
        self.trace_stream = self.trace_path.open("x", encoding="utf-8")
        self.step = -1
        self.capture_index = 0
        self._agent = None
        self._weather_applied = False
        self._route_xy = ()
        self.finish_extension_m = float(os.environ.get("GLARE_CAPTURE_FINISH_EXTENSION_M", "6.0"))

    def sensors(self):
        sensors = []
        for name in ("clean", "medium", "heavy"):
            spec = {"type": "sensor.camera.rgb", "id": self.profiles[name]["sensor_id"]}
            spec.update(RGB_GEOMETRY)
            sensors.append(spec)
        sensors.append({
            "type": "sensor.camera.rgb",
            "x": 0.0,
            "y": 0.0,
            "z": 50.0,
            "roll": 0.0,
            "pitch": -90.0,
            "yaw": 0.0,
            "width": 512,
            "height": 512,
            "fov": 50,
            "id": "bev",
        })
        return sensors

    def _initialize_navigation(self):
        hero = None
        for actor in CarlaDataProvider.get_world().get_actors():
            if actor.attributes.get("role_name") == "hero":
                hero = actor
                break
        if hero is None:
            return False
        self.hero_actor = hero
        self._agent = BasicAgent(hero, target_speed=18)
        plan = []
        route_xy = []
        previous = None
        for transform, _ in self._global_plan_world_coord:
            waypoint = CarlaDataProvider.get_map().get_waypoint(transform.location)
            route_xy.append((transform.location.x, transform.location.y))
            if previous is not None:
                plan.extend(self._agent.trace_route(previous, waypoint))
            previous = waypoint
        if not plan or len(route_xy) < 2:
            raise RuntimeError("native glare triplet route plan is incomplete")
        finish_x, finish_y = _extended_finish_xy(route_xy, self.finish_extension_m)
        extended_waypoint = CarlaDataProvider.get_map().get_waypoint(
            carla.Location(x=finish_x, y=finish_y, z=0.0),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if extended_waypoint is None:
            raise RuntimeError("could not project native glare triplet finish extension")
        plan.extend(self._agent.trace_route(previous, extended_waypoint))
        self._agent.set_global_plan(plan)
        self._route_xy = tuple(route_xy)
        return True

    def _apply_weather(self):
        if self._weather_applied:
            return
        CarlaDataProvider.get_world().set_weather(carla.WeatherParameters(**self.weather_values))
        self._weather_applied = True

    def _weather_readback(self):
        weather = CarlaDataProvider.get_world().get_weather()
        return {name: float(getattr(weather, name)) for name in self.weather_values}

    def _sensor_readback(self):
        expected_ids = set(self.profile_by_sensor_id) | {"bev"}
        result = {}
        sensor_objects = getattr(self.sensor_interface, "_sensors_objects", {})
        for sensor_id, actor in sensor_objects.items():
            if sensor_id not in expected_ids:
                continue
            if not actor.type_id.startswith("sensor.camera.rgb"):
                raise RuntimeError("registered glare sensor %s is not an RGB camera" % sensor_id)
            result[sensor_id] = {
                "actor_id": int(actor.id),
                "attributes": {
                    key: actor.attributes.get(key)
                    for key in READBACK_ATTRIBUTES
                    if key in actor.attributes
                },
                "transform": _transform_payload(actor.get_transform()),
                "registry": "AutonomousAgent.sensor_interface._sensors_objects",
            }
        missing = expected_ids - set(result)
        if missing:
            raise RuntimeError("camera sensor readback missing: %s" % sorted(missing))
        return result

    def _nearby_actors(self):
        ego_location = self.hero_actor.get_location()
        result = []
        for actor in CarlaDataProvider.get_world().get_actors():
            if actor.id == self.hero_actor.id:
                continue
            if actor.type_id.startswith("walker.pedestrian"):
                category = "walker"
            elif actor.type_id.startswith("vehicle."):
                category = "vehicle"
            else:
                continue
            transform = actor.get_transform()
            distance = transform.location.distance(ego_location)
            if distance > 80.0:
                continue
            velocity = actor.get_velocity()
            vertices = actor.bounding_box.get_world_vertices(transform)
            result.append({
                "actor_id": int(actor.id),
                "type_id": actor.type_id,
                "category": category,
                "distance_m": float(distance),
                "location": [float(transform.location.x), float(transform.location.y), float(transform.location.z)],
                "rotation": [float(transform.rotation.roll), float(transform.rotation.pitch), float(transform.rotation.yaw)],
                "velocity": [float(velocity.x), float(velocity.y), float(velocity.z)],
                "bbox_world_vertices": [
                    [float(vertex.x), float(vertex.y), float(vertex.z)]
                    for vertex in vertices
                ],
            })
        return sorted(result, key=lambda value: value["distance_m"])

    def _capture(self, input_data, timestamp, control, route_progress):
        sensor_ids = [self.profiles[name]["sensor_id"] for name in ("clean", "medium", "heavy")]
        sensor_frames = {sensor_id: int(input_data[sensor_id][0]) for sensor_id in sensor_ids}
        if len(set(sensor_frames.values())) != 1:
            raise RuntimeError("native glare RGB cameras are not on the same simulator frame")
        name = "%04d.png" % self.capture_index
        paths = {}
        for profile_name, sensor_id in zip(("clean", "medium", "heavy"), sensor_ids):
            image = input_data[sensor_id][1][:, :, :3]
            path = self.image_roots[profile_name] / name
            if not cv2.imwrite(str(path), image):
                raise RuntimeError("failed to save %s glare frame" % profile_name)
            paths[profile_name] = str(path.resolve())
        bev = input_data["bev"][1][:, :, :3]
        bev_path = self.bev_root / name
        if not cv2.imwrite(str(bev_path), bev):
            raise RuntimeError("failed to save native glare BEV frame")
        transform = self.hero_actor.get_transform()
        velocity = self.hero_actor.get_velocity()
        row = {
            "schema": SCHEMA,
            "protocol": str(self.protocol_path),
            "capture_index": self.capture_index,
            "step": self.step,
            "sim_time_seconds": float(timestamp),
            "sensor_frames": sensor_frames,
            "same_tick": len(set(sensor_frames.values())) == 1,
            "front": paths,
            "bev": str(bev_path.resolve()),
            "camera_profiles_requested": self.profiles,
            "sensor_readback": self._sensor_readback(),
            "weather_requested": self.weather_values,
            "weather_readback": self._weather_readback(),
            "ego_transform": _transform_payload(transform),
            "ego_speed_mps": float(math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)),
            "route_progress": float(route_progress),
            "control": {
                "throttle": float(control.throttle),
                "steer": float(control.steer),
                "brake": float(control.brake),
            },
            "nearby_actors": self._nearby_actors(),
            "orion_loaded": False,
            "adapter_loaded": False,
        }
        self.trace_stream.write(json.dumps(row, sort_keys=True) + "\n")
        self.trace_stream.flush()
        self.capture_index += 1

    def run_step(self, input_data, timestamp):
        self.step += 1
        self._apply_weather()
        if self._agent is None and not self._initialize_navigation():
            return carla.VehicleControl()
        control = self._agent.run_step()
        location = self.hero_actor.get_location()
        progress = _polyline_progress((location.x, location.y), self._route_xy)
        if (
            self.progress_start <= progress <= self.progress_end
            and self.step % self.capture_stride == 0
            and self.capture_index < self.maximum_frames
        ):
            self._capture(input_data, timestamp, control, progress)
        return control

    def destroy(self):
        stream = getattr(self, "trace_stream", None)
        if stream is not None and not stream.closed:
            stream.close()


__all__ = [
    "NativeGlareTripletCaptureAgent",
    "RGB_GEOMETRY",
    "SCHEMA",
    "get_entry_point",
    "load_protocol",
    "protocol_path_from_agent_config",
]
