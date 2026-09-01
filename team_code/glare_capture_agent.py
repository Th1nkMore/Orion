#!/usr/bin/env python3
"""Lightweight Bench2Drive visual-capture agent for native glare bake-offs.

The agent uses CARLA's map-based BasicAgent and never imports or loads ORION.
It patches the local Leaderboard sensor wrapper only for this process so the
front RGB blueprint receives CARLA-native lens-flare/bloom attributes.  Saved
pose telemetry permits post-hoc pose matching across independent profiles.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import carla
import cv2

from agents.navigation.basic_agent import BasicAgent
from leaderboard.autoagents.agent_wrapper import AgentWrapper
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
try:
    from orion_native_motion_blur import (
        DIAGNOSTIC_PROFILES as MOTION_BLUR_DIAGNOSTIC_PROFILES,
        PROFILES as MOTION_BLUR_PROFILES,
        install_orion_native_motion_blur_sensor_patch,
        normalize_native_motion_blur_profile,
        readback_native_motion_blur_condition,
    )
except ModuleNotFoundError:
    from team_code.orion_native_motion_blur import (
        DIAGNOSTIC_PROFILES as MOTION_BLUR_DIAGNOSTIC_PROFILES,
        PROFILES as MOTION_BLUR_PROFILES,
        install_orion_native_motion_blur_sensor_patch,
        normalize_native_motion_blur_profile,
        readback_native_motion_blur_condition,
    )


SCHEMA = "orion.native_glare_capture_trace.v1"
PROFILES = {
    "clean": {"lens_flare_intensity": 0.0, "bloom_intensity": 0.0},
    "light": {"lens_flare_intensity": 0.25, "bloom_intensity": 0.675},
    "medium": {"lens_flare_intensity": 0.75, "bloom_intensity": 1.5},
    "heavy": {"lens_flare_intensity": 1.5, "bloom_intensity": 3.0},
}
WEATHER_DEFAULTS = {
    "cloudiness": 5.0,
    "precipitation": 0.0,
    "precipitation_deposits": 50.0,
    "wind_intensity": 10.0,
    "sun_altitude_angle": 8.0,
    "sun_azimuth_angle": 180.0,
    "wetness": 0.0,
    "fog_density": 10.0,
    "fog_distance": 0.0,
    "fog_falloff": 0.2,
    "scattering_intensity": 1.0,
    "mie_scattering_scale": 0.03,
    "rayleigh_scattering_scale": 0.0331,
}


def get_entry_point():
    return "NativeGlareCaptureAgent"


def native_glare_profile(name: str) -> Dict[str, float]:
    if name not in PROFILES:
        raise ValueError("unsupported native glare profile: %s" % name)
    return dict(PROFILES[name])


def native_glare_weather(environ=os.environ) -> Dict[str, float]:
    values = dict(WEATHER_DEFAULTS)
    overrides = {
        "sun_altitude_angle": "GLARE_CAPTURE_SUN_ALTITUDE",
        "sun_azimuth_angle": "GLARE_CAPTURE_SUN_AZIMUTH",
        "mie_scattering_scale": "GLARE_CAPTURE_MIE_SCATTERING",
    }
    for field, variable in overrides.items():
        if variable in environ:
            value = float(environ[variable])
            if not math.isfinite(value):
                raise ValueError("%s must be finite" % variable)
            values[field] = value
    if not -90.0 <= values["sun_altitude_angle"] <= 90.0:
        raise ValueError("sun altitude must lie in [-90,90]")
    if not -360.0 <= values["sun_azimuth_angle"] <= 360.0:
        raise ValueError("sun azimuth must lie in [-360,360]")
    if values["mie_scattering_scale"] < 0.0:
        raise ValueError("Mie scattering must be non-negative")
    return values


def _install_sensor_attribute_patch(profile_name: str) -> None:
    """Add post-process attributes ignored by the upstream B2D wrapper."""

    if getattr(AgentWrapper, "_orion_native_glare_patch", False):
        raise RuntimeError("native-glare sensor patch installed more than once")
    profile = native_glare_profile(profile_name)
    original = AgentWrapper._preprocess_sensor_spec

    def patched(wrapper, sensor_spec):
        type_, sensor_id, transform, attributes = original(wrapper, sensor_spec)
        if type_ == "sensor.camera.rgb" and sensor_id == "CAM_FRONT":
            attributes = dict(attributes)
            attributes.update({
                "enable_postprocess_effects": "true",
                "exposure_mode": "histogram",
                "lens_flare_intensity": str(profile["lens_flare_intensity"]),
                "bloom_intensity": str(profile["bloom_intensity"]),
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
    AgentWrapper._orion_native_glare_patch = True


def _polyline_progress(
    point: Tuple[float, float], route: Iterable[Tuple[float, float]]
) -> float:
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
    best_distance = float("inf")
    best_progress = 0.0
    prefix = 0.0
    for left, right, length in zip(route[:-1], route[1:], lengths):
        if length <= 1e-9:
            continue
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        fraction = max(
            0.0,
            min(1.0, ((point[0] - left[0]) * dx + (point[1] - left[1]) * dy) / (length * length)),
        )
        projected = (left[0] + fraction * dx, left[1] + fraction * dy)
        distance = math.hypot(point[0] - projected[0], point[1] - projected[1])
        if distance < best_distance:
            best_distance = distance
            best_progress = (prefix + fraction * length) / total
        prefix += length
    return float(max(0.0, min(1.0, best_progress)))


def _extended_finish_xy(
    route: Iterable[Tuple[float, float]], extension_m: float
) -> Tuple[float, float]:
    route = tuple(route)
    if len(route) < 2 or extension_m <= 0.0:
        raise ValueError("finish extension requires two route points and a positive distance")
    previous, finish = route[-2], route[-1]
    dx, dy = finish[0] - previous[0], finish[1] - previous[1]
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        raise ValueError("last two route points must differ")
    return finish[0] + extension_m * dx / norm, finish[1] + extension_m * dy / norm


class NativeGlareCaptureAgent(AutonomousAgent):
    """Map-following capture agent with no learned perception dependency."""

    def setup(self, path_to_conf_file):
        del path_to_conf_file
        self.track = Track.SENSORS
        self.capture_family = os.environ.get(
            "HARDCAPTURE_FAMILY", "native_glare"
        ).strip()
        if self.capture_family == "native_glare":
            self.trace_schema = SCHEMA
            self.profile_name = os.environ.get("GLARE_CAPTURE_PROFILE", "clean")
            self.profile = native_glare_profile(self.profile_name)
            self.weather_values = native_glare_weather()
            _install_sensor_attribute_patch(self.profile_name)
        elif self.capture_family == "native_motion_blur":
            self.trace_schema = "orion.native_motion_blur_capture_trace.v1"
            self.profile_name = normalize_native_motion_blur_profile(
                os.environ.get("MOTION_BLUR_CAPTURE_PROFILE", "clean")
            )
            if self.profile_name == "none":
                self.profile = {}
            elif self.profile_name in MOTION_BLUR_DIAGNOSTIC_PROFILES:
                self.profile = dict(
                    MOTION_BLUR_DIAGNOSTIC_PROFILES[self.profile_name]
                )
            else:
                self.profile = dict(MOTION_BLUR_PROFILES[self.profile_name])
            self.weather_values = None
            install_orion_native_motion_blur_sensor_patch(
                AgentWrapper, self.profile_name
            )
        else:
            raise ValueError(
                "HARDCAPTURE_FAMILY must be native_glare or native_motion_blur"
            )
        self.output_root = Path(os.environ["SAVE_PATH"]).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.front_root = self.output_root / "rgb_front"
        self.bev_root = self.output_root / "bev"
        self.front_root.mkdir(parents=True, exist_ok=True)
        self.bev_root.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.output_root / "capture_trace.jsonl"
        if self.trace_path.exists():
            raise FileExistsError("refusing to overwrite native glare trace")
        self.trace_stream = self.trace_path.open("x", encoding="utf-8")
        self.capture_stride = int(os.environ.get("GLARE_CAPTURE_STRIDE", "5"))
        if self.capture_stride < 1:
            raise ValueError("GLARE_CAPTURE_STRIDE must be positive")
        self.capture_start_progress = float(
            os.environ.get("HARDCAPTURE_START_PROGRESS", "0")
        )
        self.capture_end_progress = float(
            os.environ.get("HARDCAPTURE_END_PROGRESS", "1")
        )
        if not (
            0.0 <= self.capture_start_progress
            < self.capture_end_progress <= 1.0
        ):
            raise ValueError(
                "hard-case capture progress must satisfy 0 <= start < end <= 1"
            )
        self.step = -1
        self.capture_index = 0
        self._agent = None
        self._weather_applied = False
        self._render_readback = None
        self._route_xy = ()
        self.finish_extension_m = float(os.environ.get("GLARE_CAPTURE_FINISH_EXTENSION_M", "6.0"))
        if not math.isfinite(self.finish_extension_m) or self.finish_extension_m <= 0.0:
            raise ValueError("GLARE_CAPTURE_FINISH_EXTENSION_M must be finite and positive")

    def sensors(self):
        return [
            {
                "type": "sensor.camera.rgb",
                "x": 0.80,
                "y": 0.0,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 1600,
                "height": 900,
                "fov": 70,
                "id": "CAM_FRONT",
            },
            {
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
            },
        ]

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
            raise RuntimeError("native glare capture route plan is incomplete")
        finish_x, finish_y = _extended_finish_xy(route_xy, self.finish_extension_m)
        extended_waypoint = CarlaDataProvider.get_map().get_waypoint(
            carla.Location(x=finish_x, y=finish_y, z=0.0),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if extended_waypoint is None:
            raise RuntimeError("could not project native glare finish extension")
        plan.extend(self._agent.trace_route(previous, extended_waypoint))
        self._agent.set_global_plan(plan)
        self._route_xy = tuple(route_xy)
        return True

    def _apply_weather(self):
        if self._weather_applied:
            return
        if self.capture_family == "native_glare":
            weather = carla.WeatherParameters(**self.weather_values)
            CarlaDataProvider.get_world().set_weather(weather)
        self._weather_applied = True

    @staticmethod
    def _actual_weather():
        weather = CarlaDataProvider.get_world().get_weather()
        fields = (
            "cloudiness", "precipitation", "precipitation_deposits",
            "wind_intensity", "sun_azimuth_angle", "sun_altitude_angle",
            "fog_density", "fog_distance", "fog_falloff", "wetness",
            "scattering_intensity", "mie_scattering_scale",
            "rayleigh_scattering_scale",
        )
        return {
            field: float(getattr(weather, field))
            for field in fields
            if hasattr(weather, field)
        }

    def _nearby_actors(self):
        ego_location = self.hero_actor.get_location()
        result = []
        for actor in CarlaDataProvider.get_world().get_actors():
            if actor.id == self.hero_actor.id:
                continue
            if not (
                actor.type_id.startswith("walker.pedestrian")
                or actor.type_id.startswith("vehicle.")
            ):
                continue
            location = actor.get_location()
            distance = location.distance(ego_location)
            if distance > 80.0:
                continue
            result.append({
                "actor_id": int(actor.id),
                "type_id": actor.type_id,
                "distance_m": float(distance),
                "location": [float(location.x), float(location.y), float(location.z)],
            })
        return sorted(result, key=lambda value: value["distance_m"])

    def _capture(self, input_data, timestamp, control):
        if (
            self.capture_family == "native_motion_blur"
            and self._render_readback is None
        ):
            self._render_readback = readback_native_motion_blur_condition(
                self.sensor_interface, self.profile_name
            )
        front = input_data["CAM_FRONT"][1][:, :, :3]
        bev = input_data["bev"][1][:, :, :3]
        name = "%04d.png" % self.capture_index
        if not cv2.imwrite(str(self.front_root / name), front):
            raise RuntimeError("failed to save native glare front frame")
        if not cv2.imwrite(str(self.bev_root / name), bev):
            raise RuntimeError("failed to save native glare BEV frame")
        transform = self.hero_actor.get_transform()
        location = transform.location
        rotation = transform.rotation
        velocity = self.hero_actor.get_velocity()
        row = {
            "schema": self.trace_schema,
            "corruption_family": self.capture_family,
            "profile": self.profile_name,
            "camera_postprocess": self.profile,
            "camera_postprocess_readback": self._render_readback,
            "weather": self._actual_weather(),
            "capture_index": self.capture_index,
            "step": self.step,
            "sim_time_seconds": float(timestamp),
            "front": str((self.front_root / name).resolve()),
            "bev": str((self.bev_root / name).resolve()),
            "ego_location": [float(location.x), float(location.y), float(location.z)],
            "ego_rotation": [float(rotation.roll), float(rotation.pitch), float(rotation.yaw)],
            "ego_speed_mps": float(math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)),
            "route_progress": _polyline_progress((location.x, location.y), self._route_xy),
            "finish_extension_m": self.finish_extension_m,
            "control": {
                "throttle": float(control.throttle),
                "steer": float(control.steer),
                "brake": float(control.brake),
            },
            "nearby_actors": self._nearby_actors(),
            "orion_loaded": False,
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
        transform = self.hero_actor.get_transform()
        route_progress = _polyline_progress(
            (transform.location.x, transform.location.y), self._route_xy
        )
        if (
            self.capture_start_progress <= route_progress
            < self.capture_end_progress
            and self.step % self.capture_stride == 0
        ):
            self._capture(input_data, timestamp, control)
        return control

    def destroy(self):
        stream = getattr(self, "trace_stream", None)
        if stream is not None and not stream.closed:
            stream.close()


__all__ = [
    "NativeGlareCaptureAgent",
    "PROFILES",
    "SCHEMA",
    "WEATHER_DEFAULTS",
    "_extended_finish_xy",
    "get_entry_point",
    "native_glare_profile",
    "native_glare_weather",
]
