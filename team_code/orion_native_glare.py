"""Auditable CARLA-native glare configuration for the real ORION agent.

The profile changes the RGB sensor blueprints before Leaderboard spawns them.
It deliberately does not change weather: low-sun weather remains a frozen route
property and is recorded from the running world as readback evidence.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path


SCHEMA = "orion.closedloop_render_condition.v1"
READBACK_SCHEMA = "orion.closedloop_render_condition_readback.v1"
PROFILES = {
    "clean": {"lens_flare_intensity": 0.0, "bloom_intensity": 0.0},
    "medium": {"lens_flare_intensity": 0.75, "bloom_intensity": 1.5},
    "heavy": {"lens_flare_intensity": 1.5, "bloom_intensity": 3.0},
}
CAMERA_READBACK_ATTRIBUTES = (
    "enable_postprocess_effects",
    "exposure_mode",
    "lens_flare_intensity",
    "bloom_intensity",
    "image_size_x",
    "image_size_y",
    "fov",
    "role_name",
)
WEATHER_READBACK_ATTRIBUTES = (
    "cloudiness", "precipitation", "precipitation_deposits", "wind_intensity",
    "sun_azimuth_angle", "sun_altitude_angle", "fog_density", "fog_distance",
    "fog_falloff", "wetness", "scattering_intensity", "mie_scattering_scale",
    "rayleigh_scattering_scale",
)


def normalize_native_glare_profile(value):
    profile = str(value or "none").strip().lower()
    allowed = set(PROFILES)
    allowed.add("none")
    if profile not in allowed:
        raise ValueError(
            "ORION_NATIVE_GLARE_PROFILE must be none, clean, medium, or heavy; got %r"
            % value
        )
    return profile


def requested_render_condition(profile):
    profile = normalize_native_glare_profile(profile)
    if profile == "none":
        return {
            "schema": SCHEMA,
            "kind": "standard_carla_rgb",
            "native_glare_profile": "none",
            "camera_postprocess_override": False,
            "requested": {"CAM_FRONT": None, "bev": None},
        }
    values = PROFILES[profile]
    return {
        "schema": SCHEMA,
        "kind": "carla_native_low_sun_glare",
        "native_glare_profile": profile,
        "camera_postprocess_override": True,
        "requested": {
            "CAM_FRONT": {
                "enable_postprocess_effects": "true",
                "exposure_mode": "histogram",
                "lens_flare_intensity": values["lens_flare_intensity"],
                "bloom_intensity": values["bloom_intensity"],
            },
            "bev": {
                "enable_postprocess_effects": "false",
                "lens_flare_intensity": 0.0,
                "bloom_intensity": 0.0,
            },
        },
    }


def install_orion_native_glare_sensor_patch(agent_wrapper_class, profile):
    """Patch Leaderboard's blueprint attributes before sensor construction."""
    profile = normalize_native_glare_profile(profile)
    if profile == "none":
        return False
    if getattr(agent_wrapper_class, "_orion_native_glare_patch", False):
        raise RuntimeError("ORION native-glare sensor patch installed more than once")
    requested = requested_render_condition(profile)["requested"]
    original = agent_wrapper_class._preprocess_sensor_spec

    def patched(wrapper, sensor_spec):
        type_, sensor_id, transform, attributes = original(wrapper, sensor_spec)
        if type_ == "sensor.camera.rgb" and sensor_id in {"CAM_FRONT", "bev"}:
            attributes = dict(attributes)
            attributes.update({key: str(value).lower() if isinstance(value, bool) else str(value)
                               for key, value in requested[sensor_id].items()})
        return type_, sensor_id, transform, attributes

    agent_wrapper_class._preprocess_sensor_spec = patched
    agent_wrapper_class._orion_native_glare_patch = profile
    return True


def _transform_payload(transform):
    location, rotation = transform.location, transform.rotation
    return {
        "location": {"x": float(location.x), "y": float(location.y), "z": float(location.z)},
        "rotation": {"pitch": float(rotation.pitch), "yaw": float(rotation.yaw), "roll": float(rotation.roll)},
    }


def _camera_readback(sensor_interface):
    result = {}
    actors = getattr(sensor_interface, "_sensors_objects", {})
    for sensor_id in ("CAM_FRONT", "bev"):
        actor = actors.get(sensor_id)
        if actor is None:
            continue
        result[sensor_id] = {
            "actor_id": int(actor.id),
            "type_id": str(actor.type_id),
            "attributes": {key: actor.attributes.get(key) for key in CAMERA_READBACK_ATTRIBUTES
                           if key in actor.attributes},
            "transform": _transform_payload(actor.get_transform()),
            "registry": "AutonomousAgent.sensor_interface._sensors_objects",
        }
    return result


def _matches(actual, expected):
    if isinstance(expected, (float, int)):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-6)
        except (TypeError, ValueError):
            return False
    return str(actual).strip().lower() == str(expected).strip().lower()


def readback_render_condition(sensor_interface, world, profile):
    profile = normalize_native_glare_profile(profile)
    cameras = _camera_readback(sensor_interface)
    weather = world.get_weather()
    payload = {
        "schema": READBACK_SCHEMA,
        "status": "verified" if profile != "none" else "observed",
        "native_glare_profile": profile,
        "cameras": cameras,
        "weather": {key: float(getattr(weather, key)) for key in WEATHER_READBACK_ATTRIBUTES
                    if hasattr(weather, key)},
    }
    if profile == "none":
        return payload
    requested = requested_render_condition(profile)["requested"]
    errors = []
    for sensor_id in ("CAM_FRONT", "bev"):
        actual = cameras.get(sensor_id)
        if actual is None:
            errors.append("missing RGB actor %s" % sensor_id)
            continue
        if not actual["type_id"].startswith("sensor.camera.rgb"):
            errors.append("%s is not an RGB camera" % sensor_id)
            continue
        for key, expected in requested[sensor_id].items():
            value = actual["attributes"].get(key)
            if not _matches(value, expected):
                errors.append("%s.%s requested=%r actual=%r" % (sensor_id, key, expected, value))
    if errors:
        raise RuntimeError("native-glare CARLA readback mismatch: " + "; ".join(errors))
    return payload


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-%d" % os.getpid())
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def record_render_condition_readback(output_root, requested, readback):
    output_root = Path(output_root)
    readback_path = output_root / "render_condition_readback.json"
    manifest_path = output_root / "manifest.json"
    _atomic_json(readback_path, readback)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    existing = dict(manifest.get("render_condition") or {})
    existing.pop("actual_readback", None)
    if existing != requested:
        raise RuntimeError("run manifest render_condition differs from agent request")
    updated = dict(requested)
    updated["actual_readback"] = {
        "status": readback["status"],
        "path": readback_path.name,
        "schema": readback["schema"],
    }
    manifest["render_condition"] = updated
    _atomic_json(manifest_path, manifest)
    return readback_path
