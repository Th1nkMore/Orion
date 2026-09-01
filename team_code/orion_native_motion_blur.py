"""Auditable CARLA-native motion-blur profiles for ORION experiments."""

from __future__ import annotations

import math


SCHEMA = "orion.closedloop_render_condition.v1"
READBACK_SCHEMA = "orion.closedloop_render_condition_readback.v1"

# Candidate visual-bakeoff profiles.  They are implementation defaults, not a
# claim that the final experimental severities have already been selected.
PROFILES = {
    "clean": {
        "motion_blur_intensity": 0.0,
        "motion_blur_max_distortion": 0.0,
        "motion_blur_min_object_screen_size": 0.0,
    },
    "light": {
        "motion_blur_intensity": 0.35,
        "motion_blur_max_distortion": 0.20,
        "motion_blur_min_object_screen_size": 0.10,
    },
    "medium": {
        "motion_blur_intensity": 0.70,
        "motion_blur_max_distortion": 0.45,
        "motion_blur_min_object_screen_size": 0.05,
    },
    "heavy": {
        "motion_blur_intensity": 1.0,
        "motion_blur_max_distortion": 0.75,
        "motion_blur_min_object_screen_size": 0.0,
    },
}

# Diagnostic-only profile used to isolate CARLA camera blueprint failures.  It
# deliberately changes one front-camera attribute and leaves every other
# sensor attribute untouched.  It is not a candidate experimental severity.
DIAGNOSTIC_PROFILES = {
    "intensity_zero_only": {
        "motion_blur_intensity": 0.0,
    },
}

CAMERA_READBACK_ATTRIBUTES = (
    "enable_postprocess_effects",
    "motion_blur_intensity",
    "motion_blur_max_distortion",
    "motion_blur_min_object_screen_size",
    "image_size_x",
    "image_size_y",
    "fov",
    "role_name",
)


def normalize_native_motion_blur_profile(value):
    profile = str(value or "none").strip().lower()
    allowed = set(PROFILES) | set(DIAGNOSTIC_PROFILES)
    allowed.add("none")
    if profile not in allowed:
        raise ValueError(
            "ORION_NATIVE_MOTION_BLUR_PROFILE must be none, clean, light, "
            "medium, heavy, or intensity_zero_only; got %r" % value
        )
    return profile


def requested_native_motion_blur_condition(profile):
    profile = normalize_native_motion_blur_profile(profile)
    if profile == "none":
        return {
            "schema": SCHEMA,
            "kind": "standard_carla_rgb",
            "native_motion_blur_profile": "none",
            "camera_postprocess_override": False,
            "requested": {"CAM_FRONT": None, "bev": None},
        }
    diagnostic = profile in DIAGNOSTIC_PROFILES
    values = DIAGNOSTIC_PROFILES[profile] if diagnostic else PROFILES[profile]
    front_requested = {
        **({} if diagnostic else {"enable_postprocess_effects": "true"}),
        **values,
    }
    bev_requested = None if diagnostic else {
        "enable_postprocess_effects": "false",
        "motion_blur_intensity": 0.0,
        "motion_blur_max_distortion": 0.0,
        "motion_blur_min_object_screen_size": 0.0,
    }
    return {
        "schema": SCHEMA,
        "kind": (
            "carla_camera_attribute_diagnostic"
            if diagnostic
            else "carla_native_motion_blur_candidate"
        ),
        "native_motion_blur_profile": profile,
        "camera_postprocess_override": True,
        "severity_status": (
            "diagnostic_only_not_an_experimental_condition"
            if diagnostic
            else "candidate_pending_visual_bakeoff"
        ),
        "requested": {
            "CAM_FRONT": front_requested,
            "bev": bev_requested,
        },
    }


def install_orion_native_motion_blur_sensor_patch(agent_wrapper_class, profile):
    """Patch RGB blueprint attributes before Leaderboard spawns sensors."""
    profile = normalize_native_motion_blur_profile(profile)
    if profile == "none":
        return False
    if getattr(agent_wrapper_class, "_orion_native_motion_blur_patch", False):
        raise RuntimeError("ORION native-motion-blur patch installed more than once")
    requested = requested_native_motion_blur_condition(profile)["requested"]
    original = agent_wrapper_class._preprocess_sensor_spec

    def patched(wrapper, sensor_spec):
        type_, sensor_id, transform, attributes = original(wrapper, sensor_spec)
        if type_ == "sensor.camera.rgb" and sensor_id in {"CAM_FRONT", "bev"}:
            sensor_requested = requested[sensor_id]
            if sensor_requested is None:
                return type_, sensor_id, transform, attributes
            attributes = dict(attributes)
            attributes.update(
                {
                    key: str(value).lower() if isinstance(value, bool) else str(value)
                    for key, value in sensor_requested.items()
                }
            )
        return type_, sensor_id, transform, attributes

    agent_wrapper_class._preprocess_sensor_spec = patched
    agent_wrapper_class._orion_native_motion_blur_patch = profile
    return True


def _transform_payload(transform):
    location, rotation = transform.location, transform.rotation
    return {
        "location": {
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
        },
        "rotation": {
            "pitch": float(rotation.pitch),
            "yaw": float(rotation.yaw),
            "roll": float(rotation.roll),
        },
    }


def _matches(actual, expected):
    if isinstance(expected, (float, int)):
        try:
            return math.isclose(
                float(actual), float(expected), rel_tol=0.0, abs_tol=1e-6
            )
        except (TypeError, ValueError):
            return False
    return str(actual).strip().lower() == str(expected).strip().lower()


def readback_native_motion_blur_condition(sensor_interface, profile):
    """Fail closed unless spawned CARLA actors expose requested attributes."""
    profile = normalize_native_motion_blur_profile(profile)
    actors = getattr(sensor_interface, "_sensors_objects", {})
    cameras = {}
    for sensor_id in ("CAM_FRONT", "bev"):
        actor = actors.get(sensor_id)
        if actor is None:
            continue
        cameras[sensor_id] = {
            "actor_id": int(actor.id),
            "type_id": str(actor.type_id),
            "attributes": {
                key: actor.attributes.get(key)
                for key in CAMERA_READBACK_ATTRIBUTES
                if key in actor.attributes
            },
            "transform": _transform_payload(actor.get_transform()),
            "registry": "AutonomousAgent.sensor_interface._sensors_objects",
        }
    payload = {
        "schema": READBACK_SCHEMA,
        "status": "verified" if profile != "none" else "observed",
        "native_motion_blur_profile": profile,
        "cameras": cameras,
    }
    if profile == "none":
        return payload
    requested = requested_native_motion_blur_condition(profile)["requested"]
    errors = []
    for sensor_id in ("CAM_FRONT", "bev"):
        sensor_requested = requested[sensor_id]
        if sensor_requested is None:
            continue
        actual = cameras.get(sensor_id)
        if actual is None:
            errors.append("missing RGB actor %s" % sensor_id)
            continue
        if not actual["type_id"].startswith("sensor.camera.rgb"):
            errors.append("%s is not an RGB camera" % sensor_id)
            continue
        for key, expected in sensor_requested.items():
            value = actual["attributes"].get(key)
            if not _matches(value, expected):
                errors.append(
                    "%s.%s requested=%r actual=%r"
                    % (sensor_id, key, expected, value)
                )
    if errors:
        raise RuntimeError(
            "native-motion-blur CARLA readback mismatch: " + "; ".join(errors)
        )
    return payload
