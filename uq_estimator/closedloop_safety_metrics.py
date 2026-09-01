"""Geometry-only closed-loop safety telemetry.

The helpers in this module deliberately avoid CARLA dependencies.  The agent
records the underlying planar states as well as derived disc-envelope metrics,
so alternative TTC, clearance, or PET definitions can be recomputed offline.
"""

import math


SCHEMA_VERSION = "orion.closedloop_dynamic_actor_safety.v2"
_EPSILON = 1e-9


def vertical_separating_gap(
    ego_center_z,
    ego_extent_z,
    actor_center_z,
    actor_extent_z,
):
    """Return the non-negative gap between two vertical bounding intervals."""

    values = [
        float(ego_center_z),
        float(ego_extent_z),
        float(actor_center_z),
        float(actor_extent_z),
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("vertical box values must be finite")
    if ego_extent_z < 0.0 or actor_extent_z < 0.0:
        raise ValueError("vertical extents must be non-negative")
    return max(
        0.0,
        abs(float(actor_center_z) - float(ego_center_z))
        - float(actor_extent_z)
        - float(ego_extent_z),
    )


def _xy(values, name):
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    result = (float(values[0]), float(values[1]))
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def disc_collision_ttc(
    relative_position_xy,
    relative_velocity_xy,
    combined_radius_m,
    horizon_seconds=10.0,
):
    """Return constant-velocity TTC for two planar disc envelopes.

    ``relative_*`` are actor minus ego.  ``None`` means the discs do not
    intersect within the requested future horizon.  A value of zero means the
    envelopes already overlap; it does not itself assert a simulator collision.
    """

    rx, ry = _xy(relative_position_xy, "relative_position_xy")
    vx, vy = _xy(relative_velocity_xy, "relative_velocity_xy")
    radius = float(combined_radius_m)
    horizon = float(horizon_seconds)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("combined_radius_m must be finite and non-negative")
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon_seconds must be finite and positive")

    c_term = rx * rx + ry * ry - radius * radius
    if c_term <= 0.0:
        return 0.0
    a_term = vx * vx + vy * vy
    if a_term <= _EPSILON:
        return None
    b_term = 2.0 * (rx * vx + ry * vy)
    discriminant = b_term * b_term - 4.0 * a_term * c_term
    if discriminant < 0.0:
        return None
    sqrt_discriminant = math.sqrt(max(0.0, discriminant))
    first_contact = (-b_term - sqrt_discriminant) / (2.0 * a_term)
    final_contact = (-b_term + sqrt_discriminant) / (2.0 * a_term)
    if final_contact < 0.0 or first_contact > horizon:
        return None
    return max(0.0, first_contact)


def _box_axes(yaw_degrees):
    yaw = math.radians(float(yaw_degrees))
    forward = (math.cos(yaw), math.sin(yaw))
    right = (-math.sin(yaw), math.cos(yaw))
    return forward, right


def _projected_box_radius(extent_xy_m, box_axes, projection_axis):
    extent_x, extent_y = _xy(extent_xy_m, "extent_xy_m")
    return (
        abs(box_axes[0][0] * projection_axis[0]
            + box_axes[0][1] * projection_axis[1]) * extent_x
        + abs(box_axes[1][0] * projection_axis[0]
              + box_axes[1][1] * projection_axis[1]) * extent_y
    )


def obb_collision_ttc(
    relative_position_xy,
    relative_velocity_xy,
    ego_extent_xy_m,
    ego_yaw_degrees,
    actor_extent_xy_m,
    actor_yaw_degrees,
    horizon_seconds=10.0,
):
    """Return constant-velocity TTC for fixed-orientation planar boxes.

    This applies continuous separating-axis intervals over the two axes of each
    box.  It is substantially less prone than a circumscribed disc to declaring
    adjacent-lane or roadside vehicles immediate collisions.
    """

    relative_position = _xy(relative_position_xy, "relative_position_xy")
    relative_velocity = _xy(relative_velocity_xy, "relative_velocity_xy")
    ego_axes = _box_axes(ego_yaw_degrees)
    actor_axes = _box_axes(actor_yaw_degrees)
    horizon = float(horizon_seconds)
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon_seconds must be finite and positive")

    interval_start = 0.0
    interval_end = horizon
    for axis in ego_axes + actor_axes:
        center = (
            relative_position[0] * axis[0]
            + relative_position[1] * axis[1]
        )
        velocity = (
            relative_velocity[0] * axis[0]
            + relative_velocity[1] * axis[1]
        )
        radius = _projected_box_radius(ego_extent_xy_m, ego_axes, axis)
        radius += _projected_box_radius(actor_extent_xy_m, actor_axes, axis)
        if abs(velocity) <= _EPSILON:
            if abs(center) > radius:
                return None
            continue
        entry = (-radius - center) / velocity
        exit_time = (radius - center) / velocity
        if entry > exit_time:
            entry, exit_time = exit_time, entry
        interval_start = max(interval_start, entry)
        interval_end = min(interval_end, exit_time)
        if interval_start > interval_end:
            return None
    if interval_end < 0.0 or interval_start > horizon:
        return None
    return max(0.0, interval_start)


def obb_separating_axis_gap(
    relative_position_xy,
    ego_extent_xy_m,
    ego_yaw_degrees,
    actor_extent_xy_m,
    actor_yaw_degrees,
):
    """Return the largest current separating-axis gap (zero if overlapping)."""

    relative_position = _xy(relative_position_xy, "relative_position_xy")
    ego_axes = _box_axes(ego_yaw_degrees)
    actor_axes = _box_axes(actor_yaw_degrees)
    gaps = []
    for axis in ego_axes + actor_axes:
        center = abs(
            relative_position[0] * axis[0]
            + relative_position[1] * axis[1]
        )
        radius = _projected_box_radius(ego_extent_xy_m, ego_axes, axis)
        radius += _projected_box_radius(actor_extent_xy_m, actor_axes, axis)
        gaps.append(center - radius)
    return max(0.0, max(gaps, default=0.0))


def pairwise_safety_metrics(ego_state, actor_state, horizon_seconds=10.0):
    """Compute JSON-serializable relative kinematics for one dynamic actor."""

    ego_position = _xy(ego_state["position_xy"], "ego.position_xy")
    ego_velocity = _xy(ego_state["velocity_xy"], "ego.velocity_xy")
    actor_position = _xy(actor_state["position_xy"], "actor.position_xy")
    actor_velocity = _xy(actor_state["velocity_xy"], "actor.velocity_xy")
    ego_radius = float(ego_state["radius_m"])
    actor_radius = float(actor_state["radius_m"])
    if min(ego_radius, actor_radius) < 0.0:
        raise ValueError("Actor radii must be non-negative")

    relative_position = (
        actor_position[0] - ego_position[0],
        actor_position[1] - ego_position[1],
    )
    relative_velocity = (
        actor_velocity[0] - ego_velocity[0],
        actor_velocity[1] - ego_velocity[1],
    )
    range_m = math.hypot(*relative_position)
    combined_radius = ego_radius + actor_radius
    disc_clearance = range_m - combined_radius
    radial_dot = (
        relative_position[0] * relative_velocity[0]
        + relative_position[1] * relative_velocity[1]
    )
    closing_speed = max(0.0, -radial_dot / max(range_m, _EPSILON))
    relative_speed_squared = (
        relative_velocity[0] * relative_velocity[0]
        + relative_velocity[1] * relative_velocity[1]
    )
    if relative_speed_squared <= _EPSILON:
        closest_time = 0.0
    else:
        closest_time = min(
            max(-radial_dot / relative_speed_squared, 0.0),
            float(horizon_seconds),
        )
    closest_x = relative_position[0] + relative_velocity[0] * closest_time
    closest_y = relative_position[1] + relative_velocity[1] * closest_time
    predicted_clearance = math.hypot(closest_x, closest_y) - combined_radius
    collision_ttc = disc_collision_ttc(
        relative_position,
        relative_velocity,
        combined_radius,
        horizon_seconds,
    )
    obb_ttc = obb_collision_ttc(
        relative_position,
        relative_velocity,
        ego_state["extent_xy_m"],
        ego_state.get("yaw_degrees", 0.0),
        actor_state["extent_xy_m"],
        actor_state.get("yaw_degrees", 0.0),
        horizon_seconds,
    )
    obb_gap = obb_separating_axis_gap(
        relative_position,
        ego_state["extent_xy_m"],
        ego_state.get("yaw_degrees", 0.0),
        actor_state["extent_xy_m"],
        actor_state.get("yaw_degrees", 0.0),
    )

    ego_yaw = math.radians(float(ego_state.get("yaw_degrees", 0.0)))
    forward = (math.cos(ego_yaw), math.sin(ego_yaw))
    right = (-math.sin(ego_yaw), math.cos(ego_yaw))
    relative_longitudinal = (
        relative_position[0] * forward[0]
        + relative_position[1] * forward[1]
    )
    relative_lateral = (
        relative_position[0] * right[0]
        + relative_position[1] * right[1]
    )

    return {
        "actor_id": int(actor_state["actor_id"]),
        "type_id": str(actor_state.get("type_id", "")),
        "category": str(actor_state.get("category", "unknown")),
        "position_xy": list(actor_position),
        "position_z": float(actor_state.get("position_z", 0.0)),
        "velocity_xy": list(actor_velocity),
        "yaw_degrees": float(actor_state.get("yaw_degrees", 0.0)),
        "extent_xy_m": [float(value) for value in actor_state["extent_xy_m"]],
        "extent_z_m": float(actor_state.get("extent_z_m", 0.0)),
        "disc_radius_m": actor_radius,
        "relative_position_xy_m": list(relative_position),
        "relative_velocity_xy_mps": list(relative_velocity),
        "relative_longitudinal_m": relative_longitudinal,
        "relative_lateral_m": relative_lateral,
        "range_m": range_m,
        "closing_speed_mps": closing_speed,
        "disc_clearance_m": disc_clearance,
        "closest_approach_time_seconds": closest_time,
        "predicted_min_disc_clearance_m": predicted_clearance,
        "disc_collision_ttc_seconds": collision_ttc,
        "obb_separating_axis_gap_m": obb_gap,
        "obb_collision_ttc_seconds": obb_ttc,
    }


def summarize_dynamic_actor_safety(
    ego_state,
    actor_states,
    horizon_seconds=10.0,
    max_actor_records=8,
):
    """Summarize the most safety-relevant actors while retaining raw states."""

    max_records = int(max_actor_records)
    if max_records <= 0:
        raise ValueError("max_actor_records must be positive")
    records = [
        pairwise_safety_metrics(ego_state, actor, horizon_seconds)
        for actor in actor_states
    ]

    def relevance(record):
        ttc = record["obb_collision_ttc_seconds"]
        return (
            ttc is None,
            float("inf") if ttc is None else ttc,
            record["predicted_min_disc_clearance_m"],
            record["disc_clearance_m"],
        )

    records.sort(key=relevance)
    obb_ttc_values = [
        record["obb_collision_ttc_seconds"]
        for record in records
        if record["obb_collision_ttc_seconds"] is not None
    ]
    disc_ttc_values = [
        record["disc_collision_ttc_seconds"]
        for record in records
        if record["disc_collision_ttc_seconds"] is not None
    ]
    min_clearance = min(
        (record["disc_clearance_m"] for record in records), default=None
    )
    min_predicted_clearance = min(
        (record["predicted_min_disc_clearance_m"] for record in records),
        default=None,
    )
    min_obb_gap = min(
        (record["obb_separating_axis_gap_m"] for record in records),
        default=None,
    )
    return {
        "schema": SCHEMA_VERSION,
        "available": True,
        "primary_model": "constant_velocity_fixed_orientation_planar_obb",
        "diagnostic_model": "constant_velocity_planar_disc_envelope",
        "horizon_seconds": float(horizon_seconds),
        "ego": {
            "actor_id": int(ego_state["actor_id"]),
            "position_xy": list(_xy(ego_state["position_xy"], "ego.position_xy")),
            "position_z": float(ego_state.get("position_z", 0.0)),
            "velocity_xy": list(_xy(ego_state["velocity_xy"], "ego.velocity_xy")),
            "yaw_degrees": float(ego_state.get("yaw_degrees", 0.0)),
            "extent_xy_m": [float(value) for value in ego_state["extent_xy_m"]],
            "extent_z_m": float(ego_state.get("extent_z_m", 0.0)),
            "disc_radius_m": float(ego_state["radius_m"]),
        },
        "actor_count_considered": len(records),
        "min_obb_collision_ttc_seconds": min(obb_ttc_values, default=None),
        "min_obb_separating_axis_gap_m": min_obb_gap,
        "min_disc_collision_ttc_seconds": min(disc_ttc_values, default=None),
        "min_disc_clearance_m": min_clearance,
        "min_predicted_disc_clearance_m": min_predicted_clearance,
        "critical_actor": records[0] if records else None,
        "actors": records[:max_records],
    }
