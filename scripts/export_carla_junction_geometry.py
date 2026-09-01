#!/usr/bin/env python
"""Export map-derived junction-entry geometry for a recorded control trace.

This utility intentionally stays Python-2.7 compatible because the official
CARLA 0.9.15 distribution on the compute platform includes a py2.7 client.
It is an offline map query: no CARLA server or GPU is required.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os

import carla


SCHEMA_VERSION = "orion.carla_junction_geometry.v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def waypoint_payload(waypoint):
    if waypoint is None:
        return None
    return {
        "road_id": int(waypoint.road_id),
        "section_id": int(waypoint.section_id),
        "lane_id": int(waypoint.lane_id),
        "is_junction": bool(waypoint.is_junction),
    }


def query_waypoint(carla_map, xy, z):
    return carla_map.get_waypoint(
        carla.Location(x=float(xy[0]), y=float(xy[1]), z=float(z)),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )


def point_on_segment(first, second, fraction):
    return (
        first[0] + fraction * (second[0] - first[0]),
        first[1] + fraction * (second[1] - first[1]),
    )


def refine_false_to_true(carla_map, false_sample, true_sample, z, iterations=14):
    """Binary-refine one non-junction to junction transition."""

    false_distance, false_xy = false_sample
    true_distance, true_xy = true_sample
    for unused in range(iterations):
        mid_distance = 0.5 * (false_distance + true_distance)
        mid_xy = point_on_segment(false_xy, true_xy, 0.5)
        waypoint = query_waypoint(carla_map, mid_xy, z)
        if waypoint is not None and waypoint.is_junction:
            true_distance, true_xy = mid_distance, mid_xy
        else:
            false_distance, false_xy = mid_distance, mid_xy
    return true_distance, true_xy


def first_junction_entry(carla_map, points, z, resolution_m):
    """Return the first map-junction entry along a world-space polyline."""

    if not points:
        raise ValueError("polyline cannot be empty")
    first_waypoint = query_waypoint(carla_map, points[0], z)
    if first_waypoint is not None and first_waypoint.is_junction:
        return 0.0, points[0], first_waypoint

    arc_before = 0.0
    previous = (0.0, points[0])
    previous_waypoint = first_waypoint
    for start, end in zip(points, points[1:]):
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if length <= 1e-9:
            continue
        count = max(1, int(math.ceil(length / resolution_m)))
        for sample_index in range(1, count + 1):
            fraction = float(sample_index) / float(count)
            xy = point_on_segment(start, end, fraction)
            distance = arc_before + fraction * length
            waypoint = query_waypoint(carla_map, xy, z)
            is_junction = waypoint is not None and waypoint.is_junction
            previous_is_junction = (
                previous_waypoint is not None and previous_waypoint.is_junction
            )
            if is_junction and not previous_is_junction:
                refined_distance, refined_xy = refine_false_to_true(
                    carla_map, previous, (distance, xy), z
                )
                refined_waypoint = query_waypoint(carla_map, refined_xy, z)
                return refined_distance, refined_xy, refined_waypoint
            previous = (distance, xy)
            previous_waypoint = waypoint
        arc_before += length
    return None, None, None


def trace_records(path):
    with open(path, "rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                yield json.loads(raw_line)
            except Exception as error:
                raise ValueError(
                    "invalid trace JSON at line %d: %s" % (line_number, error)
                )


def raw_plan_world_xy(row):
    """Return the unmodified world-space plan and its trace field.

    The rejected v3 oracle stored the unmodified conflict calculation under
    ``planning_response.conflict``.  The v4 runtime made the provenance
    explicit by renaming that field to ``raw_conflict``.  Geometry must always
    be queried from the unmodified plan, never from an effective/suppressed
    conflict or an intervened target trajectory.
    """

    response = row.get("planning_response") or {}
    for source in ("raw_conflict", "conflict"):
        payload = response.get(source) or {}
        base_world = payload.get("base_plan_world_xy")
        if isinstance(base_world, list) and base_world:
            return base_world, source
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--xodr", required=True)
    parser.add_argument("--map-name", default="Town05")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution-m", type=float, default=0.1)
    args = parser.parse_args()
    if args.resolution_m <= 0:
        raise ValueError("resolution must be positive")

    with open(args.xodr, "rb") as handle:
        xodr = handle.read()
    carla_map = carla.Map(args.map_name, xodr)
    records = []
    for row in trace_records(args.trace):
        safety = row.get("closedloop_safety") or {}
        ego = safety.get("ego") or {}
        ego_xy = ego.get("position_xy")
        base_world, base_world_source = raw_plan_world_xy(row)
        if not isinstance(ego_xy, list) or len(ego_xy) != 2:
            raise ValueError("trace step %s has no ego XY" % row.get("step"))
        if not isinstance(base_world, list) or not base_world:
            raise ValueError(
                "trace step %s has no base world plan" % row.get("step")
            )
        points = [tuple(ego_xy)] + [tuple(point) for point in base_world]
        z = float(ego.get("position_z", 0.0))
        ego_waypoint = query_waypoint(carla_map, points[0], z)
        entry_distance, entry_xy, entry_waypoint = first_junction_entry(
            carla_map, points, z, args.resolution_m
        )
        path_length = sum(
            math.hypot(second[0] - first[0], second[1] - first[1])
            for first, second in zip(points, points[1:])
        )
        records.append({
            "step": int(row["step"]),
            "sim_time_seconds": float(row["sim_time_seconds"]),
            "ego_world_xy": [float(ego_xy[0]), float(ego_xy[1])],
            "ego_map_waypoint": waypoint_payload(ego_waypoint),
            "base_path_length_m": path_length,
            "base_plan_world_xy_source": base_world_source,
            "junction_entry_path_distance_m": entry_distance,
            "junction_entry_world_xy": (
                list(entry_xy) if entry_xy is not None else None
            ),
            "junction_entry_map_waypoint": waypoint_payload(entry_waypoint),
        })

    payload = {
        "schema_version": SCHEMA_VERSION,
        "map_name": args.map_name,
        "resolution_m": args.resolution_m,
        "trace_path": os.path.abspath(args.trace),
        "trace_sha256": sha256_file(args.trace),
        "xodr_path": os.path.abspath(args.xodr),
        "xodr_sha256": sha256_file(args.xodr),
        "record_count": len(records),
        "records": records,
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open(args.output, "wb") as handle:
        handle.write((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({
        "output": os.path.abspath(args.output),
        "record_count": len(records),
        "trace_sha256": payload["trace_sha256"],
        "xodr_sha256": payload["xodr_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
