#!/usr/bin/env python3
"""Capture exact-pose six-view/BEV pairs under native CARLA weather."""

from __future__ import annotations

import argparse
import json
import math
import queue
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


CAMERA_SPECS = (
    ("CAM_FRONT", "rgb_front", 0.80, 0.0, 1.60, 0.0, 1600, 900, 70.0),
    ("CAM_FRONT_LEFT", "rgb_front_left", 0.27, -0.55, 1.60, -55.0, 1600, 900, 70.0),
    ("CAM_FRONT_RIGHT", "rgb_front_right", 0.27, 0.55, 1.60, 55.0, 1600, 900, 70.0),
    ("CAM_BACK", "rgb_back", -2.0, 0.0, 1.60, 180.0, 1600, 900, 110.0),
    ("CAM_BACK_LEFT", "rgb_back_left", -0.32, -0.55, 1.60, -110.0, 1600, 900, 70.0),
    ("CAM_BACK_RIGHT", "rgb_back_right", -0.32, 0.55, 1.60, 110.0, 1600, 900, 70.0),
    ("bev", "bev", 0.0, 0.0, 50.0, 0.0, 512, 512, 50.0),
)


def _route_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("route must use ROUTE_ID=XML_PATH")
    route_id, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not route_id.strip() or not path.is_file():
        raise argparse.ArgumentTypeError("route ID/path is invalid")
    return route_id.strip(), path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--route", action="append", type=_route_arg, required=True)
    parser.add_argument("--positions-per-route", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--renderer-quality", choices=("Low", "Epic"), required=True
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    return parser


def _load_route(path, route_id):
    root = ET.parse(str(path)).getroot()
    routes = list(root.iter("route"))
    if len(routes) != 1:
        raise RuntimeError("%s must contain exactly one route" % path)
    route = routes[0]
    town = route.attrib.get("town", "")
    points = []
    for node in route.find("waypoints") or ():
        points.append(
            (
                float(node.attrib["x"]),
                float(node.attrib["y"]),
                float(node.attrib["z"]),
            )
        )
    if not town or len(points) < 3:
        raise RuntimeError("route %s lacks town/waypoints" % route_id)
    return {"route_id": route_id, "town": town, "points": points, "xml": str(path.resolve())}


def _select_pose_window(points, count):
    if count < 3 or len(points) < count:
        raise RuntimeError("positions-per-route must be >=3 and fit every route")
    start = (len(points) - count) // 2
    selected = points[start : start + count]
    poses = []
    for index, point in enumerate(selected):
        neighbour = selected[min(index + 1, len(selected) - 1)]
        if neighbour == point and index:
            neighbour = selected[index - 1]
            yaw = math.degrees(math.atan2(point[1] - neighbour[1], point[0] - neighbour[0]))
        else:
            yaw = math.degrees(math.atan2(neighbour[1] - point[1], neighbour[0] - point[0]))
        poses.append({"x": point[0], "y": point[1], "z": point[2], "yaw": yaw})
    return start, poses


def _weather_specs(carla):
    common = dict(
        cloudiness=0.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=0.0,
        sun_azimuth_angle=-1.0,
        sun_altitude_angle=45.0,
        wetness=0.0,
        fog_distance=0.0,
        fog_falloff=0.2,
        scattering_intensity=1.0,
        mie_scattering_scale=0.03,
        rayleigh_scattering_scale=0.0331,
    )
    result = {}
    for name, severity, fog_density in (
        ("clear", 0.0, 0.0),
        ("fog_light", 1.0, 25.0),
        ("fog_heavy", 3.0, 75.0),
    ):
        values = dict(common)
        values["fog_density"] = fog_density
        result[name] = {
            "severity": severity,
            "renderer": "CARLA-0.9.15-native-weather",
            "parameters": values,
            "weather": carla.WeatherParameters(**values),
        }
    return result


class _SensorRig:
    def __init__(self, carla, world, vehicle):
        self._queues = {}
        self.actors = []
        library = world.get_blueprint_library()
        for sensor_id, directory, x, y, z, yaw, width, height, fov in CAMERA_SPECS:
            blueprint = library.find("sensor.camera.rgb")
            blueprint.set_attribute("image_size_x", str(width))
            blueprint.set_attribute("image_size_y", str(height))
            blueprint.set_attribute("fov", str(fov))
            transform = carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(pitch=-90.0 if sensor_id == "bev" else 0.0, yaw=yaw),
            )
            actor = world.spawn_actor(blueprint, transform, attach_to=vehicle)
            stream = queue.Queue()
            actor.listen(stream.put)
            self.actors.append(actor)
            self._queues[sensor_id] = (directory, stream)

    def collect(self, frame, timeout):
        result = {}
        deadline = time.time() + timeout
        for sensor_id, (directory, stream) in self._queues.items():
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise RuntimeError("sensor timeout at frame %s" % frame)
                image = stream.get(timeout=remaining)
                if image.frame < frame:
                    continue
                if image.frame != frame:
                    raise RuntimeError("sensor skipped requested frame %s" % frame)
                result[sensor_id] = (directory, image)
                break
        return result

    def destroy(self):
        for actor in self.actors:
            actor.stop()
            actor.destroy()


def _spawn_rig(carla, world, initial_pose):
    library = world.get_blueprint_library()
    candidates = library.filter("vehicle.tesla.model3") or library.filter("vehicle.*")
    if not candidates:
        raise RuntimeError("CARLA has no vehicle blueprint")
    pose = carla.Transform(
        carla.Location(x=initial_pose["x"], y=initial_pose["y"], z=initial_pose["z"] + 0.25),
        carla.Rotation(yaw=initial_pose["yaw"]),
    )
    vehicle = world.spawn_actor(candidates[0], pose)
    vehicle.set_simulate_physics(False)
    return vehicle, _SensorRig(carla, world, vehicle)


def _capture_route(
    carla, client, route, output, positions_per_route, timeout, fixed_delta_seconds
):
    world = client.load_world(route["town"])
    settings = world.get_settings()
    original_sync = settings.synchronous_mode
    original_delta = settings.fixed_delta_seconds
    original_no_rendering = settings.no_rendering_mode
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    start_index, poses = _select_pose_window(route["points"], positions_per_route)
    vehicle = None
    rig = None
    items = []
    weather_specs = _weather_specs(carla)
    try:
        vehicle, rig = _spawn_rig(carla, world, poses[0])
        # Warm up every sensor once after spawning.
        rig.collect(world.tick(), timeout)
        for condition, weather in weather_specs.items():
            world.set_weather(weather["weather"])
            for _ in range(3):
                rig.collect(world.tick(), timeout)
            for sequence_index, pose in enumerate(poses):
                transform = carla.Transform(
                    carla.Location(x=pose["x"], y=pose["y"], z=pose["z"] + 0.25),
                    carla.Rotation(yaw=pose["yaw"]),
                )
                vehicle.set_transform(transform)
                rig.collect(world.tick(), timeout)
                captured = rig.collect(world.tick(), timeout)
                for sensor_id, (directory, image) in captured.items():
                    target = output / condition / route["route_id"].replace("/", "_") / directory / ("%04d.png" % sequence_index)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    image.save_to_disk(str(target))
                if condition == "clear":
                    items.append(
                        {
                            "sample_id": "%s/%04d" % (route["route_id"], sequence_index),
                            "route_id": route["route_id"],
                            "town": route["town"],
                            "sequence_index": sequence_index,
                            "source_waypoint_index": start_index + sequence_index,
                            "pose": pose,
                        }
                    )
    finally:
        if rig is not None:
            rig.destroy()
        if vehicle is not None:
            vehicle.destroy()
        restore = world.get_settings()
        restore.synchronous_mode = original_sync
        restore.fixed_delta_seconds = original_delta
        restore.no_rendering_mode = original_no_rendering
        world.apply_settings(restore)
    return items, {
        name: {key: value for key, value in row.items() if key != "weather"}
        for name, row in weather_specs.items()
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    if args.positions_per_route < 3 or args.timeout <= 0 or args.fixed_delta_seconds <= 0:
        raise SystemExit("capture guard values are invalid")
    if len({route_id for route_id, _ in args.route}) != len(args.route):
        raise SystemExit("route IDs must be unique")
    args.output.mkdir(parents=True)
    try:
        import carla

        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        routes = [_load_route(path, route_id) for route_id, path in args.route]
        all_items = []
        conditions = None
        for route in routes:
            items, route_conditions = _capture_route(
                carla,
                client,
                route,
                args.output,
                args.positions_per_route,
                args.timeout,
                args.fixed_delta_seconds,
            )
            all_items.extend(items)
            if conditions is None:
                conditions = route_conditions
            elif conditions != route_conditions:
                raise RuntimeError("native weather definition changed between routes")
        manifest = {
            "schema_version": "orion.native-carla-weather-capture/v1",
            "carla_client_version": client.get_client_version(),
            "conditions": conditions,
            "routes": [{key: value for key, value in route.items() if key != "points"} for route in routes],
            "items": all_items,
            "camera_order": [row[0] for row in CAMERA_SPECS[:6]],
            "renderer_quality": args.renderer_quality,
            "paired_world_pose": True,
            "pixel_corruption_generator_used": False,
            "writes_performed": True,
        }
        manifest_path = args.output / "capture_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output), "item_count": len(all_items), "manifest": str(manifest_path)}, indent=2))
        print("NATIVE_CARLA_WEATHER_CAPTURE_OK=1")
        return 0
    except Exception:
        # Never leave a partial directory that can be mistaken for a complete capture.
        shutil.rmtree(str(args.output), ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
