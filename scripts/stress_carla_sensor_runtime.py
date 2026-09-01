#!/usr/bin/env python3
"""Exercise CARLA with the exact ORION camera suite without loading ORION."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import carla


CAMERAS = (
    ("CAM_FRONT", 0.80, 0.0, 1.60, 0.0, 0.0, 0.0, 1600, 900, 70),
    ("CAM_FRONT_LEFT", 0.27, -0.55, 1.60, 0.0, 0.0, -55.0, 1600, 900, 70),
    ("CAM_FRONT_RIGHT", 0.27, 0.55, 1.60, 0.0, 0.0, 55.0, 1600, 900, 70),
    ("CAM_BACK", -2.0, 0.0, 1.60, 0.0, 0.0, 180.0, 1600, 900, 110),
    ("CAM_BACK_LEFT", -0.32, -0.55, 1.60, 0.0, 0.0, -110.0, 1600, 900, 70),
    ("CAM_BACK_RIGHT", -0.32, 0.55, 1.60, 0.0, 0.0, 110.0, 1600, 900, 70),
    ("bev", 0.0, 0.0, 50.0, 0.0, -90.0, 0.0, 512, 512, 50),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--map", dest="map_name", default="Town05")
    parser.add_argument("--ticks", type=int, default=1600)
    parser.add_argument("--fixed-delta", type=float, default=0.05)
    parser.add_argument("--rpc-timeout", type=float, default=120.0)
    parser.add_argument("--sensor-timeout", type=float, default=15.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def choose_vehicle_blueprint(world: carla.World) -> carla.ActorBlueprint:
    library = world.get_blueprint_library()
    for blueprint_id in ("vehicle.lincoln.mkz_2020", "vehicle.tesla.model3"):
        matches = library.filter(blueprint_id)
        if matches:
            return matches[0]
    matches = library.filter("vehicle.*")
    if not matches:
        raise RuntimeError("CARLA map exposes no vehicle blueprint")
    return matches[0]


def make_callback(
    name: str,
    condition: threading.Condition,
    seen_frames: dict[str, int],
    counts: dict[str, int],
    checksums: dict[str, int],
) -> Callable[[Any], None]:
    def callback(data: Any) -> None:
        checksum = 0
        raw = getattr(data, "raw_data", None)
        if raw:
            checksum = int(raw[0]) + int(raw[len(raw) // 2]) + int(raw[-1])
        with condition:
            seen_frames[name] = int(data.frame)
            counts[name] += 1
            checksums[name] = (checksums[name] + checksum) % 1_000_000_007
            condition.notify_all()

    return callback


def wait_for_sensor_frame(
    frame: int,
    sensor_names: list[str],
    condition: threading.Condition,
    seen_frames: dict[str, int],
    timeout: float,
) -> float:
    started = time.monotonic()
    deadline = started + timeout
    with condition:
        while True:
            missing = [name for name in sensor_names if seen_frames.get(name, -1) < frame]
            if not missing:
                return time.monotonic() - started
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    f"sensor frame {frame} timed out after {timeout:.1f}s; "
                    f"missing={missing}, seen={seen_frames}"
                )
            condition.wait(timeout=remaining)


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.ticks <= 0 or args.fixed_delta <= 0.0:
        raise ValueError("ticks and fixed-delta must be positive")

    payload: dict[str, Any] = {
        "schema": "orion.carla_sensor_runtime_gate.v1",
        "status": "running",
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "map_requested": args.map_name,
        "ticks_requested": args.ticks,
        "fixed_delta_seconds": args.fixed_delta,
        "camera_specs": [
            {
                "id": item[0],
                "transform": list(item[1:7]),
                "width": item[7],
                "height": item[8],
                "fov": item[9],
            }
            for item in CAMERAS
        ],
    }
    write_result(args.output, payload)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.rpc_timeout)
    world = None
    vehicle = None
    sensors: list[carla.Actor] = []
    original_settings = None
    wall_started = time.monotonic()
    try:
        world = client.load_world(args.map_name)
        payload["client_version"] = client.get_client_version()
        payload["server_version"] = client.get_server_version()
        payload["map_loaded"] = world.get_map().name

        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta
        settings.no_rendering_mode = False
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        world.apply_settings(settings)
        world.set_weather(
            carla.WeatherParameters(
                cloudiness=100.0,
                fog_density=50.0,
                precipitation=60.0,
                precipitation_deposits=60.0,
                sun_altitude_angle=45.0,
                sun_azimuth_angle=-1.0,
                wetness=0.0,
                wind_intensity=60.0,
            )
        )

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("map exposes no vehicle spawn point")
        vehicle_blueprint = choose_vehicle_blueprint(world)
        if vehicle_blueprint.has_attribute("role_name"):
            vehicle_blueprint.set_attribute("role_name", "hero")
        for spawn in spawn_points:
            vehicle = world.try_spawn_actor(vehicle_blueprint, spawn)
            if vehicle is not None:
                break
        if vehicle is None:
            raise RuntimeError("failed to spawn ego vehicle at every map spawn point")
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

        condition = threading.Condition()
        names = [item[0] for item in CAMERAS] + ["IMU", "GPS"]
        seen_frames = {name: -1 for name in names}
        counts = {name: 0 for name in names}
        checksums = {name: 0 for name in names}
        library = world.get_blueprint_library()

        for name, x, y, z, roll, pitch, yaw, width, height, fov in CAMERAS:
            blueprint = library.find("sensor.camera.rgb")
            blueprint.set_attribute("image_size_x", str(width))
            blueprint.set_attribute("image_size_y", str(height))
            blueprint.set_attribute("fov", str(fov))
            blueprint.set_attribute("sensor_tick", "0.0")
            transform = carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(roll=roll, pitch=pitch, yaw=yaw),
            )
            sensor = world.spawn_actor(blueprint, transform, attach_to=vehicle)
            sensor.listen(make_callback(name, condition, seen_frames, counts, checksums))
            sensors.append(sensor)

        auxiliary = (
            ("IMU", "sensor.other.imu", 0.05),
            ("GPS", "sensor.other.gnss", 0.01),
        )
        for name, blueprint_id, sensor_tick in auxiliary:
            blueprint = library.find(blueprint_id)
            blueprint.set_attribute("sensor_tick", str(sensor_tick))
            transform = carla.Transform(carla.Location(x=-1.4, y=0.0, z=0.0))
            sensor = world.spawn_actor(blueprint, transform, attach_to=vehicle)
            sensor.listen(make_callback(name, condition, seen_frames, counts, checksums))
            sensors.append(sensor)

        tick_latencies: list[float] = []
        sensor_wait_latencies: list[float] = []
        first_frame = None
        last_frame = None
        for index in range(args.ticks):
            tick_started = time.monotonic()
            frame = int(world.tick(args.rpc_timeout))
            tick_latencies.append(time.monotonic() - tick_started)
            sensor_wait_latencies.append(
                wait_for_sensor_frame(
                    frame, names, condition, seen_frames, args.sensor_timeout
                )
            )
            first_frame = frame if first_frame is None else first_frame
            last_frame = frame
            if index == 0 or (index + 1) % 100 == 0:
                print(
                    "SENSOR_GATE "
                    f"tick={index + 1}/{args.ticks} frame={frame} "
                    f"tick_p50={statistics.median(tick_latencies):.4f}s "
                    f"tick_p99={percentile(tick_latencies, 0.99):.4f}s",
                    flush=True,
                )

        payload.update(
            {
                "status": "passed",
                "ticks_completed": args.ticks,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "simulated_seconds": args.ticks * args.fixed_delta,
                "wall_seconds": time.monotonic() - wall_started,
                "sensor_counts": counts,
                "sensor_checksums": checksums,
                "tick_latency_seconds": {
                    "median": statistics.median(tick_latencies),
                    "p95": percentile(tick_latencies, 0.95),
                    "p99": percentile(tick_latencies, 0.99),
                    "max": max(tick_latencies),
                },
                "sensor_wait_latency_seconds": {
                    "median": statistics.median(sensor_wait_latencies),
                    "p99": percentile(sensor_wait_latencies, 0.99),
                    "max": max(sensor_wait_latencies),
                },
            }
        )
        write_result(args.output, payload)
        print("CARLA_SENSOR_RUNTIME_GATE_OK", flush=True)
        return 0
    except BaseException as error:
        payload.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.monotonic() - wall_started,
            }
        )
        write_result(args.output, payload)
        raise
    finally:
        for sensor in sensors:
            try:
                sensor.stop()
            except BaseException:
                pass
        actors = [actor for actor in sensors + ([vehicle] if vehicle is not None else [])]
        for actor in reversed(actors):
            try:
                actor.destroy()
            except BaseException:
                pass
        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except BaseException:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
