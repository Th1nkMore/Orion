#!/usr/bin/env python3
"""Run repeated real Qwen-Drive inferences from three Bench2Drive images."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_PATH = PROJECT_ROOT / "uq_estimator" / "qwen_drive_bridge.py"


def _load_bridge():
    spec = importlib.util.spec_from_file_location("_qwen_drive_bridge_smoke", BRIDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _jpeg_versions(path, history_size, current_size, quality):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    encoded = []
    for target in (history_size, current_size):
        resized = image.resize(tuple(target), resample=Image.Resampling.BICUBIC)
        stream = io.BytesIO()
        resized.save(stream, format="JPEG", quality=int(quality))
        encoded.append(stream.getvalue())
    return encoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--front-left", type=Path, required=True)
    parser.add_argument("--front-right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="number of inferences in one loaded sidecar (default: 2)",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite %s" % args.output)
    if args.repeat < 2:
        raise ValueError("--repeat must be at least 2 to measure cold and warm latency")

    bridge = _load_bridge()
    config = bridge.load_bridge_config(args.config)
    image_config = config["images"]
    source_by_sensor = {
        "CAM_FRONT": args.front.resolve(),
        "CAM_FRONT_LEFT": args.front_left.resolve(),
        "CAM_FRONT_RIGHT": args.front_right.resolve(),
    }
    views = {}
    input_hashes = {}
    for sensor_id, view_name in bridge.QWEN_VIEW_BY_SENSOR.items():
        source = source_by_sensor[sensor_id]
        low, current = _jpeg_versions(
            source,
            image_config["history_size"],
            image_config["current_size"],
            image_config["jpeg_quality"],
        )
        views[view_name] = [low, low, low, current]
        input_hashes[sensor_id] = hashlib.sha256(source.read_bytes()).hexdigest()

    ego = {
        "history": np.zeros((16, 3), dtype=np.float32),
        "history_velocity": np.zeros((16, 2), dtype=np.float32),
        "history_acceleration": np.zeros((16, 2), dtype=np.float32),
        "ego_velocity": np.zeros(2, dtype=np.float32),
        "ego_acceleration": np.zeros(2, dtype=np.float32),
    }
    request = bridge.make_inference_request(
        views,
        ego,
        [0.0, 1.0, 0.0, 0.0],
        nav_command=0,
        token="bench2drive-bridge-real-smoke",
    )
    client = bridge.QwenDriveClient(args.config, config)
    log_path = args.output.with_suffix(".sidecar.log")
    inference_runs = []
    try:
        client.start(log_path)
        for index in range(args.repeat):
            request["token"] = "bench2drive-bridge-real-smoke-%02d" % index
            trajectory, inference_seconds = client.infer(request)
            inference_runs.append(
                {
                    "index": index,
                    "inference_seconds": inference_seconds,
                    "runtime_metrics": dict(client.last_metrics or {}),
                }
            )
    finally:
        client.close()

    world = bridge.qwen_trajectory_to_world(trajectory, [0.0, 0.0, 0.0])
    pid = bridge.world_trajectory_to_pid(
        world,
        [0.0, 0.0, 0.0],
        elapsed_seconds=0.0,
        source_hz=int(config["planning"]["source_hz"]),
        target_hz=int(config["planning"]["controller_hz"]),
        horizon_points=int(config["planning"]["controller_horizon_points"]),
    )
    report = {
        "schema": "orion.qwen-drive-bench2drive-bridge-smoke/v2",
        "status": "complete",
        "config": str(args.config.resolve()),
        "input_sha256": input_hashes,
        "inference_runs": inference_runs,
        "cold_inference_seconds": inference_runs[0]["inference_seconds"],
        "warm_inference_seconds": inference_runs[-1]["inference_seconds"],
        "runtime_metrics": inference_runs[-1]["runtime_metrics"],
        "trajectory_shape": list(trajectory.shape),
        "trajectory_10hz_x_forward_y_left_heading": trajectory.tolist(),
        "pid_2hz_right_forward": pid.tolist(),
        "sidecar_log": str(log_path.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "inference_seconds": [run["inference_seconds"] for run in inference_runs],
                "runtime_metrics": inference_runs[-1]["runtime_metrics"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
