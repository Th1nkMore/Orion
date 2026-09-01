#!/usr/bin/env python3
"""Audit frozen temporal observation-UQ on native CARLA fog pairs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uq_estimator.native_weather_audit import (  # noqa: E402
    audit_native_weather_features,
)
from uq_estimator.observation_uq_shard import (  # noqa: E402
    examples_from_feature_shard,
    load_feature_shard,
)
from uq_estimator.observation_uq_signal_audit import (  # noqa: E402
    fit_clean_position_calibrator,
    temporal_cosine_residual,
)
from uq_estimator.observation_uq_v3 import _batches, _collate  # noqa: E402


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_temporal_maps(examples, batch_size):
    result = {}
    for batch in _batches(examples, batch_size, False, 0):
        current, previous, valid = _collate(batch, torch.device("cpu"))
        scores = temporal_cosine_residual(current, previous, valid)
        for index, item in enumerate(batch):
            result[item.sample_id] = scores[index].cpu().float()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-calibration-shard", type=Path, required=True)
    parser.add_argument("--native-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    if args.batch_size <= 0:
        raise SystemExit("batch size must be positive")

    calibration_payload = load_feature_shard(args.clean_calibration_shard)
    examples = examples_from_feature_shard(calibration_payload)
    clean_train = [
        item for item in examples if item.split == "train" and item.family == "clean"
    ]
    if len(clean_train) != 560 or len({item.route_id for item in clean_train}) != 35:
        raise RuntimeError("native gate requires the frozen 560-frame/35-route calibrator")
    clean_maps = _clean_temporal_maps(clean_train, args.batch_size)
    calibrator = fit_clean_position_calibrator(
        clean_maps, clean_train, tail="positive"
    )
    del clean_maps, clean_train, examples, calibration_payload
    gc.collect()

    native_payload = torch.load(args.native_features, map_location="cpu")
    report = audit_native_weather_features(native_payload, calibrator)
    report["inputs"] = {
        "clean_calibration_shard": str(args.clean_calibration_shard.resolve()),
        "clean_calibration_shard_sha256": _sha256(args.clean_calibration_shard),
        "native_features": str(args.native_features.resolve()),
        "native_features_sha256": _sha256(args.native_features),
    }
    report["calibration"] = {
        "source": "temporal_raw",
        "tail": calibrator.tail,
        "example_count": calibrator.example_count,
        "route_count": 35,
        "families": ["clean"],
        "view_position_shape": list(calibrator.center.shape),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "candidate_passed": report["candidate_gate"]["passed"],
                "sample_severity_spearman": report["sample_severity_spearman"],
                "condition_metrics": report["condition_metrics"],
            },
            indent=2,
        )
    )
    print("NATIVE_CARLA_WEATHER_UQ_AUDIT_OK=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
