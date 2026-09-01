#!/usr/bin/env python3
"""Audit frozen clean-only appearance candidates on Epic native weather."""

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

from uq_estimator.native_appearance_audit import (  # noqa: E402
    CANDIDATE_TAILS,
    appearance_candidate_raw_maps,
    audit_native_appearance_score_maps,
    fit_clean_appearance_statistics,
)
from uq_estimator.native_weather_audit import validate_native_weather_payload  # noqa: E402
from uq_estimator.observation_uq_shard import (  # noqa: E402
    examples_from_feature_shard,
    load_feature_shard,
)
from uq_estimator.observation_uq_signal_audit import (  # noqa: E402
    fit_clean_position_calibrator,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_maps_for_examples(examples, statistics, batch_size, device):
    result = {name: {} for name in CANDIDATE_TAILS}
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        current = torch.stack([item.current for item in batch]).to(device)
        raw = appearance_candidate_raw_maps(current, statistics)
        for name, maps in raw.items():
            for index, item in enumerate(batch):
                result[name][item.sample_id] = maps[index].cpu().float()
    return result


def _native_scores(payload, statistics, calibrators, batch_size, device):
    result = {
        name: {condition: [] for condition in payload["features_by_condition"]}
        for name in CANDIDATE_TAILS
    }
    for condition, features in payload["features_by_condition"].items():
        for start in range(0, features.shape[0], batch_size):
            raw = appearance_candidate_raw_maps(
                features[start : start + batch_size].to(device), statistics
            )
            for name, maps in raw.items():
                result[name][condition].append(
                    torch.stack(
                        [calibrators[name].transform(item.cpu()) for item in maps]
                    )
                )
    return {
        name: {
            condition: torch.cat(chunks)
            for condition, chunks in conditions.items()
        }
        for name, conditions in result.items()
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-calibration-shard", type=Path, required=True)
    parser.add_argument("--native-features", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    if args.batch_size <= 0:
        raise SystemExit("batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA appearance audit requested but unavailable")
    device = torch.device(args.device)

    registration = json.loads(args.config.read_text())
    if registration.get("schema_version") != "orion.observation-uq-native-appearance-candidates/v1":
        raise RuntimeError("unexpected appearance candidate registration")
    if set(registration.get("candidates", {})) != set(CANDIDATE_TAILS):
        raise RuntimeError("registered appearance candidate set changed")

    calibration_payload = load_feature_shard(args.clean_calibration_shard)
    examples = examples_from_feature_shard(calibration_payload)
    clean_train = [
        item for item in examples if item.split == "train" and item.family == "clean"
    ]
    if len(clean_train) != 560 or len({item.route_id for item in clean_train}) != 35:
        raise RuntimeError("appearance audit requires frozen 560-frame/35-route clean data")
    statistics = fit_clean_appearance_statistics(
        [item.current for item in clean_train], device=device
    )
    raw_clean = _raw_maps_for_examples(
        clean_train, statistics, args.batch_size, device
    )
    calibrators = {
        name: fit_clean_position_calibrator(
            maps, clean_train, tail=CANDIDATE_TAILS[name]
        )
        for name, maps in raw_clean.items()
    }
    del raw_clean, examples, calibration_payload
    gc.collect()

    native_payload = torch.load(args.native_features, map_location="cpu")
    validate_native_weather_payload(native_payload)
    expected_native_sha = registration["inputs"]["native_feature_sha256"]
    native_sha = _sha256(args.native_features)
    if native_sha != expected_native_sha:
        raise RuntimeError("native feature shard differs from preregistration")
    scores = _native_scores(
        native_payload, statistics, calibrators, args.batch_size, device
    )
    report = audit_native_appearance_score_maps(native_payload, scores)
    report["inputs"] = {
        "clean_calibration_shard": str(args.clean_calibration_shard.resolve()),
        "clean_calibration_shard_sha256": _sha256(args.clean_calibration_shard),
        "native_features": str(args.native_features.resolve()),
        "native_features_sha256": native_sha,
        "preregistration": str(args.config.resolve()),
        "preregistration_sha256": _sha256(args.config),
    }
    report["calibration"] = {
        "example_count": statistics.example_count,
        "route_count": 35,
        "families": ["clean"],
        "feature_shape": list(statistics.mean.shape),
        "candidate_tails": CANDIDATE_TAILS,
        "compute_device": args.device,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "candidate_passes": report["candidate_passes"],
            },
            indent=2,
        )
    )
    print("NATIVE_CARLA_APPEARANCE_CANDIDATES_OK=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
