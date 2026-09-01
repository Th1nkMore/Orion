#!/usr/bin/env python3
"""Audit route-balanced clean-manifold scores on frozen Epic weather features."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uq_estimator.native_manifold_audit import (  # noqa: E402
    MANIFOLD_CANDIDATE_TAILS,
    audit_native_manifold_score_maps,
    route_balanced_manifold_raw_maps,
)
from uq_estimator.native_weather_audit import (  # noqa: E402
    EXPECTED_CONDITIONS,
    validate_native_weather_payload,
)
from uq_estimator.observation_uq_shard import load_feature_shard  # noqa: E402
from uq_estimator.observation_uq_signal_audit import (  # noqa: E402
    fit_clean_position_calibrator,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_calibrators(raw_clean, sample_ids):
    lightweight_examples = [
        SimpleNamespace(sample_id=sample_id, family="clean")
        for sample_id in sample_ids
    ]
    return {
        name: fit_clean_position_calibrator(
            {
                sample_id: maps[index]
                for index, sample_id in enumerate(sample_ids)
            },
            lightweight_examples,
            tail=MANIFOLD_CANDIDATE_TAILS[name],
        )
        for name, maps in raw_clean.items()
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-calibration-shard", type=Path, required=True)
    parser.add_argument("--native-features", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA manifold audit requested but unavailable")

    registration_bytes = args.config.read_bytes()
    registration = json.loads(registration_bytes.decode("utf-8"))
    if registration.get("schema_version") != (
        "orion.observation-uq-native-manifold-candidates/v1"
    ):
        raise RuntimeError("unexpected native manifold registration")
    if set(registration.get("candidates", {})) != set(MANIFOLD_CANDIDATE_TAILS):
        raise RuntimeError("registered native manifold candidate set changed")
    reference = registration["reference"]
    nearest_route_count = int(reference["nearest_route_count"])
    position_chunk_size = int(reference["position_chunk_size"])

    clean_sha = _sha256(args.clean_calibration_shard)
    native_sha = _sha256(args.native_features)
    if clean_sha != registration["inputs"]["clean_feature_sha256"]:
        raise RuntimeError("clean feature shard differs from preregistration")
    if native_sha != registration["inputs"]["native_feature_sha256"]:
        raise RuntimeError("native feature shard differs from preregistration")

    started = time.time()
    print("[NativeManifold] loading immutable clean shard", flush=True)
    clean_payload = load_feature_shard(args.clean_calibration_shard)
    selected = [
        (item, feature)
        for item, feature in zip(
            clean_payload["clean_items"], clean_payload["clean_features"]
        )
        if item["split"] == "train"
    ]
    if len(selected) != 560 or len({item[0]["route_id"] for item in selected}) != 35:
        raise RuntimeError("manifold audit requires frozen 560-frame/35-route clean data")
    clean_route_ids = [str(item[0]["route_id"]) for item in selected]
    clean_sample_ids = [str(item[0]["sample_id"]) + "/clean" for item in selected]
    clean = torch.stack([item[1] for item in selected])
    del selected, clean_payload
    gc.collect()

    print("[NativeManifold] loading immutable Epic native-weather shard", flush=True)
    native_payload = torch.load(args.native_features, map_location="cpu")
    validate_native_weather_payload(native_payload)
    condition_sizes = [
        int(native_payload["features_by_condition"][name].shape[0])
        for name in EXPECTED_CONDITIONS
    ]
    native_queries = torch.cat(
        [native_payload["features_by_condition"][name] for name in EXPECTED_CONDITIONS],
        dim=0,
    )
    combined_queries = torch.cat((clean, native_queries), dim=0)
    print(
        "[NativeManifold] computing leave-route-out clean calibration and native maps",
        flush=True,
    )
    combined_raw = route_balanced_manifold_raw_maps(
        clean,
        combined_queries,
        clean_route_ids,
        nearest_route_count=nearest_route_count,
        position_chunk_size=position_chunk_size,
        query_route_ids=clean_route_ids + [None] * int(native_queries.shape[0]),
        leave_query_route_out=True,
        device=args.device,
    )
    raw_clean = {
        name: maps[: len(clean_sample_ids)] for name, maps in combined_raw.items()
    }
    raw_native = {
        name: maps[len(clean_sample_ids) :] for name, maps in combined_raw.items()
    }
    calibrators = _fit_calibrators(raw_clean, clean_sample_ids)
    del combined_raw, combined_queries, native_queries, raw_clean, clean
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    scores = {name: {} for name in MANIFOLD_CANDIDATE_TAILS}
    for candidate, raw_maps in raw_native.items():
        start = 0
        for condition, count in zip(EXPECTED_CONDITIONS, condition_sizes):
            maps = raw_maps[start : start + count]
            scores[candidate][condition] = torch.stack(
                [calibrators[candidate].transform(item) for item in maps]
            )
            start += count
    report = audit_native_manifold_score_maps(native_payload, scores)
    report["inputs"] = {
        "clean_calibration_shard": str(args.clean_calibration_shard.resolve()),
        "clean_calibration_shard_sha256": clean_sha,
        "native_features": str(args.native_features.resolve()),
        "native_features_sha256": native_sha,
        "preregistration": str(args.config.resolve()),
        "preregistration_sha256": hashlib.sha256(registration_bytes).hexdigest(),
    }
    report["reference"] = {
        "clean_example_count": 560,
        "clean_route_count": 35,
        "nearest_route_count": nearest_route_count,
        "position_chunk_size": position_chunk_size,
        "calibration": "leave_one_route_out_then_view_position_median_mad",
        "native_query_reference": "all_clean_routes",
        "compute_device": args.device,
    }
    report["runtime_seconds"] = time.time() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "candidate_passes": report["candidate_passes"],
                "runtime_seconds": report["runtime_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    print("NATIVE_CARLA_MANIFOLD_CANDIDATES_OK=1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
