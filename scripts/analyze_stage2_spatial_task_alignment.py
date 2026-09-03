#!/usr/bin/env python3
"""Diagnose task alignment of a frozen Stage-1 spatial-UQ capture.

This report deliberately does not assign task-risk semantics to the Stage-1
map.  It measures whether map magnitude happens to align with an independently
computed privileged planning-response window and exposes camera-view bias that
Stage 2 must learn to ignore.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uq_estimator.stage2_artifact_capture import ARTIFACT_INDEX_SCHEMA, sha256_file


SCHEMA_VERSION = "orion.stage2-spatial-task-alignment-diagnostic/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-frames", type=int, default=60)
    return parser.parse_args()


def _load_uq(path: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    value = payload["observation_uq"] if isinstance(payload, dict) else payload
    value = value.detach().float()
    if value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError("observation UQ must have shape [V,H,W,3]")
    if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
        raise ValueError("observation UQ must be finite and non-negative")
    return value


def _binary_auroc(scores: list[float], labels: list[bool]) -> float | None:
    positive = [score for score, label in zip(scores, labels) if label]
    negative = [score for score, label in zip(scores, labels) if not label]
    if not positive or not negative:
        return None
    wins = 0.0
    for pos in positive:
        for neg in negative:
            wins += float(pos > neg) + 0.5 * float(pos == neg)
    return wins / (len(positive) * len(negative))


def _summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _delta(positive: list[float], negative: list[float]) -> dict:
    pos = _summary(positive)
    neg = _summary(negative)
    difference = None
    ratio = None
    if pos["mean"] is not None and neg["mean"] is not None:
        difference = pos["mean"] - neg["mean"]
        if not math.isclose(neg["mean"], 0.0):
            ratio = pos["mean"] / neg["mean"]
    return {
        "response_window": pos,
        "ordinary_go": neg,
        "response_minus_go_mean": difference,
        "response_over_go_mean": ratio,
    }


def main() -> None:
    args = parse_args()
    if args.warmup_frames < 0:
        raise ValueError("warmup frames must be non-negative")
    run_dir = args.run_dir.resolve()
    index_path = run_dir / "stage2_artifacts" / "artifact_index.json"
    index = json.loads(index_path.read_text())
    if index.get("schema_version") != ARTIFACT_INDEX_SCHEMA:
        raise ValueError("unexpected Stage-2 artifact-index schema")
    records = index.get("records") or []
    if index.get("record_count") != len(records) or not records:
        raise ValueError("artifact index is empty or internally inconsistent")
    camera_order = list(index.get("camera_order") or [])
    if "CAM_FRONT" not in camera_order:
        raise ValueError("artifact index lacks CAM_FRONT")
    front_index = camera_order.index("CAM_FRONT")
    trace_paths = list(run_dir.rglob("control_trace.jsonl"))
    if len(trace_paths) != 1:
        raise ValueError("run must contain exactly one control trace")
    trace_path = trace_paths[0]
    trace = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    rows_by_step = {int(row["step"]): row for row in trace}

    samples = []
    dominant_view_counts = {name: 0 for name in camera_order}
    for record in records:
        step = int(record["step"])
        if step < args.warmup_frames:
            continue
        row = rows_by_step.get(step)
        if row is None:
            raise ValueError(f"trace lacks captured step {step}")
        uq = _load_uq(record["observation_uq_path"])
        if uq.shape[0] != len(camera_order):
            raise ValueError("UQ view count differs from camera-order provenance")
        scalar = uq.mean(dim=-1)
        view_mean = scalar.mean(dim=(1, 2))
        view_q95 = torch.quantile(scalar.flatten(1), 0.95, dim=1)
        dominant_view_counts[camera_order[int(torch.argmax(view_mean))]] += 1
        state = ((row.get("planning_response") or {}).get("yield_label") or {}).get("state")
        if state not in {"go", "prepare_yield", "hold", "release"}:
            raise ValueError(f"captured step {step} lacks a valid planning-response state")
        samples.append({
            "step": step,
            "sim_time_seconds": float(row["sim_time_seconds"]),
            "route_progress": float(row["route_progress"]),
            "yield_state": state,
            "response_window": state != "go",
            "global_mean": float(scalar.mean()),
            "global_q95": float(torch.quantile(scalar, 0.95)),
            "front_mean": float(view_mean[front_index]),
            "front_q95": float(view_q95[front_index]),
            "per_view_mean": {
                name: float(value) for name, value in zip(camera_order, view_mean)
            },
        })
    if not samples:
        raise ValueError("no post-warmup records available")

    labels = [bool(sample["response_window"]) for sample in samples]
    metrics = ("global_mean", "global_q95", "front_mean", "front_q95")
    response_alignment = {}
    for metric in metrics:
        scores = [float(sample[metric]) for sample in samples]
        response_alignment[metric] = {
            "auroc_for_privileged_response_window": _binary_auroc(scores, labels),
            **_delta(
                [score for score, label in zip(scores, labels) if label],
                [score for score, label in zip(scores, labels) if not label],
            ),
        }
    per_view = {}
    for name in camera_order:
        values = [float(sample["per_view_mean"][name]) for sample in samples]
        per_view[name] = {
            **_summary(values),
            "response_alignment": _delta(
                [value for value, label in zip(values, labels) if label],
                [value for value, label in zip(values, labels) if not label],
            ),
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": (
            "Single-route diagnostic only. Privileged response is not an uncertainty "
            "target, and association here does not establish semantic correctness or safety."
        ),
        "run_dir": str(run_dir),
        "artifact_index_path": str(index_path),
        "artifact_index_sha256": sha256_file(index_path),
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "warmup_frames": args.warmup_frames,
        "sample_count": len(samples),
        "response_sample_count": sum(labels),
        "go_sample_count": len(labels) - sum(labels),
        "response_alignment": response_alignment,
        "per_view_mean": per_view,
        "dominant_view_counts": dominant_view_counts,
        "samples": samples,
    }
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite task-alignment diagnostic")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
