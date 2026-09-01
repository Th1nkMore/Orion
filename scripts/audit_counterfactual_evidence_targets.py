#!/usr/bin/env python3
"""Audit train-only counterfactual targets before adapter optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uq_estimator.counterfactual_evidence_training import (  # noqa: E402
    audit_train_target_distribution,
    records_from_counterfactual_shard,
    select_records,
)


SCHEMA_VERSION = "orion.counterfactual-evidence-target-audit/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-shard", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--artifact-sha256", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--response-floor", type=float, default=1e-6)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("refusing to overwrite target audit")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA target audit requested but unavailable")

    protocol_sha = _sha256(args.protocol)
    artifact_line = args.artifact_sha256.read_text(encoding="utf-8").strip().split()
    if len(artifact_line) < 2 or not artifact_line[0]:
        raise RuntimeError("invalid extraction artifact hash sidecar")
    declared_feature_sha = artifact_line[0]
    declared_path = Path(artifact_line[-1])
    if declared_path.resolve() != args.feature_shard.resolve():
        raise RuntimeError("artifact hash sidecar names another feature shard")

    # Loading the 66 GiB shard dominates runtime.  The extraction job already
    # computed its SHA-256 sidecar; provenance and the exact sidecar are carried
    # into this read-only report rather than hashing the artifact a second time.
    payload = torch.load(args.feature_shard, map_location="cpu")
    provenance = payload.get("provenance", {})
    if provenance.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("feature shard protocol lineage changed")
    records = records_from_counterfactual_shard(payload)
    train = select_records(records, ["train"], ["local_blur", "local_dark"])
    if len(train) != 2240 or len({record.route_id for record in train}) != 35:
        raise RuntimeError("frozen train-only target population changed")

    audit = audit_train_target_distribution(
        train,
        torch.device(args.device),
        batch_size=args.batch_size,
        quantile=args.quantile,
        response_floor=args.response_floor,
    )
    monotonic_values = [
        family["combined"]
        for family in audit["paired_severity_higher_target_fraction"].values()
    ]
    diagnostics = {
        "all_components_have_responsive_cells": all(
            row["responsive_cell_count"] > 0
            for row in audit["components"].values()
        ),
        "all_selected_scales_above_numeric_floor": all(
            value > 1e-4 for value in audit["component_scales"].values()
        ),
        "both_optimizer_families_have_majority_combined_monotonic_pairs": (
            len(monotonic_values) == 2 and min(monotonic_values) > 0.5
        ),
    }
    extraction_version = provenance.get("extraction_schema_version")
    schedule_audit = None
    if extraction_version == "orion.counterfactual-evidence-extraction/v2":
        unique_views = provenance.get(
            "view_schedule_unique_views_per_route_condition", {}
        )
        schedule_audit = {
            "view_schedule": provenance.get("view_schedule"),
            "route_condition_count": len(unique_views),
            "minimum_unique_views_per_route_condition": (
                min(int(value) for value in unique_views.values())
                if unique_views
                else 0
            ),
            "maximum_unique_views_per_route_condition": (
                max(int(value) for value in unique_views.values())
                if unique_views
                else 0
            ),
            "view_counts": provenance.get("view_schedule_counts", {}),
            "exact_nonzero_presence_label_authorized": provenance.get(
                "exact_nonzero_presence_label_authorized"
            ),
        }
        diagnostics.update(
            {
                "v2_every_route_condition_covers_at_least_four_views": (
                    bool(unique_views) and min(unique_views.values()) >= 4
                ),
                "v2_exact_nonzero_presence_label_is_disabled": (
                    provenance.get("exact_nonzero_presence_label_authorized")
                    is False
                ),
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "train-only paired-target distribution; no adapter optimization",
        "inputs": {
            "feature_shard": str(args.feature_shard.resolve()),
            "feature_shard_sha256_from_extraction_sidecar": declared_feature_sha,
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": protocol_sha,
        },
        "audit": audit,
        "schedule_audit": schedule_audit,
        "diagnostics": diagnostics,
        "diagnostics_passed": all(diagnostics.values()),
        "continuation": {
            "adapter_training_started": False,
            "adapter_training_authorized_by_this_report": False,
            "required_next_step": (
                "freeze a separate training-run config using this exact artifact "
                "hash, scale population, scales, and record counts"
            ),
            "orion_finetuning_authorized": False,
            "stage_b_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=True), flush=True)
    print("COUNTERFACTUAL_EVIDENCE_TARGET_AUDIT_OK=1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
