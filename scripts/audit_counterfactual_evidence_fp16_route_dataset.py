#!/usr/bin/env python3
"""Audit train-only targets from the complete direct FP16 route dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uq_estimator.counterfactual_evidence import EVIDENCE_COMPONENTS  # noqa: E402
from uq_estimator.counterfactual_evidence_training import (  # noqa: E402
    audit_target_spatial_support,
    audit_train_target_distribution,
)
from uq_estimator.counterfactual_sharded_dataset import (  # noqa: E402
    FP16_DIRECT_DATASET_SCHEMA_VERSION,
    load_fp16_dataset_manifest,
    load_fp16_dataset_records,
)


TARGET_AUDIT_SCHEMA = "orion.counterfactual-evidence-target-audit/v2"
SPATIAL_AUDIT_SCHEMA = "orion.counterfactual-evidence-spatial-support-audit/v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def _spatial_checks(audit: dict) -> list[dict]:
    overall = audit["overall"]
    checks = [
        {
            "metric": "combined_median_within_view_mask_auroc",
            "value": overall["combined"]["within_view_mask_auroc"]["median"],
            "threshold": 0.70,
        },
        {
            "metric": "combined_p10_within_view_mask_auroc",
            "value": overall["combined"]["within_view_mask_auroc"]["p10"],
            "threshold": 0.55,
        },
        {
            "metric": "combined_median_inside_outside_ratio",
            "value": overall["combined"]["inside_outside_ratio"]["median"],
            "threshold": 2.0,
        },
        {
            "metric": "combined_median_equal_area_top_iou",
            "value": overall["combined"]["equal_area_top_iou"]["median"],
            "threshold": 0.15,
        },
    ]
    for component, threshold in {
        "persistent_direction": 0.65,
        "persistent_magnitude": 0.65,
        "transient_inconsistency": 0.60,
    }.items():
        checks.append(
            {
                "metric": "%s_median_within_view_mask_auroc" % component,
                "value": overall[component]["within_view_mask_auroc"]["median"],
                "threshold": threshold,
            }
        )
    for condition, rows in audit["by_family_severity"].items():
        checks.append(
            {
                "metric": "%s_combined_median_within_view_mask_auroc" % condition,
                "value": rows["combined"]["within_view_mask_auroc"]["median"],
                "threshold": 0.65,
            }
        )
    for check in checks:
        value = float(check["value"])
        check["passed"] = math.isfinite(value) and value >= float(check["threshold"])
    return checks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--response-floor", type=float, default=1e-6)
    parser.add_argument("--mask-label-floor", type=float, default=0.25)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError("refusing to reuse route-dataset audit output")
    if args.batch_size <= 0:
        raise SystemExit("audit batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA audit requested but unavailable")

    manifest = load_fp16_dataset_manifest(args.dataset_manifest, verify_shards=False)
    if (
        manifest.get("schema_version") != FP16_DIRECT_DATASET_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("written_route_count") != 90
        or manifest.get("written_clean_count") != 1440
        or manifest.get("written_observed_count") != 5760
    ):
        raise RuntimeError("formal direct FP16 dataset counts/status differ")
    manifest_sha = _sha256(args.dataset_manifest)
    protocol_sha = _sha256(args.protocol)
    contract_path = Path(manifest["source"]["extraction_contract"])
    if (
        not contract_path.is_file()
        or _sha256(contract_path) != manifest["source"]["extraction_contract_sha256"]
    ):
        raise RuntimeError("direct FP16 extraction contract differs")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("direct_extraction_fingerprint")
        != manifest["source"]["direct_extraction_fingerprint"]
        or contract["input_lineage"]["protocol"]["sha256"] != protocol_sha
        or contract.get("corruption_mask_is_primary_target") is not False
        or contract.get("exact_nonzero_presence_label_authorized") is not False
    ):
        raise RuntimeError("formal direct FP16 lineage/target contract differs")

    train_rows = [row for row in manifest["shards"] if row.get("split") == "train"]
    if len(train_rows) != 70:
        raise RuntimeError("formal train route-shard count differs")
    route_condition_views = [
        len(views)
        for row in train_rows
        for views in row.get("view_schedule_unique_views", {}).values()
    ]
    schedule_counts = {}
    for row in train_rows:
        for key, value in row.get("view_schedule_counts", {}).items():
            schedule_counts[key] = schedule_counts.get(key, 0) + int(value)
    schedule_audit = {
        "view_schedule": contract["view_schedule"],
        "route_condition_count": len(route_condition_views),
        "minimum_unique_views_per_route_condition": (
            min(route_condition_views) if route_condition_views else 0
        ),
        "maximum_unique_views_per_route_condition": (
            max(route_condition_views) if route_condition_views else 0
        ),
        "view_counts": dict(sorted(schedule_counts.items())),
        "exact_nonzero_presence_label_authorized": False,
    }

    records = load_fp16_dataset_records(
        args.dataset_manifest,
        splits=["train"],
        families=["local_blur", "local_dark"],
        verify_shards=True,
    )
    if len(records) != 4480 or len({record.route_id for record in records}) != 70:
        raise RuntimeError("formal train target population differs")
    device = torch.device(args.device)
    target_audit = audit_train_target_distribution(
        records,
        device,
        batch_size=args.batch_size,
        quantile=args.quantile,
        response_floor=args.response_floor,
    )
    monotonic_values = [
        family["combined"]
        for family in target_audit["paired_severity_higher_target_fraction"].values()
    ]
    diagnostics = {
        "all_components_have_responsive_cells": all(
            row["responsive_cell_count"] > 0
            for row in target_audit["components"].values()
        ),
        "all_selected_scales_above_numeric_floor": all(
            value > 1e-4 for value in target_audit["component_scales"].values()
        ),
        "both_optimizer_families_have_majority_combined_monotonic_pairs": (
            len(monotonic_values) == 2 and min(monotonic_values) > 0.5
        ),
        "every_train_route_condition_covers_at_least_four_views": (
            bool(route_condition_views) and min(route_condition_views) >= 4
        ),
        "train_route_condition_count_is_280": len(route_condition_views) == 280,
        "train_schedule_count_is_4480": sum(schedule_counts.values()) == 4480,
        "exact_nonzero_presence_label_is_disabled": True,
    }
    target_report = {
        "schema_version": TARGET_AUDIT_SCHEMA,
        "scope": "expanded train-only stored FP32 paired-target distribution; no optimization",
        "inputs": {
            "dataset_manifest": str(args.dataset_manifest.resolve()),
            "dataset_manifest_sha256": manifest_sha,
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": protocol_sha,
            "extraction_contract_sha256": _sha256(contract_path),
        },
        "audit": target_audit,
        "schedule_audit": schedule_audit,
        "diagnostics": diagnostics,
        "diagnostics_passed": all(diagnostics.values()),
        "continuation": {
            "adapter_training_started": False,
            "adapter_training_authorized_by_this_report": False,
            "required_next_step": "inspect target and spatial audit before freezing training",
            "orion_finetuning_authorized": False,
            "stage_b_authorized": False,
        },
    }

    scales = torch.tensor(
        [target_audit["component_scales"][name] for name in EVIDENCE_COMPONENTS],
        dtype=torch.float32,
    )
    spatial_audit = audit_target_spatial_support(
        records,
        scales,
        device,
        batch_size=args.batch_size,
        mask_label_floor=args.mask_label_floor,
    )
    spatial_checks = _spatial_checks(spatial_audit)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    target_path = args.output_dir / "counterfactual_evidence_target_audit.json"
    _write(target_path, target_report)
    spatial_report = {
        "schema_version": SPATIAL_AUDIT_SCHEMA,
        "scope": "expanded train-only target spatial diagnostic; no optimization",
        "inputs": {
            "dataset_manifest": str(args.dataset_manifest.resolve()),
            "dataset_manifest_sha256": manifest_sha,
            "target_audit": str(target_path.resolve()),
            "target_audit_sha256": _sha256(target_path),
            "protocol_sha256": protocol_sha,
        },
        "audit": spatial_audit,
        "gate": {"passed": all(row["passed"] for row in spatial_checks), "checks": spatial_checks},
        "claim_boundary": {
            "mask_used_for_optimizer": False,
            "mask_used_only_to_diagnose_spatial_support": True,
            "synthetic_localization_implies_real_uncertainty_truth": False,
        },
        "continuation": {
            "adapter_training_started": False,
            "adapter_training_authorized_if_gate_passes": False,
            "required_next_step_if_gate_passes": "freeze a separate adapter training run",
            "orion_finetuning_authorized": False,
            "stage_b_authorized": False,
        },
    }
    spatial_path = args.output_dir / "counterfactual_evidence_spatial_support.json"
    _write(spatial_path, spatial_report)
    print(
        json.dumps(
            {
                "target_audit": str(target_path.resolve()),
                "target_diagnostics_passed": target_report["diagnostics_passed"],
                "spatial_audit": str(spatial_path.resolve()),
                "spatial_gate_passed": spatial_report["gate"]["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("COUNTERFACTUAL_DIRECT_FP16_ROUTE_AUDIT_OK=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
