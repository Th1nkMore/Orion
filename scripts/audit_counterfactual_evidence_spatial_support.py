#!/usr/bin/env python3
"""Audit whether paired feature targets retain local intervention support."""

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

from uq_estimator.counterfactual_evidence import EVIDENCE_COMPONENTS  # noqa: E402
from uq_estimator.counterfactual_evidence_training import (  # noqa: E402
    audit_target_spatial_support,
    records_from_counterfactual_shard,
    select_records,
)


SCHEMA_VERSION = "orion.counterfactual-evidence-spatial-support-audit/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-shard", type=Path, required=True)
    parser.add_argument("--target-audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--mask-label-floor", type=float, default=0.25)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("refusing to overwrite spatial-support audit")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA spatial-support audit requested but unavailable")

    target_report = json.loads(args.target_audit.read_text(encoding="utf-8"))
    if target_report.get("schema_version") != (
        "orion.counterfactual-evidence-target-audit/v1"
    ) or not target_report.get("diagnostics_passed"):
        raise RuntimeError("spatial audit requires a passed train-target audit")
    expected_feature_sha = target_report["inputs"][
        "feature_shard_sha256_from_extraction_sidecar"
    ]
    protocol_sha = _sha256(args.protocol)
    if target_report["inputs"]["protocol_sha256"] != protocol_sha:
        raise RuntimeError("target-audit protocol lineage changed")

    payload = torch.load(args.feature_shard, map_location="cpu")
    if payload.get("provenance", {}).get("protocol_sha256") != protocol_sha:
        raise RuntimeError("feature-shard protocol lineage changed")
    records = records_from_counterfactual_shard(payload)
    train = select_records(records, ["train"], ["local_blur", "local_dark"])
    scales = torch.tensor(
        [
            target_report["audit"]["component_scales"][component]
            for component in EVIDENCE_COMPONENTS
        ],
        dtype=torch.float32,
    )
    audit = audit_target_spatial_support(
        train,
        scales,
        torch.device(args.device),
        batch_size=args.batch_size,
        mask_label_floor=args.mask_label_floor,
    )

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
    component_thresholds = {
        "persistent_direction": 0.65,
        "persistent_magnitude": 0.65,
        "transient_inconsistency": 0.60,
    }
    for component, threshold in component_thresholds.items():
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
        check["passed"] = value >= float(check["threshold"])

    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "train-only target spatial diagnostic; no optimization",
        "inputs": {
            "feature_shard": str(args.feature_shard.resolve()),
            "feature_shard_sha256_from_prior_attestation": expected_feature_sha,
            "target_audit": str(args.target_audit.resolve()),
            "target_audit_sha256": _sha256(args.target_audit),
            "protocol_sha256": protocol_sha,
        },
        "audit": audit,
        "gate": {"passed": all(row["passed"] for row in checks), "checks": checks},
        "claim_boundary": {
            "mask_used_for_optimizer": False,
            "mask_used_only_to_diagnose_spatial_support": True,
            "synthetic_localization_implies_real_uncertainty_truth": False,
        },
        "continuation": {
            "adapter_training_started": False,
            "adapter_training_authorized_if_gate_passes": False,
            "required_next_step_if_gate_passes": (
                "freeze a separate bounded adapter architecture/loss smoke; "
                "this target-only diagnostic does not authorize optimization"
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
    print(json.dumps({"output": str(args.output), "gate": report["gate"]}, indent=2))
    print("COUNTERFACTUAL_EVIDENCE_SPATIAL_SUPPORT_AUDIT_OK=1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
