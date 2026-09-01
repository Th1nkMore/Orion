#!/usr/bin/env python3
"""Diagnose whether held-out interventions alter frozen EVAViT features.

This script does not train a model and does not treat the paired difference or
synthetic mask as uncertainty truth.  It is only a failure-localization check:
if the backbone feature itself is unchanged, a downstream clean-conditional
teacher cannot be expected to detect the intervention.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uq_estimator.observation_uq_v3 import (  # noqa: E402
    _binary_auc,
    _record_family,
    _spearman,
    route_splits_from_manifest,
)
from uq_estimator.spatial_training import load_paired_feature_records  # noqa: E402


def _finite(value):
    return None if not math.isfinite(float(value)) else float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    records = load_paired_feature_records(args.records)
    route_splits = route_splits_from_manifest(
        json.loads(args.manifest.read_text(encoding="utf-8"))
    )
    rows = defaultdict(
        lambda: {
            "record_mean": [],
            "severity": [],
            "inside_sum": 0.0,
            "inside_count": 0,
            "outside_sum": 0.0,
            "outside_count": 0,
            "scores": [],
            "labels": [],
        }
    )
    for record in records:
        family = _record_family(record)
        split = route_splits[record.route_id]
        error = 1.0 - F.cosine_similarity(
            record.clean_patch_features.float(),
            record.observed_patch_features.float(),
            dim=-1,
            eps=1e-6,
        ).clamp(-1.0, 1.0)
        if record.corruption_mask is None:
            raise RuntimeError("input diagnostic requires recorded corruption masks")
        mask = record.corruption_mask.float() >= 0.5
        key = (split, family)
        row = rows[key]
        row["record_mean"].append(float(error.mean()))
        row["severity"].append(float(record.severity))
        row["inside_sum"] += float(error[mask].sum())
        row["inside_count"] += int(mask.sum())
        row["outside_sum"] += float(error[~mask].sum())
        row["outside_count"] += int((~mask).sum())
        row["scores"].append(error.reshape(-1))
        row["labels"].append(mask.reshape(-1))

    by_split_family = {}
    for (split, family), row in sorted(rows.items()):
        record_mean = torch.tensor(row["record_mean"])
        severity = torch.tensor(row["severity"])
        inside = row["inside_sum"] / max(row["inside_count"], 1)
        outside = row["outside_sum"] / max(row["outside_count"], 1)
        by_split_family["%s/%s" % (split, family)] = {
            "record_count": len(record_mean),
            "paired_cosine_error_mean": float(record_mean.mean()),
            "paired_cosine_error_inside_mask": inside,
            "paired_cosine_error_outside_mask": outside,
            "inside_minus_outside": inside - outside,
            "severity_record_mean_spearman": _finite(
                _spearman(severity, record_mean)
            ),
            "mask_patch_auroc_diagnostic_only": _finite(
                _binary_auc(torch.cat(row["scores"]), torch.cat(row["labels"]))
            ),
        }
    payload = {
        "schema_version": "orion.observation-uq-input-diagnostic/v1",
        "record_count": len(records),
        "claim_boundary": {
            "paired_feature_difference_is_uncertainty_truth": False,
            "corruption_mask_is_uncertainty_truth": False,
            "training_performed": False,
            "actual_target_tensor_read": False,
            "purpose": "distinguish ineffective intervention from teacher failure",
        },
        "by_split_family": by_split_family,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
