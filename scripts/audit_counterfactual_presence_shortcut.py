#!/usr/bin/env python3
"""Audit whether sparse evidence presence degenerates to a route/view schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


EVIDENCE_COMPONENTS = (
    "persistent_direction",
    "persistent_magnitude",
    "transient_inconsistency",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _condition_view(
    route_id: str, family: str, severity: int, view_count: int, seed: int
) -> int:
    raw = "%d|%s|%s|%d" % (seed, route_id, family, severity)
    return int.from_bytes(
        hashlib.sha256(raw.encode("utf-8")).digest()[:8], "big"
    ) % view_count


def _entropy_bits(counts: Counter) -> float:
    total = sum(counts.values())
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--target-audit", type=Path, required=True)
    parser.add_argument("--spatial-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--view-count", type=int, default=6)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("refusing to overwrite presence-shortcut audit")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    target_report = json.loads(args.target_audit.read_text(encoding="utf-8"))
    spatial_report = json.loads(args.spatial_audit.read_text(encoding="utf-8"))
    families = protocol["intervention_split"]["optimizer_families"]
    severities = protocol["intervention_split"]["optimizer_severities"]
    conditions = [(family, int(severity)) for family in families for severity in severities]
    frames_per_route = int(protocol["reference_data"]["frames_per_route"])

    schedule = {}
    split_summaries = {}
    for split in ("train", "validation"):
        routes = manifest["splits"][split]["route_ids"]
        view_counts = Counter()
        unique_views_per_route = Counter()
        rows = {}
        for route_id in routes:
            assigned = {
                "%s/severity_%d" % (family, severity): _condition_view(
                    route_id, family, severity, args.view_count, args.seed
                )
                for family, severity in conditions
            }
            rows[route_id] = assigned
            view_counts.update(assigned.values())
            unique_views_per_route[len(set(assigned.values()))] += 1
        schedule[split] = rows
        split_summaries[split] = {
            "route_count": len(routes),
            "route_condition_keys": len(routes) * len(conditions),
            "records_per_route_condition_key": frames_per_route,
            "positive_view_counts_by_schedule_key": {
                str(view): view_counts[view] for view in range(args.view_count)
            },
            "positive_view_entropy_bits": _entropy_bits(view_counts),
            "maximum_view_entropy_bits": math.log2(args.view_count),
            "unique_positive_views_per_route": {
                str(count): routes for count, routes in sorted(unique_views_per_route.items())
            },
            "conditional_positive_view_entropy_bits_given_route_family_severity": 0.0,
        }

    expected_single_view_fraction = 1.0 / args.view_count
    responsive = {}
    degeneracy_checks = []
    for component in EVIDENCE_COMPONENTS:
        row = target_report["audit"]["components"][component]
        fraction = float(row["responsive_fraction"])
        delta = abs(fraction - expected_single_view_fraction)
        responsive[component] = {
            "fraction": fraction,
            "expected_one_full_view_fraction": expected_single_view_fraction,
            "absolute_delta": delta,
            "all_view_q80": float(row["all_cell_quantiles"]["q80"]),
            "responsive_q50": float(row["responsive_quantiles"]["q50"]),
            "responsive_q80": float(row["responsive_quantiles"]["q80"]),
        }
        degeneracy_checks.append(delta <= 0.001)

    spatial = spatial_report["audit"]["overall"]["combined"]
    exact_support_degenerates_to_view = all(degeneracy_checks)
    high_amplitude_target_remains_local = (
        float(spatial["within_view_mask_auroc"]["median"]) >= 0.7
        and float(spatial["inside_outside_ratio"]["median"]) >= 2.0
    )
    report = {
        "schema_version": "orion.counterfactual-presence-shortcut-audit/v1",
        "inputs": {
            "manifest_sha256": _sha256(args.manifest),
            "protocol_sha256": _sha256(args.protocol),
            "target_audit_sha256": _sha256(args.target_audit),
            "spatial_audit_sha256": _sha256(args.spatial_audit),
            "seed": args.seed,
            "view_count": args.view_count,
        },
        "responsive_support": responsive,
        "route_condition_view_schedule": schedule,
        "schedule_summary": split_summaries,
        "spatial_high_amplitude_evidence": {
            "within_view_mask_auroc_median": float(
                spatial["within_view_mask_auroc"]["median"]
            ),
            "inside_outside_ratio_median": float(
                spatial["inside_outside_ratio"]["median"]
            ),
            "equal_area_top_iou_median": float(
                spatial["equal_area_top_iou"]["median"]
            ),
        },
        "diagnosis": {
            "exact_nonzero_support_degenerates_to_one_full_view": exact_support_degenerates_to_view,
            "route_family_severity_determines_positive_view_for_all_16_frames": True,
            "high_amplitude_target_remains_spatially_local": high_amplitude_target_remains_local,
            "shortcut_risk_confirmed": (
                exact_support_degenerates_to_view and high_amplitude_target_remains_local
            ),
            "interpretation": (
                "The 1e-6 presence label is effectively an intervened-camera label. "
                "Because that camera is fixed for every frame of each route/family/severity "
                "key, absolute visual features plus view identity can memorize a route-specific "
                "camera schedule. Strong target amplitude is still locally meaningful, so the "
                "paired target itself should not be discarded."
            ),
        },
        "continuation": {
            "reuse_exact_nonzero_presence_label": False,
            "reuse_route_condition_fixed_view_schedule": False,
            "recommended_data_schedule": (
                "sample-conditioned or short-window-balanced camera assignment so each "
                "route/family/severity observes multiple cameras"
            ),
            "recommended_supervision": (
                "continuous amplitude or target-derived high-response/soft support; never "
                "the numerical nonzero footprint of globally mixed ViT features"
            ),
            "adapter_training_authorized_by_this_audit": False,
            "orion_finetuning_authorized": False,
            "stage_b_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "diagnosis": report["diagnosis"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
