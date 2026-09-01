#!/usr/bin/env python3
"""Freeze the expanded direct-FP16 counterfactual extraction protocol."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--infos", type=Path, required=True)
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--expansion-plan", type=Path, required=True)
    parser.add_argument("--frozen-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite expanded protocol")
    parent = json.loads(args.parent_protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.route_manifest.read_text(encoding="utf-8"))
    plan = json.loads(args.expansion_plan.read_text(encoding="utf-8"))
    if parent.get("schema_version") != "orion.observation-uq-counterfactual-evidence/v2":
        raise RuntimeError("expanded protocol parent must be frozen v2")
    if manifest.get("schema_version") != "spatial-uq-route-manifest/v1":
        raise RuntimeError("expanded route manifest schema differs")
    split_counts = {
        split: len(manifest.get("splits", {}).get(split, {}).get("route_ids", []))
        for split in ("train", "validation", "calibration", "held_out")
    }
    if split_counts != {
        "train": 70,
        "validation": 10,
        "calibration": 10,
        "held_out": 10,
    }:
        raise RuntimeError("expanded route split counts differ")
    leakage = manifest.get("lineage_audit", {}).get("leakage_checks", {})
    if leakage.get("passed") is not True or leakage.get("heldout_town_overlap"):
        raise RuntimeError("expanded route manifest leakage gate failed")
    infos_sha = _sha256(args.infos)
    manifest_sha = _sha256(args.route_manifest)
    plan_sha = _sha256(args.expansion_plan)
    audit = manifest["lineage_audit"]
    if audit["source"]["sha256"] != infos_sha:
        raise RuntimeError("expanded infos lineage differs from formal manifest")
    if audit["expansion_plan"]["sha256"] != plan_sha:
        raise RuntimeError("expansion plan lineage differs from formal manifest")
    if plan.get("schema_version") != "b2d-expansion-plan/v1":
        raise RuntimeError("expanded plan schema differs")

    payload = copy.deepcopy(parent)
    payload["schema_version"] = "orion.observation-uq-counterfactual-evidence/v3"
    payload["frozen_at"] = args.frozen_at
    payload["purpose"] = (
        "Expanded route-disjoint Stage-1 evidence supervision written directly "
        "as lossless FP16 whole-route shards; no monolithic feature cache or "
        "adapter optimization is part of extraction."
    )
    payload["parent_protocol_sha256"] = _sha256(args.parent_protocol)
    payload["expanded_data_lineage"] = {
        "infos": str(args.infos.resolve()),
        "infos_sha256": infos_sha,
        "route_manifest": str(args.route_manifest.resolve()),
        "route_manifest_sha256": manifest_sha,
        "expansion_plan": str(args.expansion_plan.resolve()),
        "expansion_plan_sha256": plan_sha,
        "split_counts": split_counts,
        "heldout_towns": audit["selection"]["heldout_towns"],
        "heldout_town_overlap": leakage["heldout_town_overlap"],
    }
    payload["reference_data"] = {
        "source": "formal expanded B2D infos plus frozen route-disjoint manifest",
        "historical_name": "clean",
        "corrected_name": "unintervened mixed-weather reference",
        "selected_route_quotas": {
            "train": 70,
            "validation": 10,
            "held_out": 10,
        },
        "calibration_routes_reserved_from_extraction": 10,
        "frames_per_route": 16,
        "reference_frames": 1440,
        "selection_rule": "first metadata-verified contiguous 16-frame run per selected route",
    }
    payload["intervention_split"]["route_validation"] = (
        "same optimizer families on ten disjoint B2D validation routes"
    )
    payload["intervention_split"]["heldout_family_development"] = (
        "local_glare on validation and held-out B2D routes; read only after train diagnostics"
    )
    payload["extraction"] = {
        "seed": 20260827,
        "reference_frames": 1440,
        "observed_frames": 5760,
        "total_feature_grids": 7200,
        "storage_dtype": "float16",
        "target_storage_dtype": "float32",
        "projected_feature_storage_gib": 131.84,
        "feature_only": True,
        "whole_route_shards": True,
        "direct_from_frozen_backbone": True,
        "monolithic_intermediate_created": False,
        "resumable_at_verified_route_boundaries": True,
    }
    payload["projected_schedule_audit"] = {
        "assumption": (
            "the frozen selector takes each route's first 16-frame contiguous run; "
            "runtime rechecks actual identities and route boundaries"
        ),
        "expected_selected_routes": 90,
        "expected_route_conditions": 360,
        "unique_views_per_16_frame_route_condition_minimum": 4,
        "runtime_gate": (
            "every actual route/family/severity sequence covers at least four views "
            "before its atomic route shard is committed"
        ),
        "exact_nonzero_presence_label_authorized": False,
    }
    payload["continuation"] = {
        "required_after_extraction": [
            "complete manifest/count/hash/lineage validation",
            "expanded train-only continuous-target amplitude audit",
            "expanded train-only target spatial-support diagnostic",
            "freeze a separate adapter architecture/loss run only after audit review",
        ],
        "automatic_adapter_training": False,
        "heldout_glare_read_before_train_gate": False,
        "native_weather_read_before_train_gate": False,
        "orion_finetuning_authorized": False,
        "stage_b_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": _sha256(args.output),
                "split_counts": split_counts,
                "reference_frames": 1440,
                "observed_frames": 5760,
                "adapter_training_started": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("COUNTERFACTUAL_EVIDENCE_EXPANDED_PROTOCOL_OK=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
