"""Build an infos-backed manifest while preserving a frozen B2D expansion plan."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Mapping, Sequence

from uq_estimator.b2d_route_manifest import (
    CANONICALIZATION_VERSION,
    LINEAGE_AUDIT_SCHEMA_VERSION,
    REQUIRED_SPLITS,
    ROUTE_MANIFEST_SCHEMA_VERSION,
    B2DManifestError,
    B2DRouteGroup,
    extract_b2d_frame_metadata,
    group_b2d_routes,
    normalize_folder,
)


EXPANSION_MANIFEST_AUDIT_SCHEMA_VERSION = "b2d-expansion-manifest-audit/v1"


def _split_stats(routes: Sequence[B2DRouteGroup]) -> Dict[str, Any]:
    scenario_frames: Counter[str] = Counter()
    town_frames: Counter[str] = Counter()
    for route in routes:
        scenario_frames[route.stratum_scenario] += route.frame_count
        town_frames[route.town] += route.frame_count
    return {
        "canonical_route_count": len(routes),
        "canonical_route_keys": sorted(route.canonical_route_key for route in routes),
        "folder_count": sum(len(route.folders) for route in routes),
        "folder_route_ids": sorted(folder for route in routes for folder in route.folders),
        "frame_count": sum(route.frame_count for route in routes),
        "towns": sorted({route.town for route in routes}),
        "scenario_types": sorted(
            {scenario for route in routes for scenario in route.scenarios}
        ),
        "frames_by_town": dict(sorted(town_frames.items())),
        "frames_by_scenario_stratum": dict(sorted(scenario_frames.items())),
    }


def build_b2d_expansion_manifest(
    infos: Sequence[Mapping[str, Any]],
    baseline_manifest: Mapping[str, Any],
    expansion_plan: Mapping[str, Any],
    *,
    source_lineage: Mapping[str, Any],
    baseline_lineage: Mapping[str, Any],
    expansion_plan_lineage: Mapping[str, Any],
) -> Dict[str, Any]:
    if baseline_manifest.get("schema_version") != ROUTE_MANIFEST_SCHEMA_VERSION:
        raise B2DManifestError("baseline route manifest schema differs")
    if expansion_plan.get("schema_version") != "b2d-expansion-plan/v1" or expansion_plan.get(
        "status"
    ) != "pre_download_plan_not_a_training_manifest":
        raise B2DManifestError("expansion plan schema/status differs")

    baseline_splits = baseline_manifest.get("splits")
    if not isinstance(baseline_splits, Mapping):
        raise B2DManifestError("baseline route manifest has no splits")
    owner: Dict[str, str] = {}
    baseline_folders = set()
    for split in REQUIRED_SPLITS:
        route_ids = baseline_splits.get(split, {}).get("route_ids")
        if not isinstance(route_ids, list):
            raise B2DManifestError("baseline split %s differs" % split)
        for value in route_ids:
            folder = normalize_folder(value)
            previous = owner.setdefault(folder, split)
            if previous != split:
                raise B2DManifestError("baseline folder appears in multiple splits")
            baseline_folders.add(folder)
    if len(baseline_folders) != 50:
        raise B2DManifestError("baseline manifest must preserve exactly 50 folders")

    additions = expansion_plan.get("additions")
    if not isinstance(additions, list) or len(additions) != 50:
        raise B2DManifestError("expansion plan must add exactly 50 folders")
    addition_folders = set()
    for row in additions:
        if not isinstance(row, Mapping):
            raise B2DManifestError("expansion addition is not a mapping")
        split = str(row.get("split", ""))
        if split not in REQUIRED_SPLITS:
            raise B2DManifestError("expansion addition split differs")
        folder = normalize_folder(row.get("folder"))
        if folder in baseline_folders or folder in addition_folders:
            raise B2DManifestError("expansion folder overlaps/duplicates baseline")
        addition_folders.add(folder)
        owner[folder] = split

    frames = extract_b2d_frame_metadata(infos)
    groups = group_b2d_routes(frames)
    observed_folders = {frame.folder for frame in frames}
    expected_folders = baseline_folders | addition_folders
    if observed_folders != expected_folders:
        raise B2DManifestError(
            "expanded infos folders differ; missing=%s unexpected=%s"
            % (sorted(expected_folders - observed_folders), sorted(observed_folders - expected_folders))
        )
    if len(groups) != 100:
        raise B2DManifestError("expanded infos must contain 100 canonical routes")

    assignment: Dict[str, list[B2DRouteGroup]] = {
        split: [] for split in REQUIRED_SPLITS
    }
    canonical_owner = {}
    folder_owner = {}
    for group in groups:
        owners = {owner[folder] for folder in group.folders}
        if len(owners) != 1:
            raise B2DManifestError(
                "weather/repetition siblings cross expansion splits: %s"
                % group.canonical_route_key
            )
        split = next(iter(owners))
        previous = canonical_owner.setdefault(group.canonical_route_key, split)
        if previous != split:
            raise B2DManifestError("canonical route crosses expansion splits")
        for folder in group.folders:
            previous_folder = folder_owner.setdefault(folder, split)
            if previous_folder != split:
                raise B2DManifestError("folder crosses expansion splits")
        assignment[split].append(group)
    assignment = {
        split: sorted(routes, key=lambda route: route.canonical_route_key)
        for split, routes in assignment.items()
    }
    split_counts = {split: len(routes) for split, routes in assignment.items()}
    expected_split_counts = {
        "train": 70,
        "validation": 10,
        "calibration": 10,
        "held_out": 10,
    }
    if split_counts != expected_split_counts:
        raise B2DManifestError(
            "expanded split counts differ: %s" % split_counts
        )
    heldout_towns = {route.town for route in assignment["held_out"]}
    nonheld_towns = {
        route.town
        for split in ("train", "validation", "calibration")
        for route in assignment[split]
    }
    heldout_overlap = sorted(heldout_towns & nonheld_towns)
    if heldout_overlap:
        raise B2DManifestError("expanded held-out towns overlap: %s" % heldout_overlap)

    manifest_splits = {
        split: sorted(folder for route in assignment[split] for folder in route.folders)
        for split in REQUIRED_SPLITS
    }
    for split in REQUIRED_SPLITS:
        preserved = {
            normalize_folder(value)
            for value in baseline_splits[split]["route_ids"]
        }
        if not preserved.issubset(manifest_splits[split]):
            raise B2DManifestError("baseline split membership was not preserved")
        planned = {
            normalize_folder(row["folder"])
            for row in additions
            if row["split"] == split
        }
        if not planned.issubset(manifest_splits[split]):
            raise B2DManifestError("planned addition split membership changed")

    route_catalog = {
        route.canonical_route_key: {
            "town": route.town,
            "route_token": route.route_token,
            "scenario_types": list(route.scenarios),
            "folders": list(route.folders),
            "weather_variants": list(route.weather_variants),
            "repetition_variants": list(route.repetition_variants),
            "frame_count": route.frame_count,
            "frame_min": route.frame_min,
            "frame_max": route.frame_max,
        }
        for route in groups
    }
    payload = {
        "schema_version": ROUTE_MANIFEST_SCHEMA_VERSION,
        "split_unit": "route_id",
        "seed": int(expansion_plan["seed"]),
        "route_disjoint": True,
        "splits": {
            split: {"route_ids": manifest_splits[split]}
            for split in REQUIRED_SPLITS
        },
    }
    payload["lineage_audit"] = {
        "schema_version": LINEAGE_AUDIT_SCHEMA_VERSION,
        "expansion_audit_schema_version": EXPANSION_MANIFEST_AUDIT_SCHEMA_VERSION,
        "source": dict(source_lineage),
        "baseline_manifest": dict(baseline_lineage),
        "expansion_plan": dict(expansion_plan_lineage),
        "canonicalization": {
            "version": CANONICALIZATION_VERSION,
            "key_format": "<Town>/<RouteN>",
            "manifest_route_ids": "original B2D folder strings",
            "split_assignment_unit": "canonical physical-route group",
            "weather_repetition_siblings_grouped": True,
        },
        "input_summary": {
            "raw_frame_records": len(frames),
            "folder_count": len(observed_folders),
            "canonical_route_count": len(groups),
            "towns": sorted({route.town for route in groups}),
            "scenario_types": sorted(
                {scenario for route in groups for scenario in route.scenarios}
            ),
        },
        "selection": {
            "kind": "frozen_expansion_plan_preserved_without_rerandomization",
            "baseline_folder_count": len(baseline_folders),
            "addition_folder_count": len(addition_folders),
            "baseline_membership_preserved": True,
            "addition_membership_preserved": True,
            "resulting_split_counts": split_counts,
            "heldout_towns": sorted(heldout_towns),
        },
        "split_statistics": {
            split: _split_stats(assignment[split]) for split in REQUIRED_SPLITS
        },
        "leakage_checks": {
            "canonical_route_overlap": [],
            "folder_overlap": [],
            "heldout_town_overlap": heldout_overlap,
            "weather_repetition_siblings_grouped_before_split": True,
            "passed": True,
        },
        "claim_boundary": {
            "held_out_means": "town-disjoint within this expanded offline infos file",
            "proves_absent_from_orion_pretraining": False,
            "proves_closed_loop_route_is_training_heldout": False,
        },
        "route_catalog": route_catalog,
        "writes_performed": True,
    }
    return payload


__all__ = [
    "EXPANSION_MANIFEST_AUDIT_SCHEMA_VERSION",
    "build_b2d_expansion_manifest",
]
