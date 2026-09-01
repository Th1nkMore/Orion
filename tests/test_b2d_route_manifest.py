"""Tests for metadata-only Bench2Drive route-manifest construction."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from uq_estimator.b2d_expansion_manifest import build_b2d_expansion_manifest
from uq_estimator.b2d_route_manifest import (
    B2DManifestError,
    build_b2d_route_manifest,
    canonicalize_b2d_folder,
)
from uq_estimator.spatial_training import RouteDisjointManifest


def _infos(n_routes=16, towns=("Town01", "Town02", "Town04", "Town05")):
    records = []
    scenarios = ("DynamicObjectCrossing", "ParkingCutIn", "ControlLoss")
    for route_index in range(n_routes):
        town = towns[route_index % len(towns)]
        scenario = scenarios[route_index % len(scenarios)]
        for weather, repetition in ((0, 0), (13, 1)):
            folder = (
                f"v1/{scenario}_{town}_Route{100 + route_index}_"
                f"Weather{weather}_Rep{repetition}"
            )
            for frame_idx in range(3 + route_index % 3):
                records.append(
                    {
                        "folder": folder,
                        "town_name": town,
                        "frame_idx": frame_idx,
                        # These fields must remain untouched and unopened.
                        "camera_path": "/definitely/not/a/real/image.jpg",
                        "model_checkpoint": "/not/a/model.pth",
                    }
                )
    return records


def _route_owner(payload):
    return {
        route: split
        for split, values in payload["splits"].items()
        for route in values["route_ids"]
    }


def _canonical_owner(payload):
    return {
        route: split
        for split, stats in payload["lineage_audit"]["split_statistics"].items()
        for route in stats["canonical_route_keys"]
    }


def _frozen_expansion_fixture():
    split_specs = (
        ("train", 35, "Town01", 100),
        ("validation", 5, "Town03", 200),
        ("calibration", 5, "Town04", 300),
        ("held_out", 5, "Town02", 400),
    )
    addition_specs = (
        ("train", 35, "Town12", 1000),
        ("validation", 5, "Town13", 1100),
        ("calibration", 5, "Town15", 1200),
        ("held_out", 5, "Town11", 1300),
    )
    baseline_splits = {split: {"route_ids": []} for split, *_ in split_specs}
    additions = []
    infos = []
    for split, count, town, start in split_specs:
        for offset in range(count):
            folder = (
                f"v1/BaselineScenario_{town}_Route{start + offset}_Weather0"
            )
            baseline_splits[split]["route_ids"].append(folder)
            infos.append({"folder": folder, "town_name": town, "frame_idx": 0})
    for split, count, town, start in addition_specs:
        for offset in range(count):
            folder = (
                f"v1/ExpansionScenario_{town}_Route{start + offset}_Weather7"
            )
            additions.append({"folder": folder, "split": split})
            infos.append({"folder": folder, "town_name": town, "frame_idx": 0})
    baseline = {
        "schema_version": "spatial-uq-route-manifest/v1",
        "split_unit": "route_id",
        "seed": 17,
        "route_disjoint": True,
        "splits": baseline_splits,
    }
    plan = {
        "schema_version": "b2d-expansion-plan/v1",
        "status": "pre_download_plan_not_a_training_manifest",
        "seed": 20260827,
        "additions": additions,
    }
    return infos, baseline, plan


def test_canonical_route_strips_weather_and_repetition_but_not_town_route():
    a = canonicalize_b2d_folder(
        "v1/DynamicObjectCrossing_Town04_Route166_Weather0_Rep0"
    )
    b = canonicalize_b2d_folder(
        "v2/DynamicObjectCrossing_Town04_Route166_Weather23_Repetition7"
    )
    c = canonicalize_b2d_folder(
        "DynamicObjectCrossing_Town05_Route166_Weather23_7"
    )

    assert a.canonical_route_key == b.canonical_route_key == "Town04/Route166"
    assert a.weather == 0 and b.weather == 23
    assert a.repetition == 0 and b.repetition == 7 and c.repetition == 7
    assert c.canonical_route_key == "Town05/Route166"


def test_build_is_compatible_route_disjoint_and_groups_all_variants():
    payload = build_b2d_route_manifest(
        _infos(),
        seed=17,
        minimum_canonical_routes=12,
        min_routes_per_split=2,
        closed_loop_development_routes=("route146", "route203"),
        closed_loop_headline_routes=("candidate147",),
    )
    manifest = RouteDisjointManifest.from_dict(payload)
    owner = _route_owner(payload)
    canonical_owner = _canonical_owner(payload)

    assert len(owner) == 32
    assert len(canonical_owner) == 16
    assert all(len(routes) >= 2 for routes in manifest.splits.values())
    assert payload["lineage_audit"]["leakage_checks"]["passed"] is True
    assert payload["lineage_audit"]["selection"]["town_disjoint_heldout_achieved"] is True
    heldout_towns = set(
        payload["lineage_audit"]["split_statistics"]["held_out"]["towns"]
    )
    other_towns = {
        town
        for split in ("train", "validation", "calibration")
        for town in payload["lineage_audit"]["split_statistics"][split]["towns"]
    }
    assert heldout_towns.isdisjoint(other_towns)

    catalog = payload["lineage_audit"]["route_catalog"]
    assert catalog["Town01/Route100"]["weather_variants"] == [0, 13]
    assert catalog["Town01/Route100"]["repetition_variants"] == [0, 1]
    route100_folders = catalog["Town01/Route100"]["folders"]
    assert {owner[folder] for folder in route100_folders} == {
        canonical_owner["Town01/Route100"]
    }
    semantics = payload["lineage_audit"]["closed_loop_route_semantics"]
    assert semantics["used_for_offline_id_matching"] is False
    assert semantics["training_heldout_claim_supported_by_these_labels"] is False


def test_excluding_one_weather_folder_removes_whole_canonical_route():
    infos = _infos()
    excluded = "v1/DynamicObjectCrossing_Town01_Route100_Weather0_Rep0"
    payload = build_b2d_route_manifest(
        infos,
        exclude_folders=(excluded,),
        minimum_canonical_routes=12,
        min_routes_per_split=2,
    )
    owner = _route_owner(payload)
    canonical_owner = _canonical_owner(payload)
    assert "Town01/Route100" not in canonical_owner
    assert not any("_Route100_" in folder for folder in owner)
    audit = payload["lineage_audit"]["exclusions"]
    assert audit["expanded_canonical_route_keys"] == ["Town01/Route100"]
    assert len(audit["matched_input_folders"]) == 2


def test_unmatched_exclusion_and_small_dataset_fail_closed():
    with pytest.raises(B2DManifestError, match="matched no input"):
        build_b2d_route_manifest(
            _infos(),
            exclude_folders=("Unknown_Town01_Route999_Weather0",),
        )
    with pytest.raises(B2DManifestError, match="fail-closed minimum"):
        build_b2d_route_manifest(
            _infos(n_routes=7),
            minimum_canonical_routes=8,
            min_routes_per_split=1,
        )


def test_town_metadata_conflict_and_duplicate_frame_fail_closed():
    conflict = [
        {
            "folder": "Scenario_Town01_Route1_Weather0",
            "town_name": "Town02",
            "frame_idx": 0,
        }
    ]
    with pytest.raises(B2DManifestError, match="conflicts"):
        build_b2d_route_manifest(conflict, minimum_canonical_routes=4)

    duplicate = _infos()
    duplicate.append(dict(duplicate[0]))
    with pytest.raises(B2DManifestError, match="duplicate folder/frame"):
        build_b2d_route_manifest(duplicate)


def test_cli_dry_run_writes_nothing_and_reports_lineage(tmp_path):
    infos_path = tmp_path / "infos.json"
    infos_path.write_text(json.dumps({"infos": _infos()}), encoding="utf-8")
    output = tmp_path / "manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_b2d_route_manifest.py",
            "--infos",
            str(infos_path),
            "--output",
            str(output),
            "--dry-run",
            "--closed-loop-development-route",
            "route146",
            "--closed-loop-headline-route",
            "candidate147",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert not output.exists()
    assert payload["lineage_audit"]["writes_performed"] is False
    assert payload["lineage_audit"]["source"]["image_files_opened"] is False
    assert payload["lineage_audit"]["source"]["model_loaded"] is False
    RouteDisjointManifest.from_dict(payload)


def test_frozen_expansion_preserves_membership_and_heldout_town_isolation():
    infos, baseline, plan = _frozen_expansion_fixture()
    payload = build_b2d_expansion_manifest(
        infos,
        baseline,
        plan,
        source_lineage={"kind": "synthetic_infos"},
        baseline_lineage={"kind": "synthetic_baseline"},
        expansion_plan_lineage={"kind": "synthetic_expansion_plan"},
    )
    RouteDisjointManifest.from_dict(payload)

    assert {
        split: len(values["route_ids"])
        for split, values in payload["splits"].items()
    } == {"train": 70, "validation": 10, "calibration": 10, "held_out": 10}
    for split, values in baseline["splits"].items():
        assert set(values["route_ids"]).issubset(payload["splits"][split]["route_ids"])
    for row in plan["additions"]:
        assert row["folder"] in payload["splits"][row["split"]]["route_ids"]

    audit = payload["lineage_audit"]
    assert audit["selection"]["heldout_towns"] == ["Town02", "Town11"]
    assert audit["leakage_checks"]["passed"] is True
    assert audit["leakage_checks"]["heldout_town_overlap"] == []
