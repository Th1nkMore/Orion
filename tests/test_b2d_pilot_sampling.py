"""CPU/mock tests for auditable B2D pilot frame sampling."""

import hashlib
import json
import subprocess
import sys

import pytest

from uq_estimator.b2d_pilot_sampling import (
    B2DPilotSamplingError,
    annotation_candidate_from_info,
    build_b2d_pilot_submanifest,
    load_parent_manifest,
)


SPLITS = ("train", "validation", "calibration", "held_out")


def _mock_parent_and_infos(routes_per_split=4, frames_per_folder=60):
    split_folders = {split: [] for split in SPLITS}
    catalog = {}
    infos = []
    route_index = 0
    split_towns = ("Town01", "Town02", "Town12", "Town04")
    for split_index, split in enumerate(SPLITS):
        for local_index in range(routes_per_split):
            town = split_towns[split_index]
            route = f"Route{100 + route_index}"
            scenario = ("DynamicObjectCrossing", "ParkingCutIn", "ControlLoss")[
                route_index % 3
            ]
            folders = []
            for weather, repetition in ((0, 0), (13, 1)):
                folder = f"v1/{scenario}_{town}_{route}_Weather{weather}_Rep{repetition}"
                folders.append(folder)
                split_folders[split].append(folder)
                for frame_idx in range(frames_per_folder):
                    is_candidate = frame_idx % 2 == 0
                    infos.append(
                        {
                            "folder": folder,
                            "town_name": town,
                            "frame_idx": frame_idx,
                            "gt_names": (
                                ["walker.pedestrian.0001"] if is_candidate else ["traffic.stop"]
                            ),
                            "gt_boxes": (
                                [[12.0, 2.0, 0.0, 1.0, 1.0, 2.0, 0.0]]
                                if is_candidate
                                else [[8.0, 1.0, 0.0, 1.0, 1.0, 2.0, 0.0]]
                            ),
                            "num_points": [5],
                            "camera_path": "/must/not/be/opened.jpg",
                        }
                    )
            key = f"{town}/{route}"
            catalog[key] = {
                "town": town,
                "route_token": route,
                "scenario_types": [scenario],
                "folders": folders,
                "weather_variants": [0, 13],
                "repetition_variants": [0, 1],
                "frame_count": 2 * frames_per_folder,
            }
            route_index += 1
    parent = {
        "schema_version": "spatial-uq-route-manifest/v1",
        "split_unit": "route_id",
        "seed": 7,
        "route_disjoint": True,
        "splits": {
            split: {"route_ids": folders} for split, folders in split_folders.items()
        },
        "lineage_audit": {"route_catalog": catalog},
    }
    return parent, infos


def _lineage():
    return {
        "path": "/mock/parent.json",
        "sha256": "a" * 64,
        "size_bytes": 123,
        "schema_version": "spatial-uq-route-manifest/v1",
        "seed": 7,
    }


def test_annotation_candidate_is_not_visibility_and_missing_gt_is_not_background():
    candidate = annotation_candidate_from_info(
        {
            "gt_names": ["walker.pedestrian.0001", "traffic.stop"],
            "gt_boxes": [[10.0, 1.0], [5.0, 0.0]],
            "num_points": [3, 9],
        }
    )
    assert candidate.candidate is True
    assert candidate.candidate_classes == ("pedestrian",)
    with pytest.raises(B2DPilotSamplingError, match="missing GT is not background"):
        annotation_candidate_from_info({})


def test_pilot_preserves_parent_split_groups_variants_and_unique_identities():
    parent, infos = _mock_parent_and_infos()
    payload = build_b2d_pilot_submanifest(
        infos,
        parent,
        _lineage(),
        seed=23,
        route_count=8,
        target_states=800,
    )
    assert payload["selection"]["sampled_state_count"] == 800
    assert payload["selection"]["route_count"] == 8
    assert set(payload["split_statistics"]) == set(SPLITS)
    assert all(
        payload["split_statistics"][split]["canonical_route_count"] == 2
        for split in SPLITS
    )
    ids = [sample["sample_id"] for sample in payload["samples"]]
    assert len(ids) == len(set(ids)) == 800
    temporal = payload["temporal_execution_contract"]
    assert temporal["measurement_frame_count"] == 800
    assert temporal["chronological_replay_frame_count"] == 960
    assert temporal["estimated_minimum_forward_count_clean_plus_observed"] == 1920
    assert temporal["frame_independent_inference_forbidden"] is True
    assert temporal["memory_reset_between_folders_required"] is True
    assert temporal["memory_reset_between_clean_and_observed_passes_required"] is True
    assert len(temporal["folder_replay_statistics"]) == 16
    for stats in payload["route_statistics"].values():
        assert len(stats["folders"]) == 2
        assert stats["weather_variants"] == [0, 13]
        assert stats["repetition_variants"] == [0, 1]
        assert set(stats["folder_statistics"]) == set(stats["folders"])
        assert sum(
            item["sampled_frame_count"]
            for item in stats["folder_statistics"].values()
        ) == stats["sampled_states"]
        assert all(
            item["replay_frame_count"] == item["available_frame_count"] == 60
            and item["replay_frame_start"] == 0
            and item["replay_frame_end"] == 59
            for item in stats["folder_statistics"].values()
        )
    assert payload["audit"]["parent_split_preserved"] is True
    assert payload["audit"]["weather_repetition_variants_selected_as_canonical_groups"] is True
    assert payload["claim_boundary"]["annotation_candidate_equals_camera_visible"] is False
    assert payload["claim_boundary"]["unbiased_validation_metrics_supported"] is False
    assert payload["audit"]["closed_loop_carla_map_filter_applied"] is False
    assert payload["audit"]["offline_town12_allowed"] is True
    assert payload["claim_boundary"]["installed_carla_runtime_compatibility_required"] is False
    assert payload["claim_boundary"]["sampled_states_are_measurement_frames_only"] is True
    assert payload["claim_boundary"]["frame_independent_orion_inference_permitted"] is False
    assert any(
        stats["town"] == "Town12"
        for stats in payload["route_statistics"].values()
    )


def test_route_and_frame_sampling_are_deterministic_and_balanced():
    parent, infos = _mock_parent_and_infos()
    one = build_b2d_pilot_submanifest(infos, parent, _lineage(), seed=99)
    two = build_b2d_pilot_submanifest(infos, parent, _lineage(), seed=99)
    assert [item["sample_id"] for item in one["samples"]] == [
        item["sample_id"] for item in two["samples"]
    ]
    assert one["selection"]["route_quota_by_parent_split"] == {
        # The mock parent has four canonical routes in every split, so the two
        # post-minimum slots are distributed across the first tied splits.
        "train": 3,
        "validation": 3,
        "calibration": 2,
        "held_out": 2,
    }
    assert one["selection"]["annotation_candidate_count"] == 450
    assert one["selection"]["background_count"] == 450


def test_parent_variant_cross_split_and_missing_parent_folder_fail_closed():
    parent, infos = _mock_parent_and_infos()
    moved = parent["splits"]["train"]["route_ids"].pop()
    parent["splits"]["validation"]["route_ids"].append(moved)
    with pytest.raises(B2DPilotSamplingError, match="variants.*cross parent splits"):
        build_b2d_pilot_submanifest(infos, parent, _lineage(), route_count=8, target_states=800)

    parent, infos = _mock_parent_and_infos()
    missing_folder = parent["splits"]["train"]["route_ids"][0]
    infos = [info for info in infos if info["folder"] != missing_folder]
    with pytest.raises(B2DPilotSamplingError, match="parent manifest folders are absent"):
        build_b2d_pilot_submanifest(
            infos, parent, _lineage(), route_count=8, target_states=800
        )


def test_selected_folder_with_missing_chronological_frame_fails_closed():
    parent, infos = _mock_parent_and_infos()
    # Remove frame 7 from every folder so whichever routes are selected cannot
    # be misrepresented as a valid full chronological replay.
    infos = [info for info in infos if info["frame_idx"] != 7]
    with pytest.raises(B2DPilotSamplingError, match="not chronologically contiguous"):
        build_b2d_pilot_submanifest(
            infos, parent, _lineage(), route_count=8, target_states=800
        )

def test_load_parent_records_exact_file_sha(tmp_path):
    parent, _ = _mock_parent_and_infos()
    path = tmp_path / "parent.json"
    raw = json.dumps(parent, indent=2).encode("utf-8")
    path.write_bytes(raw)
    loaded, lineage = load_parent_manifest(path)
    assert loaded == parent
    assert lineage["sha256"] == hashlib.sha256(raw).hexdigest()


def test_cli_mock_dry_run_does_not_write(tmp_path):
    parent, infos = _mock_parent_and_infos()
    parent_path = tmp_path / "parent.json"
    infos_path = tmp_path / "infos.json"
    output = tmp_path / "pilot.json"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    infos_path.write_text(json.dumps({"infos": infos}), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_b2d_pilot_submanifest.py",
            "--infos",
            str(infos_path),
            "--parent-manifest",
            str(parent_path),
            "--output",
            str(output),
            "--route-count",
            "8",
            "--target-states",
            "800",
            "--dry-run",
            "--summary-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert not output.exists()
    assert summary["writes_performed"] is False
    assert summary["selection"]["sampled_state_count"] == 800
    assert summary["temporal_execution_contract"]["measurement_frame_count"] == 800
    assert summary["temporal_execution_contract"]["chronological_replay_frame_count"] == 960
    assert summary["temporal_execution_contract"]["estimated_minimum_forward_count_clean_plus_observed"] == 1920
    assert summary["audit"]["gpu_used"] is False
    assert summary["audit"]["scheduler_job_submitted"] is False
