"""CPU tests for the auditable paired EVAViT feature extractor."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch

from uq_estimator.paired_feature_extraction import (
    PAIRED_EXTRACTION_SCHEMA_VERSION,
    PairedFeatureExtractionError,
    RouteFrameIdentity,
    build_info_identity_index,
    camera_view_names_from_info,
    exact_mask_to_patch_coverage,
    feature_map_to_patch_tokens,
    find_info_for_image_meta,
    make_representation_proxy_record,
    resolve_route_frame_identity,
    select_contiguous_route_balanced_infos,
)
from uq_estimator.spatial_training import (
    TARGET_REPRESENTATION_PROXY,
    load_paired_feature_records,
)


def _info(route="Route_Town01_Weather0", frame=7):
    return {
        "folder": route,
        "frame_idx": frame,
        "town_name": "Town01",
        "sensors": {"LIDAR_TOP": {}, "CAM_LEFT": {}, "CAM_FRONT": {}},
    }


def _meta(route="Route_Town01_Weather0", frame=7):
    return {"scene_token": route, "folder": route, "frame_idx": frame}


def test_feature_map_conversion_preserves_view_patch_and_channel_axes():
    feature_map = torch.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2).float()
    tokens = feature_map_to_patch_tokens(feature_map, batch_size=1, views=2)

    assert tokens.shape == (1, 2, 4, 3)
    torch.testing.assert_close(tokens[0, 0, 0], feature_map[0, :, 0, 0])
    torch.testing.assert_close(tokens[0, 1, 3], feature_map[1, :, 1, 1])


def test_exact_boolean_mask_becomes_fractional_patch_coverage():
    mask = torch.zeros(1, 2, 1, 4, 4, dtype=torch.bool)
    mask[0, 0, 0, 0:3, 0:3] = True
    coverage = exact_mask_to_patch_coverage(mask, patch_height=2, patch_width=2)

    assert coverage.shape == (1, 2, 4)
    torch.testing.assert_close(
        coverage[0, 0], torch.tensor([1.0, 0.5, 0.5, 0.25])
    )
    assert coverage[0, 1].sum() == 0
    with pytest.raises(PairedFeatureExtractionError, match="exact boolean"):
        exact_mask_to_patch_coverage(mask.float(), 2, 2)
    with pytest.raises(PairedFeatureExtractionError, match="divide evenly"):
        exact_mask_to_patch_coverage(mask[:, :, :, :3, :], 2, 2)


def test_route_identity_uses_info_and_scene_token_and_fails_closed():
    identity = resolve_route_frame_identity(_info(), _meta())
    assert identity.route_id == "Route_Town01_Weather0"
    assert identity.sample_token.endswith("frame_000007")

    with pytest.raises(PairedFeatureExtractionError, match="route mismatch"):
        resolve_route_frame_identity(_info(), _meta(route="other"))
    with pytest.raises(PairedFeatureExtractionError, match="scene_token"):
        # A filename is intentionally insufficient as a fallback.
        resolve_route_frame_identity(_info(), {"filename": ["route/frame.jpg"], "frame_idx": 7})


def test_info_index_rejects_duplicates_and_missing_tokens():
    index = build_info_identity_index([_info()])
    assert find_info_for_image_meta(index, _meta()) is index[("Route_Town01_Weather0", 7)]
    with pytest.raises(PairedFeatureExtractionError, match="duplicate"):
        build_info_identity_index([_info(), _info()])
    with pytest.raises(PairedFeatureExtractionError, match="absent"):
        find_info_for_image_meta(index, _meta(frame=8))


def test_camera_order_comes_from_annotations_without_front_assumption():
    assert camera_view_names_from_info(_info()) == ("CAM_LEFT", "CAM_FRONT")
    with pytest.raises(PairedFeatureExtractionError, match="no camera"):
        camera_view_names_from_info({"sensors": {"LIDAR_TOP": {}}})


def test_route_balanced_selector_preserves_splits_and_contiguous_frames():
    infos = []
    splits = {"train": [], "validation": [], "held_out": []}
    for split_index, split in enumerate(splits):
        for route_index in range(2):
            route = "%s_route_%d" % (split, route_index)
            splits[split].append(route)
            for frame in (0, 1, 2, 5):
                infos.append({"folder": route, "frame_idx": frame})
    manifest = {
        "schema_version": "spatial-uq-route-manifest/v1",
        "splits": {
            split: {"route_ids": route_ids} for split, route_ids in splits.items()
        },
    }
    selected = select_contiguous_route_balanced_infos(
        infos,
        manifest,
        {"train": 2, "validation": 1, "held_out": 1},
        samples_per_route=3,
    )
    assert len(selected) == 12
    by_route = {}
    for info in selected:
        by_route.setdefault(info["folder"], []).append(info["frame_idx"])
    assert len(by_route) == 4
    assert all(frames == [0, 1, 2] for frames in by_route.values())


def test_record_is_explicit_representation_proxy_not_semantic_uq():
    clean = torch.randn(2, 4, 6)
    corrupt = clean + 0.1
    record = make_representation_proxy_record(
        identity=RouteFrameIdentity("route", "Town01", 1, "route__frame_000001"),
        corruption="local_blur",
        severity=2,
        clean_patch_features=clean,
        corrupt_patch_features=corrupt,
        patch_corruption_coverage=torch.ones(2, 4),
        corruption_metadata={"schema_version": "orion.spatial_corruption.v1"},
        backbone_metadata={"type": "EVAViT", "global_pooling": False},
    )

    assert record.observed_patch_features.shape == (2, 4, 6)
    assert record.target_provenance == TARGET_REPRESENTATION_PROXY
    assert record.error_severity_target is None
    assert record.failure_event_target is None
    assert record.target_valid_mask is None
    contract = record.metadata["target_contract"]
    assert contract["semantic_uncertainty"] is False
    assert contract["supports_closed_loop_safety_claim"] is False
    assert record.metadata["source_identity"]["filename_fallback_used"] is False


def test_cli_mock_dry_run_is_cpu_only_and_writes_nothing(tmp_path):
    output = tmp_path / "must_not_exist.pt"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/extract_paired_spatial_features.py",
            "--mock",
            "--dry-run",
            "--output",
            str(output),
            "--mock-samples",
            "2",
            "--severities",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["schema_version"] == PAIRED_EXTRACTION_SCHEMA_VERSION
    assert summary["record_count"] == 2
    assert summary["target_provenance"] == [TARGET_REPRESENTATION_PROXY]
    assert summary["writes_performed"] is False
    assert not output.exists()


def test_cli_rejects_non_positive_output_guard():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/extract_paired_spatial_features.py",
            "--mock",
            "--dry-run",
            "--max-output-gb",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "--max-output-gb must be positive" in completed.stderr


def test_cli_mock_output_round_trips_into_spatial_training(tmp_path):
    output = tmp_path / "paired.pt"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/extract_paired_spatial_features.py",
            "--mock",
            "--output",
            str(output),
            "--mock-samples",
            "4",
            "--corruption",
            "local_dark",
            "--severities",
            "1",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    records = load_paired_feature_records(output)

    assert summary["record_count"] == len(records) == 8
    assert summary["route_count"] == 4
    assert summary["global_pooling"] is False
    assert {record.observed_patch_features.shape for record in records} == {
        torch.Size([2, 20, 6])
    }
    assert all(record.target_provenance == TARGET_REPRESENTATION_PROXY for record in records)
