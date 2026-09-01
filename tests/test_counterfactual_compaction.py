import json

import pytest
import torch

from uq_estimator.counterfactual_compaction import (
    deterministic_rademacher_projection,
    dynamic_symmetric_int8_dequantize,
    dynamic_symmetric_int8_quantize,
    project_feature_grid,
    projection_sha256,
)
from uq_estimator.counterfactual_evidence_training import (
    records_from_counterfactual_shard,
    targets_for_records,
)
from uq_estimator.counterfactual_sharded_dataset import (
    FP16_DIRECT_DATASET_SCHEMA_VERSION,
    load_fp16_dataset_manifest,
    load_fp16_dataset_records,
    load_fp16_route_shard_records,
    load_fp16_route_shard_records_selective,
    load_int8_dataset_manifest,
    load_int8_route_shard_records,
    write_fp16_route_shards_from_payload,
    write_direct_fp16_route_shard,
    write_int8_route_shards_from_payload,
)
from uq_estimator.observation_uq_shard import FEATURE_SHARD_SCHEMA_VERSION


def test_rademacher_projection_is_frozen_by_seed():
    first = deterministic_rademacher_projection(32, 8, 17)
    repeated = deterministic_rademacher_projection(32, 8, 17)
    different = deterministic_rademacher_projection(32, 8, 18)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, different)
    assert first.shape == (32, 8)
    assert projection_sha256(first) == projection_sha256(repeated)


def test_dynamic_int8_quantization_is_per_grid_channel_and_bounded():
    generator = torch.Generator().manual_seed(19)
    grid = torch.randn(2, 3, 4, 8, generator=generator)
    quantized, scale = dynamic_symmetric_int8_quantize(grid)
    reconstructed = dynamic_symmetric_int8_dequantize(
        quantized, scale, output_dtype=torch.float32
    )
    assert quantized.dtype == torch.int8
    assert scale.shape == (8,)
    assert reconstructed.shape == grid.shape
    error = (reconstructed - grid).abs().amax(dim=(0, 1, 2))
    assert bool((error <= scale * 0.51).all())


def test_project_feature_grid_preserves_nonfeature_axes_and_is_deterministic():
    generator = torch.Generator().manual_seed(9)
    grid = torch.randn(2, 3, 4, 32, generator=generator)
    projection = deterministic_rademacher_projection(32, 8, 17)
    first = project_feature_grid(
        grid, projection, torch.device("cpu"), output_dtype=torch.float32
    )
    repeated = project_feature_grid(
        grid, projection, torch.device("cpu"), output_dtype=torch.float32
    )
    assert first.shape == (2, 3, 4, 8)
    assert torch.equal(first, repeated)


def _tiny_counterfactual_shard():
    generator = torch.Generator().manual_seed(29)
    clean_features = []
    clean_items = []
    observed_features = []
    observed_items = []
    for frame_idx in range(2):
        clean = torch.randn(2, 2, 2, 4, generator=generator).half()
        clean_index = len(clean_features)
        clean_features.append(clean)
        clean_items.append(
            {
                "clean_index": clean_index,
                "sample_id": "route_a__frame_%06d" % frame_idx,
                "route_id": "route_a",
                "town": "Town01",
                "frame_idx": frame_idx,
                "split": "train",
            }
        )
        observed = clean.clone()
        observed[0, :1, :1] += 0.137
        observed_features.append(observed)
        mask = torch.zeros(2, 2, 2)
        mask[0, :1, :1] = 1.0
        observed_items.append(
            {
                "observed_index": len(observed_items),
                "clean_index": clean_index,
                "sample_id": clean_items[-1]["sample_id"] + "/local_blur/severity_1",
                "route_id": "route_a",
                "town": "Town01",
                "frame_idx": frame_idx,
                "split": "train",
                "family": "local_blur",
                "severity": 1.0,
                "corruption_mask": mask,
            }
        )
    return {
        "schema_version": FEATURE_SHARD_SCHEMA_VERSION,
        "clean_features": clean_features,
        "clean_items": clean_items,
        "observed_features": observed_features,
        "observed_items": observed_items,
        "provenance": {
            "extraction_schema_version": "orion.counterfactual-evidence-extraction/v2",
            "corruption_mask_is_primary_target": False,
        },
    }


def test_route_int8_shards_preserve_prequantization_targets(tmp_path):
    payload = _tiny_counterfactual_shard()
    original_records = records_from_counterfactual_shard(payload)
    original_target = targets_for_records(original_records, torch.device("cpu"))
    output = tmp_path / "int8-routes"
    manifest = write_int8_route_shards_from_payload(
        payload,
        output,
        source_feature_shard_sha256="a" * 64,
        quantization_batch_size=1,
        target_batch_size=1,
    )
    assert manifest["status"] == "complete"
    loaded_manifest = load_int8_dataset_manifest(output / "manifest.json")
    assert loaded_manifest == manifest
    compact_records = load_int8_route_shard_records(
        output / manifest["shards"][0]["file"]
    )
    compact_target = targets_for_records(compact_records, torch.device("cpu"))
    assert torch.equal(compact_target.values, original_target.values)
    assert torch.equal(
        compact_target.component_valid, original_target.component_valid
    )
    assert all(record.stored_target_values is not None for record in compact_records)
    assert manifest["shards"][0]["clean_quantization_error"][
        "maximum_error_over_scale"
    ] <= 0.501
    assert manifest["shards"][0]["observed_quantization_error"][
        "maximum_error_over_scale"
    ] <= 0.501


def test_route_fp16_shards_bitwise_preserve_inputs_and_targets(tmp_path):
    payload = _tiny_counterfactual_shard()
    original_records = records_from_counterfactual_shard(payload)
    original_target = targets_for_records(original_records, torch.device("cpu"))
    output = tmp_path / "fp16-routes"
    manifest = write_fp16_route_shards_from_payload(
        payload,
        output,
        source_feature_shard_sha256="b" * 64,
        target_batch_size=1,
    )
    assert manifest["status"] == "complete"
    assert manifest["shards"][0]["source_fp16_features_bitwise_preserved"] is True
    loaded_manifest = load_fp16_dataset_manifest(output / "manifest.json")
    assert loaded_manifest == manifest
    route_records = load_fp16_route_shard_records(
        output / manifest["shards"][0]["file"]
    )
    assert len(route_records) == len(original_records)
    for original, route_record in zip(original_records, route_records):
        assert torch.equal(route_record.reference_current, original.reference_current)
        assert torch.equal(route_record.observed_current, original.observed_current)
        assert torch.equal(route_record.reference_previous, original.reference_previous)
        assert torch.equal(route_record.observed_previous, original.observed_previous)
    route_target = targets_for_records(route_records, torch.device("cpu"))
    assert torch.equal(route_target.values, original_target.values)
    assert torch.equal(route_target.component_valid, original_target.component_valid)


def test_direct_fp16_route_shard_needs_no_monolithic_source(tmp_path):
    payload = _tiny_counterfactual_shard()
    output = tmp_path / "direct-fp16-routes"
    row = write_direct_fp16_route_shard(
        output,
        route_id="route_a",
        clean_features=payload["clean_features"],
        clean_items=payload["clean_items"],
        observed_features=payload["observed_features"],
        observed_items=payload["observed_items"],
        direct_extraction_fingerprint="c" * 64,
        extraction_schema_version="orion.counterfactual-evidence-extraction/v2",
        target_batch_size=1,
    )
    records = load_fp16_route_shard_records(output / row["file"])
    assert len(records) == 2
    assert row["direct_backbone_fp16_features_bitwise_preserved"] is True
    assert all(record.stored_target_values is not None for record in records)

    manifest = {
        "schema_version": FP16_DIRECT_DATASET_SCHEMA_VERSION,
        "status": "complete",
        "shards": [row],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_fp16_dataset_manifest(manifest_path) == manifest
    manifest["shards"].append(
        {
            "file": "heldout-must-not-be-read.pt",
            "sha256": "0" * 64,
            "split": "held_out",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    selected = load_fp16_dataset_records(
        manifest_path, splits=["train"], families=["local_blur"]
    )
    assert len(selected) == 2


def test_selective_fp16_loader_does_not_scan_excluded_family_values(tmp_path):
    payload = _tiny_counterfactual_shard()
    original_observed = list(payload["observed_features"])
    original_items = list(payload["observed_items"])
    for feature, item in zip(original_observed, original_items):
        copied_item = dict(item)
        copied_item["observed_index"] = len(payload["observed_features"])
        copied_item["sample_id"] = copied_item["sample_id"].replace(
            "local_blur", "local_glare"
        )
        copied_item["family"] = "local_glare"
        payload["observed_features"].append(feature.clone())
        payload["observed_items"].append(copied_item)

    output = tmp_path / "selective-fp16"
    row = write_direct_fp16_route_shard(
        output,
        route_id="route_a",
        clean_features=payload["clean_features"],
        clean_items=payload["clean_items"],
        observed_features=payload["observed_features"],
        observed_items=payload["observed_items"],
        direct_extraction_fingerprint="d" * 64,
        extraction_schema_version="orion.counterfactual-evidence-extraction/v2",
        target_batch_size=1,
    )
    shard_path = output / row["file"]
    stored = torch.load(shard_path, map_location="cpu", weights_only=False)
    for index, item in enumerate(stored["observed_items"]):
        if item["family"] == "local_glare":
            stored["observed_features"][index].fill_(float("nan"))
    torch.save(stored, shard_path)

    with pytest.raises(Exception):
        load_fp16_route_shard_records(shard_path)
    selected = load_fp16_route_shard_records_selective(
        shard_path, families=["local_blur"]
    )
    assert len(selected) == 2
    assert {record.family for record in selected} == {"local_blur"}
    assert all(torch.isfinite(record.observed_current).all() for record in selected)
