"""Route-local int8 shards with pre-quantization counterfactual targets.

The monolithic FP16 feature shard is privileged conversion input.  Targets are
computed from those original features first; only adapter inputs are quantized.
Each output shard owns complete routes so temporal predecessors never cross a
file boundary and shards can be appended without rewriting older payloads.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from uq_estimator.counterfactual_compaction import (
    dynamic_symmetric_int8_dequantize,
    dynamic_symmetric_int8_quantize,
)
from uq_estimator.counterfactual_evidence import (
    CounterfactualEvidenceError,
    counterfactual_evidence_target,
)
from uq_estimator.counterfactual_evidence_training import CounterfactualEvidenceRecord
from uq_estimator.observation_uq_shard import validate_feature_shard


INT8_DATASET_SCHEMA_VERSION = "orion.counterfactual-int8-route-dataset/v1"
INT8_ROUTE_SHARD_SCHEMA_VERSION = "orion.counterfactual-int8-route-shard/v1"
INT8_QUANTIZATION_KIND = "dynamic_symmetric_int8_per_grid_channel"
FP16_DATASET_SCHEMA_VERSION = "orion.counterfactual-fp16-route-dataset/v1"
FP16_ROUTE_SHARD_SCHEMA_VERSION = "orion.counterfactual-fp16-route-shard/v1"
FP16_DIRECT_DATASET_SCHEMA_VERSION = "orion.counterfactual-fp16-route-dataset/v2"
FP16_DIRECT_ROUTE_SHARD_SCHEMA_VERSION = "orion.counterfactual-fp16-route-shard/v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_sha256(values: torch.Tensor, validity: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(values.detach().cpu().float().contiguous().numpy().tobytes())
    digest.update(validity.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _feature_sha256(clean: torch.Tensor, observed: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in (clean, observed):
        if value.dtype != torch.float16 or value.ndim != 5:
            raise CounterfactualEvidenceError("FP16 route feature tensor differs")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _quantize_features(
    features: Sequence[torch.Tensor],
    indices: Sequence[int],
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    if not indices or batch_size <= 0:
        raise CounterfactualEvidenceError("int8 shard quantization request is empty")
    quantized_chunks = []
    scale_chunks = []
    maximum_absolute_error = 0.0
    maximum_error_over_scale = 0.0
    for start in range(0, len(indices), batch_size):
        batch = torch.stack([features[index] for index in indices[start : start + batch_size]])
        quantized, scale = dynamic_symmetric_int8_quantize(batch)
        reconstructed = dynamic_symmetric_int8_dequantize(
            quantized, scale, output_dtype=torch.float32
        )
        error = (reconstructed - batch.float()).abs()
        scale_grid = scale[:, None, None, None, :]
        maximum_absolute_error = max(maximum_absolute_error, float(error.max()))
        maximum_error_over_scale = max(
            maximum_error_over_scale,
            float((error / scale_grid).max()),
        )
        quantized_chunks.append(quantized.cpu())
        scale_chunks.append(scale.cpu().float())
    return (
        torch.cat(quantized_chunks),
        torch.cat(scale_chunks),
        {
            "maximum_absolute_error": maximum_absolute_error,
            "maximum_error_over_scale": maximum_error_over_scale,
        },
    )


def _targets_for_observed_indices(
    clean_features: Sequence[torch.Tensor],
    clean_items: Sequence[Mapping[str, Any]],
    observed_features: Sequence[torch.Tensor],
    observed_items: Sequence[Mapping[str, Any]],
    observed_indices: Sequence[int],
    batch_size: int,
    target_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[bool]]:
    if not observed_indices or batch_size <= 0:
        raise CounterfactualEvidenceError("int8 shard target request is empty")
    clean_key = {
        (str(item["route_id"]), int(item["frame_idx"])): index
        for index, item in enumerate(clean_items)
    }
    observed_key = {
        (
            str(item["route_id"]),
            int(item["frame_idx"]),
            str(item["family"]),
            float(item["severity"]),
        ): index
        for index, item in enumerate(observed_items)
    }
    target_values = []
    target_validity = []
    previous_flags: List[bool] = []
    for start in range(0, len(observed_indices), batch_size):
        current_indices = observed_indices[start : start + batch_size]
        references = []
        observations = []
        reference_previous = []
        observed_previous = []
        valid = []
        for observed_index in current_indices:
            item = observed_items[observed_index]
            route_id = str(item["route_id"])
            frame_idx = int(item["frame_idx"])
            family = str(item["family"])
            severity = float(item["severity"])
            clean_index = int(item["clean_index"])
            previous_clean_index = clean_key.get((route_id, frame_idx - 1))
            previous_observed_index = observed_key.get(
                (route_id, frame_idx - 1, family, severity)
            )
            if (previous_clean_index is None) != (previous_observed_index is None):
                raise CounterfactualEvidenceError(
                    "reference/observed temporal sequence availability differs"
                )
            current_reference = clean_features[clean_index]
            current_observed = observed_features[observed_index]
            references.append(current_reference)
            observations.append(current_observed)
            if previous_clean_index is None:
                reference_previous.append(torch.zeros_like(current_reference))
                observed_previous.append(torch.zeros_like(current_observed))
                valid.append(False)
            else:
                assert previous_observed_index is not None
                reference_previous.append(clean_features[previous_clean_index])
                observed_previous.append(observed_features[previous_observed_index])
                valid.append(True)
        valid_tensor = torch.tensor(valid, dtype=torch.bool)
        device = target_device or references[0].device
        target = counterfactual_evidence_target(
            torch.stack(references).to(device),
            torch.stack(observations).to(device),
            torch.stack(reference_previous).to(device),
            torch.stack(observed_previous).to(device),
            valid_tensor.to(device),
        )
        target_values.append(target.values.cpu().float())
        target_validity.append(target.component_valid.cpu())
        previous_flags.extend(valid)
    return torch.cat(target_values), torch.cat(target_validity), previous_flags


def _localize_items(
    clean_items: Sequence[Mapping[str, Any]],
    observed_items: Sequence[Mapping[str, Any]],
    clean_indices: Sequence[int],
    observed_indices: Sequence[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    clean_local = {global_index: local for local, global_index in enumerate(clean_indices)}
    localized_clean = []
    for local, global_index in enumerate(clean_indices):
        item = dict(clean_items[global_index])
        item["source_clean_index"] = int(item.get("clean_index", global_index))
        item["clean_index"] = local
        localized_clean.append(item)
    localized_observed = []
    for local, global_index in enumerate(observed_indices):
        item = dict(observed_items[global_index])
        source_clean = int(item["clean_index"])
        if source_clean not in clean_local:
            raise CounterfactualEvidenceError("route shard observed item crosses routes")
        item["source_observed_index"] = int(item.get("observed_index", global_index))
        item["source_clean_index"] = source_clean
        item["observed_index"] = local
        item["clean_index"] = clean_local[source_clean]
        localized_observed.append(item)
    return localized_clean, localized_observed


def write_int8_route_shards_from_payload(
    payload: Mapping[str, Any],
    output_dir: Path,
    *,
    source_feature_shard_sha256: str,
    max_routes: Optional[int] = None,
    quantization_batch_size: int = 2,
    target_batch_size: int = 2,
) -> Dict[str, Any]:
    """Convert a validated FP16 payload into atomic, whole-route int8 shards."""

    summary = validate_feature_shard(payload)
    provenance = payload.get("provenance", {})
    if provenance.get("extraction_schema_version") not in {
        "orion.counterfactual-evidence-extraction/v1",
        "orion.counterfactual-evidence-extraction/v2",
    } or provenance.get("corruption_mask_is_primary_target") is not False:
        raise CounterfactualEvidenceError("source shard attestation is unsafe")
    if len(source_feature_shard_sha256) != 64:
        raise CounterfactualEvidenceError("source feature shard SHA256 is invalid")
    if max_routes is not None and max_routes <= 0:
        raise CounterfactualEvidenceError("max_routes must be positive")

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite int8 dataset %s" % output_dir)
    temporary = output_dir.parent / (output_dir.name + ".tmp-" + uuid.uuid4().hex)
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        clean_features = payload["clean_features"]
        clean_items = payload["clean_items"]
        observed_features = payload["observed_features"]
        observed_items = payload["observed_items"]
        clean_by_route: Dict[str, List[int]] = defaultdict(list)
        observed_by_route: Dict[str, List[int]] = defaultdict(list)
        for index, item in enumerate(clean_items):
            clean_by_route[str(item["route_id"])].append(index)
        for index, item in enumerate(observed_items):
            observed_by_route[str(item["route_id"])].append(index)
        route_ids = sorted(clean_by_route)
        if set(route_ids) != set(observed_by_route):
            raise CounterfactualEvidenceError("clean/observed route sets differ")
        selected_route_ids = route_ids[:max_routes] if max_routes is not None else route_ids

        shard_rows = []
        for shard_index, route_id in enumerate(selected_route_ids):
            clean_indices = sorted(
                clean_by_route[route_id],
                key=lambda index: int(clean_items[index]["frame_idx"]),
            )
            observed_indices = sorted(
                observed_by_route[route_id],
                key=lambda index: (
                    int(observed_items[index]["frame_idx"]),
                    str(observed_items[index]["family"]),
                    float(observed_items[index]["severity"]),
                ),
            )
            localized_clean, localized_observed = _localize_items(
                clean_items,
                observed_items,
                clean_indices,
                observed_indices,
            )
            clean_q, clean_scale, clean_error = _quantize_features(
                clean_features, clean_indices, quantization_batch_size
            )
            observed_q, observed_scale, observed_error = _quantize_features(
                observed_features, observed_indices, quantization_batch_size
            )
            target_values, target_validity, previous_flags = _targets_for_observed_indices(
                clean_features,
                clean_items,
                observed_features,
                observed_items,
                observed_indices,
                target_batch_size,
            )
            for item, previous_valid in zip(localized_observed, previous_flags):
                item["previous_valid"] = previous_valid
            shard_payload = {
                "schema_version": INT8_ROUTE_SHARD_SCHEMA_VERSION,
                "route_ids": [route_id],
                "feature_shape": summary["feature_shape"],
                "quantization": {
                    "kind": INT8_QUANTIZATION_KIND,
                    "scale_scope": "per-grid-per-feature-channel",
                    "scale_dtype": "float32",
                    "dequantized_dtype": "float16",
                },
                "target": {
                    "computed_before_input_quantization": True,
                    "storage_dtype": "float32",
                    "corruption_mask_is_primary_target": False,
                    "values": target_values,
                    "component_valid": target_validity,
                },
                "clean_quantized": clean_q,
                "clean_scale": clean_scale,
                "clean_items": localized_clean,
                "observed_quantized": observed_q,
                "observed_scale": observed_scale,
                "observed_items": localized_observed,
                "source": {
                    "feature_shard_sha256": source_feature_shard_sha256,
                    "extraction_schema_version": provenance.get(
                        "extraction_schema_version"
                    ),
                },
            }
            shard_name = "route-%04d.pt" % shard_index
            shard_path = temporary / shard_name
            torch.save(shard_payload, shard_path)
            shard_rows.append(
                {
                    "file": shard_name,
                    "sha256": _sha256(shard_path),
                    "route_ids": [route_id],
                    "split": str(localized_clean[0]["split"]),
                    "clean_count": len(clean_indices),
                    "observed_count": len(observed_indices),
                    "size_bytes": shard_path.stat().st_size,
                    "prequantization_target_sha256": _target_sha256(
                        target_values, target_validity
                    ),
                    "clean_quantization_error": clean_error,
                    "observed_quantization_error": observed_error,
                }
            )
        is_complete = len(selected_route_ids) == len(route_ids)
        manifest = {
            "schema_version": INT8_DATASET_SCHEMA_VERSION,
            "status": "complete" if is_complete else "partial_probe_not_for_training",
            "source": {
                "feature_shard_sha256": source_feature_shard_sha256,
                "route_count": len(route_ids),
                "clean_count": summary["clean_count"],
                "observed_count": summary["observed_count"],
                "feature_shape": summary["feature_shape"],
            },
            "storage_contract": {
                "whole_routes_per_shard": True,
                "append_without_rewriting_existing_shards": True,
                "input_quantization": INT8_QUANTIZATION_KIND,
                "targets_computed_from_original_fp16_before_quantization": True,
                "corruption_mask_is_optimizer_target": False,
            },
            "written_route_count": len(selected_route_ids),
            "written_clean_count": sum(row["clean_count"] for row in shard_rows),
            "written_observed_count": sum(row["observed_count"] for row in shard_rows),
            "shards": shard_rows,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def write_fp16_route_shards_from_payload(
    payload: Mapping[str, Any],
    output_dir: Path,
    *,
    source_feature_shard_sha256: str,
    max_routes: Optional[int] = None,
    target_batch_size: int = 2,
) -> Dict[str, Any]:
    """Copy exact FP16 inputs into atomic route shards and store FP32 targets."""

    summary = validate_feature_shard(payload)
    provenance = payload.get("provenance", {})
    if provenance.get("extraction_schema_version") not in {
        "orion.counterfactual-evidence-extraction/v1",
        "orion.counterfactual-evidence-extraction/v2",
    } or provenance.get("corruption_mask_is_primary_target") is not False:
        raise CounterfactualEvidenceError("source shard attestation is unsafe")
    if len(source_feature_shard_sha256) != 64:
        raise CounterfactualEvidenceError("source feature shard SHA256 is invalid")
    if max_routes is not None and max_routes <= 0:
        raise CounterfactualEvidenceError("max_routes must be positive")
    if target_batch_size <= 0:
        raise CounterfactualEvidenceError("target batch size must be positive")

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite FP16 dataset %s" % output_dir)
    temporary = output_dir.parent / (output_dir.name + ".tmp-" + uuid.uuid4().hex)
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        clean_features = payload["clean_features"]
        clean_items = payload["clean_items"]
        observed_features = payload["observed_features"]
        observed_items = payload["observed_items"]
        if any(value.dtype != torch.float16 for value in clean_features + observed_features):
            raise CounterfactualEvidenceError(
                "lossless FP16 route conversion requires FP16 source features"
            )
        clean_by_route: Dict[str, List[int]] = defaultdict(list)
        observed_by_route: Dict[str, List[int]] = defaultdict(list)
        for index, item in enumerate(clean_items):
            clean_by_route[str(item["route_id"])].append(index)
        for index, item in enumerate(observed_items):
            observed_by_route[str(item["route_id"])].append(index)
        route_ids = sorted(clean_by_route)
        if set(route_ids) != set(observed_by_route):
            raise CounterfactualEvidenceError("clean/observed route sets differ")
        selected_route_ids = route_ids[:max_routes] if max_routes is not None else route_ids

        shard_rows = []
        for shard_index, route_id in enumerate(selected_route_ids):
            clean_indices = sorted(
                clean_by_route[route_id],
                key=lambda index: int(clean_items[index]["frame_idx"]),
            )
            observed_indices = sorted(
                observed_by_route[route_id],
                key=lambda index: (
                    int(observed_items[index]["frame_idx"]),
                    str(observed_items[index]["family"]),
                    float(observed_items[index]["severity"]),
                ),
            )
            localized_clean, localized_observed = _localize_items(
                clean_items,
                observed_items,
                clean_indices,
                observed_indices,
            )
            clean_fp16 = torch.stack(
                [clean_features[index] for index in clean_indices]
            ).contiguous()
            observed_fp16 = torch.stack(
                [observed_features[index] for index in observed_indices]
            ).contiguous()
            feature_sha = _feature_sha256(clean_fp16, observed_fp16)
            target_values, target_validity, previous_flags = _targets_for_observed_indices(
                clean_features,
                clean_items,
                observed_features,
                observed_items,
                observed_indices,
                target_batch_size,
            )
            for item, previous_valid in zip(localized_observed, previous_flags):
                item["previous_valid"] = previous_valid
            shard_payload = {
                "schema_version": FP16_ROUTE_SHARD_SCHEMA_VERSION,
                "route_ids": [route_id],
                "feature_shape": summary["feature_shape"],
                "storage": {
                    "feature_dtype": "float16",
                    "lossless_copy_from_source_features": True,
                },
                "target": {
                    "computed_from_source_fp16": True,
                    "storage_dtype": "float32",
                    "corruption_mask_is_primary_target": False,
                    "values": target_values,
                    "component_valid": target_validity,
                },
                "clean_features": clean_fp16,
                "clean_items": localized_clean,
                "observed_features": observed_fp16,
                "observed_items": localized_observed,
                "source": {
                    "feature_shard_sha256": source_feature_shard_sha256,
                    "route_feature_sha256": feature_sha,
                    "extraction_schema_version": provenance.get(
                        "extraction_schema_version"
                    ),
                },
            }
            shard_name = "route-%04d.pt" % shard_index
            shard_path = temporary / shard_name
            torch.save(shard_payload, shard_path)
            loaded = torch.load(shard_path, map_location="cpu")
            if _feature_sha256(
                loaded["clean_features"], loaded["observed_features"]
            ) != feature_sha:
                raise CounterfactualEvidenceError("saved FP16 route feature hash differs")
            target_sha = _target_sha256(target_values, target_validity)
            if _target_sha256(
                loaded["target"]["values"], loaded["target"]["component_valid"]
            ) != target_sha:
                raise CounterfactualEvidenceError("saved FP16 route target hash differs")
            del loaded
            shard_rows.append(
                {
                    "file": shard_name,
                    "sha256": _sha256(shard_path),
                    "route_ids": [route_id],
                    "split": str(localized_clean[0]["split"]),
                    "clean_count": len(clean_indices),
                    "observed_count": len(observed_indices),
                    "size_bytes": shard_path.stat().st_size,
                    "lossless_fp16_feature_sha256": feature_sha,
                    "source_fp16_features_bitwise_preserved": True,
                    "source_fp16_target_sha256": target_sha,
                }
            )
        is_complete = len(selected_route_ids) == len(route_ids)
        manifest = {
            "schema_version": FP16_DATASET_SCHEMA_VERSION,
            "status": "complete" if is_complete else "partial_probe_not_for_training",
            "source": {
                "feature_shard_sha256": source_feature_shard_sha256,
                "route_count": len(route_ids),
                "clean_count": summary["clean_count"],
                "observed_count": summary["observed_count"],
                "feature_shape": summary["feature_shape"],
            },
            "storage_contract": {
                "whole_routes_per_shard": True,
                "feature_dtype": "float16",
                "lossless_copy_from_source_features": True,
                "targets_computed_from_source_fp16": True,
                "target_dtype": "float32",
                "corruption_mask_is_optimizer_target": False,
            },
            "written_route_count": len(selected_route_ids),
            "written_clean_count": sum(row["clean_count"] for row in shard_rows),
            "written_observed_count": sum(row["observed_count"] for row in shard_rows),
            "shards": shard_rows,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def write_direct_fp16_route_shard(
    output_dir: Path,
    *,
    route_id: str,
    clean_features: Sequence[torch.Tensor],
    clean_items: Sequence[Mapping[str, Any]],
    observed_features: Sequence[torch.Tensor],
    observed_items: Sequence[Mapping[str, Any]],
    direct_extraction_fingerprint: str,
    extraction_schema_version: str,
    target_batch_size: int = 2,
    target_device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Atomically persist one route directly from the frozen backbone output.

    Unlike :func:`write_fp16_route_shards_from_payload`, this path never creates
    a monolithic intermediate shard.  It computes the FP32 evidence target from
    the in-memory FP16 route tensors before the route buffer is released.
    """

    route_id = str(route_id).strip()
    if not route_id or len(direct_extraction_fingerprint) != 64:
        raise CounterfactualEvidenceError("direct FP16 route lineage is invalid")
    if not str(extraction_schema_version).strip() or target_batch_size <= 0:
        raise CounterfactualEvidenceError("direct FP16 route settings are invalid")
    if not clean_features or not observed_features:
        raise CounterfactualEvidenceError("direct FP16 route buffers are empty")
    if len(clean_features) != len(clean_items) or len(observed_features) != len(
        observed_items
    ):
        raise CounterfactualEvidenceError("direct FP16 route item counts differ")
    if any(value.dtype != torch.float16 or value.ndim != 4 for value in clean_features):
        raise CounterfactualEvidenceError("direct clean features must be FP16 grids")
    if any(
        value.dtype != torch.float16 or value.ndim != 4 for value in observed_features
    ):
        raise CounterfactualEvidenceError("direct observed features must be FP16 grids")
    feature_shape = tuple(clean_features[0].shape)
    if any(tuple(value.shape) != feature_shape for value in clean_features) or any(
        tuple(value.shape) != feature_shape for value in observed_features
    ):
        raise CounterfactualEvidenceError("direct FP16 route feature shapes differ")
    if {
        str(item.get("route_id", "")) for item in list(clean_items) + list(observed_items)
    } != {route_id}:
        raise CounterfactualEvidenceError("direct FP16 shard must own exactly one route")

    clean_indices = sorted(
        range(len(clean_items)), key=lambda index: int(clean_items[index]["frame_idx"])
    )
    observed_indices = sorted(
        range(len(observed_items)),
        key=lambda index: (
            int(observed_items[index]["frame_idx"]),
            str(observed_items[index]["family"]),
            float(observed_items[index]["severity"]),
        ),
    )
    localized_clean, localized_observed = _localize_items(
        clean_items, observed_items, clean_indices, observed_indices
    )
    clean_fp16 = torch.stack([clean_features[index] for index in clean_indices]).contiguous()
    observed_fp16 = torch.stack(
        [observed_features[index] for index in observed_indices]
    ).contiguous()
    feature_sha = _feature_sha256(clean_fp16, observed_fp16)
    target_values, target_validity, previous_flags = _targets_for_observed_indices(
        clean_features,
        clean_items,
        observed_features,
        observed_items,
        observed_indices,
        target_batch_size,
        target_device,
    )
    for item, previous_valid in zip(localized_observed, previous_flags):
        item["previous_valid"] = previous_valid
    target_sha = _target_sha256(target_values, target_validity)
    payload = {
        "schema_version": FP16_DIRECT_ROUTE_SHARD_SCHEMA_VERSION,
        "route_ids": [route_id],
        "feature_shape": list(feature_shape),
        "storage": {
            "feature_dtype": "float16",
            "direct_from_frozen_backbone": True,
            "monolithic_intermediate_created": False,
        },
        "target": {
            "computed_from_source_fp16": True,
            "storage_dtype": "float32",
            "corruption_mask_is_primary_target": False,
            "values": target_values,
            "component_valid": target_validity,
        },
        "clean_features": clean_fp16,
        "clean_items": localized_clean,
        "observed_features": observed_fp16,
        "observed_items": localized_observed,
        "source": {
            "direct_extraction_fingerprint": direct_extraction_fingerprint,
            "direct_from_frozen_backbone": True,
            "route_feature_sha256": feature_sha,
            "extraction_schema_version": extraction_schema_version,
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    route_token = hashlib.sha256(route_id.encode("utf-8")).hexdigest()[:20]
    shard_name = "route-%s.pt" % route_token
    shard_path = output_dir / shard_name
    if shard_path.exists():
        raise FileExistsError("refusing to overwrite direct FP16 route shard %s" % shard_path)
    temporary = output_dir / (shard_name + ".tmp-" + uuid.uuid4().hex)
    try:
        torch.save(payload, temporary)
        loaded = torch.load(temporary, map_location="cpu")
        validate_fp16_route_shard_payload(loaded)
        if _target_sha256(
            loaded["target"]["values"], loaded["target"]["component_valid"]
        ) != target_sha:
            raise CounterfactualEvidenceError("saved direct FP16 target hash differs")
        del loaded
        temporary.rename(shard_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "file": shard_name,
        "sha256": _sha256(shard_path),
        "route_ids": [route_id],
        "split": str(localized_clean[0]["split"]),
        "clean_count": len(clean_indices),
        "observed_count": len(observed_indices),
        "feature_shape": list(feature_shape),
        "size_bytes": shard_path.stat().st_size,
        "lossless_fp16_feature_sha256": feature_sha,
        "direct_backbone_fp16_features_bitwise_preserved": True,
        "source_fp16_target_sha256": target_sha,
    }


def _validate_fp16_route_shard_structure(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate metadata and shapes without scanning mapped feature storage."""

    if payload.get("schema_version") not in {
        FP16_ROUTE_SHARD_SCHEMA_VERSION,
        FP16_DIRECT_ROUTE_SHARD_SCHEMA_VERSION,
    }:
        raise CounterfactualEvidenceError("unsupported FP16 route shard schema")
    clean = payload.get("clean_features")
    observed = payload.get("observed_features")
    clean_items = payload.get("clean_items")
    observed_items = payload.get("observed_items")
    if (
        not torch.is_tensor(clean)
        or not torch.is_tensor(observed)
        or clean.dtype != torch.float16
        or observed.dtype != torch.float16
        or clean.ndim != 5
        or observed.ndim != 5
        or clean.shape[1:] != observed.shape[1:]
    ):
        raise CounterfactualEvidenceError("FP16 route feature payload differs")
    if (
        not isinstance(clean_items, list)
        or len(clean_items) != clean.shape[0]
        or not isinstance(observed_items, list)
        or len(observed_items) != observed.shape[0]
    ):
        raise CounterfactualEvidenceError("FP16 route item counts differ")
    source = payload.get("source")
    if (
        not isinstance(source, Mapping)
        or len(str(source.get("route_feature_sha256", ""))) != 64
    ):
        raise CounterfactualEvidenceError("FP16 route feature lineage differs")
    if payload.get("schema_version") == FP16_DIRECT_ROUTE_SHARD_SCHEMA_VERSION and (
        source.get("direct_from_frozen_backbone") is not True
        or len(str(source.get("direct_extraction_fingerprint", ""))) != 64
    ):
        raise CounterfactualEvidenceError("direct FP16 route lineage differs")
    target = payload.get("target")
    if not isinstance(target, Mapping) or target.get("computed_from_source_fp16") is not True:
        raise CounterfactualEvidenceError("FP16 route target attestation differs")
    values = target.get("values")
    validity = target.get("component_valid")
    expected_target_shape = tuple(observed.shape[:-1]) + (3,)
    if (
        not torch.is_tensor(values)
        or tuple(values.shape) != expected_target_shape
        or values.dtype != torch.float32
        or not torch.is_tensor(validity)
        or tuple(validity.shape) != expected_target_shape
        or validity.dtype != torch.bool
    ):
        raise CounterfactualEvidenceError("FP16 route stored target differs")
    return {
        "clean_count": clean.shape[0],
        "observed_count": observed.shape[0],
        "feature_shape": list(clean.shape[1:]),
    }


def validate_fp16_route_shard_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = _validate_fp16_route_shard_structure(payload)
    clean = payload["clean_features"]
    observed = payload["observed_features"]
    source = payload["source"]
    values = payload["target"]["values"]
    if (
        not bool(torch.isfinite(clean).all())
        or not bool(torch.isfinite(observed).all())
        or source.get("route_feature_sha256") != _feature_sha256(clean, observed)
    ):
        raise CounterfactualEvidenceError("FP16 route feature hash differs")
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise CounterfactualEvidenceError("FP16 route stored target differs")
    return summary


def _fp16_records_from_payload(
    payload: Mapping[str, Any], observed_indices: Sequence[int]
) -> List[CounterfactualEvidenceRecord]:
    if not isinstance(payload, Mapping):
        raise CounterfactualEvidenceError("FP16 route shard must be a mapping")
    clean = payload["clean_features"]
    observed = payload["observed_features"]
    clean_items = payload["clean_items"]
    observed_items = payload["observed_items"]
    clean_key = {
        (str(item["route_id"]), int(item["frame_idx"])): index
        for index, item in enumerate(clean_items)
    }
    observed_key = {
        (
            str(item["route_id"]),
            int(item["frame_idx"]),
            str(item["family"]),
            float(item["severity"]),
        ): index
        for index, item in enumerate(observed_items)
    }
    target = payload["target"]
    records = []
    for index in observed_indices:
        item = observed_items[index]
        route_id = str(item["route_id"])
        frame_idx = int(item["frame_idx"])
        family = str(item["family"])
        severity = float(item["severity"])
        clean_index = int(item["clean_index"])
        previous_clean = clean_key.get((route_id, frame_idx - 1))
        previous_observed = observed_key.get(
            (route_id, frame_idx - 1, family, severity)
        )
        if (previous_clean is None) != (previous_observed is None):
            raise CounterfactualEvidenceError(
                "FP16 route reference/observed temporal availability differs"
            )
        declared_previous_valid = bool(item.get("previous_valid"))
        if declared_previous_valid != (previous_clean is not None):
            raise CounterfactualEvidenceError("FP16 route previous_valid differs")
        records.append(
            CounterfactualEvidenceRecord(
                sample_id=str(item["sample_id"]),
                pair_id=str(clean_items[clean_index]["sample_id"]),
                route_id=route_id,
                frame_idx=frame_idx,
                split=str(item["split"]),
                family=family,
                severity=severity,
                reference_current=clean[clean_index],
                observed_current=observed[index],
                reference_previous=(
                    clean[previous_clean]
                    if previous_clean is not None
                    else torch.zeros_like(clean[clean_index])
                ),
                observed_previous=(
                    observed[previous_observed]
                    if previous_observed is not None
                    else torch.zeros_like(observed[index])
                ),
                previous_valid=declared_previous_valid,
                corruption_mask=item.get("corruption_mask"),
                stored_target_values=target["values"][index],
                stored_target_component_valid=target["component_valid"][index],
            )
        )
    return records


def load_fp16_route_shard_records(path: Path) -> List[CounterfactualEvidenceRecord]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise CounterfactualEvidenceError("FP16 route shard must be a mapping")
    validate_fp16_route_shard_payload(payload)
    return _fp16_records_from_payload(payload, range(len(payload["observed_items"])))


def load_fp16_route_shard_records_selective(
    path: Path,
    *,
    families: Sequence[str],
) -> List[CounterfactualEvidenceRecord]:
    """Mmap a route and expose only explicitly selected family records.

    The caller must pin a previously audited dataset-manifest hash.  Excluded
    co-sharded family tensor pages are never selected, stacked, or evaluated.
    """

    selected_families = {str(value) for value in families}
    if not selected_families:
        raise CounterfactualEvidenceError("selective FP16 family set is empty")
    payload = torch.load(
        Path(path), map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(payload, Mapping):
        raise CounterfactualEvidenceError("FP16 route shard must be a mapping")
    _validate_fp16_route_shard_structure(payload)
    observed_indices = [
        index
        for index, item in enumerate(payload["observed_items"])
        if str(item.get("family")) in selected_families
    ]
    records = _fp16_records_from_payload(payload, observed_indices)
    if not records:
        raise CounterfactualEvidenceError("selective FP16 route contains no records")
    if {record.family for record in records} - selected_families:
        raise CounterfactualEvidenceError("selective FP16 route leaked a family")
    return records


def load_fp16_dataset_manifest(path: Path, verify_shards: bool = True) -> Dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {
        FP16_DATASET_SCHEMA_VERSION,
        FP16_DIRECT_DATASET_SCHEMA_VERSION,
    }:
        raise CounterfactualEvidenceError("unsupported FP16 dataset manifest schema")
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise CounterfactualEvidenceError("FP16 dataset manifest has no shards")
    if verify_shards:
        for row in shards:
            shard_path = path.parent / str(row["file"])
            if not shard_path.is_file() or _sha256(shard_path) != row["sha256"]:
                raise CounterfactualEvidenceError("FP16 dataset shard hash differs")
    return payload


def load_fp16_dataset_records(
    path: Path,
    *,
    splits: Optional[Sequence[str]] = None,
    families: Optional[Sequence[str]] = None,
    verify_shards: bool = True,
) -> List[CounterfactualEvidenceRecord]:
    """Load selected whole-route shards without reading excluded splits."""

    path = Path(path)
    manifest = load_fp16_dataset_manifest(path, verify_shards=False)
    split_filter = {str(value) for value in splits} if splits is not None else None
    family_filter = {str(value) for value in families} if families is not None else None
    records: List[CounterfactualEvidenceRecord] = []
    selected_rows = [
        row
        for row in manifest["shards"]
        if split_filter is None or str(row.get("split", "")) in split_filter
    ]
    if not selected_rows:
        raise CounterfactualEvidenceError("FP16 dataset selection contains no shards")
    for row in selected_rows:
        shard_path = path.parent / str(row["file"])
        if verify_shards and (
            not shard_path.is_file() or _sha256(shard_path) != row["sha256"]
        ):
            raise CounterfactualEvidenceError("FP16 selected dataset shard hash differs")
        route_records = load_fp16_route_shard_records(shard_path)
        if family_filter is not None:
            route_records = [
                record for record in route_records if record.family in family_filter
            ]
        records.extend(route_records)
    if not records:
        raise CounterfactualEvidenceError("FP16 dataset selection contains no records")
    return records


def validate_int8_route_shard_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if payload.get("schema_version") != INT8_ROUTE_SHARD_SCHEMA_VERSION:
        raise CounterfactualEvidenceError("unsupported int8 route shard schema")
    clean_q = payload.get("clean_quantized")
    observed_q = payload.get("observed_quantized")
    clean_scale = payload.get("clean_scale")
    observed_scale = payload.get("observed_scale")
    clean_items = payload.get("clean_items")
    observed_items = payload.get("observed_items")
    if (
        not torch.is_tensor(clean_q)
        or not torch.is_tensor(observed_q)
        or clean_q.dtype != torch.int8
        or observed_q.dtype != torch.int8
        or clean_q.ndim != 5
        or observed_q.ndim != 5
        or clean_q.shape[1:] != observed_q.shape[1:]
    ):
        raise CounterfactualEvidenceError("int8 route feature payload differs")
    if (
        not torch.is_tensor(clean_scale)
        or not torch.is_tensor(observed_scale)
        or clean_scale.shape != (clean_q.shape[0], clean_q.shape[-1])
        or observed_scale.shape != (observed_q.shape[0], observed_q.shape[-1])
        or not clean_scale.is_floating_point()
        or not observed_scale.is_floating_point()
        or not bool(torch.isfinite(clean_scale).all())
        or not bool(torch.isfinite(observed_scale).all())
        or bool((clean_scale <= 0).any())
        or bool((observed_scale <= 0).any())
    ):
        raise CounterfactualEvidenceError("int8 route scales differ")
    if (
        not isinstance(clean_items, list)
        or len(clean_items) != clean_q.shape[0]
        or not isinstance(observed_items, list)
        or len(observed_items) != observed_q.shape[0]
    ):
        raise CounterfactualEvidenceError("int8 route item counts differ")
    target = payload.get("target")
    if not isinstance(target, Mapping) or target.get(
        "computed_before_input_quantization"
    ) is not True:
        raise CounterfactualEvidenceError("int8 route target attestation differs")
    values = target.get("values")
    validity = target.get("component_valid")
    expected_target_shape = tuple(observed_q.shape[:-1]) + (3,)
    if (
        not torch.is_tensor(values)
        or tuple(values.shape) != expected_target_shape
        or values.dtype != torch.float32
        or not bool(torch.isfinite(values).all())
        or bool((values < 0).any())
        or not torch.is_tensor(validity)
        or tuple(validity.shape) != expected_target_shape
        or validity.dtype != torch.bool
    ):
        raise CounterfactualEvidenceError("int8 route stored target differs")
    return {
        "clean_count": clean_q.shape[0],
        "observed_count": observed_q.shape[0],
        "feature_shape": list(clean_q.shape[1:]),
    }


def load_int8_route_shard_records(path: Path) -> List[CounterfactualEvidenceRecord]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise CounterfactualEvidenceError("int8 route shard must be a mapping")
    validate_int8_route_shard_payload(payload)
    clean = dynamic_symmetric_int8_dequantize(
        payload["clean_quantized"], payload["clean_scale"], output_dtype=torch.float16
    )
    observed = dynamic_symmetric_int8_dequantize(
        payload["observed_quantized"],
        payload["observed_scale"],
        output_dtype=torch.float16,
    )
    clean_items = payload["clean_items"]
    observed_items = payload["observed_items"]
    clean_key = {
        (str(item["route_id"]), int(item["frame_idx"])): index
        for index, item in enumerate(clean_items)
    }
    observed_key = {
        (
            str(item["route_id"]),
            int(item["frame_idx"]),
            str(item["family"]),
            float(item["severity"]),
        ): index
        for index, item in enumerate(observed_items)
    }
    target = payload["target"]
    records = []
    for index, item in enumerate(observed_items):
        route_id = str(item["route_id"])
        frame_idx = int(item["frame_idx"])
        family = str(item["family"])
        severity = float(item["severity"])
        clean_index = int(item["clean_index"])
        previous_clean = clean_key.get((route_id, frame_idx - 1))
        previous_observed = observed_key.get(
            (route_id, frame_idx - 1, family, severity)
        )
        if (previous_clean is None) != (previous_observed is None):
            raise CounterfactualEvidenceError(
                "int8 route reference/observed temporal availability differs"
            )
        declared_previous_valid = bool(item.get("previous_valid"))
        if declared_previous_valid != (previous_clean is not None):
            raise CounterfactualEvidenceError("int8 route previous_valid differs")
        records.append(
            CounterfactualEvidenceRecord(
                sample_id=str(item["sample_id"]),
                pair_id=str(clean_items[clean_index]["sample_id"]),
                route_id=route_id,
                frame_idx=frame_idx,
                split=str(item["split"]),
                family=family,
                severity=severity,
                reference_current=clean[clean_index],
                observed_current=observed[index],
                reference_previous=(
                    clean[previous_clean]
                    if previous_clean is not None
                    else torch.zeros_like(clean[clean_index])
                ),
                observed_previous=(
                    observed[previous_observed]
                    if previous_observed is not None
                    else torch.zeros_like(observed[index])
                ),
                previous_valid=declared_previous_valid,
                corruption_mask=item.get("corruption_mask"),
                stored_target_values=target["values"][index],
                stored_target_component_valid=target["component_valid"][index],
            )
        )
    return records


def load_int8_dataset_manifest(path: Path, verify_shards: bool = True) -> Dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != INT8_DATASET_SCHEMA_VERSION:
        raise CounterfactualEvidenceError("unsupported int8 dataset manifest schema")
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise CounterfactualEvidenceError("int8 dataset manifest has no shards")
    if verify_shards:
        for row in shards:
            shard_path = path.parent / str(row["file"])
            if not shard_path.is_file() or _sha256(shard_path) != row["sha256"]:
                raise CounterfactualEvidenceError("int8 dataset shard hash differs")
    return payload


__all__ = [
    "FP16_DIRECT_DATASET_SCHEMA_VERSION",
    "FP16_DIRECT_ROUTE_SHARD_SCHEMA_VERSION",
    "FP16_DATASET_SCHEMA_VERSION",
    "FP16_ROUTE_SHARD_SCHEMA_VERSION",
    "INT8_DATASET_SCHEMA_VERSION",
    "INT8_QUANTIZATION_KIND",
    "INT8_ROUTE_SHARD_SCHEMA_VERSION",
    "load_fp16_dataset_manifest",
    "load_fp16_dataset_records",
    "load_fp16_route_shard_records",
    "load_fp16_route_shard_records_selective",
    "write_direct_fp16_route_shard",
    "load_int8_dataset_manifest",
    "load_int8_route_shard_records",
    "validate_fp16_route_shard_payload",
    "validate_int8_route_shard_payload",
    "write_fp16_route_shards_from_payload",
    "write_int8_route_shards_from_payload",
]
