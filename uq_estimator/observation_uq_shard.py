"""Deduplicated FP16 feature shard for clean-first observation UQ."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from uq_estimator.observation_uq_v3 import ObservationUQError, ObservationUQExample


FEATURE_SHARD_SCHEMA_VERSION = "orion.observation-uq-feature-shard/v1"


def _feature_shape(tensor: torch.Tensor) -> Tuple[int, int, int, int]:
    if not torch.is_tensor(tensor) or tensor.ndim != 4:
        raise ObservationUQError("shard features must have shape [V,H,W,D]")
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise ObservationUQError("shard features must be finite floating point")
    return tuple(int(value) for value in tensor.shape)


def _required_item_text(item: Mapping[str, Any], key: str) -> str:
    value = str(item.get(key, "")).strip()
    if not value:
        raise ObservationUQError("shard item is missing %s" % key)
    return value


def validate_feature_shard(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if payload.get("schema_version") != FEATURE_SHARD_SCHEMA_VERSION:
        raise ObservationUQError("unsupported observation feature shard schema")
    clean_features = payload.get("clean_features")
    clean_items = payload.get("clean_items")
    observed_features = payload.get("observed_features")
    observed_items = payload.get("observed_items")
    if not isinstance(clean_features, list) or not clean_features:
        raise ObservationUQError("feature shard requires clean feature tensors")
    if not isinstance(clean_items, list) or len(clean_items) != len(clean_features):
        raise ObservationUQError("clean item/feature counts do not match")
    if not isinstance(observed_features, list) or not isinstance(observed_items, list):
        raise ObservationUQError("observed shard fields must be lists")
    if len(observed_features) != len(observed_items):
        raise ObservationUQError("observed item/feature counts do not match")
    shapes = {_feature_shape(tensor) for tensor in clean_features + observed_features}
    if len(shapes) != 1:
        raise ObservationUQError("all shard feature grids must share one shape")
    feature_shape = next(iter(shapes))
    sample_ids = []
    route_frame = set()
    route_splits = {}
    for index, item in enumerate(clean_items):
        if not isinstance(item, Mapping):
            raise ObservationUQError("clean shard item must be a mapping")
        sample_id = _required_item_text(item, "sample_id")
        route_id = _required_item_text(item, "route_id")
        split = _required_item_text(item, "split")
        frame_idx = int(item.get("frame_idx"))
        if frame_idx < 0:
            raise ObservationUQError("frame_idx must be non-negative")
        identity = (route_id, frame_idx)
        if identity in route_frame:
            raise ObservationUQError("duplicate clean route/frame")
        route_frame.add(identity)
        previous_split = route_splits.setdefault(route_id, split)
        if previous_split != split:
            raise ObservationUQError("one route appears in multiple splits")
        sample_ids.append(sample_id)
        declared_index = item.get("clean_index", index)
        if int(declared_index) != index:
            raise ObservationUQError("clean_index does not match list position")
    if len(sample_ids) != len(set(sample_ids)):
        raise ObservationUQError("duplicate clean sample_id")
    observed_ids = []
    observed_keys = set()
    views, height, width, _ = feature_shape
    for index, item in enumerate(observed_items):
        if not isinstance(item, Mapping):
            raise ObservationUQError("observed shard item must be a mapping")
        observed_id = _required_item_text(item, "sample_id")
        family = _required_item_text(item, "family")
        if family == "clean":
            raise ObservationUQError("observed family cannot be clean")
        clean_index = int(item.get("clean_index"))
        if clean_index < 0 or clean_index >= len(clean_items):
            raise ObservationUQError("observed clean_index is out of range")
        clean_item = clean_items[clean_index]
        route_id = _required_item_text(item, "route_id")
        split = _required_item_text(item, "split")
        frame_idx = int(item.get("frame_idx"))
        if (
            route_id != clean_item["route_id"]
            or split != clean_item["split"]
            or frame_idx != int(clean_item["frame_idx"])
        ):
            raise ObservationUQError("observed source identity disagrees with clean item")
        severity = float(item.get("severity"))
        if severity <= 0:
            raise ObservationUQError("observed severity must be positive")
        key = (route_id, frame_idx, family, severity)
        if key in observed_keys:
            raise ObservationUQError("duplicate observed route/frame/family/severity")
        observed_keys.add(key)
        mask = item.get("corruption_mask")
        if not torch.is_tensor(mask) or mask.shape != (views, height, width):
            raise ObservationUQError("observed mask must have shape [V,H,W]")
        if not mask.is_floating_point() or bool((mask < 0).any()) or bool((mask > 1).any()):
            raise ObservationUQError("observed mask must be floating point in [0,1]")
        observed_ids.append(observed_id)
        declared_index = item.get("observed_index", index)
        if int(declared_index) != index:
            raise ObservationUQError("observed_index does not match list position")
    if len(observed_ids) != len(set(observed_ids)):
        raise ObservationUQError("duplicate observed sample_id")
    return {
        "clean_count": len(clean_features),
        "observed_count": len(observed_features),
        "route_count": len(route_splits),
        "routes_by_split": {
            split: len({route for route, owner in route_splits.items() if owner == split})
            for split in sorted(set(route_splits.values()))
        },
        "feature_shape": list(feature_shape),
        "clean_bytes": sum(t.numel() * t.element_size() for t in clean_features),
        "observed_bytes": sum(t.numel() * t.element_size() for t in observed_features),
    }


def save_feature_shard(payload: Mapping[str, Any], output_path: Path) -> Dict[str, Any]:
    summary = validate_feature_shard(payload)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("refusing to overwrite feature shard %s" % output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), output_path)
    return summary


def load_feature_shard(path: Path) -> Dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ObservationUQError("feature shard payload must be a mapping")
    payload = dict(payload)
    validate_feature_shard(payload)
    return payload


def examples_from_feature_shard(payload: Mapping[str, Any]) -> List[ObservationUQExample]:
    """Build clean and diagnostic examples without duplicating stored tensors."""

    validate_feature_shard(payload)
    clean_features = payload["clean_features"]
    clean_items = payload["clean_items"]
    observed_features = payload["observed_features"]
    observed_items = payload["observed_items"]
    clean_key_to_index = {
        (item["route_id"], int(item["frame_idx"])): index
        for index, item in enumerate(clean_items)
    }
    observed_key_to_index = {
        (
            item["route_id"],
            int(item["frame_idx"]),
            item["family"],
            float(item["severity"]),
        ): index
        for index, item in enumerate(observed_items)
    }
    examples = []
    for index, item in enumerate(clean_items):
        previous_index = clean_key_to_index.get(
            (item["route_id"], int(item["frame_idx"]) - 1)
        )
        current = clean_features[index]
        examples.append(
            ObservationUQExample(
                sample_id=item["sample_id"] + "/clean",
                route_id=item["route_id"],
                split=item["split"],
                family="clean",
                severity=0.0,
                current=current,
                previous=(
                    clean_features[previous_index]
                    if previous_index is not None
                    else torch.zeros_like(current)
                ),
                previous_valid=previous_index is not None,
                corruption_mask=torch.zeros(current.shape[:-1], dtype=torch.float16),
            )
        )
    for index, item in enumerate(observed_items):
        key = (
            item["route_id"],
            int(item["frame_idx"]) - 1,
            item["family"],
            float(item["severity"]),
        )
        previous_index = observed_key_to_index.get(key)
        current = observed_features[index]
        examples.append(
            ObservationUQExample(
                sample_id=item["sample_id"],
                route_id=item["route_id"],
                split=item["split"],
                family=item["family"],
                severity=float(item["severity"]),
                current=current,
                previous=(
                    observed_features[previous_index]
                    if previous_index is not None
                    else torch.zeros_like(current)
                ),
                previous_valid=previous_index is not None,
                corruption_mask=item["corruption_mask"].float(),
            )
        )
    return examples


__all__ = [
    "FEATURE_SHARD_SCHEMA_VERSION",
    "validate_feature_shard",
    "save_feature_shard",
    "load_feature_shard",
    "examples_from_feature_shard",
]
