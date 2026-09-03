"""Standalone Stage-1 training utilities for spatial perception uncertainty.

The data and checkpoint schemas in this module make an important distinction:
an actual perception-failure target is used whenever it is available; paired
clean/corrupt representation error is an explicitly labelled fallback proxy.
Neither a corruption mask nor a route identifier is treated as uncertainty.

This module depends only on PyTorch and :mod:`uq_estimator.spatial_uq`.  It can
therefore be exercised with extracted features or CPU mocks without importing
ORION, MMCV, CARLA, or a vision backbone.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from uq_estimator.spatial_uq import (
    SpatialPatchUQHead,
    SpatialUQOutput,
    brier_loss,
    decompose_ensemble_variance,
    heteroscedastic_gaussian_nll,
    paired_cosine_representation_error,
)


PAIRED_RECORD_SCHEMA_VERSION = "spatial-uq-paired-feature/v2"
PAIRED_DATASET_SCHEMA_VERSION = "spatial-uq-paired-dataset/v2"
LEGACY_PAIRED_RECORD_SCHEMA_VERSION = "spatial-uq-paired-feature/v1"
LEGACY_PAIRED_DATASET_SCHEMA_VERSION = "spatial-uq-paired-dataset/v1"
ROUTE_MANIFEST_SCHEMA_VERSION = "spatial-uq-route-manifest/v1"
SPATIAL_CHECKPOINT_SCHEMA_VERSION = "spatial-uq-stage1-checkpoint/v2"
TARGET_CONTRACT_SCHEMA_VERSION = "spatial-uq-target-contract/v2"

TARGET_ACTUAL_FAILURE = "actual_perception_failure"
TARGET_REPRESENTATION_PROXY = "paired_cosine_representation_error_proxy"

DEFAULT_CLAIM_BOUNDARY: Dict[str, Any] = {
    "output_claim": (
        "spatial error severity; calibrated failure probability only for records "
        "with explicit failure-event targets"
    ),
    "actual_failure_preferred": True,
    "representation_error_proxy_is_semantic_uq": False,
    "corruption_mask_is_primary_target": False,
    "corruption_mask_role": "auxiliary_localization_only",
    "route_is_uq_head_input": False,
    "epistemic_scope": "head_level_shared_frozen_backbone",
    "supports_closed_loop_safety_claim": False,
    "supports_llm_understanding_claim": False,
}


class SpatialTrainingDataError(ValueError):
    """Raised when paired records or split manifests violate the contract."""


def _check_optional_spatial_tensor(
    value: Optional[torch.Tensor],
    expected_shape: torch.Size,
    name: str,
    non_negative: bool = False,
    unit_interval: bool = False,
    allow_bool: bool = False,
) -> None:
    if value is None:
        return
    if not isinstance(value, torch.Tensor):
        raise SpatialTrainingDataError(f"{name} must be a tensor or None")
    if value.shape != expected_shape:
        raise SpatialTrainingDataError(
            f"{name} must have spatial shape {tuple(expected_shape)}, "
            f"got {tuple(value.shape)}"
        )
    if not value.is_floating_point() and not (allow_bool and value.dtype == torch.bool):
        expected = "floating point or boolean" if allow_bool else "floating point"
        raise SpatialTrainingDataError(f"{name} must be {expected}")
    if not torch.isfinite(value).all():
        raise SpatialTrainingDataError(f"{name} must contain only finite values")
    if non_negative and torch.any(value < 0):
        raise SpatialTrainingDataError(f"{name} must be non-negative")
    if unit_interval and (torch.any(value < 0) or torch.any(value > 1)):
        raise SpatialTrainingDataError(f"{name} must lie in [0, 1]")


def _check_optional_valid_mask(
    value: Optional[torch.Tensor], expected_shape: torch.Size, name: str
) -> None:
    if value is None:
        return
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
        raise SpatialTrainingDataError(f"{name} must be a boolean tensor or None")
    if value.shape != expected_shape:
        raise SpatialTrainingDataError(
            f"{name} must have spatial shape {tuple(expected_shape)}, "
            f"got {tuple(value.shape)}"
        )


def _check_optional_component_tensor(
    value: Optional[torch.Tensor],
    spatial_shape: torch.Size,
    names: Tuple[str, ...],
    axis: int,
    name: str,
) -> None:
    if value is None:
        return
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise SpatialTrainingDataError(f"{name} must be a floating tensor or None")
    if not torch.isfinite(value).all() or torch.any(value < 0):
        raise SpatialTrainingDataError(f"{name} must be finite and non-negative")
    if value.ndim != len(spatial_shape) + 1:
        raise SpatialTrainingDataError(
            f"{name} must add exactly one component axis to the spatial target"
        )
    normalized_axis = axis if axis >= 0 else value.ndim + axis
    if not 0 <= normalized_axis < value.ndim:
        raise SpatialTrainingDataError(f"component_error_axis {axis} is out of range")
    remaining = value.shape[:normalized_axis] + value.shape[normalized_axis + 1 :]
    if remaining != spatial_shape:
        raise SpatialTrainingDataError(
            f"{name} non-component axes must equal {tuple(spatial_shape)}, "
            f"got {tuple(remaining)}"
        )
    if value.shape[normalized_axis] != len(names):
        raise SpatialTrainingDataError(
            f"{name} component axis has {value.shape[normalized_axis]} entries but "
            f"component_error_names has {len(names)}"
        )


@dataclass(frozen=True)
class PairedSpatialFeatureRecord:
    """Versioned, route-addressable paired feature record.

    ``observed_patch_features`` is the input seen by the UQ head.  For severity
    above zero it is normally the corrupted observation.  ``clean_patch_features``
    is extracted at the same simulated state and provides the fallback paired
    representation target. Optional actual targets always take priority.
    Continuous error severity and failure-event probability are deliberately
    separate. Invalid or ambiguous patches are excluded by a boolean valid
    mask. Clean targets, when present, are independently measured and are
    never inferred as zero from the observed targets.
    """

    record_id: str
    pair_id: str
    route_id: str
    town: str
    severity: float
    observed_patch_features: torch.Tensor
    clean_patch_features: torch.Tensor
    error_severity_target: Optional[torch.Tensor] = None
    failure_event_target: Optional[torch.Tensor] = None
    target_valid_mask: Optional[torch.Tensor] = None
    clean_error_severity_target: Optional[torch.Tensor] = None
    clean_failure_event_target: Optional[torch.Tensor] = None
    clean_target_valid_mask: Optional[torch.Tensor] = None
    component_errors: Optional[torch.Tensor] = None
    clean_component_errors: Optional[torch.Tensor] = None
    component_error_names: Tuple[str, ...] = ()
    component_error_axis: int = -1
    corruption_mask: Optional[torch.Tensor] = None
    ensemble_teacher_variance: Optional[torch.Tensor] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = PAIRED_RECORD_SCHEMA_VERSION

    CURRENT_SCHEMA_VERSION: ClassVar[str] = PAIRED_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != self.CURRENT_SCHEMA_VERSION:
            raise SpatialTrainingDataError(
                f"unsupported paired-record schema {self.schema_version!r}"
            )
        for name, value in (
            ("record_id", self.record_id),
            ("pair_id", self.pair_id),
            ("route_id", self.route_id),
            ("town", self.town),
        ):
            if not str(value).strip():
                raise SpatialTrainingDataError(f"{name} must be non-empty")
        if self.severity < 0:
            raise SpatialTrainingDataError("severity must be non-negative")
        if self.observed_patch_features.shape != self.clean_patch_features.shape:
            raise SpatialTrainingDataError(
                "observed and clean patch features must have identical shapes"
            )
        if self.observed_patch_features.ndim < 2:
            raise SpatialTrainingDataError(
                "patch features must have shape [..., feature_dim]"
            )
        if not self.observed_patch_features.is_floating_point():
            raise SpatialTrainingDataError("observed patch features must be floating point")
        if not self.clean_patch_features.is_floating_point():
            raise SpatialTrainingDataError("clean patch features must be floating point")
        if not torch.isfinite(self.observed_patch_features).all():
            raise SpatialTrainingDataError("observed patch features must be finite")
        if not torch.isfinite(self.clean_patch_features).all():
            raise SpatialTrainingDataError("clean patch features must be finite")

        spatial_shape = self.observed_patch_features.shape[:-1]
        _check_optional_spatial_tensor(
            self.error_severity_target,
            spatial_shape,
            "error_severity_target",
            non_negative=True,
        )
        _check_optional_spatial_tensor(
            self.failure_event_target,
            spatial_shape,
            "failure_event_target",
            unit_interval=True,
            allow_bool=True,
        )
        _check_optional_valid_mask(
            self.target_valid_mask, spatial_shape, "target_valid_mask"
        )
        observed_actual_fields = (
            self.error_severity_target,
            self.failure_event_target,
            self.target_valid_mask,
        )
        if any(value is not None for value in observed_actual_fields) and not all(
            value is not None for value in observed_actual_fields
        ):
            raise SpatialTrainingDataError(
                "actual observed targets require error_severity_target, "
                "failure_event_target, and target_valid_mask together"
            )

        _check_optional_spatial_tensor(
            self.clean_error_severity_target,
            spatial_shape,
            "clean_error_severity_target",
            non_negative=True,
        )
        _check_optional_spatial_tensor(
            self.clean_failure_event_target,
            spatial_shape,
            "clean_failure_event_target",
            unit_interval=True,
            allow_bool=True,
        )
        _check_optional_valid_mask(
            self.clean_target_valid_mask, spatial_shape, "clean_target_valid_mask"
        )
        clean_actual_fields = (
            self.clean_error_severity_target,
            self.clean_failure_event_target,
            self.clean_target_valid_mask,
        )
        if any(value is not None for value in clean_actual_fields) and not all(
            value is not None for value in clean_actual_fields
        ):
            raise SpatialTrainingDataError(
                "clean targets require clean_error_severity_target, "
                "clean_failure_event_target, and clean_target_valid_mask together"
            )
        if self.error_severity_target is None and any(
            value is not None for value in clean_actual_fields
        ):
            raise SpatialTrainingDataError(
                "clean actual targets cannot be attached to a representation-proxy record"
            )

        names = tuple(str(value).strip() for value in self.component_error_names)
        if names != self.component_error_names or any(not value for value in names):
            raise SpatialTrainingDataError(
                "component_error_names must be a tuple of non-empty normalized names"
            )
        if len(names) != len(set(names)):
            raise SpatialTrainingDataError("component_error_names must be unique")
        if (self.component_errors is None) != (len(names) == 0):
            raise SpatialTrainingDataError(
                "component_errors and component_error_names must be provided together"
            )
        if self.component_errors is not None and self.error_severity_target is None:
            raise SpatialTrainingDataError(
                "component_errors are only valid for actual perception-error targets"
            )
        _check_optional_component_tensor(
            self.component_errors,
            spatial_shape,
            names,
            self.component_error_axis,
            "component_errors",
        )
        if self.clean_component_errors is not None and self.clean_error_severity_target is None:
            raise SpatialTrainingDataError(
                "clean_component_errors require explicit clean actual targets"
            )
        _check_optional_component_tensor(
            self.clean_component_errors,
            spatial_shape,
            names,
            self.component_error_axis,
            "clean_component_errors",
        )
        _check_optional_spatial_tensor(
            self.corruption_mask,
            spatial_shape,
            "corruption_mask",
            unit_interval=True,
        )
        _check_optional_spatial_tensor(
            self.ensemble_teacher_variance,
            spatial_shape,
            "ensemble_teacher_variance",
            non_negative=True,
        )

    @property
    def target_provenance(self) -> str:
        return (
            TARGET_ACTUAL_FAILURE
            if self.error_severity_target is not None
            else TARGET_REPRESENTATION_PROXY
        )

    @property
    def corrupt_patch_features(self) -> torch.Tensor:
        """Explicit alias for records whose observed input is corrupted."""
        return self.observed_patch_features

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "pair_id": self.pair_id,
            "route_id": self.route_id,
            "town": self.town,
            "severity": float(self.severity),
            "observed_patch_features": self.observed_patch_features,
            "clean_patch_features": self.clean_patch_features,
            "target_contract_schema_version": TARGET_CONTRACT_SCHEMA_VERSION,
            "error_severity_target": self.error_severity_target,
            "failure_event_target": self.failure_event_target,
            "target_valid_mask": self.target_valid_mask,
            "clean_error_severity_target": self.clean_error_severity_target,
            "clean_failure_event_target": self.clean_failure_event_target,
            "clean_target_valid_mask": self.clean_target_valid_mask,
            "component_errors": self.component_errors,
            "clean_component_errors": self.clean_component_errors,
            "component_error_names": self.component_error_names,
            "component_error_axis": self.component_error_axis,
            "corruption_mask": self.corruption_mask,
            "ensemble_teacher_variance": self.ensemble_teacher_variance,
            "metadata": dict(self.metadata),
            "target_provenance": self.target_provenance,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PairedSpatialFeatureRecord":
        if payload.get("target_contract_schema_version") != TARGET_CONTRACT_SCHEMA_VERSION:
            raise SpatialTrainingDataError(
                "paired record target-contract schema is missing or incompatible"
            )
        declared = payload.get("target_provenance")
        record = cls(
            schema_version=str(payload.get("schema_version", "")),
            record_id=str(payload["record_id"]),
            pair_id=str(payload["pair_id"]),
            route_id=str(payload["route_id"]),
            town=str(payload.get("town", "unknown")),
            severity=float(payload["severity"]),
            observed_patch_features=payload["observed_patch_features"],
            clean_patch_features=payload["clean_patch_features"],
            error_severity_target=payload.get("error_severity_target"),
            failure_event_target=payload.get("failure_event_target"),
            target_valid_mask=payload.get("target_valid_mask"),
            clean_error_severity_target=payload.get("clean_error_severity_target"),
            clean_failure_event_target=payload.get("clean_failure_event_target"),
            clean_target_valid_mask=payload.get("clean_target_valid_mask"),
            component_errors=payload.get("component_errors"),
            clean_component_errors=payload.get("clean_component_errors"),
            component_error_names=tuple(payload.get("component_error_names", ())),
            component_error_axis=int(payload.get("component_error_axis", -1)),
            corruption_mask=payload.get("corruption_mask"),
            ensemble_teacher_variance=payload.get("ensemble_teacher_variance"),
            metadata=payload.get("metadata", {}),
        )
        if declared is not None and declared != record.target_provenance:
            raise SpatialTrainingDataError(
                f"record {record.record_id} declares target provenance {declared!r}, "
                f"but its tensors imply {record.target_provenance!r}"
            )
        return record


def migrate_legacy_v1_proxy_payload(
    payload: Mapping[str, Any],
) -> PairedSpatialFeatureRecord:
    """Migrate only unambiguous v1 representation-proxy records.

    A v1 ``failure_target`` was simultaneously interpreted as continuous
    severity and an event probability. Automatically guessing either meaning
    for an actual-target record would recreate the bug this schema fixes.
    """
    if payload.get("schema_version") != LEGACY_PAIRED_RECORD_SCHEMA_VERSION:
        raise SpatialTrainingDataError("legacy migration requires a v1 record payload")
    if payload.get("failure_target") is not None or payload.get("clean_failure_target") is not None:
        raise SpatialTrainingDataError(
            "cannot automatically migrate a v1 actual-target record: regenerate or "
            "explicitly map failure_target to separate error_severity_target and "
            "failure_event_target tensors with valid masks"
        )
    declared = payload.get("target_provenance")
    if declared not in (None, TARGET_REPRESENTATION_PROXY):
        raise SpatialTrainingDataError(
            "legacy proxy record declares incompatible target provenance"
        )
    metadata = dict(payload.get("metadata", {}))
    metadata["legacy_migration"] = {
        "from_record_schema": LEGACY_PAIRED_RECORD_SCHEMA_VERSION,
        "to_record_schema": PAIRED_RECORD_SCHEMA_VERSION,
        "actual_targets_inferred": False,
        "failure_event_target_available": False,
    }
    return PairedSpatialFeatureRecord(
        record_id=str(payload["record_id"]),
        pair_id=str(payload["pair_id"]),
        route_id=str(payload["route_id"]),
        town=str(payload.get("town", "unknown")),
        severity=float(payload["severity"]),
        observed_patch_features=payload["observed_patch_features"],
        clean_patch_features=payload["clean_patch_features"],
        corruption_mask=payload.get("corruption_mask"),
        ensemble_teacher_variance=payload.get("ensemble_teacher_variance"),
        metadata=metadata,
    )


def save_paired_feature_records(
    path: Path | str,
    records: Sequence[PairedSpatialFeatureRecord],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": PAIRED_DATASET_SCHEMA_VERSION,
            "record_schema_version": PAIRED_RECORD_SCHEMA_VERSION,
            "records": [record.to_payload() for record in records],
        },
        path,
    )


def load_paired_feature_records(path: Path | str) -> List[PairedSpatialFeatureRecord]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    dataset_schema = payload.get("schema_version")
    record_schema = payload.get("record_schema_version")
    if dataset_schema == LEGACY_PAIRED_DATASET_SCHEMA_VERSION:
        if record_schema != LEGACY_PAIRED_RECORD_SCHEMA_VERSION:
            raise SpatialTrainingDataError(
                "legacy paired dataset declares an incompatible record schema"
            )
        records = [
            migrate_legacy_v1_proxy_payload(item)
            for item in payload.get("records", [])
        ]
    elif dataset_schema != PAIRED_DATASET_SCHEMA_VERSION:
        raise SpatialTrainingDataError(
            f"unsupported paired dataset schema {dataset_schema!r}"
        )
    else:
        if record_schema != PAIRED_RECORD_SCHEMA_VERSION:
            raise SpatialTrainingDataError(
                "paired dataset declares an unsupported record schema"
            )
        records = [
            PairedSpatialFeatureRecord.from_payload(item)
            for item in payload.get("records", [])
        ]
    if not records:
        raise SpatialTrainingDataError("paired dataset contains no records")
    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise SpatialTrainingDataError("paired dataset contains duplicate record_id values")
    return records


@dataclass(frozen=True)
class RouteDisjointManifest:
    """Versioned route-level data split; route IDs may occur in one split only."""

    splits: Mapping[str, Tuple[str, ...]]
    seed: int = 0
    split_unit: str = "route_id"
    schema_version: str = ROUTE_MANIFEST_SCHEMA_VERSION

    CURRENT_SCHEMA_VERSION: ClassVar[str] = ROUTE_MANIFEST_SCHEMA_VERSION
    REQUIRED_SPLITS: ClassVar[Tuple[str, ...]] = (
        "train",
        "validation",
        "calibration",
        "held_out",
    )

    def __post_init__(self) -> None:
        if self.schema_version != self.CURRENT_SCHEMA_VERSION:
            raise SpatialTrainingDataError(
                f"unsupported route-manifest schema {self.schema_version!r}"
            )
        if self.split_unit != "route_id":
            raise SpatialTrainingDataError("manifest split_unit must be 'route_id'")
        missing = set(self.REQUIRED_SPLITS) - set(self.splits)
        if missing:
            raise SpatialTrainingDataError(f"manifest is missing splits: {sorted(missing)}")

        owner: Dict[str, str] = {}
        for split in self.REQUIRED_SPLITS:
            routes = tuple(str(route) for route in self.splits[split])
            if not routes:
                raise SpatialTrainingDataError(f"manifest split {split!r} is empty")
            if len(routes) != len(set(routes)):
                raise SpatialTrainingDataError(
                    f"manifest split {split!r} contains duplicate routes"
                )
            for route in routes:
                if not route:
                    raise SpatialTrainingDataError("route IDs must be non-empty")
                if route in owner:
                    raise SpatialTrainingDataError(
                        f"route {route!r} appears in both {owner[route]!r} and {split!r}"
                    )
                owner[route] = split

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split_unit": self.split_unit,
            "seed": int(self.seed),
            "route_disjoint": True,
            "splits": {
                split: {"route_ids": list(self.splits[split])}
                for split in self.REQUIRED_SPLITS
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RouteDisjointManifest":
        if payload.get("route_disjoint") is not True:
            raise SpatialTrainingDataError("manifest must assert route_disjoint=true")
        raw_splits = payload.get("splits", {})
        splits = {
            split: tuple(raw_splits.get(split, {}).get("route_ids", []))
            for split in cls.REQUIRED_SPLITS
        }
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            split_unit=str(payload.get("split_unit", "")),
            seed=int(payload.get("seed", 0)),
            splits=splits,
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "RouteDisjointManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_route_disjoint_manifest(
    records: Sequence[PairedSpatialFeatureRecord],
    seed: int = 0,
) -> RouteDisjointManifest:
    """Create a deterministic minimal four-way route-disjoint split.

    Real experiments should normally persist and preregister an explicit
    manifest.  This helper is primarily for pilots and mocks.
    """
    routes = sorted({record.route_id for record in records})
    if len(routes) < 4:
        raise SpatialTrainingDataError(
            "at least four distinct routes are required for train/validation/"
            "calibration/held_out splits"
        )
    rng = random.Random(seed)
    rng.shuffle(routes)
    splits = {
        "train": tuple(routes[:-3]),
        "validation": (routes[-3],),
        "calibration": (routes[-2],),
        "held_out": (routes[-1],),
    }
    return RouteDisjointManifest(splits=splits, seed=seed)


def validate_manifest_coverage(
    records: Sequence[PairedSpatialFeatureRecord],
    manifest: RouteDisjointManifest,
) -> None:
    record_routes = {record.route_id for record in records}
    manifest_routes = {
        route for routes in manifest.splits.values() for route in routes
    }
    missing = record_routes - manifest_routes
    unknown = manifest_routes - record_routes
    if missing or unknown:
        raise SpatialTrainingDataError(
            f"manifest/record route mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


class PairedSpatialFeatureDataset(Dataset):
    """Dataset view filtered by one route-disjoint manifest split."""

    def __init__(
        self,
        records: Sequence[PairedSpatialFeatureRecord],
        manifest: RouteDisjointManifest,
        split: str,
    ) -> None:
        validate_manifest_coverage(records, manifest)
        if split not in manifest.splits:
            raise SpatialTrainingDataError(f"unknown manifest split {split!r}")
        allowed = set(manifest.splits[split])
        self.records = [record for record in records if record.route_id in allowed]
        if not self.records:
            raise SpatialTrainingDataError(f"no records found for split {split!r}")
        feature_shapes = {tuple(record.observed_patch_features.shape) for record in self.records}
        if len(feature_shapes) != 1:
            raise SpatialTrainingDataError(
                "records within one minibatched split must share a feature shape"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> PairedSpatialFeatureRecord:
        return self.records[index]


class PairGroupedBatchSampler(Sampler[List[int]]):
    """Keep every severity of one ``pair_id`` in the same training batch.

    Without this sampler, random record-level shuffling can silently disable
    the paired severity-ranking loss because its lower/higher observations no
    longer coexist in a minibatch.
    """

    def __init__(
        self,
        dataset: PairedSpatialFeatureDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        grouped: Dict[str, List[int]] = {}
        for index, record in enumerate(dataset.records):
            grouped.setdefault(record.pair_id, []).append(index)
        largest = max(len(indices) for indices in grouped.values())
        if largest > batch_size:
            raise SpatialTrainingDataError(
                f"batch_size={batch_size} cannot hold the largest pair group ({largest})"
            )
        self.groups = list(grouped.values())
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        groups = list(self.groups)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(groups)
        self.epoch += 1
        batch: List[int] = []
        for group in groups:
            if batch and len(batch) + len(group) > self.batch_size:
                yield batch
                batch = []
            batch.extend(group)
        if batch:
            yield batch

    def __len__(self) -> int:
        # Exact for the stable underlying group order; shuffled packing may use
        # the same or more batches, so compute a conservative exact simulation.
        count = 0
        used = 0
        for group in self.groups:
            if used and used + len(group) > self.batch_size:
                count += 1
                used = 0
            used += len(group)
        return count + int(used > 0)


@dataclass(frozen=True)
class SpatialTrainingBatch:
    """Resolved batch whose target source remains inspectable per sample."""

    observed_features: torch.Tensor
    clean_features: torch.Tensor
    error_severity_target: torch.Tensor
    error_severity_valid_mask: torch.Tensor
    failure_event_target: torch.Tensor
    failure_event_valid_mask: torch.Tensor
    target_is_actual: torch.Tensor
    corruption_mask: torch.Tensor
    corruption_mask_present: torch.Tensor
    clean_error_severity_target: torch.Tensor
    clean_failure_event_target: torch.Tensor
    clean_target_valid_mask: torch.Tensor
    ensemble_teacher_variance: torch.Tensor
    ensemble_teacher_present: torch.Tensor
    severity: torch.Tensor
    record_ids: Tuple[str, ...]
    pair_ids: Tuple[str, ...]
    route_ids: Tuple[str, ...]
    target_provenance: Tuple[str, ...]

    def to(self, device: torch.device | str) -> "SpatialTrainingBatch":
        values = {}
        for name, value in self.__dict__.items():
            values[name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return SpatialTrainingBatch(**values)


def collate_paired_spatial_records(
    records: Sequence[PairedSpatialFeatureRecord],
) -> SpatialTrainingBatch:
    if not records:
        raise SpatialTrainingDataError("cannot collate an empty record list")
    observed = torch.stack([record.observed_patch_features for record in records])
    clean = torch.stack([record.clean_patch_features for record in records])
    proxy = paired_cosine_representation_error(clean, observed)
    spatial_shape = proxy.shape[1:]

    error_targets = []
    error_valid_masks = []
    failure_targets = []
    failure_valid_masks = []
    target_is_actual = []
    masks = []
    mask_present = []
    clean_error_targets = []
    clean_failure_targets = []
    clean_valid_masks = []
    teacher_targets = []
    teacher_present = []

    for index, record in enumerate(records):
        if record.error_severity_target is not None:
            error_target = record.error_severity_target
            failure_target = record.failure_event_target
            valid_mask = record.target_valid_mask
            actual = True
        else:
            error_target = proxy[index]
            failure_target = torch.zeros(spatial_shape, dtype=observed.dtype)
            valid_mask = torch.ones(spatial_shape, dtype=torch.bool)
            actual = False
        error_targets.append(error_target)
        error_valid_masks.append(valid_mask)
        failure_targets.append(failure_target)
        failure_valid_masks.append(
            valid_mask if actual else torch.zeros(spatial_shape, dtype=torch.bool)
        )
        target_is_actual.append(actual)

        masks.append(
            record.corruption_mask
            if record.corruption_mask is not None
            else torch.zeros(spatial_shape, dtype=observed.dtype)
        )
        mask_present.append(record.corruption_mask is not None)

        # A clean image can contain real ORION failures. Missing clean targets
        # therefore remain missing and contribute no clean loss.
        if record.clean_error_severity_target is not None:
            clean_error_targets.append(record.clean_error_severity_target)
            clean_failure_targets.append(record.clean_failure_event_target)
            clean_valid_masks.append(record.clean_target_valid_mask)
        else:
            clean_error_targets.append(
                torch.zeros(spatial_shape, dtype=observed.dtype)
            )
            clean_failure_targets.append(
                torch.zeros(spatial_shape, dtype=observed.dtype)
            )
            clean_valid_masks.append(torch.zeros(spatial_shape, dtype=torch.bool))

        teacher_targets.append(
            record.ensemble_teacher_variance
            if record.ensemble_teacher_variance is not None
            else torch.zeros(spatial_shape, dtype=observed.dtype)
        )
        teacher_present.append(record.ensemble_teacher_variance is not None)

    return SpatialTrainingBatch(
        observed_features=observed,
        clean_features=clean,
        error_severity_target=torch.stack(error_targets),
        error_severity_valid_mask=torch.stack(error_valid_masks),
        failure_event_target=torch.stack(failure_targets).to(observed.dtype),
        failure_event_valid_mask=torch.stack(failure_valid_masks),
        target_is_actual=torch.tensor(target_is_actual, dtype=torch.bool),
        corruption_mask=torch.stack(masks),
        corruption_mask_present=torch.tensor(mask_present, dtype=torch.bool),
        clean_error_severity_target=torch.stack(clean_error_targets),
        clean_failure_event_target=torch.stack(clean_failure_targets).to(observed.dtype),
        clean_target_valid_mask=torch.stack(clean_valid_masks),
        ensemble_teacher_variance=torch.stack(teacher_targets),
        ensemble_teacher_present=torch.tensor(teacher_present, dtype=torch.bool),
        severity=torch.tensor([record.severity for record in records], dtype=torch.float32),
        record_ids=tuple(record.record_id for record in records),
        pair_ids=tuple(record.pair_id for record in records),
        route_ids=tuple(record.route_id for record in records),
        target_provenance=tuple(record.target_provenance for record in records),
    )


@dataclass(frozen=True)
class SpatialLossWeights:
    gaussian_nll: float = 1.0
    failure_brier: float = 1.0
    error_ranking: float = 0.25
    epistemic_distill: float = 0.25
    mask_aux: float = 0.05
    clean_gaussian_nll: float = 0.5
    clean_failure_brier: float = 0.5
    ranking_margin: float = 0.1
    ranking_min_target_increase: float = 1e-6


@dataclass(frozen=True)
class SpatialEnsembleOutput:
    member_outputs: Tuple[SpatialUQOutput, ...]
    expected_error: torch.Tensor
    aleatoric_variance: torch.Tensor
    epistemic_variance: torch.Tensor
    total_variance: torch.Tensor
    failure_probability: torch.Tensor


class SpatialHeadEnsemble(nn.Module):
    """Independently initialized heads sharing only frozen input features."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        n_members: int = 3,
        min_log_variance: float = -8.0,
        max_log_variance: float = 4.0,
    ) -> None:
        super().__init__()
        if n_members < 2:
            raise ValueError("an epistemic ensemble requires at least two members")
        self.members = nn.ModuleList(
            [
                SpatialPatchUQHead(
                    feature_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    min_log_variance=min_log_variance,
                    max_log_variance=max_log_variance,
                    predict_epistemic=False,
                )
                for _ in range(n_members)
            ]
        )

    def forward(self, patch_features: torch.Tensor) -> SpatialEnsembleOutput:
        outputs = tuple(member(patch_features) for member in self.members)
        means = torch.stack([output.expected_error for output in outputs], dim=0)
        aleatoric = torch.stack(
            [output.aleatoric_variance for output in outputs], dim=0
        )
        decomposition = decompose_ensemble_variance(means, aleatoric, member_dim=0)
        failure_probability = torch.stack(
            [output.failure_probability for output in outputs], dim=0
        ).mean(dim=0)
        return SpatialEnsembleOutput(
            member_outputs=outputs,
            expected_error=decomposition.predictive_mean,
            aleatoric_variance=decomposition.aleatoric_variance,
            epistemic_variance=decomposition.epistemic_variance,
            total_variance=decomposition.total_variance,
            failure_probability=failure_probability,
        )


def _expand_sample_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mask.reshape(mask.shape + (1,) * (target.ndim - 1)).expand_as(target)


def _severity_ranking_loss(
    predicted_error: torch.Tensor,
    target_error: torch.Tensor,
    target_valid_mask: torch.Tensor,
    pair_ids: Sequence[str],
    severity: torch.Tensor,
    margin: float,
    min_target_increase: float,
) -> torch.Tensor:
    total = predicted_error.sum() * 0.0
    active_count = torch.zeros((), dtype=predicted_error.dtype, device=predicted_error.device)
    for low in range(len(pair_ids)):
        for high in range(len(pair_ids)):
            if pair_ids[low] != pair_ids[high] or severity[high] <= severity[low]:
                continue
            active = (
                target_valid_mask[low]
                & target_valid_mask[high]
                & (
                    target_error[high]
                    > target_error[low] + min_target_increase
                )
            )
            raw = F.relu(
                margin - (predicted_error[high] - predicted_error[low])
            )
            total = total + torch.where(active, raw, torch.zeros_like(raw)).sum()
            active_count = active_count + active.to(predicted_error.dtype).sum()
    return total / active_count.clamp_min(1.0)


def compute_spatial_training_loss(
    output: SpatialUQOutput,
    batch: SpatialTrainingBatch,
    weights: SpatialLossWeights,
    clean_output: Optional[SpatialUQOutput] = None,
    live_teacher_epistemic: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute all Stage-1 loss terms while preserving target provenance."""
    gaussian = heteroscedastic_gaussian_nll(
        output.expected_error,
        batch.error_severity_target,
        output.log_variance,
        weight=batch.error_severity_valid_mask,
    )
    failure = brier_loss(
        output.failure_probability,
        batch.failure_event_target,
        weight=batch.failure_event_valid_mask,
    )
    ranking = _severity_ranking_loss(
        output.expected_error,
        batch.error_severity_target,
        batch.error_severity_valid_mask,
        batch.pair_ids,
        batch.severity,
        margin=weights.ranking_margin,
        min_target_increase=weights.ranking_min_target_increase,
    )

    mask_weight = _expand_sample_mask(
        batch.corruption_mask_present, output.failure_probability
    ) & batch.error_severity_valid_mask
    mask_aux = brier_loss(
        output.failure_probability,
        batch.corruption_mask,
        weight=mask_weight,
    )

    if clean_output is None:
        clean_gaussian = output.failure_probability.sum() * 0.0
        clean_failure = output.failure_probability.sum() * 0.0
    else:
        clean_gaussian = heteroscedastic_gaussian_nll(
            clean_output.expected_error,
            batch.clean_error_severity_target,
            clean_output.log_variance,
            weight=batch.clean_target_valid_mask,
        )
        clean_failure = brier_loss(
            clean_output.failure_probability,
            batch.clean_failure_event_target,
            weight=batch.clean_target_valid_mask,
        )

    if output.epistemic_variance is None:
        epistemic = output.failure_probability.sum() * 0.0
    else:
        provided_mask = _expand_sample_mask(
            batch.ensemble_teacher_present,
            output.epistemic_variance,
        ) & batch.error_severity_valid_mask
        if live_teacher_epistemic is None:
            teacher_target = batch.ensemble_teacher_variance
            teacher_weight = provided_mask
        else:
            live_teacher_epistemic = live_teacher_epistemic.detach()
            teacher_target = torch.where(
                provided_mask,
                batch.ensemble_teacher_variance,
                live_teacher_epistemic,
            )
            teacher_weight = batch.error_severity_valid_mask
        epistemic_raw = F.smooth_l1_loss(
            output.epistemic_variance,
            teacher_target,
            reduction="none",
        )
        epistemic = (
            epistemic_raw * teacher_weight.to(epistemic_raw.dtype)
        ).sum() / teacher_weight.to(epistemic_raw.dtype).sum().clamp_min(1.0)

    total = (
        weights.gaussian_nll * gaussian
        + weights.failure_brier * failure
        + weights.error_ranking * ranking
        + weights.epistemic_distill * epistemic
        + weights.mask_aux * mask_aux
        + weights.clean_gaussian_nll * clean_gaussian
        + weights.clean_failure_brier * clean_failure
    )
    return {
        "total": total,
        "gaussian_nll": gaussian,
        "failure_brier": failure,
        "error_ranking": ranking,
        "epistemic_distill": epistemic,
        "mask_aux": mask_aux,
        "clean_gaussian_nll": clean_gaussian,
        "clean_failure_brier": clean_failure,
    }


def summarize_target_provenance(
    records: Sequence[PairedSpatialFeatureRecord],
) -> Dict[str, Any]:
    counts = {TARGET_ACTUAL_FAILURE: 0, TARGET_REPRESENTATION_PROXY: 0}
    clean_actual = 0
    actual_valid_cells = 0
    actual_invalid_cells = 0
    failure_event_records = 0
    component_names: set[str] = set()
    legacy_migrated_records = 0
    teacher_recorded = 0
    mask_recorded = 0
    for record in records:
        counts[record.target_provenance] += 1
        clean_actual += int(record.clean_error_severity_target is not None)
        failure_event_records += int(record.failure_event_target is not None)
        if record.target_valid_mask is not None:
            actual_valid_cells += int(record.target_valid_mask.sum())
            actual_invalid_cells += int((~record.target_valid_mask).sum())
        component_names.update(record.component_error_names)
        legacy_migrated_records += int("legacy_migration" in record.metadata)
        teacher_recorded += int(record.ensemble_teacher_variance is not None)
        mask_recorded += int(record.corruption_mask is not None)
    return {
        "primary_target_counts": counts,
        "actual_failure_priority_rule": True,
        "representation_proxy_definition": "1-cosine(clean_feature, observed_feature)",
        "representation_proxy_has_failure_event_target": False,
        "clean_actual_target_records": clean_actual,
        "failure_event_target_records": failure_event_records,
        "actual_valid_patch_cells": actual_valid_cells,
        "actual_invalid_patch_cells": actual_invalid_cells,
        "component_error_names": sorted(component_names),
        "legacy_v1_proxy_migrated_records": legacy_migrated_records,
        "ensemble_teacher_records": teacher_recorded,
        "corruption_mask_aux_records": mask_recorded,
        "record_count": len(records),
    }


def _mean_epoch_metrics(metrics: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {}
    return {
        key: sum(item[key] for item in metrics) / len(metrics)
        for key in metrics[0]
    }


def _loader(
    dataset: PairedSpatialFeatureDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    if shuffle:
        return DataLoader(
            dataset,
            batch_sampler=PairGroupedBatchSampler(
                dataset, batch_size=batch_size, shuffle=True, seed=seed
            ),
            collate_fn=collate_paired_spatial_records,
            num_workers=0,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_paired_spatial_records,
        num_workers=0,
    )


def train_head_ensemble_epoch(
    ensemble: SpatialHeadEnsemble,
    loader: Iterable[SpatialTrainingBatch],
    optimizer: torch.optim.Optimizer,
    weights: SpatialLossWeights,
    device: torch.device,
) -> Dict[str, float]:
    ensemble.train()
    epoch_metrics: List[Dict[str, float]] = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        member_losses = []
        component_sums: Dict[str, torch.Tensor] = {}
        for member in ensemble.members:
            output = member(batch.observed_features)
            clean_output = member(batch.clean_features)
            losses = compute_spatial_training_loss(
                output, batch, weights, clean_output=clean_output
            )
            member_losses.append(losses["total"])
            for name, value in losses.items():
                component_sums[name] = component_sums.get(name, value * 0.0) + value
        loss = torch.stack(member_losses).mean()
        loss.backward()
        optimizer.step()
        epoch_metrics.append(
            {
                name: float((value / len(ensemble.members)).detach().cpu())
                for name, value in component_sums.items()
            }
        )
    return _mean_epoch_metrics(epoch_metrics)


def train_student_epoch(
    student: SpatialPatchUQHead,
    ensemble: SpatialHeadEnsemble,
    loader: Iterable[SpatialTrainingBatch],
    optimizer: torch.optim.Optimizer,
    weights: SpatialLossWeights,
    device: torch.device,
) -> Dict[str, float]:
    student.train()
    ensemble.eval()
    epoch_metrics: List[Dict[str, float]] = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher = ensemble(batch.observed_features)
        output = student(batch.observed_features)
        clean_output = student(batch.clean_features)
        losses = compute_spatial_training_loss(
            output,
            batch,
            weights,
            clean_output=clean_output,
            live_teacher_epistemic=teacher.epistemic_variance,
        )
        losses["total"].backward()
        optimizer.step()
        epoch_metrics.append(
            {name: float(value.detach().cpu()) for name, value in losses.items()}
        )
    return _mean_epoch_metrics(epoch_metrics)


def evaluate_student(
    student: SpatialPatchUQHead,
    loader: Iterable[SpatialTrainingBatch],
    weights: SpatialLossWeights,
    device: torch.device,
) -> Dict[str, float]:
    student.eval()
    epoch_metrics: List[Dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = student(batch.observed_features)
            clean_output = student(batch.clean_features)
            losses = compute_spatial_training_loss(
                output, batch, weights, clean_output=clean_output
            )
            epoch_metrics.append(
                {name: float(value.detach().cpu()) for name, value in losses.items()}
            )
    return _mean_epoch_metrics(epoch_metrics)


def make_mock_paired_records(
    feature_dim: int = 8,
    n_routes: int = 7,
    pairs_per_route: int = 2,
    severities: Sequence[float] = (0.0, 1.0, 2.0),
    seed: int = 0,
) -> List[PairedSpatialFeatureRecord]:
    """Generate deterministic CPU records containing real and proxy targets."""
    if n_routes < 4:
        raise ValueError("mock data requires at least four routes")
    generator = torch.Generator().manual_seed(seed)
    records: List[PairedSpatialFeatureRecord] = []
    for route_index in range(n_routes):
        route_id = f"route_{route_index:03d}"
        town = f"Town{1 + route_index % 5:02d}"
        for pair_index in range(pairs_per_route):
            pair_id = f"{route_id}/pair_{pair_index:02d}"
            clean = torch.randn(2, 4, feature_dim, generator=generator)
            mask = torch.zeros(2, 4)
            mask[:, :2] = 1.0
            for severity in severities:
                noise = torch.randn(clean.shape, generator=generator)
                observed = clean + mask.unsqueeze(-1) * noise * (0.15 * severity)
                proxy = paired_cosine_representation_error(clean, observed)
                # Even routes provide privileged actual failures; odd routes
                # exercise the explicitly labelled representation proxy path.
                actual_severity = None
                actual_event = None
                actual_valid = None
                clean_severity = None
                clean_event = None
                clean_valid = None
                if route_index % 2 == 0:
                    actual_severity = proxy * (0.4 + 0.2 * severity)
                    actual_event = (actual_severity >= 0.1).to(proxy.dtype)
                    actual_valid = torch.ones_like(proxy, dtype=torch.bool)
                    # Non-zero clean errors are intentional: clean ORION output
                    # is an observation, not privileged ground truth.
                    clean_severity = torch.full_like(proxy, 0.03)
                    clean_event = torch.zeros_like(proxy)
                    clean_valid = torch.ones_like(proxy, dtype=torch.bool)
                teacher = (
                    torch.full_like(proxy, 0.01 + 0.02 * severity)
                    if pair_index == 0
                    else None
                )
                records.append(
                    PairedSpatialFeatureRecord(
                        record_id=f"{pair_id}/severity_{severity:g}",
                        pair_id=pair_id,
                        route_id=route_id,
                        town=town,
                        severity=float(severity),
                        observed_patch_features=observed,
                        clean_patch_features=clean,
                        error_severity_target=actual_severity,
                        failure_event_target=actual_event,
                        target_valid_mask=actual_valid,
                        clean_error_severity_target=clean_severity,
                        clean_failure_event_target=clean_event,
                        clean_target_valid_mask=clean_valid,
                        corruption_mask=mask,
                        ensemble_teacher_variance=teacher,
                        metadata={"mock": True, "corruption": "local_feature_noise"},
                    )
                )
    return records


def run_stage1_training(
    records: Sequence[PairedSpatialFeatureRecord],
    manifest: RouteDisjointManifest,
    output_path: Path | str,
    feature_dim: int,
    hidden_dim: int = 128,
    ensemble_members: int = 3,
    min_log_variance: float = -6.0,
    max_log_variance: float = 3.0,
    teacher_epochs: int = 1,
    student_epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    seed: int = 0,
    device: torch.device | str = "cpu",
    weights: Optional[SpatialLossWeights] = None,
) -> Dict[str, Any]:
    """Train an ensemble teacher and distilled single-pass spatial UQ head."""
    validate_manifest_coverage(records, manifest)
    if not records:
        raise SpatialTrainingDataError("training requires at least one record")
    if any(record.observed_patch_features.shape[-1] != feature_dim for record in records):
        raise SpatialTrainingDataError("feature_dim does not match all paired records")
    if teacher_epochs < 0 or student_epochs < 0:
        raise ValueError("epoch counts must be non-negative")
    if teacher_epochs == 0 and student_epochs > 0:
        raise ValueError("student distillation requires a trained teacher phase")

    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)
    weights = weights or SpatialLossWeights()

    train_dataset = PairedSpatialFeatureDataset(records, manifest, "train")
    validation_dataset = PairedSpatialFeatureDataset(records, manifest, "validation")
    train_loader = _loader(train_dataset, batch_size, shuffle=True, seed=seed)
    validation_loader = _loader(
        validation_dataset, batch_size, shuffle=False, seed=seed
    )

    ensemble = SpatialHeadEnsemble(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        n_members=ensemble_members,
        min_log_variance=min_log_variance,
        max_log_variance=max_log_variance,
    ).to(device)
    teacher_optimizer = torch.optim.AdamW(
        ensemble.parameters(), lr=learning_rate
    )
    history: Dict[str, List[Dict[str, float]]] = {
        "teacher_train": [],
        "student_train": [],
        "validation": [],
    }
    for _ in range(teacher_epochs):
        history["teacher_train"].append(
            train_head_ensemble_epoch(
                ensemble, train_loader, teacher_optimizer, weights, device
            )
        )

    student = SpatialPatchUQHead(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        min_log_variance=min_log_variance,
        max_log_variance=max_log_variance,
        predict_epistemic=True,
    ).to(device)
    student_optimizer = torch.optim.AdamW(
        student.parameters(), lr=learning_rate
    )
    for _ in range(student_epochs):
        history["student_train"].append(
            train_student_epoch(
                student, ensemble, train_loader, student_optimizer, weights, device
            )
        )
        history["validation"].append(
            evaluate_student(student, validation_loader, weights, device)
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": SPATIAL_CHECKPOINT_SCHEMA_VERSION,
        "target_contract_schema_version": TARGET_CONTRACT_SCHEMA_VERSION,
        "spatial_output_schema_version": SpatialUQOutput.CURRENT_SCHEMA_VERSION,
        "paired_record_schema_version": PAIRED_RECORD_SCHEMA_VERSION,
        "student_state": student.state_dict(),
        "ensemble_state": ensemble.state_dict(),
        "model_config": {
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "min_log_variance": min_log_variance,
            "max_log_variance": max_log_variance,
            "predict_epistemic": True,
            "ensemble_members": ensemble_members,
        },
        "training_config": {
            "teacher_epochs": teacher_epochs,
            "student_epochs": student_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "loss_weights": dict(weights.__dict__),
        },
        "route_disjoint_manifest": manifest.to_dict(),
        "target_provenance": summarize_target_provenance(records),
        "claim_boundary": dict(DEFAULT_CLAIM_BOUNDARY),
        "history": history,
    }
    torch.save(checkpoint, output_path)
    return checkpoint


__all__ = [
    "PAIRED_RECORD_SCHEMA_VERSION",
    "PAIRED_DATASET_SCHEMA_VERSION",
    "LEGACY_PAIRED_RECORD_SCHEMA_VERSION",
    "LEGACY_PAIRED_DATASET_SCHEMA_VERSION",
    "ROUTE_MANIFEST_SCHEMA_VERSION",
    "SPATIAL_CHECKPOINT_SCHEMA_VERSION",
    "TARGET_CONTRACT_SCHEMA_VERSION",
    "TARGET_ACTUAL_FAILURE",
    "TARGET_REPRESENTATION_PROXY",
    "DEFAULT_CLAIM_BOUNDARY",
    "SpatialTrainingDataError",
    "PairedSpatialFeatureRecord",
    "migrate_legacy_v1_proxy_payload",
    "save_paired_feature_records",
    "load_paired_feature_records",
    "RouteDisjointManifest",
    "build_route_disjoint_manifest",
    "validate_manifest_coverage",
    "PairedSpatialFeatureDataset",
    "PairGroupedBatchSampler",
    "SpatialTrainingBatch",
    "collate_paired_spatial_records",
    "SpatialLossWeights",
    "SpatialEnsembleOutput",
    "SpatialHeadEnsemble",
    "compute_spatial_training_loss",
    "summarize_target_provenance",
    "train_head_ensemble_epoch",
    "train_student_epoch",
    "evaluate_student",
    "make_mock_paired_records",
    "run_stage1_training",
]
