"""Route-disjoint training helpers for counterfactual observation evidence."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from uq_estimator.counterfactual_compaction import dynamic_symmetric_int8_roundtrip
from uq_estimator.counterfactual_evidence import (
    EVIDENCE_COMPONENTS,
    CounterfactualEvidenceError,
    CounterfactualEvidenceTarget,
    ObservationEvidenceAdapter,
    ObservationEvidenceHurdleAdapter,
    balanced_evidence_presence_loss,
    balanced_counterfactual_evidence_regression_loss,
    counterfactual_evidence_regression_loss,
    counterfactual_evidence_target,
    responsive_evidence_magnitude_loss,
    scale_counterfactual_target,
)
from uq_estimator.observation_uq_shard import validate_feature_shard
from uq_estimator.observation_uq_v3 import _binary_auc, _spearman


COUNTERFACTUAL_TRAINING_SCHEMA_VERSION = "orion.counterfactual-evidence-training/v1"


def _exact_quantiles_1d(
    values: torch.Tensor, quantiles: Sequence[float]
) -> List[torch.Tensor]:
    """Return linearly interpolated quantiles without torch.quantile's size cap."""

    if values.ndim != 1 or values.numel() == 0:
        raise CounterfactualEvidenceError("quantile input must be non-empty and 1D")
    levels = [float(level) for level in quantiles]
    if any(not math.isfinite(level) or not 0.0 <= level <= 1.0 for level in levels):
        raise CounterfactualEvidenceError("quantile levels must be finite in [0,1]")
    ordered = torch.sort(values).values
    maximum_index = ordered.numel() - 1
    result = []
    for level in levels:
        position = level * maximum_index
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            result.append(ordered[lower])
            continue
        weight = position - lower
        result.append(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)
    return result


def _exact_quantile_1d(values: torch.Tensor, quantile: float) -> torch.Tensor:
    return _exact_quantiles_1d(values, [quantile])[0]


def _project_model_input(
    value: torch.Tensor,
    input_projection: Optional[torch.Tensor],
    input_quantization: Optional[str] = None,
) -> torch.Tensor:
    """Apply an optional frozen feature projection only to model inputs.

    Counterfactual targets must be computed before this transform so the
    original frozen-feature supervision remains unchanged.
    """

    transformed = value
    if input_projection is not None:
        if (
            input_projection.ndim != 2
            or input_projection.shape[0] != value.shape[-1]
            or input_projection.device != value.device
            or not input_projection.is_floating_point()
        ):
            raise CounterfactualEvidenceError("model-input projection differs")
        transformed = (value.float() @ input_projection.float()).to(dtype=value.dtype)
    if input_quantization is not None:
        if input_quantization != "dynamic_symmetric_int8_per_grid_channel":
            raise CounterfactualEvidenceError("unsupported model-input quantization")
        transformed = dynamic_symmetric_int8_roundtrip(transformed)
    return transformed


@dataclass(frozen=True)
class CounterfactualEvidenceRecord:
    sample_id: str
    pair_id: str
    route_id: str
    frame_idx: int
    split: str
    family: str
    severity: float
    reference_current: torch.Tensor
    observed_current: torch.Tensor
    reference_previous: torch.Tensor
    observed_previous: torch.Tensor
    previous_valid: bool
    corruption_mask: Optional[torch.Tensor] = None
    stored_target_values: Optional[torch.Tensor] = None
    stored_target_component_valid: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if not all(
            str(value).strip()
            for value in (
                self.sample_id,
                self.pair_id,
                self.route_id,
                self.split,
                self.family,
            )
        ):
            raise CounterfactualEvidenceError("counterfactual record identity is empty")
        if int(self.frame_idx) < 0:
            raise CounterfactualEvidenceError("counterfactual frame index is negative")
        tensors = (
            self.reference_current,
            self.observed_current,
            self.reference_previous,
            self.observed_previous,
        )
        if len({tuple(value.shape) for value in tensors}) != 1:
            raise CounterfactualEvidenceError("counterfactual record feature shapes differ")
        if any(value.ndim != 4 or not value.is_floating_point() for value in tensors):
            raise CounterfactualEvidenceError("record features must be floating [V,H,W,D]")
        stored = (self.stored_target_values, self.stored_target_component_valid)
        if (stored[0] is None) != (stored[1] is None):
            raise CounterfactualEvidenceError(
                "stored target values and validity must appear together"
            )
        if stored[0] is not None:
            assert stored[1] is not None
            expected = tuple(self.reference_current.shape[:-1]) + (
                len(EVIDENCE_COMPONENTS),
            )
            if (
                tuple(stored[0].shape) != expected
                or not stored[0].is_floating_point()
                or not bool(torch.isfinite(stored[0]).all())
                or bool((stored[0] < 0).any())
            ):
                raise CounterfactualEvidenceError("stored target values differ")
            if tuple(stored[1].shape) != expected or stored[1].dtype != torch.bool:
                raise CounterfactualEvidenceError("stored target validity differs")


def records_from_counterfactual_shard(
    payload: Mapping[str, Any]
) -> List[CounterfactualEvidenceRecord]:
    validate_feature_shard(payload)
    provenance = payload.get("provenance", {})
    if provenance.get("extraction_schema_version") not in {
        "orion.counterfactual-evidence-extraction/v1",
        "orion.counterfactual-evidence-extraction/v2",
    }:
        raise CounterfactualEvidenceError("feature shard is not counterfactual evidence data")
    if provenance.get("corruption_mask_is_primary_target") is not False:
        raise CounterfactualEvidenceError("feature shard target attestation is unsafe")
    clean_features = payload["clean_features"]
    clean_items = payload["clean_items"]
    observed_features = payload["observed_features"]
    observed_items = payload["observed_items"]
    clean_key_to_index = {
        (str(item["route_id"]), int(item["frame_idx"])): index
        for index, item in enumerate(clean_items)
    }
    observed_key_to_index = {
        (
            str(item["route_id"]),
            int(item["frame_idx"]),
            str(item["family"]),
            float(item["severity"]),
        ): index
        for index, item in enumerate(observed_items)
    }
    records = []
    for observed_index, item in enumerate(observed_items):
        route_id = str(item["route_id"])
        frame_idx = int(item["frame_idx"])
        family = str(item["family"])
        severity = float(item["severity"])
        clean_index = int(item["clean_index"])
        previous_clean_index = clean_key_to_index.get((route_id, frame_idx - 1))
        previous_observed_index = observed_key_to_index.get(
            (route_id, frame_idx - 1, family, severity)
        )
        if (previous_clean_index is None) != (previous_observed_index is None):
            raise CounterfactualEvidenceError(
                "reference/observed temporal sequence availability differs"
            )
        reference_current = clean_features[clean_index]
        observed_current = observed_features[observed_index]
        records.append(
            CounterfactualEvidenceRecord(
                sample_id=str(item["sample_id"]),
                pair_id=str(clean_items[clean_index]["sample_id"]),
                route_id=route_id,
                frame_idx=frame_idx,
                split=str(item["split"]),
                family=family,
                severity=severity,
                reference_current=reference_current,
                observed_current=observed_current,
                reference_previous=(
                    clean_features[previous_clean_index]
                    if previous_clean_index is not None
                    else torch.zeros_like(reference_current)
                ),
                observed_previous=(
                    observed_features[previous_observed_index]
                    if previous_observed_index is not None
                    else torch.zeros_like(observed_current)
                ),
                previous_valid=previous_clean_index is not None,
                corruption_mask=item.get("corruption_mask"),
            )
        )
    return records


def select_records(
    records: Sequence[CounterfactualEvidenceRecord],
    splits: Sequence[str],
    families: Sequence[str],
) -> List[CounterfactualEvidenceRecord]:
    split_set = set(splits)
    family_set = set(families)
    result = [
        record
        for record in records
        if record.split in split_set and record.family in family_set
    ]
    if not result:
        raise CounterfactualEvidenceError("counterfactual record selection is empty")
    return result


def _collate_record_features(
    records: Sequence[CounterfactualEvidenceRecord], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not records:
        raise CounterfactualEvidenceError("cannot collate empty evidence records")
    values = []
    for name in (
        "reference_current",
        "observed_current",
        "reference_previous",
        "observed_previous",
    ):
        values.append(
            torch.stack([getattr(record, name) for record in records]).to(
                device=device, dtype=torch.float32
            )
        )
    valid = torch.tensor(
        [record.previous_valid for record in records],
        dtype=torch.bool,
        device=device,
    )
    return values[0], values[1], values[2], values[3], valid


def targets_for_records(
    records: Sequence[CounterfactualEvidenceRecord],
    device: torch.device,
    scales: Optional[torch.Tensor] = None,
) -> CounterfactualEvidenceTarget:
    stored = [record.stored_target_values is not None for record in records]
    if any(stored):
        if not all(stored):
            raise CounterfactualEvidenceError(
                "cannot mix stored and recomputed targets in one batch"
            )
        target = CounterfactualEvidenceTarget(
            values=torch.stack(
                [record.stored_target_values for record in records]  # type: ignore[arg-type]
            ).to(device=device, dtype=torch.float32),
            component_valid=torch.stack(
                [record.stored_target_component_valid for record in records]  # type: ignore[arg-type]
            ).to(device=device, dtype=torch.bool),
        )
        return scale_counterfactual_target(target, scales) if scales is not None else target
    reference, observed, reference_previous, observed_previous, valid = (
        _collate_record_features(records, device)
    )
    target = _targets_from_records_or_features(
        records,
        reference,
        observed,
        reference_previous,
        observed_previous,
        valid,
    )
    return scale_counterfactual_target(target, scales) if scales is not None else target


def _targets_from_records_or_features(
    records: Sequence[CounterfactualEvidenceRecord],
    reference: torch.Tensor,
    observed: torch.Tensor,
    reference_previous: torch.Tensor,
    observed_previous: torch.Tensor,
    valid: torch.Tensor,
) -> CounterfactualEvidenceTarget:
    """Prefer pre-quantization targets stored by compact-shard writers.

    A batch may contain either only stored targets or none.  Mixing the two is
    rejected so a training batch cannot silently change supervision semantics.
    """

    stored = [record.stored_target_values is not None for record in records]
    if any(stored):
        if not all(stored):
            raise CounterfactualEvidenceError(
                "cannot mix stored and recomputed targets in one batch"
            )
        values = torch.stack(
            [record.stored_target_values for record in records]  # type: ignore[arg-type]
        ).to(device=reference.device, dtype=torch.float32)
        component_valid = torch.stack(
            [record.stored_target_component_valid for record in records]  # type: ignore[arg-type]
        ).to(device=reference.device, dtype=torch.bool)
        return CounterfactualEvidenceTarget(
            values=values,
            component_valid=component_valid,
        )
    return counterfactual_evidence_target(
        reference,
        observed,
        reference_previous,
        observed_previous,
        valid,
    )


def fit_train_component_scales(
    records: Sequence[CounterfactualEvidenceRecord],
    device: torch.device,
    batch_size: int = 4,
    quantile: float = 0.95,
    response_floor: float = 1e-6,
) -> torch.Tensor:
    """Fit train-only scales on intervention-responsive target cells.

    A local intervention changes only one of six camera views.  Including the
    structurally unchanged views in the quantile would therefore make q95 zero
    by construction.  Responsiveness is defined from the measured paired
    target, never from the corruption mask.
    """

    if (
        not records
        or batch_size <= 0
        or not 0.5 <= quantile < 1.0
        or not math.isfinite(response_floor)
        or response_floor < 0
    ):
        raise CounterfactualEvidenceError("invalid component scale request")
    chunks = [[] for _ in EVIDENCE_COMPONENTS]
    for start in range(0, len(records), batch_size):
        target = targets_for_records(records[start : start + batch_size], device)
        for component in range(len(EVIDENCE_COMPONENTS)):
            values = target.values[..., component]
            valid = target.component_valid[..., component] & (
                values > response_floor
            )
            chunks[component].append(
                values[valid].detach().cpu().float()
            )
    scales = []
    for name, component_chunks in zip(EVIDENCE_COMPONENTS, chunks):
        responsive = [values for values in component_chunks if values.numel()]
        if not responsive:
            raise CounterfactualEvidenceError(
                "%s has no intervention-responsive train targets" % name
            )
        scales.append(_exact_quantile_1d(torch.cat(responsive), quantile).clamp_min(1e-4))
    return torch.stack(scales)


@torch.no_grad()
def audit_train_target_distribution(
    records: Sequence[CounterfactualEvidenceRecord],
    device: torch.device,
    batch_size: int = 4,
    quantile: float = 0.95,
    response_floor: float = 1e-6,
) -> Dict[str, Any]:
    """Describe train targets before any adapter optimization.

    This audit deliberately reads neither corruption masks nor validation
    records.  It freezes the scaling population and checks whether the paired
    feature targets themselves contain a measurable severity response.
    """

    if not records:
        raise CounterfactualEvidenceError("target-distribution audit is empty")
    component_chunks: List[List[torch.Tensor]] = [
        [] for _ in EVIDENCE_COMPONENTS
    ]
    condition_rows: Dict[Tuple[str, float], List[torch.Tensor]] = {}
    pair_rows: Dict[Tuple[str, str], Dict[float, torch.Tensor]] = {}
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        target = targets_for_records(batch, device)
        values = target.values.detach().cpu().float()
        validity = target.component_valid.detach().cpu()
        for component in range(len(EVIDENCE_COMPONENTS)):
            component_chunks[component].append(
                values[..., component][validity[..., component]]
            )
        valid_count = validity.sum(dim=-1).clamp_min(1)
        combined = (
            values * validity.to(values.dtype)
        ).sum(dim=-1) / valid_count
        for index, record in enumerate(batch):
            component_means = []
            for component in range(len(EVIDENCE_COMPONENTS)):
                component_valid = validity[index, ..., component]
                component_means.append(
                    values[index, ..., component][component_valid].mean()
                    if bool(component_valid.any())
                    else torch.tensor(float("nan"))
                )
            row = torch.stack(component_means + [combined[index].mean()])
            condition_rows.setdefault((record.family, record.severity), []).append(row)
            pair_rows.setdefault((record.pair_id, record.family), {})[
                record.severity
            ] = row

    components = {}
    scales = []
    quantiles = (0.50, 0.80, 0.90, 0.95, 0.99)
    for name, chunks in zip(EVIDENCE_COMPONENTS, component_chunks):
        values = torch.cat(chunks)
        responsive = values[values > response_floor]
        if responsive.numel() == 0:
            raise CounterfactualEvidenceError(
                "%s has no intervention-responsive train targets" % name
            )
        scale = _exact_quantile_1d(responsive, quantile).clamp_min(1e-4)
        scales.append(scale)
        all_quantiles = _exact_quantiles_1d(values, quantiles)
        responsive_quantiles = _exact_quantiles_1d(responsive, quantiles)
        components[name] = {
            "valid_cell_count": int(values.numel()),
            "responsive_cell_count": int(responsive.numel()),
            "responsive_fraction": float(responsive.numel() / values.numel()),
            "all_cell_quantiles": {
                "q%02d" % int(level * 100): float(value)
                for level, value in zip(quantiles, all_quantiles)
            },
            "responsive_quantiles": {
                "q%02d" % int(level * 100): float(value)
                for level, value in zip(quantiles, responsive_quantiles)
            },
            "selected_scale": float(scale),
        }

    by_family_severity = {}
    for (family, severity), rows in sorted(condition_rows.items()):
        stacked = torch.stack(rows)
        means = {}
        for index, name in enumerate(EVIDENCE_COMPONENTS + ("combined",)):
            finite = stacked[:, index][torch.isfinite(stacked[:, index])]
            means[name] = float(finite.mean()) if finite.numel() else float("nan")
        by_family_severity["%s/severity_%g" % (family, severity)] = {
            "record_count": len(rows),
            "target_means": means,
        }

    by_family_monotonicity = {}
    families = sorted({record.family for record in records})
    for family in families:
        comparisons = []
        for (pair_id, owner_family), severity_rows in pair_rows.items():
            if owner_family != family or len(severity_rows) < 2:
                continue
            low_severity, high_severity = min(severity_rows), max(severity_rows)
            comparisons.append(
                severity_rows[high_severity] - severity_rows[low_severity]
            )
        if not comparisons:
            raise CounterfactualEvidenceError(
                "%s has no paired severity comparisons" % family
            )
        comparison = torch.stack(comparisons)
        monotonicity = {}
        for index, name in enumerate(EVIDENCE_COMPONENTS + ("combined",)):
            difference = comparison[:, index].float()
            finite = torch.isfinite(difference)
            if bool(finite.any()):
                monotonicity[name] = float((difference[finite] > 0).float().mean())
        by_family_monotonicity[family] = monotonicity

    return {
        "record_count": len(records),
        "route_count": len({record.route_id for record in records}),
        "families": sorted({record.family for record in records}),
        "target_scale_quantile": quantile,
        "response_floor": response_floor,
        "scale_population": (
            "train-only valid cells with measured paired target > response_floor"
        ),
        "component_scales": {
            name: float(value)
            for name, value in zip(EVIDENCE_COMPONENTS, scales)
        },
        "components": components,
        "by_family_severity": by_family_severity,
        "paired_severity_higher_target_fraction": by_family_monotonicity,
        "corruption_mask_read": False,
        "validation_records_read": False,
    }


def _metric_summary(values: Sequence[float]) -> Dict[str, float]:
    tensor = torch.tensor(
        [value for value in values if math.isfinite(value)], dtype=torch.float32
    )
    if tensor.numel() == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "p10": float("nan")}
    median, p10 = _exact_quantiles_1d(tensor, [0.50, 0.10])
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "median": float(median),
        "p10": float(p10),
    }


@torch.no_grad()
def audit_target_spatial_support(
    records: Sequence[CounterfactualEvidenceRecord],
    scales: torch.Tensor,
    device: torch.device,
    batch_size: int = 2,
    mask_label_floor: float = 0.25,
) -> Dict[str, Any]:
    """Check target localization against intervention masks without optimizing on them."""

    if not records or batch_size <= 0 or not 0.0 < mask_label_floor <= 1.0:
        raise CounterfactualEvidenceError("invalid spatial-support audit request")
    rows = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        target = targets_for_records(batch, device, scales=scales)
        values = target.values.detach().cpu().float()
        validity = target.component_valid.detach().cpu()
        valid_count = validity.sum(dim=-1).clamp_min(1)
        combined = (
            values * validity.to(values.dtype)
        ).sum(dim=-1) / valid_count
        for index, record in enumerate(batch):
            if record.corruption_mask is None:
                raise CounterfactualEvidenceError("spatial audit requires audit-only mask")
            coverage = record.corruption_mask.detach().cpu().float()
            affected_views = coverage.reshape(coverage.shape[0], -1).max(dim=1).values > 0
            if int(affected_views.sum()) != 1:
                raise CounterfactualEvidenceError(
                    "bounded spatial audit expects one intervened camera"
                )
            view = int(torch.nonzero(affected_views, as_tuple=False)[0, 0])
            coverage_view = coverage[view].reshape(-1)
            labels = coverage_view >= mask_label_floor
            if not bool(labels.any()) or bool(labels.all()):
                raise CounterfactualEvidenceError("mask threshold has one class")
            maps = {
                name: values[index, view, ..., component].reshape(-1)
                for component, name in enumerate(EVIDENCE_COMPONENTS)
                if bool(validity[index, view, ..., component].any())
            }
            maps["combined"] = combined[index, view].reshape(-1)
            metrics = {}
            for name, score in maps.items():
                positive_mean = float(score[labels].mean())
                negative_mean = float(score[~labels].mean())
                top_count = int(labels.sum())
                top_indices = torch.topk(score, top_count).indices
                predicted = torch.zeros_like(labels)
                predicted[top_indices] = True
                intersection = int((predicted & labels).sum())
                union = int((predicted | labels).sum())
                metrics[name] = {
                    "within_view_mask_auroc": _binary_auc(score, labels),
                    "within_view_mask_coverage_spearman": _spearman(
                        score, coverage_view
                    ),
                    "inside_mean": positive_mean,
                    "outside_mean": negative_mean,
                    "inside_outside_ratio": positive_mean / max(negative_mean, 1e-8),
                    "equal_area_top_iou": intersection / max(union, 1),
                }
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "route_id": record.route_id,
                    "family": record.family,
                    "severity": record.severity,
                    "affected_view": view,
                    "metrics": metrics,
                }
            )

    metric_names = (
        "within_view_mask_auroc",
        "within_view_mask_coverage_spearman",
        "inside_outside_ratio",
        "equal_area_top_iou",
    )

    def summarize_rows(selected):
        return {
            component: {
                metric: _metric_summary(
                    [
                        row["metrics"][component][metric]
                        for row in selected
                        if component in row["metrics"]
                    ]
                )
                for metric in metric_names
            }
            for component in EVIDENCE_COMPONENTS + ("combined",)
        }

    by_family_severity = {}
    for family, severity in sorted(
        {(record.family, record.severity) for record in records}
    ):
        selected = [
            row
            for row in rows
            if row["family"] == family and row["severity"] == severity
        ]
        by_family_severity["%s/severity_%g" % (family, severity)] = summarize_rows(
            selected
        )
    return {
        "record_count": len(rows),
        "route_count": len({row["route_id"] for row in rows}),
        "mask_label_floor": mask_label_floor,
        "scope": "within-intervened-view localization of paired feature targets",
        "overall": summarize_rows(rows),
        "by_family_severity": by_family_severity,
        "corruption_mask_role": "audit metric only; optimizer weight remains zero",
        "validation_records_read": False,
    }


def _group_batches(
    records: Sequence[CounterfactualEvidenceRecord],
    pair_batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterable[List[CounterfactualEvidenceRecord]]:
    if pair_batch_size <= 0:
        raise CounterfactualEvidenceError("pair batch size must be positive")
    grouped: Dict[str, List[CounterfactualEvidenceRecord]] = {}
    for record in records:
        grouped.setdefault(record.pair_id, []).append(record)
    pair_ids = sorted(grouped)
    if shuffle:
        random.Random(seed).shuffle(pair_ids)
    for start in range(0, len(pair_ids), pair_batch_size):
        yield [
            record
            for pair_id in pair_ids[start : start + pair_batch_size]
            for record in sorted(
                grouped[pair_id], key=lambda item: (item.family, item.severity)
            )
        ]


def measured_target_ranking_loss(
    prediction: torch.Tensor,
    target: CounterfactualEvidenceTarget,
    records: Sequence[CounterfactualEvidenceRecord],
    margin: float = 0.10,
) -> torch.Tensor:
    if prediction.shape != target.values.shape or prediction.shape[0] != len(records):
        raise CounterfactualEvidenceError("ranking tensors/records disagree")
    total = prediction.sum() * 0.0
    active_count = prediction.new_zeros(())
    groups: Dict[Tuple[str, str], List[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault((record.pair_id, record.family), []).append(index)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: records[index].severity)
        for low, high in zip(ordered[:-1], ordered[1:]):
            active = (
                target.component_valid[low]
                & target.component_valid[high]
                & (target.values[high] > target.values[low] + 1e-6)
            )
            raw = F.relu(margin - (prediction[high] - prediction[low]))
            total = total + torch.where(active, raw, torch.zeros_like(raw)).sum()
            active_count = active_count + active.to(prediction.dtype).sum()
    return total / active_count.clamp_min(1.0)


def _reference_batch(
    records: Sequence[CounterfactualEvidenceRecord], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unique = {}
    for record in records:
        unique.setdefault(record.pair_id, record)
    representatives = list(unique.values())
    current = torch.stack([record.reference_current for record in representatives]).to(
        device=device, dtype=torch.float32
    )
    previous = torch.stack([record.reference_previous for record in representatives]).to(
        device=device, dtype=torch.float32
    )
    valid = torch.tensor(
        [record.previous_valid for record in representatives],
        dtype=torch.bool,
        device=device,
    )
    return current, previous, valid


def _responsive_top_fraction_labels(
    target: torch.Tensor,
    fraction: float = 0.20,
    response_floor: float = 1e-6,
) -> torch.Tensor:
    """Label the strongest measured responses without promoting zero ties.

    With one intervened camera, more than 80% of all-view cells can be exactly
    zero.  A conventional all-cell q80 threshold would consequently label
    every zero as positive.  The quantile is fitted only on measured responses;
    unchanged cells remain legitimate negatives for AUROC.
    """

    flat = target.reshape(-1)
    responsive = flat > response_floor
    if int(responsive.sum()) < 2 or not 0.0 < fraction < 1.0:
        raise CounterfactualEvidenceError(
            "top-fraction metric needs at least two responsive targets"
        )
    threshold = _exact_quantile_1d(flat[responsive], 1.0 - fraction)
    return (responsive & (flat >= threshold)).reshape(target.shape)


def run_evidence_epoch(
    model: ObservationEvidenceAdapter,
    records: Sequence[CounterfactualEvidenceRecord],
    scales: torch.Tensor,
    device: torch.device,
    pair_batch_size: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    seed: int = 0,
    ranking_weight: float = 0.25,
    reference_weight: float = 0.50,
    responsive_weight: Optional[float] = None,
    response_floor: float = 1e-6,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "observed": 0.0, "ranking": 0.0, "reference": 0.0}
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in _group_batches(
            records, pair_batch_size, shuffle=training, seed=seed
        ):
            reference, observed, reference_previous, observed_previous, valid = (
                _collate_record_features(batch, device)
            )
            target = _targets_from_records_or_features(
                batch,
                reference,
                observed,
                reference_previous,
                observed_previous,
                valid,
            )
            target = scale_counterfactual_target(target, scales)
            prediction = model(observed, observed_previous, valid)
            observed_loss = (
                counterfactual_evidence_regression_loss(prediction, target)
                if responsive_weight is None
                else balanced_counterfactual_evidence_regression_loss(
                    prediction,
                    target,
                    responsive_weight=responsive_weight,
                    response_floor=response_floor,
                )
            )
            ranking_loss = measured_target_ranking_loss(prediction, target, batch)
            clean_current, clean_previous, clean_valid = _reference_batch(batch, device)
            clean_prediction = model(clean_current, clean_previous, clean_valid)
            reference_loss = F.smooth_l1_loss(
                torch.log1p(clean_prediction),
                torch.zeros_like(clean_prediction),
            )
            loss = (
                observed_loss
                + ranking_weight * ranking_loss
                + reference_weight * reference_loss
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            weight = len(batch)
            totals["total"] += float(loss.detach()) * weight
            totals["observed"] += float(observed_loss.detach()) * weight
            totals["ranking"] += float(ranking_loss.detach()) * weight
            totals["reference"] += float(reference_loss.detach()) * weight
            count += weight
    return {name: value / max(count, 1) for name, value in totals.items()}


def run_hurdle_evidence_epoch(
    model: ObservationEvidenceHurdleAdapter,
    records: Sequence[CounterfactualEvidenceRecord],
    scales: torch.Tensor,
    device: torch.device,
    pair_batch_size: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    seed: int = 0,
    presence_responsive_weight: float = 0.5,
    magnitude_weight: float = 1.0,
    ranking_weight: float = 0.25,
    reference_weight: float = 0.5,
    response_floor: float = 1e-6,
    support_thresholds: Optional[torch.Tensor] = None,
    input_projection: Optional[torch.Tensor] = None,
    input_quantization: Optional[str] = None,
) -> Dict[str, float]:
    """Train/evaluate the two-part sparse evidence head."""

    training = optimizer is not None
    model.train(training)
    totals = {
        "total": 0.0,
        "presence": 0.0,
        "magnitude": 0.0,
        "ranking": 0.0,
        "reference_presence": 0.0,
        "reference_score": 0.0,
    }
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in _group_batches(
            records, pair_batch_size, shuffle=training, seed=seed
        ):
            reference, observed, reference_previous, observed_previous, valid = (
                _collate_record_features(batch, device)
            )
            target = scale_counterfactual_target(
                _targets_from_records_or_features(
                    batch,
                    reference,
                    observed,
                    reference_previous,
                    observed_previous,
                    valid,
                ),
                scales,
            )
            prediction = model.predict_parts(
                _project_model_input(observed, input_projection, input_quantization),
                _project_model_input(
                    observed_previous, input_projection, input_quantization
                ),
                valid,
            )
            presence_loss = balanced_evidence_presence_loss(
                prediction.presence_logits,
                target,
                responsive_weight=presence_responsive_weight,
                response_floor=response_floor,
                support_thresholds=support_thresholds,
            )
            magnitude_loss = responsive_evidence_magnitude_loss(
                prediction.conditional_magnitude,
                target,
                response_floor=response_floor,
            )
            ranking_loss = measured_target_ranking_loss(
                prediction.score, target, batch
            )
            clean_current, clean_previous, clean_valid = _reference_batch(batch, device)
            clean = model.predict_parts(
                _project_model_input(
                    clean_current, input_projection, input_quantization
                ),
                _project_model_input(
                    clean_previous, input_projection, input_quantization
                ),
                clean_valid,
            )
            reference_presence = F.binary_cross_entropy_with_logits(
                clean.presence_logits, torch.zeros_like(clean.presence_logits)
            )
            reference_score = F.smooth_l1_loss(
                torch.log1p(clean.score), torch.zeros_like(clean.score)
            )
            loss = (
                presence_loss
                + magnitude_weight * magnitude_loss
                + ranking_weight * ranking_loss
                + reference_weight * (reference_presence + reference_score)
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            weight = len(batch)
            for name, value in (
                ("total", loss),
                ("presence", presence_loss),
                ("magnitude", magnitude_loss),
                ("ranking", ranking_loss),
                ("reference_presence", reference_presence),
                ("reference_score", reference_score),
            ):
                totals[name] += float(value.detach()) * weight
            count += weight
    return {name: value / max(count, 1) for name, value in totals.items()}


@torch.no_grad()
def evaluate_hurdle_diagnostics(
    model: ObservationEvidenceHurdleAdapter,
    records: Sequence[CounterfactualEvidenceRecord],
    scales: torch.Tensor,
    device: torch.device,
    batch_size: int = 4,
    response_floor: float = 1e-6,
    support_thresholds: Optional[torch.Tensor] = None,
    input_projection: Optional[torch.Tensor] = None,
    input_quantization: Optional[str] = None,
) -> Dict[str, Any]:
    """Report whether the two hurdle factors behave as intended."""

    model.eval()
    presence_chunks = []
    magnitude_chunks = []
    target_chunks = []
    validity_chunks = []
    clean_presence = {}
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        reference, observed, reference_previous, observed_previous, valid = (
            _collate_record_features(batch, device)
        )
        target = scale_counterfactual_target(
            _targets_from_records_or_features(
                batch,
                reference,
                observed,
                reference_previous,
                observed_previous,
                valid,
            ),
            scales,
        )
        prediction = model.predict_parts(
            _project_model_input(observed, input_projection, input_quantization),
            _project_model_input(
                observed_previous, input_projection, input_quantization
            ),
            valid,
        )
        presence_chunks.append(prediction.presence_probability.cpu())
        magnitude_chunks.append(prediction.conditional_magnitude.cpu())
        target_chunks.append(target.values.cpu())
        validity_chunks.append(target.component_valid.cpu())
        for index, record in enumerate(batch):
            if record.pair_id not in clean_presence:
                clean_presence[record.pair_id] = model.predict_parts(
                    _project_model_input(
                        reference[index : index + 1],
                        input_projection,
                        input_quantization,
                    ),
                    _project_model_input(
                        reference_previous[index : index + 1],
                        input_projection,
                        input_quantization,
                    ),
                    valid[index : index + 1],
                ).presence_probability[0].cpu()
    presence = torch.cat(presence_chunks)
    magnitude = torch.cat(magnitude_chunks)
    target_values = torch.cat(target_chunks)
    validity = torch.cat(validity_chunks)
    if support_thresholds is None:
        thresholds = torch.full(
            (len(EVIDENCE_COMPONENTS),), float(response_floor)
        )
        support_definition = "target > numerical response floor"
    else:
        if support_thresholds.shape != (len(EVIDENCE_COMPONENTS),):
            raise CounterfactualEvidenceError("diagnostic support thresholds differ")
        thresholds = support_thresholds.detach().cpu().float()
        support_definition = "target > frozen train-responsive q80 per component"
    components = {}
    for index, name in enumerate(EVIDENCE_COMPONENTS):
        valid = validity[..., index]
        responsive = valid & (target_values[..., index] > thresholds[index])
        background = valid & ~responsive
        components[name] = {
            "presence_auroc": _binary_auc(
                presence[..., index][valid], responsive[valid]
            ),
            "responsive_presence_mean": float(
                presence[..., index][responsive].mean()
            ),
            "background_presence_mean": float(
                presence[..., index][background].mean()
            ),
            "responsive_conditional_magnitude_mae": float(
                (
                    magnitude[..., index][responsive]
                    - target_values[..., index][responsive]
                )
                .abs()
                .mean()
            ),
        }
    clean_values = torch.stack(list(clean_presence.values())).reshape(-1)
    return {
        "components": components,
        "reference_presence_mean": float(clean_values.mean()),
        "reference_presence_p95": float(_exact_quantile_1d(clean_values, 0.95)),
        "support_definition": support_definition,
        "support_thresholds": {
            name: float(value) for name, value in zip(EVIDENCE_COMPONENTS, thresholds)
        },
        "corruption_mask_read": False,
    }


@torch.no_grad()
def evaluate_evidence_records(
    model: ObservationEvidenceAdapter,
    records: Sequence[CounterfactualEvidenceRecord],
    scales: torch.Tensor,
    device: torch.device,
    batch_size: int = 4,
    input_projection: Optional[torch.Tensor] = None,
    input_quantization: Optional[str] = None,
) -> Dict[str, Any]:
    model.eval()
    predictions = []
    targets = []
    validities = []
    route_rows: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    within_view_rows: Dict[str, List[Dict[str, float]]] = {}
    family_rows: Dict[str, Dict[str, List[float]]] = {}
    family_severity_rows: Dict[Tuple[str, float], Dict[str, List[float]]] = {}
    reference_predictions = {}
    within_view_skipped_records = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        reference, observed, reference_previous, observed_previous, valid = (
            _collate_record_features(batch, device)
        )
        target = scale_counterfactual_target(
            _targets_from_records_or_features(
                batch,
                reference,
                observed,
                reference_previous,
                observed_previous,
                valid,
            ),
            scales,
        )
        prediction = model(
            _project_model_input(observed, input_projection, input_quantization),
            _project_model_input(
                observed_previous, input_projection, input_quantization
            ),
            valid,
        )
        predictions.append(prediction.cpu())
        targets.append(target.values.cpu())
        validities.append(target.component_valid.cpu())
        for index, record in enumerate(batch):
            component_valid = target.component_valid[index]
            valid_count = component_valid.sum(dim=-1).clamp_min(1)
            combined_prediction = (
                prediction[index] * component_valid.to(prediction.dtype)
            ).sum(dim=-1) / valid_count
            combined_target = (
                target.values[index]
                * component_valid.to(target.values.dtype)
            ).sum(dim=-1) / valid_count
            route = route_rows.setdefault(record.route_id, {"pred": [], "target": []})
            route["pred"].append(combined_prediction.cpu())
            route["target"].append(combined_target.cpu())
            # Window-balanced schedules can change the corrupted camera between
            # adjacent frames.  At those boundaries the transient component
            # correctly responds in both the previous and current cameras, so
            # it cannot identify the current intervention view.  The two
            # persistent components depend only on the current clean/observed
            # pair and therefore remain a mask-free, metadata-free selector.
            persistent_valid = target.component_valid[index, ..., :2]
            persistent_mass = (
                target.values[index, ..., :2]
                * persistent_valid.to(target.values.dtype)
            ).sum(dim=(1, 2, 3))
            if not bool(torch.isfinite(persistent_mass).all()) or not bool(
                (persistent_mass > 1e-6).any()
            ):
                raise CounterfactualEvidenceError(
                    "persistent evaluation target does not identify an intervened view"
                )
            view = int(torch.argmax(persistent_mass))
            view_prediction = combined_prediction[view].reshape(-1).cpu()
            view_target = combined_target[view].reshape(-1).cpu()
            responsive_count = int((view_target > 1e-6).sum())
            if responsive_count < 2:
                # AUROC is undefined when the measured paired target contains
                # no usable positive cells.  Keep the record in every global,
                # route, family, severity, and clean-FP metric, but exclude it
                # from this one per-record within-view summary and report the
                # exact exclusion.  Lowering the response floor after seeing
                # validation data would silently change the frozen metric.
                within_view_skipped_records.append(
                    {
                        "sample_id": record.sample_id,
                        "route_id": record.route_id,
                        "frame_idx": int(record.frame_idx),
                        "family": record.family,
                        "severity": float(record.severity),
                        "selected_view": view,
                        "responsive_cell_count": responsive_count,
                    }
                )
            else:
                view_labels = _responsive_top_fraction_labels(view_target)
                within_view_rows.setdefault(record.route_id, []).append(
                    {
                        "target_top20_auroc": _binary_auc(
                            view_prediction, view_labels.reshape(-1)
                        ),
                        "patch_spearman": _spearman(view_prediction, view_target),
                    }
                )
            family = family_rows.setdefault(
                record.family, {"score": [], "target": [], "severity": []}
            )
            family["score"].append(float(combined_prediction.mean()))
            family["target"].append(float(combined_target.mean()))
            family["severity"].append(float(record.severity))
            family_severity = family_severity_rows.setdefault(
                (record.family, float(record.severity)),
                {"score": [], "target": []},
            )
            family_severity["score"].append(float(combined_prediction.mean()))
            family_severity["target"].append(float(combined_target.mean()))
            if record.pair_id not in reference_predictions:
                reference_predictions[record.pair_id] = model(
                    _project_model_input(
                        reference[index : index + 1],
                        input_projection,
                        input_quantization,
                    ),
                    _project_model_input(
                        reference_previous[index : index + 1],
                        input_projection,
                        input_quantization,
                    ),
                    valid[index : index + 1],
                )[0].cpu()
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    validity = torch.cat(validities)
    components = {}
    for index, name in enumerate(EVIDENCE_COMPONENTS):
        mask = validity[..., index]
        components[name] = {
            "patch_spearman": _spearman(
                prediction[..., index][mask], target[..., index][mask]
            ),
            "mae": float(
                (prediction[..., index][mask] - target[..., index][mask]).abs().mean()
            ),
        }
    valid_count = validity.sum(dim=-1).clamp_min(1)
    combined_prediction = (
        prediction * validity.to(prediction.dtype)
    ).sum(dim=-1) / valid_count
    combined_target = (
        target * validity.to(target.dtype)
    ).sum(dim=-1) / valid_count
    flat_target = combined_target.reshape(-1)
    labels = _responsive_top_fraction_labels(combined_target)
    route_metrics = {}
    for route_id, rows in route_rows.items():
        within_rows = within_view_rows.get(route_id, [])
        if not within_rows:
            raise CounterfactualEvidenceError(
                "route has no defined within-view target metric"
            )
        route_prediction = torch.stack(rows["pred"]).reshape(-1)
        route_target = torch.stack(rows["target"]).reshape(-1)
        route_labels = _responsive_top_fraction_labels(route_target)
        route_metrics[route_id] = {
            "target_top20_auroc": _binary_auc(route_prediction, route_labels),
            "patch_spearman": _spearman(route_prediction, route_target),
            "median_within_intervened_view_target_top20_auroc": float(
                torch.tensor(
                    [
                        row["target_top20_auroc"]
                        for row in within_rows
                    ]
                ).median()
            ),
            "median_within_intervened_view_patch_spearman": float(
                torch.tensor(
                    [row["patch_spearman"] for row in within_rows]
                ).median()
            ),
        }
    by_family = {}
    for family, rows in family_rows.items():
        score = torch.tensor(rows["score"])
        family_target = torch.tensor(rows["target"])
        severity = torch.tensor(rows["severity"])
        by_family[family] = {
            "score_mean": float(score.mean()),
            "target_mean": float(family_target.mean()),
            "severity_score_spearman": _spearman(severity, score),
            "severity_target_spearman": _spearman(severity, family_target),
        }
    clean_values = torch.stack(list(reference_predictions.values())).reshape(-1)
    by_family_severity = {
        "%s/severity_%g" % (family, severity): {
            "score_mean": float(torch.tensor(rows["score"]).mean()),
            "target_mean": float(torch.tensor(rows["target"]).mean()),
            "record_count": len(rows["score"]),
        }
        for (family, severity), rows in sorted(family_severity_rows.items())
    }
    return {
        "record_count": len(records),
        "components": components,
        "combined_patch_spearman": _spearman(
            combined_prediction.reshape(-1), combined_target.reshape(-1)
        ),
        "combined_target_top20_auroc": _binary_auc(
            combined_prediction.reshape(-1), labels.reshape(-1)
        ),
        "target_top20_definition": (
            "top 20% of measured intervention-responsive combined targets; "
            "unchanged cells remain negatives"
        ),
        "reference_prediction_p95": float(_exact_quantile_1d(clean_values, 0.95)),
        "reference_prediction_mean": float(clean_values.mean()),
        "by_family": by_family,
        "by_family_severity": by_family_severity,
        "by_route": route_metrics,
        "median_route_target_top20_auroc": float(
            torch.tensor(
                [row["target_top20_auroc"] for row in route_metrics.values()]
            ).median()
        ),
        "median_record_within_intervened_view_target_top20_auroc": float(
            torch.tensor(
                [
                    row["target_top20_auroc"]
                    for rows in within_view_rows.values()
                    for row in rows
                ]
            ).median()
        ),
        "within_intervened_view_evaluated_record_count": sum(
            len(rows) for rows in within_view_rows.values()
        ),
        "within_intervened_view_skipped_record_count": len(
            within_view_skipped_records
        ),
        "within_intervened_view_skipped_records": within_view_skipped_records,
        "median_route_within_intervened_view_target_top20_auroc": float(
            torch.tensor(
                [
                    row["median_within_intervened_view_target_top20_auroc"]
                    for row in route_metrics.values()
                ]
            ).median()
        ),
        "within_intervened_view_definition": (
            "view selected by maximum measured persistent direction-plus-magnitude "
            "target mass; transient response, corruption mask, and intervention "
            "metadata are not used for view selection; records with fewer than "
            "two response-floor-positive cells are reported and excluded only "
            "from the undefined per-record within-view AUROC/Spearman summary"
        ),
        "corruption_mask_read_for_metrics": False,
    }


__all__ = [
    "COUNTERFACTUAL_TRAINING_SCHEMA_VERSION",
    "CounterfactualEvidenceRecord",
    "audit_train_target_distribution",
    "audit_target_spatial_support",
    "evaluate_evidence_records",
    "evaluate_hurdle_diagnostics",
    "fit_train_component_scales",
    "measured_target_ranking_loss",
    "records_from_counterfactual_shard",
    "run_evidence_epoch",
    "run_hurdle_evidence_epoch",
    "select_records",
    "targets_for_records",
]
