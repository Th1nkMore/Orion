"""Clean-calibrated observation-UQ signal diagnostics.

This module does not train an adapter.  It compares inexpensive signals whose
calibration is fitted on clean training routes only.  Corruption family,
severity, and masks are audit-only metadata used after every score map has been
computed.

The paired clean/observed delta is included as a non-deployable diagnostic
upper bound.  It is never eligible as an adapter target.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

from uq_estimator.observation_uq_v3 import (
    ObservationUQError,
    ObservationUQExample,
    _binary_auc,
    _spearman,
)


SIGNAL_AUDIT_SCHEMA_VERSION = "orion.observation-uq-signal-audit/v1.2"


def _require_batched_pair(
    current: torch.Tensor, previous: torch.Tensor, previous_valid: torch.Tensor
) -> None:
    if current.ndim != 5 or previous.shape != current.shape:
        raise ObservationUQError(
            "signal inputs must have matching [B,V,H,W,D] shape"
        )
    if previous_valid.shape != (current.shape[0],):
        raise ObservationUQError("previous_valid must have shape [B]")
    if not current.is_floating_point() or not previous.is_floating_point():
        raise ObservationUQError("signal inputs must be floating point")


def temporal_cosine_residual(
    current: torch.Tensor, previous: torch.Tensor, previous_valid: torch.Tensor
) -> torch.Tensor:
    """Patchwise temporal change without using intervention metadata."""

    _require_batched_pair(current, previous, previous_valid)
    score = 1.0 - F.cosine_similarity(current, previous, dim=-1, eps=1e-6)
    valid = previous_valid.to(device=current.device, dtype=torch.bool)
    return torch.where(
        valid[:, None, None, None], score, torch.zeros_like(score)
    )


def spatial_neighbor_residual(current: torch.Tensor) -> torch.Tensor:
    """Difference from the mean of valid 8-neighbours at each feature patch."""

    if current.ndim != 5 or not current.is_floating_point():
        raise ObservationUQError("current must have floating [B,V,H,W,D] shape")
    batch, views, height, width, feature_dim = current.shape
    normalized = F.normalize(current.float(), dim=-1, eps=1e-6)
    chw = normalized.permute(0, 1, 4, 2, 3).reshape(
        batch * views, feature_dim, height, width
    )
    neighbour_sum = F.avg_pool2d(
        chw, kernel_size=3, stride=1, padding=1, count_include_pad=True
    ) * 9.0 - chw
    ones = torch.ones(
        batch * views, 1, height, width, device=current.device, dtype=chw.dtype
    )
    neighbour_count = F.avg_pool2d(
        ones, kernel_size=3, stride=1, padding=1, count_include_pad=True
    ) * 9.0 - 1.0
    neighbour_mean = neighbour_sum / neighbour_count.clamp_min(1.0)
    score = 1.0 - F.cosine_similarity(chw, neighbour_mean, dim=1, eps=1e-6)
    return score.reshape(batch, views, height, width).to(dtype=current.dtype)


def feature_rms(current: torch.Tensor) -> torch.Tensor:
    """Per-patch feature magnitude; deviations are calibrated on clean routes."""

    if current.ndim != 5 or not current.is_floating_point():
        raise ObservationUQError("current must have floating [B,V,H,W,D] shape")
    return current.float().square().mean(dim=-1).clamp_min(0.0).sqrt()


@dataclass(frozen=True)
class CleanPositionCalibrator:
    """Robust clean center/scale for every view and feature-grid position."""

    center: torch.Tensor
    scale: torch.Tensor
    tail: str
    example_count: int

    def transform(self, score: torch.Tensor) -> torch.Tensor:
        if score.shape != self.center.shape:
            raise ObservationUQError("score map does not match calibrator shape")
        z = (score.float() - self.center) / self.scale
        if self.tail == "positive":
            z = z.clamp_min(0.0)
        elif self.tail == "absolute":
            z = z.abs()
        else:  # pragma: no cover - constructor is internal
            raise ObservationUQError("unsupported calibration tail")
        return z.clamp_max(20.0)


def fit_clean_position_calibrator(
    score_maps: Mapping[str, torch.Tensor],
    clean_examples: Sequence[ObservationUQExample],
    tail: str = "positive",
) -> CleanPositionCalibrator:
    """Fit a robust view/position baseline using clean examples exclusively."""

    if tail not in ("positive", "absolute"):
        raise ObservationUQError("tail must be positive or absolute")
    if not clean_examples:
        raise ObservationUQError("clean calibration requires examples")
    if any(item.family != "clean" for item in clean_examples):
        raise ObservationUQError("clean calibrator received a corruption")
    missing = [item.sample_id for item in clean_examples if item.sample_id not in score_maps]
    if missing:
        raise ObservationUQError("calibration score cache is incomplete")
    values = torch.stack(
        [score_maps[item.sample_id].detach().cpu().float() for item in clean_examples]
    )
    center = values.median(dim=0).values
    mad = (values - center).abs().median(dim=0).values * 1.4826
    flat = mad.flatten(start_dim=1)
    per_view_floor = torch.quantile(flat, 0.10, dim=1).clamp_min(1e-4)
    scale = torch.maximum(mad, per_view_floor[:, None, None])
    return CleanPositionCalibrator(
        center=center,
        scale=scale,
        tail=tail,
        example_count=len(clean_examples),
    )


def apply_calibrator(
    score_maps: Mapping[str, torch.Tensor], calibrator: CleanPositionCalibrator
) -> Dict[str, torch.Tensor]:
    return {
        sample_id: calibrator.transform(score.detach().cpu())
        for sample_id, score in score_maps.items()
    }


def paired_clean_delta_maps(
    payload: Mapping[str, Any],
) -> Dict[str, torch.Tensor]:
    """Build a diagnostic upper bound from paired clean/observed tensors."""

    clean_features = payload.get("clean_features")
    clean_items = payload.get("clean_items")
    observed_features = payload.get("observed_features")
    observed_items = payload.get("observed_items")
    if not all(
        isinstance(value, list)
        for value in (clean_features, clean_items, observed_features, observed_items)
    ):
        raise ObservationUQError("paired delta requires a validated feature shard")
    result = {}
    for item, feature in zip(clean_items, clean_features):
        result[str(item["sample_id"]) + "/clean"] = torch.zeros(
            feature.shape[:-1], dtype=torch.float32
        )
    for item, observed in zip(observed_items, observed_features):
        clean_index = int(item["clean_index"])
        clean = clean_features[clean_index]
        result[str(item["sample_id"])] = (
            1.0
            - F.cosine_similarity(observed.float(), clean.float(), dim=-1, eps=1e-6)
        ).cpu()
    return result


def evaluate_score_maps(
    examples: Sequence[ObservationUQExample],
    score_maps: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    """Use corruption metadata only here, after score computation is complete."""

    if not examples:
        raise ObservationUQError("signal evaluation requires examples")
    family_rows = defaultdict(lambda: {"score": [], "severity": []})
    mask_scores = []
    mask_labels = []
    for item in examples:
        if item.sample_id not in score_maps:
            raise ObservationUQError("evaluation score cache is incomplete")
        score = score_maps[item.sample_id].detach().cpu().float()
        if score.shape != item.current.shape[:-1]:
            raise ObservationUQError("score map has the wrong spatial shape")
        family_rows[item.family]["score"].append(float(score.mean()))
        family_rows[item.family]["severity"].append(float(item.severity))
        if item.family != "clean" and item.corruption_mask is not None:
            mask_scores.append(score.reshape(-1))
            mask_labels.append((item.corruption_mask.reshape(-1) >= 0.5).cpu())
    by_family = {}
    for family, rows in sorted(family_rows.items()):
        score = torch.tensor(rows["score"], dtype=torch.float32)
        severity = torch.tensor(rows["severity"], dtype=torch.float32)
        by_family[family] = {
            "example_count": int(score.numel()),
            "score_mean": float(score.mean()),
            "score_std": float(score.std(unbiased=False)),
            "severity_score_spearman": _spearman(severity, score),
        }
    clean = by_family.get("clean")
    if clean is not None:
        for family, row in by_family.items():
            if family == "clean":
                continue
            uplift = float(row["score_mean"] - clean["score_mean"])
            pooled = math.sqrt(
                0.5 * (row["score_std"] ** 2 + clean["score_std"] ** 2)
            )
            row["score_uplift_over_clean"] = uplift
            row["uplift_standardized_effect"] = uplift / max(pooled, 1e-8)
    auc = float("nan")
    if mask_scores:
        auc = _binary_auc(torch.cat(mask_scores), torch.cat(mask_labels))
    return {
        "example_count": len(examples),
        "corruption_mask_patch_auroc_diagnostic_only": auc,
        "by_family": by_family,
    }


def attach_route_shift_diagnostics(
    clean_train: Mapping[str, Any], evaluation: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare intervention uplift with nuisance clean route shift."""

    train_mean = float(clean_train["by_family"]["clean"]["score_mean"])
    clean_mean = float(evaluation["by_family"]["clean"]["score_mean"])
    route_shift = abs(clean_mean - train_mean)
    evaluation = dict(evaluation)
    evaluation["clean_route_shift_from_train"] = route_shift
    updated = {}
    for family, row in evaluation["by_family"].items():
        row = dict(row)
        if family != "clean":
            uplift = float(row["score_uplift_over_clean"])
            row["uplift_to_clean_route_shift_ratio"] = uplift / max(
                route_shift, 1e-8
            )
        updated[family] = row
    evaluation["by_family"] = updated
    return evaluation


def evaluate_detailed_score_maps(
    examples: Sequence[ObservationUQExample],
    score_maps: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    """Route, severity, temporal-validity, and camera-view diagnostics."""

    if not examples:
        raise ObservationUQError("detailed evaluation requires examples")
    by_route = {}
    routes = sorted({item.route_id for item in examples})
    for route_id in routes:
        route_examples = [item for item in examples if item.route_id == route_id]
        families = {item.family for item in route_examples}
        if "clean" not in families or len(families) < 2:
            raise ObservationUQError("each diagnostic route needs clean and observed data")
        by_route[route_id] = evaluate_score_maps(route_examples, score_maps)

    valid_examples = [item for item in examples if item.previous_valid]
    if not valid_examples:
        raise ObservationUQError("temporal-valid diagnostic subset is empty")
    previous_valid_only = evaluate_score_maps(valid_examples, score_maps)

    severity_rows = defaultdict(
        lambda: {"example": [], "inside": [], "outside": []}
    )
    max_views = max(item.current.shape[0] for item in examples)
    view_scores = [[] for _ in range(max_views)]
    view_labels = [[] for _ in range(max_views)]
    for item in examples:
        if item.family == "clean" or item.corruption_mask is None:
            continue
        score = score_maps[item.sample_id].detach().cpu().float()
        mask = (item.corruption_mask.detach().cpu() >= 0.5)
        key = (item.family, float(item.severity))
        severity_rows[key]["example"].append(float(score.mean()))
        if bool(mask.any()):
            severity_rows[key]["inside"].append(float(score[mask].mean()))
        if bool((~mask).any()):
            severity_rows[key]["outside"].append(float(score[~mask].mean()))
        for view in range(score.shape[0]):
            view_scores[view].append(score[view].reshape(-1))
            view_labels[view].append(mask[view].reshape(-1))

    by_severity = {}
    for (family, severity), rows in sorted(severity_rows.items()):
        example = torch.tensor(rows["example"], dtype=torch.float32)
        inside = torch.tensor(rows["inside"], dtype=torch.float32)
        outside = torch.tensor(rows["outside"], dtype=torch.float32)
        if inside.numel() == 0 or outside.numel() == 0:
            raise ObservationUQError("severity diagnostic has an empty mask region")
        inside_mean = float(inside.mean())
        outside_mean = float(outside.mean())
        by_severity.setdefault(family, {})[str(severity)] = {
            "example_count": int(example.numel()),
            "example_score_mean": float(example.mean()),
            "mask_inside_mean": inside_mean,
            "mask_outside_mean": outside_mean,
            "inside_minus_outside": inside_mean - outside_mean,
        }

    by_view = {}
    for view in range(max_views):
        if not view_scores[view]:
            continue
        scores = torch.cat(view_scores[view])
        labels = torch.cat(view_labels[view]).bool()
        by_view[str(view)] = {
            "patch_count": int(scores.numel()),
            "positive_patch_count": int(labels.sum()),
            "mask_auroc": _binary_auc(scores, labels),
        }
    return {
        "by_route": by_route,
        "previous_valid_only": previous_valid_only,
        "by_severity": by_severity,
        "by_view": by_view,
    }
