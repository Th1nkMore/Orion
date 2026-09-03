"""Clean-only appearance candidates for paired native-weather diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

from uq_estimator.native_weather_audit import (
    CONDITION_SEVERITY,
    EXPECTED_CONDITIONS,
    _oracle_top_fraction_labels,
    validate_native_weather_payload,
)
from uq_estimator.observation_uq_signal_audit import spatial_neighbor_residual
from uq_estimator.observation_uq_v3 import ObservationUQError, _binary_auc, _spearman


NATIVE_APPEARANCE_AUDIT_SCHEMA_VERSION = "orion.native-appearance-audit/v1"
CANDIDATE_TAILS = {
    "feature_rms_viewpos_abs_z": "absolute",
    "spatial_neighbor_viewpos_abs_z": "absolute",
    "appearance_prototype_viewpos_z": "positive",
    "appearance_diagonal_viewpos_z": "positive",
}


@dataclass(frozen=True)
class CleanAppearanceStatistics:
    mean: torch.Tensor
    scale: torch.Tensor
    prototype: torch.Tensor
    example_count: int


def fit_clean_appearance_statistics(
    clean_features: Sequence[torch.Tensor],
    device="cpu",
) -> CleanAppearanceStatistics:
    """Fit view/position/channel moments using clean observations only."""

    if not clean_features:
        raise ObservationUQError("appearance statistics require clean features")
    shape = tuple(clean_features[0].shape)
    if len(shape) != 4:
        raise ObservationUQError("clean features must have [V,H,W,D] shape")
    target = torch.device(device)
    total = torch.zeros(shape, dtype=torch.float32, device=target)
    total_square = torch.zeros(shape, dtype=torch.float32, device=target)
    for feature in clean_features:
        if tuple(feature.shape) != shape or not feature.is_floating_point():
            raise ObservationUQError("clean appearance feature shapes disagree")
        value = feature.detach().to(device=target, dtype=torch.float32)
        total.add_(value)
        total_square.addcmul_(value, value)
    count = len(clean_features)
    mean = total / float(count)
    variance = (total_square / float(count) - mean.square()).clamp_min(0.0)
    scale = variance.sqrt()
    view_channel_floor = torch.quantile(
        scale.permute(0, 3, 1, 2).reshape(shape[0], shape[-1], -1),
        0.10,
        dim=-1,
    ).clamp_min(1e-3)
    scale = torch.maximum(
        scale,
        view_channel_floor[:, None, None, :],
    )
    prototype = F.normalize(mean, dim=-1, eps=1e-6)
    return CleanAppearanceStatistics(
        mean=mean,
        scale=scale,
        prototype=prototype,
        example_count=count,
    )


def appearance_candidate_raw_maps(
    current: torch.Tensor, statistics: CleanAppearanceStatistics
) -> Dict[str, torch.Tensor]:
    """Compute deployable scores from current features and clean statistics."""

    if current.ndim != 5 or not current.is_floating_point():
        raise ObservationUQError("current must have floating [B,V,H,W,D] shape")
    if tuple(current.shape[1:]) != tuple(statistics.mean.shape):
        raise ObservationUQError("current feature shape does not match clean statistics")
    value = current.float()
    rms = value.square().mean(dim=-1).clamp_min(0.0).sqrt()
    spatial = spatial_neighbor_residual(value).float()
    prototype = 1.0 - F.cosine_similarity(
        value, statistics.prototype[None], dim=-1, eps=1e-6
    )
    standardized = (value - statistics.mean[None]) / statistics.scale[None]
    diagonal = standardized.square().clamp_max(100.0).mean(dim=-1).sqrt()
    return {
        "feature_rms_viewpos_abs_z": rms,
        "spatial_neighbor_viewpos_abs_z": spatial,
        "appearance_prototype_viewpos_z": prototype,
        "appearance_diagonal_viewpos_z": diagonal,
    }


def audit_native_appearance_score_maps(
    payload: Mapping[str, Any],
    scores_by_candidate: Mapping[str, Mapping[str, torch.Tensor]],
    candidate_tails: Mapping[str, str] = None,
) -> Dict[str, Any]:
    """Apply the frozen native gate; paired clean is evaluation-only."""

    validate_native_weather_payload(payload)
    tails = dict(CANDIDATE_TAILS if candidate_tails is None else candidate_tails)
    if not tails or any(value not in ("positive", "absolute") for value in tails.values()):
        raise ObservationUQError("candidate tails must be positive or absolute")
    items = payload["items"]
    features = {
        name: payload["features_by_condition"][name].detach().cpu().float()
        for name in EXPECTED_CONDITIONS
    }
    clean = features["clear"]
    candidate_reports = {}
    for candidate, condition_scores in scores_by_candidate.items():
        if candidate not in tails:
            raise ObservationUQError("unregistered native appearance candidate")
        if tuple(sorted(condition_scores)) != tuple(sorted(EXPECTED_CONDITIONS)):
            raise ObservationUQError("appearance candidate conditions are incomplete")
        scores = {}
        for condition in EXPECTED_CONDITIONS:
            score = condition_scores[condition].detach().cpu().float()
            if tuple(score.shape) != tuple(features[condition].shape[:-1]):
                raise ObservationUQError("appearance score shape is invalid")
            scores[condition] = score

        condition_metrics = {}
        severity_scores = []
        severity_labels = []
        for condition in EXPECTED_CONDITIONS:
            example_mean = scores[condition].flatten(start_dim=1).mean(dim=1)
            row = {
                "severity": CONDITION_SEVERITY[condition],
                "example_count": int(example_mean.numel()),
                "score_mean": float(example_mean.mean()),
                "score_std": float(example_mean.std(unbiased=False)),
            }
            if condition != "clear":
                oracle = 1.0 - F.cosine_similarity(
                    features[condition], clean, dim=-1, eps=1e-6
                )
                labels = _oracle_top_fraction_labels(oracle)
                row.update(
                    {
                        "paired_clean_delta_mean": float(oracle.mean()),
                        "score_to_paired_delta_patch_spearman": _spearman(
                            scores[condition].reshape(-1), oracle.reshape(-1)
                        ),
                        "paired_delta_top20_patch_auroc": _binary_auc(
                            scores[condition].reshape(-1), labels.reshape(-1)
                        ),
                        "score_uplift_over_clear": float(
                            example_mean.mean()
                            - scores["clear"].flatten(start_dim=1).mean(dim=1).mean()
                        ),
                    }
                )
                severity_scores.append(example_mean)
                severity_labels.append(
                    torch.full_like(example_mean, CONDITION_SEVERITY[condition])
                )
            condition_metrics[condition] = row
        severity_rho = _spearman(torch.cat(severity_labels), torch.cat(severity_scores))

        by_route = {}
        for route_id in sorted({str(item["route_id"]) for item in items}):
            route_mask = torch.tensor(
                [str(item["route_id"]) == route_id for item in items],
                dtype=torch.bool,
            )
            by_route[route_id] = {
                condition: float(scores[condition][route_mask].mean())
                for condition in EXPECTED_CONDITIONS
            }

        light = condition_metrics["fog_light"]
        heavy = condition_metrics["fog_heavy"]
        checks = [
            {
                "metric": "fog_light_positive_uplift",
                "value": light["score_uplift_over_clear"],
                "threshold": 0.0,
                "passed": light["score_uplift_over_clear"] > 0.0,
            },
            {
                "metric": "fog_heavy_higher_than_light",
                "value": heavy["score_mean"] - light["score_mean"],
                "threshold": 0.0,
                "passed": heavy["score_mean"] > light["score_mean"],
            },
            {
                "metric": "sample_severity_spearman",
                "value": severity_rho,
                "threshold": 0.10,
                "passed": severity_rho >= 0.10,
            },
        ]
        for condition in ("fog_light", "fog_heavy"):
            row = condition_metrics[condition]
            for metric, threshold in (
                ("score_to_paired_delta_patch_spearman", 0.10),
                ("paired_delta_top20_patch_auroc", 0.60),
            ):
                value = float(row[metric])
                checks.append(
                    {
                        "condition": condition,
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "passed": value >= threshold,
                    }
                )
        checks.append(
            {
                "metric": "paired_delta_heavy_higher_than_light",
                "value": heavy["paired_clean_delta_mean"] - light["paired_clean_delta_mean"],
                "threshold": 0.0,
                "passed": heavy["paired_clean_delta_mean"] > light["paired_clean_delta_mean"] > 0.0,
            }
        )
        for route_id, row in by_route.items():
            checks.append(
                {
                    "route_id": route_id,
                    "metric": "route_heavy_higher_than_light_higher_than_clear",
                    "value": row["fog_heavy"] - row["clear"],
                    "threshold": 0.0,
                    "passed": row["fog_heavy"] > row["fog_light"] > row["clear"],
                }
            )
        candidate_reports[candidate] = {
            "tail": tails[candidate],
            "condition_metrics": condition_metrics,
            "sample_severity_spearman": severity_rho,
            "by_route_score_mean": by_route,
            "candidate_gate": {
                "passed": all(row["passed"] for row in checks),
                "checks": checks,
            },
        }

    return {
        "schema_version": NATIVE_APPEARANCE_AUDIT_SCHEMA_VERSION,
        "candidates": candidate_reports,
        "candidate_passes": {
            name: row["candidate_gate"]["passed"]
            for name, row in candidate_reports.items()
        },
        "data_attestation": {
            "candidate_training": False,
            "corruption_metadata_used_for_candidate": False,
            "paired_clean_used_for_candidate_or_calibration": False,
            "paired_clean_used_for_evaluation_only": True,
            "adapter_trained": False,
            "actual_target_read": False,
            "stage_b_authorized": False,
        },
    }


__all__ = [
    "CANDIDATE_TAILS",
    "CleanAppearanceStatistics",
    "NATIVE_APPEARANCE_AUDIT_SCHEMA_VERSION",
    "appearance_candidate_raw_maps",
    "audit_native_appearance_score_maps",
    "fit_clean_appearance_statistics",
]
