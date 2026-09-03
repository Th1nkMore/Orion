"""Audit a frozen observation-UQ score on paired native CARLA weather.

The native weather renderer is an independent intervention source: no pixel
corruption mask, family label, or paired-clean feature may be consumed by the
deployable score.  Paired clean features are used only after score computation
as a diagnostic localization reference.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

from uq_estimator.observation_uq_signal_audit import CleanPositionCalibrator
from uq_estimator.observation_uq_v3 import (
    ObservationUQError,
    _binary_auc,
    _spearman,
)


NATIVE_WEATHER_FEATURE_SCHEMA_VERSION = "orion.native-weather-features/v1"
NATIVE_WEATHER_AUDIT_SCHEMA_VERSION = "orion.native-weather-uq-audit/v1"
EXPECTED_CONDITIONS = ("clear", "fog_light", "fog_heavy")
CONDITION_SEVERITY = {"clear": 0.0, "fog_light": 1.0, "fog_heavy": 3.0}


def validate_native_weather_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed on mismatched pairing or undocumented renderer inputs."""

    if payload.get("schema_version") != NATIVE_WEATHER_FEATURE_SCHEMA_VERSION:
        raise ObservationUQError("unexpected native weather feature schema")
    items = payload.get("items")
    features = payload.get("features_by_condition")
    conditions = payload.get("conditions")
    if not isinstance(items, list) or not items:
        raise ObservationUQError("native weather payload needs ordered items")
    if not isinstance(features, Mapping) or tuple(sorted(features)) != tuple(
        sorted(EXPECTED_CONDITIONS)
    ):
        raise ObservationUQError("native weather conditions are incomplete")
    if not isinstance(conditions, Mapping):
        raise ObservationUQError("native weather condition metadata is missing")
    sample_ids = [str(item.get("sample_id", "")) for item in items]
    if any(not value for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ObservationUQError("native sample IDs must be non-empty and unique")
    route_positions = defaultdict(list)
    for index, item in enumerate(items):
        route_id = str(item.get("route_id", ""))
        sequence_index = item.get("sequence_index")
        if not route_id or not isinstance(sequence_index, int):
            raise ObservationUQError("native items need route and sequence identity")
        route_positions[route_id].append((sequence_index, index))
    if len(route_positions) < 2:
        raise ObservationUQError("native audit requires at least two routes")
    for route_id, positions in route_positions.items():
        ordered = sorted(sequence_index for sequence_index, _ in positions)
        if ordered != list(range(len(ordered))):
            raise ObservationUQError(
                "route %s sequence indices are not contiguous" % route_id
            )
    expected_shape = None
    for condition in EXPECTED_CONDITIONS:
        tensor = features[condition]
        if not torch.is_tensor(tensor) or tensor.ndim != 5:
            raise ObservationUQError(
                "native features must have [N,V,H,W,D] shape"
            )
        if tensor.shape[0] != len(items):
            raise ObservationUQError("native feature/item counts disagree")
        if not tensor.is_floating_point():
            raise ObservationUQError("native features must be floating point")
        if expected_shape is None:
            expected_shape = tuple(tensor.shape)
        elif tuple(tensor.shape) != expected_shape:
            raise ObservationUQError("native condition feature shapes disagree")
        metadata = conditions.get(condition)
        if not isinstance(metadata, Mapping):
            raise ObservationUQError("native condition metadata is malformed")
        if float(metadata.get("severity", -1.0)) != CONDITION_SEVERITY[condition]:
            raise ObservationUQError("native condition severity is not frozen")
        if metadata.get("renderer") != "CARLA-0.9.15-native-weather":
            raise ObservationUQError("native weather renderer attestation is missing")
    if payload.get("pixel_corruption_generator_used") is not False:
        raise ObservationUQError("native audit prohibits pixel corruption generators")
    if payload.get("paired_world_pose") is not True:
        raise ObservationUQError("native weather observations must share world poses")
    if payload.get("renderer_quality") != "Epic":
        raise ObservationUQError("native weather audit requires Epic rendering")


def _temporal_raw_for_condition(
    features: torch.Tensor, items: Sequence[Mapping[str, Any]]
) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 5:
        raise ObservationUQError("condition features must have [N,V,H,W,D] shape")
    previous = torch.empty_like(features)
    previous_valid = torch.zeros(features.shape[0], dtype=torch.bool)
    last_by_route: Dict[str, int] = {}
    for index, item in enumerate(items):
        route_id = str(item["route_id"])
        prior = last_by_route.get(route_id)
        if prior is None:
            previous[index] = features[index]
        else:
            expected = int(items[prior]["sequence_index"]) + 1
            if int(item["sequence_index"]) != expected:
                raise ObservationUQError("native route order is not contiguous")
            previous[index] = features[prior]
            previous_valid[index] = True
        last_by_route[route_id] = index
    raw = 1.0 - F.cosine_similarity(
        features.float(), previous.float(), dim=-1, eps=1e-6
    )
    raw = torch.where(
        previous_valid[:, None, None, None], raw, torch.zeros_like(raw)
    )
    return raw, previous_valid


def _calibrate_batch(
    raw: torch.Tensor, calibrator: CleanPositionCalibrator
) -> torch.Tensor:
    return torch.stack([calibrator.transform(item.cpu()) for item in raw])


def _oracle_top_fraction_labels(
    oracle: torch.Tensor, fraction: float = 0.20
) -> torch.Tensor:
    if not 0.0 < fraction < 1.0:
        raise ObservationUQError("oracle top fraction must lie in (0,1)")
    flat = oracle.flatten(start_dim=1)
    threshold = torch.quantile(flat, 1.0 - fraction, dim=1, keepdim=True)
    return (flat >= threshold).reshape_as(oracle)


def audit_native_weather_features(
    payload: Mapping[str, Any], calibrator: CleanPositionCalibrator
) -> Dict[str, Any]:
    """Evaluate temporal UQ without leaking paired-clean data into the score."""

    validate_native_weather_payload(payload)
    items = payload["items"]
    features = {
        name: payload["features_by_condition"][name].detach().cpu()
        for name in EXPECTED_CONDITIONS
    }
    raw = {}
    scores = {}
    previous_valid = None
    for condition in EXPECTED_CONDITIONS:
        raw[condition], valid = _temporal_raw_for_condition(features[condition], items)
        scores[condition] = _calibrate_batch(raw[condition], calibrator)
        if previous_valid is None:
            previous_valid = valid
        elif not torch.equal(previous_valid, valid):
            raise ObservationUQError("native conditions have inconsistent pairing")
    assert previous_valid is not None
    if int(previous_valid.sum()) == 0:
        raise ObservationUQError("native audit has no temporal-valid frames")

    clean = features["clear"].float()
    condition_metrics = {}
    observed_score_means = []
    observed_severities = []
    valid = previous_valid
    for condition in EXPECTED_CONDITIONS:
        condition_score = scores[condition][valid]
        example_mean = condition_score.flatten(start_dim=1).mean(dim=1)
        row = {
            "severity": CONDITION_SEVERITY[condition],
            "example_count": int(example_mean.numel()),
            "score_mean": float(example_mean.mean()),
            "score_std": float(example_mean.std(unbiased=False)),
        }
        if condition != "clear":
            oracle = 1.0 - F.cosine_similarity(
                features[condition].float(), clean, dim=-1, eps=1e-6
            )
            oracle_valid = oracle[valid]
            labels = _oracle_top_fraction_labels(oracle_valid)
            row.update(
                {
                    "paired_clean_delta_mean": float(oracle_valid.mean()),
                    "score_to_paired_delta_patch_spearman": _spearman(
                        condition_score.reshape(-1), oracle_valid.reshape(-1)
                    ),
                    "paired_delta_top20_patch_auroc": _binary_auc(
                        condition_score.reshape(-1), labels.reshape(-1)
                    ),
                    "score_uplift_over_clear": float(
                        example_mean.mean()
                        - scores["clear"][valid].flatten(start_dim=1).mean(dim=1).mean()
                    ),
                }
            )
            observed_score_means.append(example_mean)
            observed_severities.append(
                torch.full_like(example_mean, CONDITION_SEVERITY[condition])
            )
        condition_metrics[condition] = row

    severity_rho = _spearman(
        torch.cat(observed_severities), torch.cat(observed_score_means)
    )
    by_route = {}
    for route_id in sorted({str(item["route_id"]) for item in items}):
        route_mask = torch.tensor(
            [str(item["route_id"]) == route_id for item in items], dtype=torch.bool
        ) & valid
        if not bool(route_mask.any()):
            raise ObservationUQError("native route has no temporal-valid frames")
        by_route[route_id] = {
            name: float(scores[name][route_mask].mean())
            for name in EXPECTED_CONDITIONS
        }

    checks = []
    light = condition_metrics["fog_light"]
    heavy = condition_metrics["fog_heavy"]
    checks.extend(
        [
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
    )
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
            "value": (
                heavy["paired_clean_delta_mean"]
                - light["paired_clean_delta_mean"]
            ),
            "threshold": 0.0,
            "passed": (
                heavy["paired_clean_delta_mean"]
                > light["paired_clean_delta_mean"]
                > 0.0
            ),
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

    return {
        "schema_version": NATIVE_WEATHER_AUDIT_SCHEMA_VERSION,
        "score": "temporal_viewpos_z",
        "condition_metrics": condition_metrics,
        "sample_severity_spearman": severity_rho,
        "by_route_score_mean": by_route,
        "candidate_gate": {"passed": all(row["passed"] for row in checks), "checks": checks},
        "data_attestation": {
            "pixel_corruption_generator_used": False,
            "paired_clean_delta_used_for_score_or_calibration": False,
            "paired_clean_delta_used_for_evaluation_only": True,
            "adapter_trained": False,
            "actual_target_read": False,
            "stage_b_authorized": False,
        },
    }


__all__ = [
    "CONDITION_SEVERITY",
    "EXPECTED_CONDITIONS",
    "NATIVE_WEATHER_AUDIT_SCHEMA_VERSION",
    "NATIVE_WEATHER_FEATURE_SCHEMA_VERSION",
    "audit_native_weather_features",
    "validate_native_weather_payload",
]
