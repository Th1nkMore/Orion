import copy

import pytest
import torch

from uq_estimator.native_weather_audit import (
    CONDITION_SEVERITY,
    EXPECTED_CONDITIONS,
    NATIVE_WEATHER_FEATURE_SCHEMA_VERSION,
    audit_native_weather_features,
    validate_native_weather_payload,
)
from uq_estimator.observation_uq_signal_audit import CleanPositionCalibrator
from uq_estimator.observation_uq_v3 import ObservationUQError


def _feature(angle):
    return torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)


def _payload():
    items = []
    for route_id in ("Town01/Route146", "Town04/Route203"):
        for sequence_index in range(3):
            items.append(
                {
                    "sample_id": "%s/%02d" % (route_id, sequence_index),
                    "route_id": route_id,
                    "sequence_index": sequence_index,
                }
            )
    patch_angle = torch.tensor([0.03, 0.08, 0.14, 0.22, 0.34])
    features = {}
    for condition, multiplier in (
        ("clear", 0.0),
        ("fog_light", 1.0),
        ("fog_heavy", 2.2),
    ):
        rows = []
        for _route in range(2):
            for sequence_index in range(3):
                angle = patch_angle * multiplier * sequence_index
                rows.append(_feature(angle).reshape(1, 1, 5, 2))
        features[condition] = torch.stack(rows)
    return {
        "schema_version": NATIVE_WEATHER_FEATURE_SCHEMA_VERSION,
        "items": items,
        "features_by_condition": features,
        "conditions": {
            condition: {
                "severity": CONDITION_SEVERITY[condition],
                "renderer": "CARLA-0.9.15-native-weather",
            }
            for condition in EXPECTED_CONDITIONS
        },
        "pixel_corruption_generator_used": False,
        "paired_world_pose": True,
        "renderer_quality": "Epic",
    }


def test_native_weather_audit_passes_for_monotonic_localized_temporal_signal():
    payload = _payload()
    calibrator = CleanPositionCalibrator(
        center=torch.zeros(1, 1, 5),
        scale=torch.ones(1, 1, 5),
        tail="positive",
        example_count=560,
    )

    report = audit_native_weather_features(payload, calibrator)

    assert report["candidate_gate"]["passed"] is True
    assert report["condition_metrics"]["fog_heavy"]["score_mean"] > report[
        "condition_metrics"
    ]["fog_light"]["score_mean"]
    assert report["data_attestation"]["paired_clean_delta_used_for_score_or_calibration"] is False


def test_native_weather_payload_rejects_unpaired_or_generator_based_inputs():
    payload = _payload()
    payload["pixel_corruption_generator_used"] = True
    with pytest.raises(ObservationUQError, match="prohibits pixel corruption"):
        validate_native_weather_payload(payload)

    payload = _payload()
    payload["features_by_condition"]["fog_heavy"] = payload[
        "features_by_condition"
    ]["fog_heavy"][:-1]
    with pytest.raises(ObservationUQError, match="counts disagree"):
        validate_native_weather_payload(payload)

    payload = _payload()
    payload["renderer_quality"] = "Low"
    with pytest.raises(ObservationUQError, match="requires Epic"):
        validate_native_weather_payload(payload)


def test_native_weather_payload_requires_two_routes_and_contiguous_sequences():
    payload = _payload()
    payload["items"] = copy.deepcopy(payload["items"][:3])
    for condition in EXPECTED_CONDITIONS:
        payload["features_by_condition"][condition] = payload[
            "features_by_condition"
        ][condition][:3]
    with pytest.raises(ObservationUQError, match="at least two routes"):
        validate_native_weather_payload(payload)
