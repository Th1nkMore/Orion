import torch

from uq_estimator.native_appearance_audit import (
    CANDIDATE_TAILS,
    appearance_candidate_raw_maps,
    audit_native_appearance_score_maps,
    fit_clean_appearance_statistics,
)
from uq_estimator.native_weather_audit import (
    CONDITION_SEVERITY,
    EXPECTED_CONDITIONS,
    NATIVE_WEATHER_FEATURE_SCHEMA_VERSION,
)


def test_clean_appearance_statistics_and_candidates_are_current_only():
    clean = [torch.randn(2, 3, 3, 8) * 0.05 + index for index in (0.0, 0.1, -0.1)]
    stats = fit_clean_appearance_statistics(clean)
    current = torch.stack((clean[0], clean[1] + 0.8))
    result = appearance_candidate_raw_maps(current, stats)
    assert set(result) == set(CANDIDATE_TAILS)
    assert all(value.shape == (2, 2, 3, 3) for value in result.values())
    assert result["appearance_diagonal_viewpos_z"][1].mean() > result[
        "appearance_diagonal_viewpos_z"
    ][0].mean()


def test_native_appearance_gate_accepts_monotonic_localized_scores():
    items = []
    for route_id in ("Town01/Route146", "Town04/Route203"):
        for sequence_index in range(3):
            items.append(
                {
                    "sample_id": "%s/%d" % (route_id, sequence_index),
                    "route_id": route_id,
                    "sequence_index": sequence_index,
                }
            )
    base = torch.zeros(len(items), 1, 1, 5, 2)
    base[..., 0] = 1.0
    features = {"clear": base.clone()}
    scores = {name: {} for name in CANDIDATE_TAILS}
    patch_strength = torch.tensor([0.02, 0.04, 0.10, 0.22, 0.45])
    for condition, multiplier in (("fog_light", 1.0), ("fog_heavy", 2.0)):
        angle = patch_strength * multiplier
        value = base.clone()
        value[..., 0] = torch.cos(angle)
        value[..., 1] = torch.sin(angle)
        features[condition] = value
    for name in CANDIDATE_TAILS:
        scores[name]["clear"] = torch.zeros(len(items), 1, 1, 5)
        scores[name]["fog_light"] = patch_strength[None, None, None].repeat(len(items), 1, 1, 1)
        scores[name]["fog_heavy"] = (patch_strength * 2.0)[None, None, None].repeat(len(items), 1, 1, 1)
    payload = {
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
    report = audit_native_appearance_score_maps(payload, scores)
    assert all(report["candidate_passes"].values())
    assert report["data_attestation"]["candidate_training"] is False
