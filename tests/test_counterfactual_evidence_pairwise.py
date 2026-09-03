import torch

from uq_estimator.counterfactual_evidence import CounterfactualEvidenceTarget
from uq_estimator.counterfactual_evidence_pairwise import (
    pairwise_evidence_delta_loss,
    records_from_native_weather_payload,
    run_pairwise_hurdle_epoch,
)
from uq_estimator.counterfactual_evidence import ObservationEvidenceHurdleAdapter
from uq_estimator.counterfactual_evidence_training import fit_train_component_scales
from uq_estimator.native_weather_audit import (
    CONDITION_SEVERITY,
    EXPECTED_CONDITIONS,
    NATIVE_WEATHER_FEATURE_SCHEMA_VERSION,
)


def _native_payload():
    items = []
    for route_id in ("Town03/Route195", "Town10HD/Route148"):
        for sequence_index in range(3):
            items.append(
                {
                    "sample_id": "%s/%02d" % (route_id, sequence_index),
                    "route_id": route_id,
                    "sequence_index": sequence_index,
                }
            )
    generator = torch.Generator().manual_seed(12)
    clear = torch.randn(6, 2, 3, 3, 8, generator=generator)
    features = {
        "clear": clear,
        "fog_light": clear + 0.08,
        "fog_heavy": clear + 0.22,
    }
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


def test_native_payload_conversion_preserves_route_temporal_pairs():
    records = records_from_native_weather_payload(_native_payload())
    assert len(records) == 12
    assert {record.family for record in records} == {"native_fog"}
    assert {record.severity for record in records} == {1.0, 3.0}
    assert all(record.corruption_mask is None for record in records)
    first = [record for record in records if record.route_id == "Town03/Route195"]
    assert [record.previous_valid for record in first] == [False, False, True, True, True, True]
    assert first[0].pair_id == first[1].pair_id
    assert torch.equal(first[2].reference_previous, first[0].reference_current)
    assert torch.equal(first[2].observed_previous, first[0].observed_current)


def test_pairwise_loss_is_offset_invariant_and_rejects_reversed_delta():
    target_values = torch.tensor([[[[[0.0, 0.5, 1.0]]]]])
    target = CounterfactualEvidenceTarget(
        values=target_values,
        component_valid=torch.ones_like(target_values, dtype=torch.bool),
    )
    reference = torch.full_like(target_values, 0.7)
    observed = reference + target_values
    matched = pairwise_evidence_delta_loss(observed, reference, target)
    shifted = pairwise_evidence_delta_loss(observed + 5.0, reference + 5.0, target)
    reversed_loss = pairwise_evidence_delta_loss(reference, observed, target)
    assert torch.allclose(matched, shifted, atol=1e-7)
    assert float(matched) < 1e-8
    assert float(reversed_loss) > float(matched) + 0.05


def test_pairwise_epoch_has_no_reference_zero_loss_term():
    records = records_from_native_weather_payload(_native_payload())
    device = torch.device("cpu")
    scales = fit_train_component_scales(records, device, batch_size=2)
    model = ObservationEvidenceHurdleAdapter(
        8, hidden_dim=8, max_views=2, use_view_embedding=False
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = run_pairwise_hurdle_epoch(
        model,
        records,
        scales,
        device,
        pair_batch_size=1,
        optimizer=optimizer,
    )
    assert set(metrics) == {"total", "delta", "ranking"}
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
