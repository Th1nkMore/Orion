import dataclasses

import torch

import uq_estimator.counterfactual_evidence_training as evidence_training

from uq_estimator.counterfactual_evidence import (
    ObservationEvidenceAdapter,
    ObservationEvidenceHurdleAdapter,
)
from uq_estimator.counterfactual_compaction import deterministic_rademacher_projection
from uq_estimator.counterfactual_evidence_training import (
    _exact_quantiles_1d,
    audit_target_spatial_support,
    audit_train_target_distribution,
    evaluate_evidence_records,
    evaluate_hurdle_diagnostics,
    fit_train_component_scales,
    records_from_counterfactual_shard,
    run_evidence_epoch,
    run_hurdle_evidence_epoch,
    select_records,
    targets_for_records,
)
from uq_estimator.observation_uq_shard import FEATURE_SHARD_SCHEMA_VERSION


def _mock_shard(cycle_views=False):
    clean_features = []
    clean_items = []
    observed_features = []
    observed_items = []
    generator = torch.Generator().manual_seed(7)
    for route_index, split in ((0, "train"), (1, "validation")):
        route = "route_%d" % route_index
        for frame in range(2):
            clean_index = len(clean_features)
            clean = torch.randn(2, 3, 3, 8, generator=generator) + frame * 0.05
            clean_features.append(clean)
            clean_items.append(
                {
                    "clean_index": clean_index,
                    "sample_id": "%s__frame_%06d" % (route, frame),
                    "route_id": route,
                    "town": "Town01",
                    "frame_idx": frame,
                    "split": split,
                }
            )
            for family in ("local_blur", "local_dark"):
                for severity in (1, 3):
                    observed_index = len(observed_features)
                    view_index = frame % 2 if cycle_views else 0
                    mask = torch.zeros(2, 3, 3)
                    mask[view_index, :2, :2] = 1.0
                    perturbation = torch.zeros_like(clean)
                    perturbation[view_index, :2, :2] = severity * (
                        0.08 if family == "local_blur" else -0.06
                    )
                    observed_features.append(clean + perturbation)
                    observed_items.append(
                        {
                            "observed_index": observed_index,
                            "clean_index": clean_index,
                            "sample_id": "%s/%s/severity_%d"
                            % (clean_items[-1]["sample_id"], family, severity),
                            "route_id": route,
                            "town": "Town01",
                            "frame_idx": frame,
                            "split": split,
                            "family": family,
                            "severity": float(severity),
                            "corruption_mask": mask,
                        }
                    )
    return {
        "schema_version": FEATURE_SHARD_SCHEMA_VERSION,
        "clean_features": clean_features,
        "clean_items": clean_items,
        "observed_features": observed_features,
        "observed_items": observed_items,
        "provenance": {
            "extraction_schema_version": (
                "orion.counterfactual-evidence-extraction/v2"
                if cycle_views
                else "orion.counterfactual-evidence-extraction/v1"
            ),
            "corruption_mask_is_primary_target": False,
        },
    }


def test_exact_quantiles_match_torch_on_small_tensor():
    values = torch.tensor([9.0, 1.0, 4.0, 7.0, 2.0, 3.0])
    levels = [0.0, 0.5, 0.8, 0.95, 1.0]
    actual = torch.stack(_exact_quantiles_1d(values, levels))
    expected = torch.quantile(values, torch.tensor(levels))
    assert torch.allclose(actual, expected)


def test_counterfactual_records_and_bounded_epoch_use_paired_targets():
    records = records_from_counterfactual_shard(_mock_shard())
    train = select_records(records, ["train"], ["local_blur", "local_dark"])
    validation = select_records(
        records, ["validation"], ["local_blur", "local_dark"]
    )
    device = torch.device("cpu")
    scales = fit_train_component_scales(train, device, batch_size=2)
    assert scales.shape == (3,)
    assert (scales > 0).all()
    model = ObservationEvidenceAdapter(8, hidden_dim=12, max_views=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_metrics = run_evidence_epoch(
        model,
        train,
        scales,
        device,
        pair_batch_size=1,
        optimizer=optimizer,
    )
    assert set(train_metrics) == {"total", "observed", "ranking", "reference"}
    report = evaluate_evidence_records(
        model, validation, scales, device, batch_size=2
    )
    assert report["record_count"] == 8
    assert set(report["components"]) == {
        "persistent_direction",
        "persistent_magnitude",
        "transient_inconsistency",
    }
    assert report["corruption_mask_read_for_metrics"] is False
    assert 0.0 <= report[
        "median_record_within_intervened_view_target_top20_auroc"
    ] <= 1.0


def test_stored_targets_skip_large_feature_collation(monkeypatch):
    record = records_from_counterfactual_shard(_mock_shard())[0]
    measured = targets_for_records([record], torch.device("cpu"))
    stored = dataclasses.replace(
        record,
        stored_target_values=measured.values[0],
        stored_target_component_valid=measured.component_valid[0],
    )

    def forbidden_collation(*args, **kwargs):
        raise AssertionError("stored-target audit must not collate FP16 feature grids")

    monkeypatch.setattr(
        evidence_training, "_collate_record_features", forbidden_collation
    )
    loaded = targets_for_records([stored], torch.device("cpu"))
    assert torch.equal(loaded.values, measured.values)
    assert torch.equal(loaded.component_valid, measured.component_valid)


def test_evaluation_handles_window_cycle_transient_response_in_two_views():
    records = records_from_counterfactual_shard(_mock_shard(cycle_views=True))
    train = select_records(records, ["train"], ["local_blur", "local_dark"])
    validation = select_records(
        records, ["validation"], ["local_blur", "local_dark"]
    )
    device = torch.device("cpu")
    scales = fit_train_component_scales(train, device, batch_size=2)
    model = ObservationEvidenceAdapter(8, hidden_dim=12, max_views=2)
    report = evaluate_evidence_records(
        model, validation, scales, device, batch_size=2
    )
    assert report["record_count"] == len(validation)
    assert report["corruption_mask_read_for_metrics"] is False
    assert report["within_intervened_view_definition"].startswith(
        "view selected by maximum measured persistent"
    )


def test_evaluation_reports_and_skips_undefined_sparse_within_view_record():
    records = records_from_counterfactual_shard(_mock_shard(cycle_views=True))
    train = select_records(records, ["train"], ["local_blur", "local_dark"])
    validation = select_records(
        records, ["validation"], ["local_blur", "local_dark"]
    )
    device = torch.device("cpu")
    scales = fit_train_component_scales(train, device, batch_size=2)
    measured = targets_for_records(validation, device)
    stored = [
        dataclasses.replace(
            record,
            stored_target_values=measured.values[index],
            stored_target_component_valid=measured.component_valid[index],
        )
        for index, record in enumerate(validation)
    ]
    sparse = torch.zeros_like(stored[0].stored_target_values)
    sparse[0, 0, :5, 0] = scales[0] * 5e-7
    validity = torch.ones_like(
        stored[0].stored_target_component_valid, dtype=torch.bool
    )
    stored[0] = dataclasses.replace(
        stored[0],
        stored_target_values=sparse,
        stored_target_component_valid=validity,
    )
    model = ObservationEvidenceAdapter(8, hidden_dim=12, max_views=2)
    report = evaluate_evidence_records(
        model, stored, scales, device, batch_size=2
    )
    assert report["within_intervened_view_skipped_record_count"] == 1
    assert report["within_intervened_view_evaluated_record_count"] == len(stored) - 1
    assert (
        report["within_intervened_view_skipped_records"][0]["sample_id"]
        == stored[0].sample_id
    )


def test_scale_fit_excludes_structurally_unchanged_target_cells():
    records = select_records(
        records_from_counterfactual_shard(_mock_shard()),
        ["train"],
        ["local_blur", "local_dark"],
    )
    scales = fit_train_component_scales(
        records,
        torch.device("cpu"),
        batch_size=2,
        quantile=0.95,
        response_floor=1e-6,
    )
    # Most of this mock grid is exactly unchanged, so an all-cell q95
    # would be structurally inappropriate for local intervention supervision.
    assert (scales > 1e-4).all()
    audit = audit_train_target_distribution(
        records,
        torch.device("cpu"),
        batch_size=2,
        quantile=0.95,
        response_floor=1e-6,
    )
    assert audit["record_count"] == len(records)
    assert audit["corruption_mask_read"] is False
    assert audit["validation_records_read"] is False
    assert set(audit["component_scales"]) == {
        "persistent_direction",
        "persistent_magnitude",
        "transient_inconsistency",
    }
    spatial = audit_target_spatial_support(
        records,
        scales,
        torch.device("cpu"),
        batch_size=2,
        mask_label_floor=0.25,
    )
    assert spatial["record_count"] == len(records)
    assert spatial["validation_records_read"] is False
    assert spatial["overall"]["combined"]["within_view_mask_auroc"]["median"] > 0.9


def test_hurdle_epoch_trains_without_corruption_mask_labels():
    records = select_records(
        records_from_counterfactual_shard(_mock_shard()),
        ["train"],
        ["local_blur", "local_dark"],
    )
    device = torch.device("cpu")
    scales = fit_train_component_scales(records, device, batch_size=2)
    model = ObservationEvidenceHurdleAdapter(8, hidden_dim=12, max_views=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = run_hurdle_evidence_epoch(
        model,
        records,
        scales,
        device,
        pair_batch_size=1,
        optimizer=optimizer,
    )
    assert set(metrics) == {
        "total",
        "presence",
        "magnitude",
        "ranking",
        "reference_presence",
        "reference_score",
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    diagnostics = evaluate_hurdle_diagnostics(
        model,
        records,
        scales,
        device,
        batch_size=2,
        support_thresholds=torch.tensor([0.1, 0.1, 0.1]),
    )
    assert diagnostics["corruption_mask_read"] is False
    assert 0.0 <= diagnostics["reference_presence_p95"] <= 1.0
    assert diagnostics["support_definition"] == (
        "target > frozen train-responsive q80 per component"
    )


def test_hurdle_can_project_inputs_without_changing_original_target_path():
    records = select_records(
        records_from_counterfactual_shard(_mock_shard()),
        ["train"],
        ["local_blur", "local_dark"],
    )
    device = torch.device("cpu")
    scales = fit_train_component_scales(records, device, batch_size=2)
    projection = deterministic_rademacher_projection(8, 4, 23)
    model = ObservationEvidenceHurdleAdapter(4, hidden_dim=8, max_views=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = run_hurdle_evidence_epoch(
        model,
        records,
        scales,
        device,
        pair_batch_size=1,
        optimizer=optimizer,
        input_projection=projection,
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    report = evaluate_evidence_records(
        model,
        records,
        scales,
        device,
        batch_size=2,
        input_projection=projection,
    )
    assert report["record_count"] == len(records)


def test_hurdle_can_roundtrip_int8_inputs_without_changing_target_path():
    records = select_records(
        records_from_counterfactual_shard(_mock_shard()),
        ["train"],
        ["local_blur", "local_dark"],
    )
    device = torch.device("cpu")
    scales = fit_train_component_scales(records, device, batch_size=2)
    model = ObservationEvidenceHurdleAdapter(8, hidden_dim=8, max_views=2)
    metrics = run_hurdle_evidence_epoch(
        model,
        records,
        scales,
        device,
        pair_batch_size=1,
        optimizer=None,
        input_quantization="dynamic_symmetric_int8_per_grid_channel",
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
