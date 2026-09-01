"""CPU tests for standalone Stage-1 spatial UQ training."""

import json
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from uq_estimator.spatial_training import (
    LEGACY_PAIRED_DATASET_SCHEMA_VERSION,
    LEGACY_PAIRED_RECORD_SCHEMA_VERSION,
    SPATIAL_CHECKPOINT_SCHEMA_VERSION,
    TARGET_CONTRACT_SCHEMA_VERSION,
    TARGET_ACTUAL_FAILURE,
    TARGET_REPRESENTATION_PROXY,
    PairGroupedBatchSampler,
    PairedSpatialFeatureDataset,
    PairedSpatialFeatureRecord,
    RouteDisjointManifest,
    SpatialHeadEnsemble,
    SpatialLossWeights,
    SpatialPatchUQHead,
    SpatialTrainingDataError,
    build_route_disjoint_manifest,
    collate_paired_spatial_records,
    compute_spatial_training_loss,
    load_paired_feature_records,
    make_mock_paired_records,
    run_stage1_training,
    save_paired_feature_records,
)


def _record(
    record_id="record",
    route_id="route_a",
    pair_id="pair_a",
    severity=1.0,
    error_severity_target=None,
    failure_event_target=None,
    target_valid_mask=None,
):
    clean = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    observed = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    return PairedSpatialFeatureRecord(
        record_id=record_id,
        pair_id=pair_id,
        route_id=route_id,
        town="Town01",
        severity=severity,
        observed_patch_features=observed,
        clean_patch_features=clean,
        error_severity_target=error_severity_target,
        failure_event_target=failure_event_target,
        target_valid_mask=target_valid_mask,
        clean_error_severity_target=(
            torch.tensor([[0.05, 0.20]])
            if error_severity_target is not None
            else None
        ),
        clean_failure_event_target=(
            torch.tensor([[0.0, 1.0]])
            if error_severity_target is not None
            else None
        ),
        clean_target_valid_mask=(
            torch.tensor([[True, True]])
            if error_severity_target is not None
            else None
        ),
        corruption_mask=torch.tensor([[1.0, 0.0]]),
        ensemble_teacher_variance=torch.tensor([[0.2, 0.1]]),
    )


def test_actual_severity_and_failure_event_targets_are_independent():
    severity = torch.tensor([[0.25, 1.75]])
    event = torch.tensor([[False, True]])
    valid = torch.tensor([[True, False]])
    record = _record(
        error_severity_target=severity,
        failure_event_target=event,
        target_valid_mask=valid,
    )
    batch = collate_paired_spatial_records([record])

    torch.testing.assert_close(batch.error_severity_target[0], severity)
    torch.testing.assert_close(batch.failure_event_target[0], event.float())
    assert torch.equal(batch.error_severity_valid_mask[0], valid)
    assert torch.equal(batch.failure_event_valid_mask[0], valid)
    assert not torch.equal(batch.error_severity_target, batch.failure_event_target)
    assert batch.target_is_actual.tolist() == [True]
    assert batch.target_provenance == (TARGET_ACTUAL_FAILURE,)


def test_missing_actual_target_is_explicit_representation_error_proxy():
    record = _record()
    batch = collate_paired_spatial_records([record])

    # First patch is orthogonal (1-cos = 1), second is unchanged (0).
    torch.testing.assert_close(
        batch.error_severity_target[0], torch.tensor([[1.0, 0.0]])
    )
    assert batch.target_is_actual.tolist() == [False]
    assert batch.target_provenance == (TARGET_REPRESENTATION_PROXY,)
    assert not batch.failure_event_valid_mask.any()
    assert not batch.clean_target_valid_mask.any()


def test_paired_record_round_trip_preserves_schema_and_provenance(tmp_path):
    records = [
        _record(
            error_severity_target=torch.tensor([[0.1, 0.2]]),
            failure_event_target=torch.tensor([[0.0, 1.0]]),
            target_valid_mask=torch.tensor([[True, True]]),
        )
    ]
    path = tmp_path / "records.pt"
    save_paired_feature_records(path, records)
    loaded = load_paired_feature_records(path)

    assert len(loaded) == 1
    assert loaded[0].schema_version == records[0].schema_version
    assert loaded[0].target_provenance == TARGET_ACTUAL_FAILURE
    torch.testing.assert_close(
        loaded[0].observed_patch_features,
        records[0].observed_patch_features,
    )


def test_v2_requires_complete_actual_contract_and_component_axis():
    with pytest.raises(SpatialTrainingDataError, match="require error_severity"):
        _record(error_severity_target=torch.tensor([[0.1, 0.2]]))

    base = _record(
        error_severity_target=torch.tensor([[0.1, 0.2]]),
        failure_event_target=torch.tensor([[0.0, 1.0]]),
        target_valid_mask=torch.tensor([[True, True]]),
    )
    component = replace(
        base,
        component_errors=torch.tensor([[[0.1, 0.0], [0.2, 1.0]]]),
        component_error_names=("localization", "miss"),
        component_error_axis=-1,
    )
    assert component.component_errors.shape[-1] == 2
    with pytest.raises(SpatialTrainingDataError, match="component axis has"):
        replace(component, component_error_names=("miss",))


def test_v1_proxy_migrates_but_ambiguous_v1_actual_fails_closed(tmp_path):
    proxy = _record()
    legacy_item = {
        **proxy.to_payload(),
        "schema_version": LEGACY_PAIRED_RECORD_SCHEMA_VERSION,
    }
    legacy_item.pop("target_contract_schema_version")
    for key in (
        "error_severity_target",
        "failure_event_target",
        "target_valid_mask",
        "clean_error_severity_target",
        "clean_failure_event_target",
        "clean_target_valid_mask",
        "component_errors",
        "clean_component_errors",
        "component_error_names",
        "component_error_axis",
    ):
        legacy_item.pop(key)
    legacy_item["failure_target"] = None
    legacy_item["clean_failure_target"] = None
    path = tmp_path / "legacy_proxy.pt"
    torch.save(
        {
            "schema_version": LEGACY_PAIRED_DATASET_SCHEMA_VERSION,
            "record_schema_version": LEGACY_PAIRED_RECORD_SCHEMA_VERSION,
            "records": [legacy_item],
        },
        path,
    )
    migrated = load_paired_feature_records(path)
    assert migrated[0].schema_version.endswith("/v2")
    assert migrated[0].target_provenance == TARGET_REPRESENTATION_PROXY
    assert migrated[0].metadata["legacy_migration"]["actual_targets_inferred"] is False

    legacy_item["failure_target"] = torch.tensor([[0.2, 0.8]])
    actual_path = tmp_path / "legacy_actual.pt"
    torch.save(
        {
            "schema_version": LEGACY_PAIRED_DATASET_SCHEMA_VERSION,
            "record_schema_version": LEGACY_PAIRED_RECORD_SCHEMA_VERSION,
            "records": [legacy_item],
        },
        actual_path,
    )
    with pytest.raises(SpatialTrainingDataError, match="cannot automatically migrate"):
        load_paired_feature_records(actual_path)


def test_manifest_enforces_route_disjointness_and_filters_dataset():
    records = make_mock_paired_records(n_routes=6, pairs_per_route=1)
    manifest = build_route_disjoint_manifest(records, seed=3)
    all_routes = [route for routes in manifest.splits.values() for route in routes]
    assert len(all_routes) == len(set(all_routes)) == 6

    train = PairedSpatialFeatureDataset(records, manifest, "train")
    assert {record.route_id for record in train.records} == set(
        manifest.splits["train"]
    )

    with pytest.raises(SpatialTrainingDataError, match="appears in both"):
        RouteDisjointManifest(
            splits={
                "train": ("route_a",),
                "validation": ("route_a",),
                "calibration": ("route_b",),
                "held_out": ("route_c",),
            }
        )


def test_pair_grouped_sampler_never_splits_severity_pairs():
    records = make_mock_paired_records(n_routes=6, pairs_per_route=2)
    manifest = build_route_disjoint_manifest(records, seed=1)
    dataset = PairedSpatialFeatureDataset(records, manifest, "train")
    sampler = PairGroupedBatchSampler(dataset, batch_size=6, shuffle=True, seed=2)

    for indices in sampler:
        in_batch = {dataset.records[index].pair_id for index in indices}
        for pair_id in in_batch:
            expected = {
                index
                for index, record in enumerate(dataset.records)
                if record.pair_id == pair_id
            }
            assert expected.issubset(set(indices))


def test_loss_exposes_all_required_components_and_backpropagates():
    records = [
        _record(record_id="low", pair_id="same", severity=0.0),
        _record(record_id="high", pair_id="same", severity=2.0),
    ]
    # Make the higher-severity representation error truly larger.
    records[0] = PairedSpatialFeatureRecord(
        **{
            **records[0].__dict__,
            "observed_patch_features": records[0].clean_patch_features.clone(),
        }
    )
    batch = collate_paired_spatial_records(records)
    model = SpatialPatchUQHead(feature_dim=2, hidden_dim=8, predict_epistemic=True)
    output = model(batch.observed_features)
    clean_output = model(batch.clean_features)
    teacher = torch.full_like(output.expected_error, 0.15)
    losses = compute_spatial_training_loss(
        output,
        batch,
        SpatialLossWeights(),
        clean_output=clean_output,
        live_teacher_epistemic=teacher,
    )

    assert set(losses) == {
        "total",
        "gaussian_nll",
        "failure_brier",
        "error_ranking",
        "epistemic_distill",
        "mask_aux",
        "clean_gaussian_nll",
        "clean_failure_brier",
    }
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_every_supervised_loss_ignores_invalid_target_cells():
    valid = torch.tensor([[True, False]])
    base = _record(
        error_severity_target=torch.tensor([[0.2, 1.0]]),
        failure_event_target=torch.tensor([[0.0, 0.0]]),
        target_valid_mask=valid,
    )
    changed_only_where_invalid = replace(
        base,
        error_severity_target=torch.tensor([[0.2, 100.0]]),
        failure_event_target=torch.tensor([[0.0, 1.0]]),
        corruption_mask=torch.tensor([[1.0, 1.0]]),
        ensemble_teacher_variance=torch.tensor([[0.2, 99.0]]),
    )
    model = SpatialPatchUQHead(feature_dim=2, hidden_dim=8, predict_epistemic=True)
    outputs = []
    for record in (base, changed_only_where_invalid):
        batch = collate_paired_spatial_records([record])
        output = model(batch.observed_features)
        clean_output = model(batch.clean_features)
        outputs.append(
            compute_spatial_training_loss(
                output,
                batch,
                SpatialLossWeights(),
                clean_output=clean_output,
            )
        )
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key])


def test_three_member_ensemble_and_student_shapes():
    features = torch.randn(2, 3, 4, 6)
    ensemble = SpatialHeadEnsemble(feature_dim=6, hidden_dim=9, n_members=3)
    teacher = ensemble(features)
    student = SpatialPatchUQHead(
        feature_dim=6, hidden_dim=9, predict_epistemic=True
    )(features)

    assert len(teacher.member_outputs) == 3
    assert teacher.expected_error.shape == (2, 3, 4)
    assert teacher.epistemic_variance.shape == (2, 3, 4)
    assert torch.all(teacher.epistemic_variance >= 0)
    assert student.epistemic_variance is not None
    assert student.epistemic_variance.shape == teacher.epistemic_variance.shape


def test_minimal_training_checkpoint_records_provenance_and_claim_boundary(tmp_path):
    records = make_mock_paired_records(
        feature_dim=5, n_routes=6, pairs_per_route=1, seed=4
    )
    manifest = build_route_disjoint_manifest(records, seed=4)
    output = tmp_path / "stage1.pt"
    checkpoint = run_stage1_training(
        records,
        manifest,
        output,
        feature_dim=5,
        hidden_dim=8,
        ensemble_members=3,
        teacher_epochs=1,
        student_epochs=1,
        batch_size=6,
        seed=4,
    )

    assert output.is_file()
    assert checkpoint["schema_version"] == SPATIAL_CHECKPOINT_SCHEMA_VERSION
    assert checkpoint["target_contract_schema_version"] == TARGET_CONTRACT_SCHEMA_VERSION
    assert checkpoint["model_config"]["ensemble_members"] == 3
    assert checkpoint["model_config"]["min_log_variance"] == -6.0
    assert checkpoint["model_config"]["max_log_variance"] == 3.0
    counts = checkpoint["target_provenance"]["primary_target_counts"]
    assert counts[TARGET_ACTUAL_FAILURE] > 0
    assert counts[TARGET_REPRESENTATION_PROXY] > 0
    boundary = checkpoint["claim_boundary"]
    assert boundary["route_is_uq_head_input"] is False
    assert boundary["representation_error_proxy_is_semantic_uq"] is False
    assert boundary["supports_closed_loop_safety_claim"] is False
    assert checkpoint["route_disjoint_manifest"]["route_disjoint"] is True


def test_cli_mock_smoke_runs_without_orion_dependencies(tmp_path):
    output = tmp_path / "cli_smoke.pt"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_spatial_uq.py",
            "--mock",
            "--smoke",
            "--feature-dim",
            "5",
            "--hidden-dim",
            "8",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["schema_version"] == SPATIAL_CHECKPOINT_SCHEMA_VERSION
    assert summary["target_contract_schema_version"] == TARGET_CONTRACT_SCHEMA_VERSION
    assert summary["ensemble_members"] == 3
    assert output.is_file()
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert payload["claim_boundary"]["supports_llm_understanding_claim"] is False
