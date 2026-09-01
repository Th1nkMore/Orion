"""CPU tests for independent Stage-1 spatial UQ evaluation/calibration."""

from dataclasses import replace
import json
import subprocess
import sys

import pytest
import torch

from uq_estimator.spatial_evaluation import (
    SPATIAL_EVALUATION_SCHEMA_VERSION,
    SpatialEvaluationError,
    evaluate_stage1_checkpoint,
    fit_monotonic_calibration,
    load_stage1_student,
    save_evaluation_report,
)
from uq_estimator.spatial_training import (
    TARGET_ACTUAL_FAILURE,
    TARGET_REPRESENTATION_PROXY,
    RouteDisjointManifest,
    make_mock_paired_records,
    run_stage1_training,
)


def _records_and_manifest():
    records = make_mock_paired_records(
        feature_dim=4,
        n_routes=10,
        pairs_per_route=2,
        seed=3,
    )
    revised = []
    for record in records:
        mask = record.corruption_mask.bool()
        observed = record.clean_patch_features.clone()
        if record.severity > 0:
            observed = torch.where(
                mask.unsqueeze(-1), -record.clean_patch_features, observed
            )
        actual_severity = None
        actual_event = None
        actual_valid = None
        clean_severity = None
        clean_event = None
        clean_valid = None
        component_errors = None
        component_names = ()
        route_number = int(record.route_id.rsplit("_", 1)[1])
        if route_number % 2 == 0:
            actual_severity = (
                mask.float()
                if record.severity > 0
                else torch.full_like(mask.float(), 0.05)
            )
            actual_event = (actual_severity >= 0.5).float()
            actual_valid = torch.ones_like(mask, dtype=torch.bool)
            actual_valid[..., -1] = False
            clean_severity = torch.full_like(mask.float(), 0.05)
            clean_event = torch.zeros_like(mask.float())
            clean_valid = torch.ones_like(mask, dtype=torch.bool)
            component_errors = torch.stack(
                (actual_severity, actual_event), dim=-1
            )
            component_names = ("occupancy", "miss")
        pair_number = int(record.pair_id.rsplit("_", 1)[1])
        metadata = {
            **record.metadata,
            "path_mask": torch.tensor(
                [[True, False, False, False], [True, False, False, False]]
            ),
            "relevance": "on_path" if pair_number == 0 else "off_path",
            "event_id": record.pair_id,
            "timestamp_seconds": float(record.severity),
            "event_active": record.severity == 1.0,
        }
        revised.append(
            replace(
                record,
                observed_patch_features=observed,
                error_severity_target=actual_severity,
                failure_event_target=actual_event,
                target_valid_mask=actual_valid,
                clean_error_severity_target=clean_severity,
                clean_failure_event_target=clean_event,
                clean_target_valid_mask=clean_valid,
                component_errors=component_errors,
                component_error_names=component_names,
                component_error_axis=-1,
                metadata=metadata,
            )
        )
    manifest = RouteDisjointManifest(
        splits={
            "train": ("route_000", "route_001", "route_002", "route_003"),
            "validation": ("route_004", "route_005"),
            "calibration": ("route_006", "route_007"),
            "held_out": ("route_008", "route_009"),
        },
        seed=3,
    )
    return revised, manifest


@pytest.fixture(scope="module")
def trained_case(tmp_path_factory):
    directory = tmp_path_factory.mktemp("spatial-evaluation")
    records, manifest = _records_and_manifest()
    checkpoint = directory / "stage1.pt"
    run_stage1_training(
        records=records,
        manifest=manifest,
        output_path=checkpoint,
        feature_dim=4,
        hidden_dim=8,
        ensemble_members=3,
        teacher_epochs=1,
        student_epochs=1,
        batch_size=12,
        seed=3,
        device="cpu",
    )
    return records, manifest, checkpoint


def test_temperature_calibration_is_monotone_and_uses_binary_targets():
    probability = torch.tensor([0.10, 0.20, 0.70, 0.90])
    target = torch.tensor([0.0, 0.0, 1.0, 1.0])
    calibration = fit_monotonic_calibration(probability, target)
    calibrated = calibration.apply(probability)

    assert calibration.fit_split == "calibration"
    assert torch.equal(torch.argsort(calibrated), torch.argsort(probability))
    assert 0.0 <= calibration.threshold <= 1.0

    with pytest.raises(SpatialEvaluationError, match="both positive and negative"):
        fit_monotonic_calibration(probability, torch.zeros(4))


def test_evaluation_separates_actual_and_proxy_and_never_evaluates_train(trained_case):
    records, manifest, checkpoint = trained_case
    report = evaluate_stage1_checkpoint(
        checkpoint,
        records,
        supplied_manifest=manifest,
        bootstrap_replicates=20,
        seed=7,
    )

    assert report["schema_version"] == SPATIAL_EVALUATION_SCHEMA_VERSION
    assert report["evaluated_splits"] == ["validation", "calibration", "held_out"]
    assert report["train_split_evaluated"] is False
    assert report["calibration"]["held_out_used_for_fitting"] is False
    assert report["calibration"]["validation_used_for_fitting"] is False
    assert (
        report["calibration"]["by_target_provenance"][TARGET_ACTUAL_FAILURE]["status"]
        == "ok"
    )
    assert (
        report["calibration"]["by_target_provenance"][TARGET_REPRESENTATION_PROXY][
            "status"
        ]
        == "unavailable"
    )

    held_out = report["splits"]["held_out"]
    assert held_out["pooled_cross_provenance_metrics"]["status"] == "prohibited"
    for provenance in (TARGET_ACTUAL_FAILURE, TARGET_REPRESENTATION_PROXY):
        section = held_out["by_target_provenance"][provenance]
        assert section["status"] == "ok"
        assert "spearman_expected_vs_target_error" in section["uncalibrated_metrics"]
        if provenance == TARGET_ACTUAL_FAILURE:
            assert (
                section["calibrated_metrics"]["failure_event_metrics"]["status"]
                == "ok"
            )
            assert section["calibrated_metrics"]["failure_event_metrics"][
                "valid_patch_cells"
            ] == 36
            assert "average_precision" in section["calibrated_metrics"][
                "failure_event_metrics"
            ]
            assert section["calibrated_metrics"]["component_error_metrics"][
                "occupancy"
            ]["status"] == "ok"
            assert section["on_off_path_calibrated"]["path_mask_cell_contrast"][
                "status"
            ] == "ok"
            assert section["temporal_calibrated"]["status"] == "ok"
        else:
            assert section["calibrated_metrics"]["status"] == "unavailable"
            assert (
                section["uncalibrated_metrics"]["failure_event_metrics"]["status"]
                == "unavailable"
            )
        # Only one held-out route exists for each provenance, so a route-level
        # interval must be honestly marked insufficient.
        assert section["bootstrap_ci"]["route"]["status"] == "insufficient"


def test_json_report_is_strict_and_contains_no_nan(trained_case, tmp_path):
    records, manifest, checkpoint = trained_case
    report = evaluate_stage1_checkpoint(
        checkpoint,
        records,
        supplied_manifest=manifest,
        bootstrap_replicates=20,
    )
    path = tmp_path / "report.json"
    save_evaluation_report(report, path)
    loaded = json.loads(path.read_text())
    assert loaded["schema_version"] == SPATIAL_EVALUATION_SCHEMA_VERSION
    assert "NaN" not in path.read_text()
    assert "Infinity" not in path.read_text()


def test_checkpoint_and_manifest_schema_drift_fail_closed(trained_case, tmp_path):
    records, manifest, checkpoint = trained_case
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["schema_version"] = "spatial-uq-stage1-checkpoint/v999"
    bad_checkpoint = tmp_path / "bad.pt"
    torch.save(payload, bad_checkpoint)
    with pytest.raises(SpatialEvaluationError, match="unsupported checkpoint schema"):
        load_stage1_student(bad_checkpoint)

    missing_bound_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    del missing_bound_payload["model_config"]["min_log_variance"]
    missing_bound_checkpoint = tmp_path / "missing_bound.pt"
    torch.save(missing_bound_payload, missing_bound_checkpoint)
    with pytest.raises(SpatialEvaluationError, match="model_config is incomplete"):
        load_stage1_student(missing_bound_checkpoint)

    invalid_bounds_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    invalid_bounds_payload["model_config"]["min_log_variance"] = 4.0
    invalid_bounds_payload["model_config"]["max_log_variance"] = 3.0
    invalid_bounds_checkpoint = tmp_path / "invalid_bounds.pt"
    torch.save(invalid_bounds_payload, invalid_bounds_checkpoint)
    with pytest.raises(SpatialEvaluationError, match="min_log_variance"):
        load_stage1_student(invalid_bounds_checkpoint)

    missing_contract_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    del missing_contract_payload["target_contract_schema_version"]
    missing_contract_checkpoint = tmp_path / "missing_contract.pt"
    torch.save(missing_contract_payload, missing_contract_checkpoint)
    with pytest.raises(SpatialEvaluationError, match="target-contract schema"):
        load_stage1_student(missing_contract_checkpoint)

    wrong_manifest = RouteDisjointManifest(
        splits={
            "train": ("route_000", "route_001", "route_002", "route_004"),
            "validation": ("route_003", "route_005"),
            "calibration": ("route_006", "route_007"),
            "held_out": ("route_008", "route_009"),
        }
    )
    with pytest.raises(SpatialEvaluationError, match="immutable checkpoint manifest"):
        evaluate_stage1_checkpoint(
            checkpoint,
            records,
            supplied_manifest=wrong_manifest,
            bootstrap_replicates=20,
        )


def test_cli_mock_cpu_smoke(tmp_path):
    report = tmp_path / "mock_report.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/eval_spatial_uq.py",
            "--mock",
            "--bootstrap-replicates",
            "20",
            "--report",
            str(report),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["train_split_evaluated"] is False
    assert summary["held_out_used_for_calibration"] is False
    assert report.is_file()
