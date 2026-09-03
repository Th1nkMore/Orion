from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts import run_orion_actual_target_route214 as launch
from uq_estimator.orion_actual_target_builder import ProductionActualTargetBranchBuilderV1
from uq_estimator.orion_decode_adapter import ORIONDecodeAdapterConfigV1
from uq_estimator.orion_route214_production_integration import (
    PersistentRoute214QAEvidenceV1,
    QA_EVIDENCE_SCHEMA_VERSION,
    QA_HOOK_IDS,
    build_route214_production_integration_v1,
)
from uq_estimator import orion_route214_production_integration as production_integration


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _context(tmp_path: Path) -> dict:
    return {
        "schema_version": "orion-route214-production-factory-context/v1",
        "repo_root": launch.REPO_ROOT,
        "plan": {
            "plan_id": "route214-fixture",
            "route": {
                "canonical_route_key": "Town04/Route214",
                "folder": "Town04/Route214_repetition0",
                "town": "Town04",
                "smoke_prefix_frame_range_inclusive": [0, 63],
            },
            "corruption": {
                "family": "local_occlusion",
                "severity": 2,
                "seed": 20260826,
                "event_window_frames_inclusive": [0, 63],
            },
        },
        "decode_config": ORIONDecodeAdapterConfigV1(
            num_classes=9,
            max_num=100,
            post_center_range=(-61.2, -61.2, -10.0, 61.2, 61.2, 10.0),
            class_mapping_id="fixture-classes/v1",
            occupancy_rasterizer_id="orion-selected-motion-occupancy-rasterizer/v1",
            with_light_state=True,
        ),
        "config_lineage": {"sha256": "b" * 64},
        "checkpoint_sha256": "a" * 64,
        "qa_evidence_paths": {
            name: tmp_path / (name + ".json") for name in QA_HOOK_IDS
        },
    }


def test_default_factory_is_concrete_and_has_readiness_markers(tmp_path: Path):
    integration = build_route214_production_integration_v1(_context(tmp_path))
    assert isinstance(integration["branch_target_builder"], ProductionActualTargetBranchBuilderV1)
    assert integration["branch_target_builder"].production_hook_id
    for name, expected_id in QA_HOOK_IDS.items():
        callback = integration[name]
        assert callback.production_hook_id == expected_id
        assert callback.evidence_path == (tmp_path / (name + ".json")).resolve()
        assert callback() is False


def test_deployed_tree_digest_is_explicit_when_git_metadata_is_absent(monkeypatch):
    monkeypatch.setattr(
        production_integration.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=128, stdout="", stderr="not a git repository"
        ),
    )
    revision = production_integration._git_revision(launch.REPO_ROOT)
    assert revision.startswith("deployed-source-tree-sha256:")
    assert len(revision.rsplit(":", 1)[1]) == 64


def test_persistent_qa_evidence_checks_artifact_and_lineage(tmp_path: Path):
    artifact = tmp_path / "overlay.png"
    artifact.write_bytes(b"not-a-bool-only-gate")
    evidence = tmp_path / "projection.json"
    callback = PersistentRoute214QAEvidenceV1(
        qa_kind="projection_overlay_check",
        evidence_path=evidence,
        expected_plan_id="route214-fixture",
        expected_checkpoint_sha256="a" * 64,
        expected_config_sha256="b" * 64,
        expected_route_folder="Town04/Route214_repetition0",
        production_hook_id=QA_HOOK_IDS["projection_overlay_check"],
    )
    evidence.write_text(json.dumps({
        "schema_version": QA_EVIDENCE_SCHEMA_VERSION,
        "qa_kind": "projection_overlay_check",
        "production_hook_id": QA_HOOK_IDS["projection_overlay_check"],
        "plan_id": "route214-fixture",
        "checkpoint_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "route_folder": "Town04/Route214_repetition0",
        "passed": True,
        "checks": {"human_reviewed_overlay": True},
        "generated_by": "test fixture",
        "artifacts": [{"path": artifact.name, "sha256": _sha(artifact)}],
    }), encoding="utf-8")
    assert callback() is True
    artifact.write_bytes(b"changed")
    assert callback() is False
    assert callback.last_audit["reason"] == "evidence_mismatch"


def test_fixed_corruption_is_front_only_and_spatially_stable():
    hook = launch.FixedRoute214LocalOcclusionV1()
    original = torch.randn(1, 6, 3, 32, 48)
    context = {
        "branch": "observed",
        "frame_idx": 0,
        "corruption": {
            "family": "local_occlusion",
            "severity": 2,
            "seed": 20260826,
            "event_window_frames_inclusive": [0, 63],
        },
    }
    first = hook({"img": [original]}, context)["img"][0]
    second = hook({"img": [original]}, dict(context, frame_idx=1))["img"][0]
    assert torch.equal(first[:, 1:], original[:, 1:])
    assert torch.equal(second[:, 1:], original[:, 1:])
    assert not torch.equal(first[:, 0], original[:, 0])
    assert hook.audit()["processed_frames"] == [0, 1]


def test_branch_batches_are_ordered_and_keep_metadata_cpu_side():
    class FakeDataContainer:
        def __init__(self, data):
            self._data = data

        @property
        def data(self):
            return self._data

    metadata = {"scene": "route214"}
    batches = launch.DevicePreparedRoute214BranchBatchesV1(
        [{
            "img": FakeDataContainer([torch.ones(1)]),
            "gt_labels_3d": torch.tensor([1]),
            "img_metas": metadata,
        }],
        torch.device("cpu"),
    )
    clean = list(batches("clean"))[0]
    observed = list(batches("observed"))[0]
    assert clean["img"].data[0].device.type == "cpu"
    assert clean["img"] is not batches.data_loader[0]["img"]
    assert clean["img_metas"] is metadata
    assert observed["gt_labels_3d"].device.type == "cpu"
    with pytest.raises(launch.Route214LaunchError):
        list(batches("clean"))


def test_dataset_filter_and_bounded_sink(tmp_path: Path):
    folder = "Town04/Route214_repetition0"
    dataset = SimpleNamespace(
        data_infos=[{"folder": folder, "frame_idx": index} for index in reversed(range(64))],
        flag=None,
    )
    audit = launch.filter_dataset_to_route214_prefix(dataset, folder)
    assert audit["frame_count"] == 64
    assert [row["frame_idx"] for row in dataset.data_infos] == list(range(64))

    sink = launch.BoundedRoute214RecordSinkV1(tmp_path / "sink")
    expected = []
    for index in range(43):
        record_id = "route214-%03d" % index
        expected.append(record_id)
        sink(SimpleNamespace(record_id=record_id), {"paired": True}, {"frame_idx": index})
    manifest = sink.finalize(expected)
    assert manifest["record_count"] == 43
    assert (tmp_path / "sink" / "record_manifest.json").is_file()


def test_cli_dry_run_loads_default_factory_and_never_executes(tmp_path: Path):
    completed = subprocess.run([
        sys.executable,
        str(launch.REPO_ROOT / "scripts/run_orion_actual_target_route214.py"),
        "--dry-run",
        "--checkpoint-sha256", "a" * 64,
        "--output-root", str(tmp_path / "must-not-be-created"),
    ], cwd=launch.REPO_ROOT, check=True, capture_output=True, text=True)
    report = json.loads(completed.stdout)
    assert report["mode"] == "dry_run"
    assert report["production_integration_error"] is None
    assert report["production_integration"]["factory"] == launch.DEFAULT_FACTORY
    assert report["hook_readiness"]["audit_results"]["concrete_production_branch_builder"] is True
    assert report["hook_readiness"]["audit_results"]["external_production_hook_ids_present"] is True
    assert report["plan"]["forward_count_total"] == 128
    assert report["plan"]["measurement_record_count"] == 43
    assert report["carla_used"] is False
    assert report["training_performed"] is False
    assert report["slurm_job_submitted_by_this_script"] is False
    assert not (tmp_path / "must-not-be-created").exists()


def test_submit_wrapper_is_valid_and_defaults_to_no_submission():
    wrapper = launch.REPO_ROOT / "scripts/submit_orion_actual_target_route214.sh"
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    completed = subprocess.run(
        ["bash", str(wrapper), "--dry-run"], cwd=launch.REPO_ROOT,
        env={"PATH": "/usr/bin:/bin", "RUN_ID": "pytest-route214"},
        check=True, capture_output=True, text=True,
    )
    assert "DRY_RUN_ONLY=1" in completed.stdout
    assert "gpu:1,cpus:8,mem:220G,time:02:00:00" in completed.stdout
    assert "--execute" in completed.stdout
