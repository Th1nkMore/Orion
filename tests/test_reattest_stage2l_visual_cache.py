import json
from pathlib import Path

import pytest
import torch

import scripts.reattest_stage2l_visual_cache as module
from scripts.reattest_stage2l_visual_cache import (
    reattest_cache,
    validate_cache_manifest_for_factory,
)
from scripts.scenario_factory_lib import sha256_file


CHECKPOINT = "4" * 64
VIEWS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _factory(tmp_path: Path, name: str, image_root: Path):
    frame_reports = []
    bundles = {}
    metas = {}
    for frame in (10, 12, 14):
        camera_rows = []
        for index, view in enumerate(VIEWS):
            image = image_root / view / ("%04d.png" % frame)
            image.parent.mkdir(parents=True, exist_ok=True)
            if not image.exists():
                image.write_bytes(("image-%d-%d" % (frame, index)).encode("ascii"))
            camera_rows.append(
                {"view": view, "path": str(image), "sha256": sha256_file(image)}
            )
        group_id = "route1_step20_saved_%04d" % frame
        bundle = _write_json(
            tmp_path / name / ("bundle_%04d.json" % frame),
            {
                "schema": "orion.uq_relevance_frame_bundle.v1",
                "counterfactual": {"variant": "observed", "group_id": group_id},
                "model_input": {"observation": {"camera_files": camera_rows}},
                "provenance": {"selected_saved_frame_index": frame},
            },
        )
        batch = _write_json(
            tmp_path / name / ("batch_%04d.json" % frame),
            {
                "schema": "orion.uq_relevance_frame_bundle_batch.v1",
                "bundles": [
                    {
                        "variant": "observed",
                        "path": str(bundle),
                        "sha256": sha256_file(bundle),
                    }
                ],
            },
        )
        meta = _write_json(image_root / "meta" / ("%04d.json" % frame), {"frame": frame})
        bundles[group_id] = bundle
        metas[group_id] = meta
        frame_reports.append(
            {
                "selected_saved_frame_index": frame,
                "frame_bundle_batch": {
                    "path": str(batch),
                    "sha256": sha256_file(batch),
                },
            }
        )
    report = _write_json(
        tmp_path / name / "factory.json",
        {
            "schema": "orion.uq_relevance_multiframe_event_factory.v1",
            "event_id": "route1_step20",
            "keyframe_count": 3,
            "frame_reports": frame_reports,
        },
    )
    return report, bundles, metas


def _fixture(tmp_path: Path, monkeypatch):
    image_root = tmp_path / "images"
    source_report, source_bundles, metas = _factory(tmp_path, "source", image_root)
    target_report, _, _ = _factory(tmp_path, "target", image_root)
    groups = sorted(source_bundles)
    cache = tmp_path / "source_cache.pt"
    cache.write_bytes(b"immutable-cache")
    source_manifest = _write_json(
        tmp_path / "source_cache.json",
        {
            "schema": "orion.stage2l_multiframe_visual_context_cache.v1",
            "status": "immutable_multiframe_visual_context_cache",
            "output": str(cache),
            "sha256": sha256_file(cache),
            "keyframe_count": 3,
            "group_ids": groups,
            "event_factory_report": {
                "path": str(source_report),
                "sha256": sha256_file(source_report),
            },
            "head_memory_reset_per_keyframe": True,
            "privileged_safety_inputs_used": False,
            "stage1_uq_inputs_used": False,
            "task_relevance_targets_used": False,
            "qa_answers_used": False,
        },
    )
    payload = {
        "schema": "orion.stage2l_multiframe_visual_context_cache.v1",
        "contexts": {
            group: torch.empty((1, 529, 4096), device="meta") for group in groups
        },
        "metadata": {
            "event_factory_report": {
                "path": str(source_report),
                "sha256": sha256_file(source_report),
            },
            "frames": [
                {
                    "group_id": group,
                    "frame_bundle": {
                        "path": str(source_bundles[group]),
                        "sha256": sha256_file(source_bundles[group]),
                    },
                    "frame_meta": {
                        "path": str(metas[group]),
                        "sha256": sha256_file(metas[group]),
                    },
                }
                for group in groups
            ],
            "orion_checkpoint": {"sha256": CHECKPOINT},
            "head_memory_reset_per_keyframe": True,
            "privileged_safety_inputs_used": False,
            "stage1_uq_inputs_used": False,
            "task_relevance_targets_used": False,
            "qa_answers_used": False,
            "llm_run_during_cache": False,
            "trajectory_decoder_run_during_cache": False,
        },
    }
    monkeypatch.setattr(module.torch, "load", lambda *_args, **_kwargs: payload)
    return source_manifest, source_report, target_report


def test_reattests_observation_equivalent_cache(tmp_path, monkeypatch):
    source_manifest, _, target_report = _fixture(tmp_path, monkeypatch)
    output_manifest = tmp_path / "reattested.json"
    output_attestation = tmp_path / "reuse_attestation.json"
    result = reattest_cache(
        source_manifest_path=source_manifest,
        target_factory_report_path=target_report,
        expected_orion_checkpoint_sha256=CHECKPOINT,
        output_manifest_path=output_manifest,
        output_attestation_path=output_attestation,
    )
    assert result["status"] == "immutable_multiframe_visual_context_cache_reattested"
    validation = validate_cache_manifest_for_factory(
        cache_manifest_path=output_manifest,
        factory_report_path=target_report,
        expected_orion_checkpoint_sha256=CHECKPOINT,
    )
    assert validation["reuse_attested"] is True
    assert validation["group_count"] == 3


def test_rejects_stale_direct_factory_binding(tmp_path, monkeypatch):
    source_manifest, _, target_report = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="not bound to the current"):
        validate_cache_manifest_for_factory(
            cache_manifest_path=source_manifest,
            factory_report_path=target_report,
            expected_orion_checkpoint_sha256=CHECKPOINT,
        )


def test_rejects_changed_target_camera_bytes(tmp_path, monkeypatch):
    source_manifest, _, _ = _fixture(tmp_path, monkeypatch)
    changed_root = tmp_path / "changed_images"
    target_report, _, _ = _factory(tmp_path, "changed_target", changed_root)
    changed = changed_root / "CAM_FRONT" / "0010.png"
    changed.write_bytes(b"different-observation")
    # Rebuild the report after changing the underlying image and its hash-bound bundles.
    target_report, _, _ = _factory(tmp_path, "changed_target_2", changed_root)
    with pytest.raises(ValueError, match="different observation bytes"):
        reattest_cache(
            source_manifest_path=source_manifest,
            target_factory_report_path=target_report,
            expected_orion_checkpoint_sha256=CHECKPOINT,
            output_manifest_path=tmp_path / "out.json",
            output_attestation_path=tmp_path / "attestation.json",
        )


def test_rejects_changed_target_frame_metadata(tmp_path, monkeypatch):
    source_manifest, _, _ = _fixture(tmp_path, monkeypatch)
    changed_root = tmp_path / "changed_meta_images"
    target_report, _, _ = _factory(tmp_path, "changed_meta_target", changed_root)
    meta = changed_root / "meta" / "0010.json"
    meta.write_text(json.dumps({"frame": 10, "changed": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="different observation bytes"):
        reattest_cache(
            source_manifest_path=source_manifest,
            target_factory_report_path=target_report,
            expected_orion_checkpoint_sha256=CHECKPOINT,
            output_manifest_path=tmp_path / "out.json",
            output_attestation_path=tmp_path / "attestation.json",
        )
