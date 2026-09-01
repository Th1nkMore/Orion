"""CPU-only tests for the fail-closed chronological ORION replay plan."""

import hashlib
import json
import subprocess
import sys

import pytest

from uq_estimator.orion_replay_smoke import (
    RUNTIME_ATTESTATION_SCHEMA_VERSION,
    OrionReplayPlanError,
    build_replay_smoke_plan,
    evaluate_runtime_attestation,
    verify_source_infos,
)


FOLDER = "v1/OppositeVehicleTakingPriority_Town04_Route214_Weather6"
ROUTE = "Town04/Route214"
CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def _pilot(info_sha="a" * 64, replay_start=0, replay_end=3):
    count = replay_end - replay_start + 1
    return {
        "schema_version": "spatial-uq-pilot-submanifest/v1",
        "parent_manifest": {"info_source": {"sha256": info_sha}},
        "route_statistics": {
            ROUTE: {
                "canonical_route_key": ROUTE,
                "parent_split": "calibration",
                "town": "Town04",
                "scenario_types": ["OppositeVehicleTakingPriority"],
                "folders": [FOLDER],
                "folder_statistics": {
                    FOLDER: {
                        "replay_frame_start": replay_start,
                        "replay_frame_end": replay_end,
                        "replay_frame_count": count,
                    }
                },
            }
        },
        "temporal_execution_contract": {
            "chronological_full_folder_replay_required": True,
            "frame_independent_inference_forbidden": True,
            "memory_reset_between_folders_required": True,
            "memory_reset_between_clean_and_observed_passes_required": True,
            "folder_replay_statistics": {
                FOLDER: {
                    "canonical_route_key": ROUTE,
                    "chronologically_contiguous": True,
                    "replay_frame_start": replay_start,
                    "replay_frame_end": replay_end,
                    "replay_frame_count": count,
                    "measurement_frame_count": 2,
                }
            },
        },
        "samples": [
            {
                "canonical_route_key": ROUTE,
                "folder": FOLDER,
                "frame_idx": 0,
                "annotation_stratum": "safety_visible_candidate_annotation_only",
            },
            {
                "canonical_route_key": ROUTE,
                "folder": FOLDER,
                "frame_idx": replay_end,
                "annotation_stratum": "background_annotation",
            },
        ],
    }


def _lineage():
    return {"path": "/mock/pilot.json", "sha256": "b" * 64, "size_bytes": 12}


def _source_infos():
    records = []
    for frame in range(4):
        sensors = {
            camera: {
                "data_path": "%s/camera/%s/%05d.jpg" % (FOLDER, camera, frame)
            }
            for camera in CAMERAS
        }
        sensors["LIDAR_TOP"] = {"data_path": "unused.las"}
        records.append({"folder": FOLDER, "frame_idx": frame, "sensors": sensors})
    return records


def _write_source_and_files(tmp_path, missing_frame=None):
    infos = [item for item in _source_infos() if item["frame_idx"] != missing_frame]
    raw = json.dumps({"infos": infos}, sort_keys=False).encode("utf-8")
    path = tmp_path / "infos.json"
    path.write_bytes(raw)
    for info in infos:
        frame = info["frame_idx"]
        for camera in CAMERAS:
            image = tmp_path / info["sensors"][camera]["data_path"]
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"image")
        annotation = tmp_path / FOLDER / "anno" / ("%05d.json.gz" % frame)
        annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation.write_bytes(b"annotation")
    return path, hashlib.sha256(raw).hexdigest()


def _passing_attestation(plan):
    execution = plan["execution"]
    expected = execution["expected_frames_each_branch"]
    measured = set(execution["measurement_frames"])
    branches = {}
    for branch in ("clean", "observed"):
        branches[branch] = {
            "frames_processed": expected,
            "measurement_frames_persisted": execution["measurement_frames"],
            "reset_called_before_frame_zero": True,
            "reset_state_verified_empty": True,
            "no_other_branch_interleaving": True,
            "paired_replay_id": "shared-replay",
            "branch_history_id": "%s-history" % branch,
            "per_frame_audit": [
                {
                    "frame_idx": frame,
                    "scene_token": plan["route"]["folder"],
                    "model_forward_completed": True,
                    "six_camera_images_loaded": True,
                    "traffic_state_shape_n_by_2": True,
                    "traffic_state_mask_matches_objects": True,
                    "post_augmentation_lidar2img_count_is_6": True,
                    "processed_image_shape_present": True,
                    "decoded_output_adapter_ready": True,
                    "actual_target_adapter_ready": True,
                    "persisted": frame in measured,
                }
                for frame in expected
            ],
        }
    return {
        "schema_version": RUNTIME_ATTESTATION_SCHEMA_VERSION,
        "plan_id": plan["plan_id"],
        "global_checks": {
            name: True
            for name in plan["runtime_attestation_contract"][
                "required_global_g1_checks"
            ]
        },
        "branches": branches,
    }


def test_route214_prefix_is_two_separate_frame_zero_replays_and_measurement_only():
    plan = build_replay_smoke_plan(_pilot(), _lineage(), prefix_end=3)
    assert plan["route"]["smoke_prefix_frame_range_inclusive"] == [0, 3]
    assert plan["execution"]["branch_order"] == ["clean", "observed"]
    assert plan["execution"]["expected_frames_each_branch"] == [0, 1, 2, 3]
    assert plan["execution"]["expected_forward_count_total"] == 8
    assert plan["execution"]["measurement_frames"] == [0, 3]
    assert plan["execution"]["expected_paired_target_record_count"] == 2
    assert plan["execution"]["warmup_or_unscored_frame_count_each_branch"] == 2
    assert plan["gates"]["g1_passed"] is False
    assert plan["resources"]["job_submitted"] is False


def test_nonzero_start_and_missing_annotation_candidate_fail_closed():
    with pytest.raises(OrionReplayPlanError, match="frame-0-through-end"):
        build_replay_smoke_plan(_pilot(replay_start=1), _lineage(), prefix_end=3)
    payload = _pilot()
    payload["samples"][0]["annotation_stratum"] = "background_annotation"
    with pytest.raises(OrionReplayPlanError, match="no annotation candidate"):
        build_replay_smoke_plan(payload, _lineage(), prefix_end=3)


def test_source_preflight_verifies_sha_frames_six_cameras_and_files(tmp_path):
    path, digest = _write_source_and_files(tmp_path)
    pilot = _pilot(info_sha=digest)
    verified = verify_source_infos(path, pilot, FOLDER, 3, dataset_root=tmp_path)
    assert verified["source_info_sha256_verified"] is True
    assert verified["exact_frame_zero_prefix_verified"] is True
    assert verified["canonical_six_camera_metadata_verified"] is True
    assert verified["camera_files_exist_for_all_frames"] is True
    assert verified["annotation_files_exist_for_all_frames"] is True


def test_source_preflight_rejects_missing_frame_and_sha_mismatch(tmp_path):
    path, digest = _write_source_and_files(tmp_path, missing_frame=2)
    with pytest.raises(OrionReplayPlanError, match=r"missing=\[2\]"):
        verify_source_infos(path, _pilot(info_sha=digest), FOLDER, 3)
    path, _ = _write_source_and_files(tmp_path)
    with pytest.raises(OrionReplayPlanError, match="SHA-256 disagrees"):
        verify_source_infos(path, _pilot(info_sha="0" * 64), FOLDER, 3)


def test_source_preflight_rejects_noncanonical_camera_insertion_order(tmp_path):
    infos = _source_infos()
    front_left = infos[0]["sensors"].pop("CAM_FRONT_LEFT")
    infos[0]["sensors"]["CAM_FRONT_LEFT"] = front_left
    raw = json.dumps({"infos": infos}, sort_keys=False).encode("utf-8")
    path = tmp_path / "infos.json"
    path.write_bytes(raw)
    pilot = _pilot(info_sha=hashlib.sha256(raw).hexdigest())
    with pytest.raises(OrionReplayPlanError, match="camera insertion order"):
        verify_source_infos(path, pilot, FOLDER, 3)


def test_runtime_attestation_can_pass_only_after_exhaustive_checks(tmp_path):
    path, digest = _write_source_and_files(tmp_path)
    pilot = _pilot(info_sha=digest)
    verified = verify_source_infos(path, pilot, FOLDER, 3, dataset_root=tmp_path)
    plan = build_replay_smoke_plan(
        pilot, _lineage(), prefix_end=3, source_verification=verified
    )
    result = evaluate_runtime_attestation(plan, _passing_attestation(plan))
    assert result["g1_passed"] is True
    assert result["failures"] == []


def test_runtime_attestation_fails_on_skipped_frame_extra_persistence_or_adapter(tmp_path):
    path, digest = _write_source_and_files(tmp_path)
    pilot = _pilot(info_sha=digest)
    verified = verify_source_infos(path, pilot, FOLDER, 3, dataset_root=tmp_path)
    plan = build_replay_smoke_plan(
        pilot, _lineage(), prefix_end=3, source_verification=verified
    )
    attestation = _passing_attestation(plan)
    attestation["branches"]["observed"]["frames_processed"] = [0, 1, 3]
    attestation["branches"]["clean"]["measurement_frames_persisted"] = [0, 1, 3]
    attestation["branches"]["observed"]["per_frame_audit"][2][
        "actual_target_adapter_ready"
    ] = False
    result = evaluate_runtime_attestation(plan, attestation)
    assert result["g1_passed"] is False
    assert any("frames_processed_exactly" in item for item in result["failures"])
    assert any("measurement_frames_persisted_exactly" in item for item in result["failures"])
    assert any("per_frame_checks_all_pass" in item for item in result["failures"])


def test_real_pilot_cli_reports_route214_prefix_without_running_orion():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/plan_orion_actual_target_smoke.py",
            "--summary-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["route"]["canonical_route_key"] == ROUTE
    assert result["route"]["full_route_frame_count"] == 136
    assert result["route"]["smoke_prefix_frame_count"] == 64
    assert result["execution"]["expected_forward_count_total"] == 128
    assert result["execution"]["measurement_frame_count"] == 43
    assert result["execution"]["annotation_candidate_measurement_count"] == 20
    assert result["gates"]["g1_passed"] is False
    assert result["resources"]["job_submitted"] is False
