import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from uq_estimator.projection_overlay_preflight import (
    DEFAULT_CANDIDATE_FRAME,
    OVERLAY_PREFLIGHT_SCHEMA_VERSION,
    ProjectionOverlayPreflightError,
    Route214ProjectionFrameV1,
    build_mock_route214_projection_frames,
    generate_route214_projection_overlays,
)
from uq_estimator.projected_visible_support import ORION_CAMERA_ORDER


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "preflight_route214_projection_overlays.py"


def test_mock_generates_two_frames_six_views_json_png_and_noncausal_manifest(tmp_path):
    output = tmp_path / "overlays"
    manifest = generate_route214_projection_overlays(
        build_mock_route214_projection_frames(), output
    )
    assert manifest["schema_version"] == OVERLAY_PREFLIGHT_SCHEMA_VERSION
    assert manifest["input_mode"] == "mock_fixture"
    assert manifest["selected_frames"] == [0, DEFAULT_CANDIDATE_FRAME]
    assert manifest["automated_preflight"]["passed"] is True
    assert manifest["automated_preflight"][
        "candidate_has_nonempty_projected_support"
    ] is True
    assert manifest["automated_preflight"][
        "all_frames_use_bottom_origin_gt"
    ] is True
    assert manifest["claim_boundary"]["model_loaded"] is False
    assert manifest["claim_boundary"]["gpu_used"] is False
    assert manifest["claim_boundary"]["slurm_job_submitted"] is False
    assert manifest["claim_boundary"]["attribution_is_causal"] is False
    assert manifest["claim_boundary"]["g1_projection_overlay_gate_passed"] is False
    assert manifest["claim_boundary"]["mock_fixture_used"] is True
    assert manifest["claim_boundary"]["real_route214_data_used"] is False

    for frame_idx in (0, 39):
        frame_dir = output / ("frame_%06d" % frame_idx)
        frame_audit = json.loads((frame_dir / "frame_audit.json").read_text())
        assert frame_audit["camera_order"] == list(ORION_CAMERA_ORDER)
        assert frame_audit["box_z_origin"] == "bottom"
        assert frame_audit["support_shape"] == [6, 1600, 1]
        assert frame_audit["nonempty_projected_support"] is True
        assert frame_audit["human_visual_review"]["performed"] is False
        assert frame_audit["human_visual_review"][
            "g1_projection_overlay_gate_passed"
        ] is False
        assert (frame_dir / "six_view_contact_sheet.png").is_file()
        for camera in ORION_CAMERA_ORDER:
            png = frame_dir / (camera + ".overlay.png")
            audit = frame_dir / (camera + ".overlay.json")
            assert png.is_file() and png.stat().st_size > 0
            payload = json.loads(audit.read_text())
            assert payload["camera_name"] == camera
            assert payload["claim_boundary"]["attribution_is_causal"] is False
            assert payload["objects"][0]["gt_class"] == 0
            assert isinstance(payload["objects"][0]["gt_actor_id"], int)

    with pytest.raises(ProjectionOverlayPreflightError, match="refusing overwrite"):
        generate_route214_projection_overlays(
            build_mock_route214_projection_frames(), output
        )


def test_frame_contract_fails_closed_on_camera_origin_axis_and_pipeline_claims():
    frame = build_mock_route214_projection_frames()[0]
    values = dict(frame.__dict__)
    values["camera_order"] = (
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    )
    with pytest.raises(ProjectionOverlayPreflightError, match="camera order"):
        Route214ProjectionFrameV1(**values)

    values = dict(frame.__dict__)
    values["box_z_origin"] = "center"
    with pytest.raises(ProjectionOverlayPreflightError, match="bottom"):
        Route214ProjectionFrameV1(**values)

    values = dict(frame.__dict__)
    values["gt_actor_ids"] = torch.tensor([1, 2])
    with pytest.raises(ProjectionOverlayPreflightError, match="misaligned"):
        Route214ProjectionFrameV1(**values)

    values = dict(frame.__dict__)
    values["pipeline_audit"] = {
        **frame.pipeline_audit,
        "post_augmentation_geometry": False,
    }
    with pytest.raises(ProjectionOverlayPreflightError, match="pipeline audit"):
        Route214ProjectionFrameV1(**values)


def test_generator_requires_frame_zero_and_candidate(tmp_path):
    frames = build_mock_route214_projection_frames()
    with pytest.raises(ProjectionOverlayPreflightError, match="frame 0"):
        generate_route214_projection_overlays([frames[1]], tmp_path / "only39")
    with pytest.raises(ProjectionOverlayPreflightError, match="candidate frame"):
        generate_route214_projection_overlays(
            [frames[0]], tmp_path / "only0", candidate_frame=39
        )


def test_cli_mock_summary_and_dataset_argument_gate(tmp_path):
    output = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mock",
            "--output-dir",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["input_mode"] == "mock_fixture"
    assert summary["selected_frames"] == [0, 39]
    assert summary["automated_preflight"]["passed"] is True
    assert summary["claim_boundary"]["gpu_used"] is False
    assert summary["claim_boundary"]["g1_projection_overlay_gate_passed"] is False
    assert Path(summary["manifest_path"]).is_file()
    assert len(summary["manifest_sha256"]) == 64

    refused = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            "--output-dir",
            str(tmp_path / "dataset"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "dataset mode requires" in refused.stderr
