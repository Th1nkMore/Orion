import json
from pathlib import Path

import numpy as np

from scripts.build_uq_relevance_frame_bundles import build_frame_bundles
from scripts.scenario_factory_lib import sha256_file
from scripts.uq_relevance_qa_factory_lib import validate_frame_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _actor():
    return {
        "actor_id": 7,
        "category": "vehicle",
        "type_id": "vehicle.test",
        "position_xy": [4.0, -6.0],
        "position_z": 0.8,
        "velocity_xy": [0.0, 6.0],
        "yaw_degrees": 90.0,
        "extent_xy_m": [2.0, 1.0],
        "extent_z_m": 0.8,
        "relative_longitudinal_m": 4.0,
        "relative_lateral_m": -6.0,
    }


def _safety():
    return {
        "available": True,
        "ego": {
            "actor_id": 1,
            "position_xy": [0.0, 0.0],
            "position_z": 0.75,
            "velocity_xy": [0.0, 0.0],
            "yaw_degrees": 0.0,
            "extent_xy_m": [2.0, 1.0],
            "extent_z_m": 0.75,
        },
        "actors": [_actor()],
    }


def _fixture(tmp_path):
    scenario = tmp_path / "scenario"
    directories = (
        "rgb_front", "rgb_front_model_input", "rgb_front_left",
        "rgb_front_right", "rgb_back", "rgb_back_left", "rgb_back_right",
        "bev",
    )
    inventory = {}
    for name in directories:
        root = scenario / name
        root.mkdir(parents=True)
        (root / "0002.png").write_bytes((name + "-frame").encode())
        inventory[name] = {"path": str(root), "frame_count": 1}
    meta = scenario / "meta"
    meta.mkdir()
    (meta / "0002.json").write_text(json.dumps({
        "command": 4,
        "plan": [[0.0, value] for value in (2, 4, 6, 8, 10, 12)],
        "speed": 4.25,
        "closedloop_safety": _safety(),
    }))
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps({
        "routes": [{
            "route_index": 42,
            "town": "Town01",
            "scenario_type": "Crossing",
        }]
    }))
    package = tmp_path / "event_package.json"
    package.write_text(json.dumps({
        "schema": "orion.scenario_event_package.v1",
        "qa_input_ready": True,
        "runtime": {"valid": True},
        "route": {"route_index": 42},
        "critical_event": {"step": 20},
        "camera_inventory": inventory,
        "source_files": {
            "batch_manifest": {"path": str(batch), "sha256": sha256_file(batch)}
        },
    }))
    uq_path = tmp_path / "uq.npz"
    uq = np.zeros((2, 6, 10, 10), dtype=np.float32)
    uq[:, 0, 5, 5] = (0.4, 0.8)
    components = np.repeat(uq[..., None], 3, axis=-1)
    np.savez_compressed(
        uq_path, uncertainty=uq, uncertainty_components=components
    )
    manifest = tmp_path / "stage1.json"
    manifest.write_text(json.dumps({
        "schema": "orion.stage1_observation_uq_sequence.v1",
        "control_influence": False,
        "latest_frame_index": 2,
        "camera_order": [
            "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        ],
        "event_package_sha256": sha256_file(package),
        "checkpoint_sha256": "a" * 64,
        "normalization": "frozen_train_calibration_v1",
        "component_names": [
            "persistent_direction",
            "persistent_magnitude",
            "transient_inconsistency"
        ],
        "uncertainty": {
            "path": str(uq_path),
            "sha256": sha256_file(uq_path),
            "shape": list(uq.shape),
            "key": "uncertainty",
            "component_key": "uncertainty_components",
            "component_shape": list(components.shape),
        },
    }))
    return package, manifest


def test_builds_aligned_observed_onpath_and_offpath_bundles(tmp_path):
    package, stage1 = _fixture(tmp_path)
    output = tmp_path / "output"
    report = build_frame_bundles(
        event_package_path=package,
        stage1_manifest_path=stage1,
        split="train",
        output_dir=output,
        variants=("observed", "on_path_uq", "off_path_uq"),
        counterfactual_peak=0.9,
    )
    assert report["relevant_actor_ids"] == [7]
    assert {item["variant"] for item in report["bundles"]} == {
        "observed", "on_path_uq", "off_path_uq"
    }
    config = json.loads(
        (PROJECT_ROOT / "configs/scenario_factory/qa_factory_v1.json").read_text()
    )
    for item in report["bundles"]:
        path = Path(item["path"])
        bundle = json.loads(path.read_text())
        validated = validate_frame_bundle(
            bundle, bundle_path=path, config=config
        )
        assert validated["relevance"].shape == (6, 10, 10)
        assert "counterfactual" not in bundle["model_input"]

    on_bundle = json.loads((output / "frame_bundle_on_path_uq.json").read_text())
    off_bundle = json.loads((output / "frame_bundle_off_path_uq.json").read_text())
    route_context = on_bundle["model_input"]["route_context"]
    assert route_context["schema"] == "orion.route_context.v2"
    assert route_context["payload"]["ego_state"] == {
        "speedometer_mps": 4.25
    }
    on_support = on_bundle["counterfactual"]["spatial_support"]
    off_support = off_bundle["counterfactual"]["spatial_support"]
    assert on_support["support_type"] == "matched_local_gaussian_region_v1"
    assert on_support["same_view_matched_pair"] is True
    assert on_support["latest_nonzero_patches"] > 1
    assert on_support["latest_nonzero_patches"] == off_support["latest_nonzero_patches"]
    assert np.isclose(on_support["latest_spatial_sum"], off_support["latest_spatial_sum"])
    assert on_support["support_weighted_relevance"] > off_support["support_weighted_relevance"]


def test_explicit_fixed_keyframe_is_recorded_in_provenance(tmp_path):
    package, stage1 = _fixture(tmp_path)
    output = tmp_path / "explicit_output"
    report = build_frame_bundles(
        event_package_path=package,
        stage1_manifest_path=stage1,
        split="train",
        output_dir=output,
        variants=("observed",),
        counterfactual_peak=0.9,
        selected_frame_index=2,
    )
    bundle = json.loads(Path(report["bundles"][0]["path"]).read_text())
    assert bundle["provenance"]["selected_saved_frame_index"] == 2
    assert bundle["provenance"]["frame_selection"] == "explicit_fixed_temporal_keyframe"


def test_route_context_rejects_missing_or_invalid_current_speed(tmp_path):
    package, stage1 = _fixture(tmp_path)
    event = json.loads(package.read_text())
    scenario = Path(event["camera_inventory"]["rgb_front"]["path"]).parent
    meta_path = scenario / "meta" / "0002.json"
    meta = json.loads(meta_path.read_text())
    meta["speed"] = float("nan")
    meta_path.write_text(json.dumps(meta))
    with np.testing.assert_raises_regex(ValueError, "speedometer"):
        build_frame_bundles(
            event_package_path=package,
            stage1_manifest_path=stage1,
            split="train",
            output_dir=tmp_path / "invalid_speed",
            variants=("observed",),
            counterfactual_peak=0.9,
        )
