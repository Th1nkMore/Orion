import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.uq_relevance_qa_factory_lib import (
    QA_DATASET_SCHEMA,
    QAFactoryError,
    audit_dataset,
    build_records_for_bundle,
    loss_policy_for_record,
    sha256_file,
    summarize_maps,
    validate_frame_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/scenario_factory/qa_factory_v1.json"


def _config():
    return json.loads(CONFIG_PATH.read_text())


def _write_npz(path, key, value):
    np.savez_compressed(path, **{key: np.asarray(value, dtype=np.float32)})
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "shape": list(np.asarray(value).shape),
    }


def _bundle(tmp_path, *, variant="observed", split="train", uq=None, relevance=None):
    config = _config()
    camera_files = []
    hashes = []
    for index, view in enumerate(config["camera_order"]):
        path = tmp_path / (view + ".png")
        path.write_bytes(("camera-%d" % index).encode())
        digest = sha256_file(path)
        hashes.append(digest)
        camera_files.append({"view": view, "path": path.name, "sha256": digest})
    observation_sha = hashlib.sha256("\n".join(hashes).encode("ascii")).hexdigest()
    route_payload = {"command": "follow_lane", "corridor_id": "corridor-1"}
    route_sha = hashlib.sha256(
        json.dumps(route_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if uq is None:
        uq = np.zeros((2, 6, 3, 3), dtype=np.float32)
        uq[-1, 0, 1, 1] = 0.9
    if relevance is None:
        relevance = np.zeros((6, 3, 3), dtype=np.float32)
        relevance[0, 1, 1] = 0.8
    uq_ref = _write_npz(tmp_path / ("uq_%s.npz" % variant), "uncertainty", uq)
    uq_ref.update({
        "source": "frozen_stage1_observation_adapter",
        "checkpoint_sha256": "a" * 64,
        "control_influence": False,
        "normalization": "frozen_train_calibration_v1",
    })
    relevance_ref = _write_npz(
        tmp_path / "relevance.npz", "relevance", relevance
    )
    relevance_ref.update({
        "source": "projected_actor_route_corridor_geometry_v1",
        "uses_corruption_label": False,
        "privileged_for_supervision_only": True,
    })
    bundle = {
        "schema": "orion.uq_relevance_frame_bundle.v1",
        "split": split,
        "event_id": "event-1",
        "frame_id": "frame-10",
        "town": "Town01",
        "scenario_family": "PedestrianCrossing",
        "route": {"route_id": "route-1"},
        "counterfactual": {"group_id": "group-1", "variant": variant},
        "model_input": {
            "observation": {
                "camera_files": camera_files,
                "observation_sha256": observation_sha,
            },
            "route_context": {"payload": route_payload, "sha256": route_sha},
            "stage1_observation_uq": uq_ref,
        },
        "supervision": {"task_relevance": relevance_ref},
    }
    path = tmp_path / ("bundle_%s.json" % variant)
    path.write_text(json.dumps(bundle))
    return bundle, path


def _built_dataset(tmp_path, variants=("observed",)):
    records = []
    for index, variant in enumerate(variants):
        uq = np.zeros((2, 6, 3, 3), dtype=np.float32)
        if variant == "off_path_uq":
            uq[-1, 5, 0, 0] = 0.9
        elif variant != "zero_uq":
            uq[-1, 0, 1, 1] = 0.9 - index * 0.1
        bundle, path = _bundle(tmp_path, variant=variant, uq=uq)
        built, _, _ = build_records_for_bundle(
            bundle,
            bundle_path=path,
            config=_config(),
            sidecar_relative_path="unused.npz",
        )
        records.extend(built)
    return {"schema": QA_DATASET_SCHEMA, "records": records}


def test_frame_bundle_builds_four_consistent_question_families(tmp_path):
    bundle, path = _bundle(tmp_path)
    validated = validate_frame_bundle(
        bundle, bundle_path=path, config=_config()
    )
    records, metadata, arrays = build_records_for_bundle(
        bundle,
        bundle_path=path,
        config=_config(),
        sidecar_relative_path="maps/sample.npz",
    )
    assert validated["uq"].shape == (2, 6, 3, 3)
    assert len(records) == 4
    assert arrays.shape == (2, 6, 3, 3)
    assert metadata["task_risk_definition"].startswith("latest")
    relevance_answer = next(
        row for row in records if row["question_family"] == "task_relevance"
    )["conversation"][1]["value"]
    assert relevance_answer.startswith("<task_relevance_map>")
    assert records[0]["model_input"].get("supervision") is None


def test_v2_loss_policy_excludes_untrusted_driving_answers(tmp_path):
    config = json.loads(
        (PROJECT_ROOT / "configs/scenario_factory/qa_factory_v2_matched_supervision.json").read_text()
    )
    for variant in ("observed", "view_shuffled_uq"):
        policy = loss_policy_for_record(variant, "driving_implication", config)
        assert policy["hard_language_target"] is False
        assert policy["hard_stance_target"] is False
        assert loss_policy_for_record(
            variant, "task_relevance", config
        )["hard_language_target"] is True
    assert loss_policy_for_record(
        "on_path_uq", "driving_implication", config
    )["hard_stance_target"] is True


def test_v2_built_records_embed_and_audit_frozen_loss_policy(tmp_path):
    config = json.loads(
        (PROJECT_ROOT / "configs/scenario_factory/qa_factory_v2_matched_supervision.json").read_text()
    )
    bundle, path = _bundle(tmp_path, variant="observed")
    records, _, _ = build_records_for_bundle(
        bundle,
        bundle_path=path,
        config=config,
        sidecar_relative_path="unused.npz",
    )
    driving = next(
        row for row in records if row["question_family"] == "driving_implication"
    )
    assert driving["loss_policy"]["hard_language_target"] is False
    dataset = {"schema": QA_DATASET_SCHEMA, "records": records}
    report = audit_dataset(dataset, config=config)
    assert report["checks"]["loss_policy_consistent"]
    records[0]["loss_policy"]["hard_language_target"] = False
    report = audit_dataset(dataset, config=config)
    assert not report["checks"]["loss_policy_consistent"]


def test_forbidden_ttc_in_model_input_fails_closed(tmp_path):
    bundle, path = _bundle(tmp_path)
    bundle["model_input"]["route_context"]["ttc"] = 1.2
    with pytest.raises(QAFactoryError, match="forbidden information"):
        validate_frame_bundle(bundle, bundle_path=path, config=_config())


def test_route_context_v2_requires_valid_current_ego_speed(tmp_path):
    bundle, path = _bundle(tmp_path)
    route_context = bundle["model_input"]["route_context"]
    route_context["schema"] = "orion.route_context.v2"
    route_context["payload"]["ego_state"] = {"speedometer_mps": 3.5}
    route_context["sha256"] = hashlib.sha256(
        json.dumps(
            route_context["payload"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    validate_frame_bundle(bundle, bundle_path=path, config=_config())

    route_context["payload"]["ego_state"]["speedometer_mps"] = float("nan")
    with pytest.raises(QAFactoryError, match="speedometer"):
        validate_frame_bundle(bundle, bundle_path=path, config=_config())


def test_on_path_and_off_path_uncertainty_change_task_risk_not_relevance():
    config = _config()
    relevance = np.zeros((6, 3, 3), dtype=np.float32)
    relevance[0, 1, 1] = 1.0
    on_path = np.zeros((1, 6, 3, 3), dtype=np.float32)
    off_path = np.zeros_like(on_path)
    on_path[0, 0, 1, 1] = 0.9
    off_path[0, 5, 0, 0] = 0.9
    on_summary = summarize_maps(on_path, relevance, config=config)
    off_summary = summarize_maps(off_path, relevance, config=config)
    assert on_summary["planning_implication"]["stance"] == "prepare_to_yield"
    assert off_summary["planning_implication"]["stance"] == "maintain"
    assert np.array_equal(relevance, relevance.copy())


def test_rearward_high_task_risk_caps_stance_at_caution():
    config = _config()
    relevance = np.zeros((6, 3, 3), dtype=np.float32)
    uq = np.zeros((1, 6, 3, 3), dtype=np.float32)
    relevance[3, 1, 1] = 1.0
    uq[0, 3, 1, 1] = 0.9

    summary = summarize_maps(uq, relevance, config=config)

    assert summary["task_risk"]["peak_view"] == "CAM_BACK"
    assert summary["planning_implication"] == {
        "stance": "caution",
        "risk_bearing": "rearward",
        "is_direct_control_command": False,
    }


def test_temporal_trend_uses_local_peak_region_not_global_map_mean():
    uq = np.zeros((4, 6, 40, 40), dtype=np.float32)
    for index, value in enumerate((0.2, 0.4, 0.6, 0.9)):
        uq[index, 1, 20:23, 5:8] = value
    relevance = np.zeros((6, 40, 40), dtype=np.float32)

    summary = summarize_maps(uq, relevance, config=_config())

    observation = summary["observation_uncertainty"]
    assert observation["temporal_trend"] == "rising"
    assert observation["temporal_peak_region_delta"] == pytest.approx(0.7)
    assert observation["temporal_summary_scope"] == "latest_peak_patch_across_time"


def test_audit_detects_counterfactual_split_leakage(tmp_path):
    dataset = _built_dataset(tmp_path, variants=("observed", "off_path_uq"))
    leaked = copy.deepcopy(dataset["records"][0])
    leaked["sample_id"] += "/leaked"
    leaked["split"] = "dev"
    dataset["records"].append(leaked)
    report = audit_dataset(dataset, config=_config())
    assert not report["checks"]["event_disjoint_splits"]
    assert not report["checks"]["counterfactual_groups_split_intact"]
    assert not report["formal_training_ready"]


def test_cpu_only_builder_cli_writes_audited_engineering_dataset(tmp_path):
    _, bundle_path = _bundle(tmp_path)
    output = tmp_path / "dataset_output"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_uq_relevance_qa_dataset.py",
            "--bundle",
            str(bundle_path),
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    dataset = json.loads((output / "dataset.json").read_text())
    audit = json.loads((output / "audit.json").read_text())
    assert len(dataset["records"]) == 4
    assert audit["checks"]["map_sidecars_valid"]
    assert not audit["formal_training_ready"]
    assert json.loads(completed.stdout)["record_count"] == 4
