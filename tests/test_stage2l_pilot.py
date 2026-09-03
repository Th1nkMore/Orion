import json
from pathlib import Path

import pytest
import torch

from uq_estimator.stage2l_pilot import (
    QUESTION_FAMILIES,
    VARIANTS,
    balanced_driving_epoch,
    binary_auroc,
    load_pilot_inputs,
    matched_answer_preference_loss,
    parse_planning_stance,
    planning_stance,
    sha256_file,
)
import random


def _pilot(tmp_path):
    records = []
    events = []
    for event_index in range(8):
        event_id = "route%d_step10" % event_index
        split = "train" if event_index < 6 else "dev"
        groups = []
        for frame in (10, 12, 14):
            group = "%s_saved_%04d" % (event_id, frame)
            groups.append(group)
            for variant in VARIANTS:
                for family in QUESTION_FAMILIES:
                    records.append({
                        "sample_id": "%s/%s/%s" % (group, variant, family),
                        "event_id": event_id,
                        "split": split,
                        "counterfactual": {"group_id": group, "variant": variant},
                        "question_family": family,
                    })
        cache = tmp_path / (event_id + ".pt")
        cache.write_bytes(("cache-" + event_id).encode())
        cache_manifest = tmp_path / (event_id + ".json")
        cache_manifest.write_text(json.dumps({
            "schema": "orion.stage2l_multiframe_visual_context_cache.v1",
            "output": str(cache),
            "sha256": sha256_file(cache),
            "group_ids": groups,
            "privileged_safety_inputs_used": False,
            "stage1_uq_inputs_used": False,
            "task_relevance_targets_used": False,
            "qa_answers_used": False,
        }))
        events.append({
            "event_id": event_id,
            "route_index": event_index,
            "split": split,
            "visual_cache": {"path": str(cache), "sha256": sha256_file(cache)},
            "visual_cache_manifest": {
                "path": str(cache_manifest),
                "sha256": sha256_file(cache_manifest),
            },
        })
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    manifest_path = tmp_path / "pilot.json"
    manifest_path.write_text(json.dumps({
        "schema": "orion.stage2_l.pilot_dataset.v1",
        "status": "assembled_ready_for_stage2l_pilot_training",
        "formal_training_ready": False,
        "event_count": 8,
        "qa_record_count": len(records),
        "records": {"path": str(records_path), "sha256": sha256_file(records_path)},
        "events": events,
    }))
    return manifest_path


def test_loads_route_disjoint_six_two_pilot(tmp_path):
    inputs = load_pilot_inputs(_pilot(tmp_path))
    assert len(inputs.records) == 480
    assert len(inputs.event_cache_paths) == 8


def test_rejects_visual_cache_group_mismatch(tmp_path):
    manifest_path = _pilot(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    cache_manifest_path = manifest["events"][0]["visual_cache_manifest"]["path"]
    cache_manifest_path = Path(cache_manifest_path)
    cache_manifest = json.loads(cache_manifest_path.read_text())
    cache_manifest["group_ids"].pop()
    cache_manifest_path.write_text(json.dumps(cache_manifest))
    manifest["events"][0]["visual_cache_manifest"]["sha256"] = sha256_file(
        cache_manifest_path
    )
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="groups differ"):
        load_pilot_inputs(manifest_path)


def test_auc_and_planning_stance_contracts():
    assert binary_auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert binary_auroc([0.5, 0.5], [False, True]) == 0.5
    assert planning_stance(0.1) == "maintain"
    assert planning_stance(0.4) == "caution"
    assert planning_stance(0.8) == "prepare_to_yield"
    assert planning_stance(0.8, "CAM_BACK") == "caution"
    assert parse_planning_stance("The stance is prepare to yield.") == "prepare_to_yield"


def test_balanced_driving_epoch_only_repeats_minority_stances():
    def row(sample_id, family, stance=None):
        value = {"sample_id": sample_id, "question_family": family}
        if stance is not None:
            value["target"] = {
                "structured_summary": {
                    "planning_implication": {"stance": stance}
                }
            }
        return value

    records = [row("semantic", "observation_semantics")]
    records += [row("m%d" % index, "driving_implication", "maintain") for index in range(3)]
    records += [row("c0", "driving_implication", "caution")]
    records += [row("p0", "driving_implication", "prepare_to_yield")]
    epoch = balanced_driving_epoch(records, random.Random(7))
    assert len(epoch) == 10
    assert sum(value["sample_id"] == "semantic" for value in epoch) == 1
    counts = {
        stance: sum(
            value.get("target", {}).get("structured_summary", {})
            .get("planning_implication", {}).get("stance") == stance
            for value in epoch
        )
        for stance in ("maintain", "caution", "prepare_to_yield")
    }
    assert counts == {"maintain": 3, "caution": 3, "prepare_to_yield": 3}


def test_matched_answer_preference_loss_enforces_target_margin():
    assert matched_answer_preference_loss(
        torch.tensor([0.1]), torch.tensor([0.5]), margin=0.2
    ).item() == pytest.approx(0.0)
    assert matched_answer_preference_loss(
        torch.tensor([0.5]), torch.tensor([0.1]), margin=0.2
    ).item() == pytest.approx(0.6)
    with pytest.raises(ValueError, match="must match"):
        matched_answer_preference_loss(torch.ones(2), torch.ones(1))
