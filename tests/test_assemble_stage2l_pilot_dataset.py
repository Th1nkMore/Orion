import json

import pytest

from scripts.assemble_stage2l_pilot_dataset import (
    _materialize_record_paths,
    assemble_pilot,
)
from scripts.scenario_factory_lib import sha256_file


@pytest.mark.parametrize(
    "bank_schema,bank_status,schedule_schema,expected_status",
    [
        (
            "orion.stage2_l.pilot_event_bank.v1",
            "frozen_before_stage2l_pilot_training",
            "orion.stage2_l.schedule.v1",
            "assembled_ready_for_stage2l_pilot_training",
        ),
        (
            "orion.stage2_l.formal_pilot_event_bank.v1",
            "frozen_bank_training_still_locked",
            "orion.stage2_l.schedule.v2",
            "assembled_data_training_launch_locked",
        ),
    ],
)
def test_assembles_eight_event_480_record_pilot(
    tmp_path, bank_schema, bank_status, schedule_schema, expected_status
):
    events = []
    reports = []
    caches = []
    variants = ("observed", "zero_uq", "on_path_uq", "off_path_uq", "view_shuffled_uq")
    questions = ("observation_semantics", "epistemic_limitation", "task_relevance", "driving_implication")
    for event_index in range(8):
        event_id = "route%d_step10" % event_index
        split = "train" if event_index < 6 else "dev"
        events.append({
            "event_id": event_id,
            "route_index": event_index,
            "town": "Town%02d" % (1 + event_index % 3),
            "scenario_family": "Family%d" % (event_index % 4),
            "pilot_split": split,
            "formal_split": split,
        })
        rows = []
        groups = []
        for frame in (10, 12, 14):
            group = "%s_saved_%04d" % (event_id, frame)
            groups.append(group)
            for variant in variants:
                for question in questions:
                    rows.append({
                        "event_id": event_id,
                        "frame_id": "saved_%04d" % frame,
                        "split": split,
                        "counterfactual": {"group_id": group, "variant": variant},
                        "question_family": question,
                    })
        records = tmp_path / (event_id + "_records.jsonl")
        records.write_text("".join(json.dumps(row) + "\n" for row in rows))
        report = tmp_path / (event_id + "_factory.json")
        report.write_text(json.dumps({
            "schema": "orion.uq_relevance_multiframe_event_factory.v1",
            "event_id": event_id,
            "keyframe_count": 3,
            "qa_dataset": {"records": {"path": str(records), "sha256": sha256_file(records)}},
        }))
        reports.append(report)
        cache = tmp_path / (event_id + "_visual.pt")
        cache.write_bytes(b"visual-cache")
        cache_manifest = tmp_path / (event_id + "_visual.json")
        cache_manifest.write_text(json.dumps({
            "schema": "orion.stage2l_multiframe_visual_context_cache.v1",
            "output": str(cache),
            "sha256": sha256_file(cache),
            "group_ids": groups,
            "event_factory_report": {"path": str(report), "sha256": sha256_file(report)},
            "privileged_safety_inputs_used": False,
            "stage1_uq_inputs_used": False,
            "task_relevance_targets_used": False,
            "qa_answers_used": False,
        }))
        caches.append(cache_manifest)
    bank = tmp_path / "pilot_bank.json"
    bank.write_text(json.dumps({
        "schema": bank_schema,
        "status": bank_status,
        "selection_policy": {"reassigns_frozen_splits": False},
        "events": events,
    }))
    schedule = tmp_path / "schedule.json"
    schedule.write_text(json.dumps({
        "schema": schedule_schema,
        "fixed_keyframe_policy": {"records_per_keyframe": 20},
        "pilot_gate": {"minimum_independent_events": 8, "expected_qa_records": [480, 800]},
    }))
    review = tmp_path / "qa_review.json"
    review.write_text(json.dumps({
        "schema": "orion.stage2_l.qa_geometry_review_bank.v1",
        "status": "frozen_human_qa_geometry_review",
        "accepted": [
            {
                "event_id": event["event_id"],
                "factory_report": {
                    "path": str(report),
                    "sha256": sha256_file(report),
                },
            }
            for event, report in zip(events, reports)
        ],
    }))
    result = assemble_pilot(
        pilot_bank_path=bank,
        schedule_path=schedule,
        qa_review_bank_path=review,
        factory_reports=reports,
        visual_cache_manifests=caches,
    )
    assert result["event_count"] == 8
    assert result["qa_record_count"] == 480
    assert result["qa_split_counts"] == {"dev": 120, "train": 360}
    assert result["formal_training_ready"] is False
    assert result["status"] == expected_status
    assert result["formal_splits_preserved"] is (
        bank_schema == "orion.stage2_l.formal_pilot_event_bank.v1"
    )


def test_materializes_relative_map_sidecar_before_combining(tmp_path):
    records_path = tmp_path / "event" / "qa_dataset" / "records.jsonl"
    records_path.parent.mkdir(parents=True)
    sidecar = records_path.parent / "map_sidecars" / "sample.npz"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"immutable-map")
    row = {
        "target": {
            "map_sidecar": {
                "path": "map_sidecars/sample.npz",
                "sha256": sha256_file(sidecar),
            }
        }
    }
    materialized = _materialize_record_paths(row, records_path)
    assert materialized["target"]["map_sidecar"]["path"] == str(sidecar.resolve())
    assert row["target"]["map_sidecar"]["path"] == "map_sidecars/sample.npz"
