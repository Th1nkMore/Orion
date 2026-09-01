import json

import pytest

from scripts.assemble_stage2l_formal_dataset import assemble_formal_dataset
from scripts.scenario_factory_lib import sha256_file
from scripts.upgrade_stage2l_v9_qa_records import upgrade_records


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(
    tmp_path,
    *,
    altered_split=False,
    qa_config_mismatch=False,
    native_glare=False,
    incomplete_protocol=False,
    stage1_checkpoint_mismatch=False,
    legacy_qa_records=False,
):
    stage1_checkpoint_sha256 = "a" * 64
    qa_config = _write_json(
        tmp_path / "qa_v5.json",
        {"schema": "orion.uq_relevance_qa_factory_config.v5"},
    )
    other_config = _write_json(tmp_path / "qa_old.json", {"schema": "qa-old"})
    schedule = _write_json(
        tmp_path / "schedule.json",
        {
            "schema": "orion.stage2_l.schedule.v2",
            "fixed_keyframe_policy": {"records_per_keyframe": 20},
            "formal_gate": {
                "independent_events": 24,
                "event_level_split": {"train": 16, "dev": 4, "test": 4},
                "target_structured_qa_records": [1500, 2500],
            },
        },
    )
    events = []
    plan_events = []
    reports = []
    caches = []
    accepted = []
    variants = (
        "observed",
        "zero_uq",
        "on_path_uq",
        "off_path_uq",
        "view_shuffled_uq",
    )
    questions = (
        "observation_semantics",
        "epistemic_limitation",
        "task_relevance",
        "driving_implication",
    )
    for index in range(24):
        if index < 16:
            split = "train"
        elif index < 20:
            split = "dev"
        else:
            split = "test"
        event_id = "route%d_step10" % index
        event = {
            "event_id": event_id,
            "route_index": index,
            "town": "Town%02d" % (1 + index % 6),
            "scenario_family": "Family%d" % (index % 9),
            "formal_split": ("dev" if altered_split and index == 0 else split),
        }
        events.append(event)
        plan_events.append(
            {
                "route_index": index,
                "town": event["town"],
                "scenario_family": event["scenario_family"],
                "formal_split": split,
            }
        )
        rows = []
        groups = []
        for frame in (10, 12, 14, 16):
            group = "%s_saved_%04d" % (event_id, frame)
            groups.append(group)
            for variant in variants:
                for question in questions:
                    rows.append(
                        {
                            "schema": "orion.uq_relevance_qa_record.v4",
                            "sample_id": "%s/%s/%s" % (group, variant, question),
                            "event_id": event_id,
                            "frame_id": "saved_%04d" % frame,
                            "split": event["formal_split"],
                            "counterfactual": {
                                "group_id": group,
                                "variant": variant,
                            },
                            "question_family": question,
                            "conversation": [
                                {"from": "human", "value": "question"},
                                {"from": "gpt", "value": "legacy answer"},
                            ],
                            "target": {
                                "structured_summary": {
                                    "observation_uncertainty": {
                                        "level": "low" if variant == "zero_uq" else "high",
                                        "peak_score": 0.0 if variant == "zero_uq" else 0.9,
                                        "peak_view": "CAM_FRONT",
                                        "peak_region": "lower_center",
                                        "temporal_trend": "stable",
                                    },
                                    "relevance_at_most_uncertain_region": {
                                        "level": (
                                            "low"
                                            if variant in {"zero_uq", "off_path_uq"}
                                            else "high"
                                        ),
                                        "score": (
                                            0.0
                                            if variant in {"zero_uq", "off_path_uq"}
                                            else 0.7
                                        ),
                                    },
                                    "task_risk": {
                                        "level": (
                                            "low"
                                            if variant in {"zero_uq", "off_path_uq"}
                                            else "medium"
                                        ),
                                        "peak_score": (
                                            0.0
                                            if variant in {"zero_uq", "off_path_uq"}
                                            else 0.6
                                        ),
                                        "peak_view": "CAM_FRONT",
                                        "peak_region": "lower_center",
                                    },
                                    "planning_implication": {
                                        "stance": (
                                            "prepare_to_yield"
                                            if variant == "on_path_uq"
                                            else "maintain"
                                        ),
                                        "risk_bearing": "forward_or_crossing",
                                        "is_direct_control_command": False,
                                    },
                                }
                            },
                        }
                    )
        if not legacy_qa_records:
            rows, audit = upgrade_records(rows)
            assert audit["passed"] is True
        records = tmp_path / (event_id + "_records.jsonl")
        records.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        run_manifest = _write_json(
            tmp_path / (event_id + "_run.json"),
            {
                "pilot_condition": "clean_off",
                "orion_closedloop_uq_mode": "none",
                "orion_closedloop_risk_mode": "off",
                "orion_planning_response_mode": "off",
                "orion_enable_legacy_density_uq": "0",
                "orion_closedloop_corruption": None,
                "render_condition": {
                    "schema": "orion.closedloop_render_condition.v1",
                    "kind": (
                        "carla_native_low_sun_glare"
                        if native_glare and index == 0
                        else "standard_carla_rgb"
                    ),
                    "native_glare_profile": (
                        "medium" if native_glare and index == 0 else "none"
                    ),
                    "camera_postprocess_override": native_glare and index == 0,
                },
            },
        )
        event_package = _write_json(
            tmp_path / (event_id + "_event_package.json"),
            {
                "schema": "orion.scenario_event_package.v1",
                "route": {"route_index": index},
                "runtime": {"valid": True},
                "qa_input_ready": True,
                "source_files": {
                    "run_manifest": {
                        "path": str(run_manifest),
                        "sha256": sha256_file(run_manifest),
                    }
                },
            },
        )
        sequence_rows = []
        for frame in (10, 12, 14, 16):
            sequence = _write_json(
                tmp_path / ("%s_%04d_stage1.json" % (event_id, frame)),
                {
                    "schema": "orion.stage1_observation_uq_sequence.v1",
                    "status": "offline_frozen_stage1_output",
                    "control_influence": False,
                    "event_package_sha256": sha256_file(event_package),
                    "checkpoint_sha256": (
                        "b" * 64
                        if stage1_checkpoint_mismatch and index == 0 and frame == 10
                        else stage1_checkpoint_sha256
                    ),
                    "forbidden_inputs": {
                        "route": False,
                        "actor_geometry": False,
                        "ttc": False,
                        "collision_outcome": False,
                        "corruption_metadata": False,
                    },
                },
            )
            sequence_rows.append(
                {
                    "selected_saved_frame_index": frame,
                    "manifest": {
                        "path": str(sequence),
                        "sha256": sha256_file(sequence),
                    },
                }
            )
        stage1_root = _write_json(
            tmp_path / (event_id + "_stage1_multiframe.json"),
            {
                "schema": "orion.stage1_observation_uq_multiframe.v1",
                "control_influence": False,
                "event_package": {
                    "path": str(event_package),
                    "sha256": sha256_file(event_package),
                },
                "sequences": sequence_rows,
            },
        )
        report = _write_json(
            tmp_path / (event_id + "_factory.json"),
            {
                "schema": "orion.uq_relevance_multiframe_event_factory.v1",
                "event_id": event_id,
                "keyframe_count": 4,
                "event_package": {
                    "path": str(event_package),
                    "sha256": sha256_file(event_package),
                },
                "stage1_multiframe_manifest": {
                    "path": str(stage1_root),
                    "sha256": sha256_file(stage1_root),
                },
                "qa_factory_config": {
                    "path": str(other_config if qa_config_mismatch and index == 0 else qa_config),
                    "sha256": sha256_file(
                        other_config if qa_config_mismatch and index == 0 else qa_config
                    ),
                },
                "qa_dataset": {
                    "records": {
                        "path": str(records),
                        "sha256": sha256_file(records),
                    }
                },
            },
        )
        reports.append(report)
        cache = tmp_path / (event_id + "_visual.pt")
        cache.write_bytes(b"cache-" + bytes(str(index), "ascii"))
        cache_manifest = _write_json(
            tmp_path / (event_id + "_cache.json"),
            {
                "schema": "orion.stage2l_multiframe_visual_context_cache.v1",
                "status": "immutable_multiframe_visual_context_cache",
                "output": str(cache),
                "sha256": sha256_file(cache),
                "group_ids": groups,
                "event_factory_report": {
                    "path": str(report),
                    "sha256": sha256_file(report),
                },
                "orion_checkpoint_sha256": "4" * 64,
                "privileged_safety_inputs_used": False,
                "stage1_uq_inputs_used": False,
                "task_relevance_targets_used": False,
                "qa_answers_used": False,
            },
        )
        caches.append(cache_manifest)
        accepted.append(
            {
                "event_id": event_id,
                "factory_report": {
                    "path": str(report),
                    "sha256": sha256_file(report),
                },
            }
        )
    plan = _write_json(
        tmp_path / "formal_plan.json",
        {
            "schema": "orion.stage2_l.formal_route_plan.v1",
            "events": plan_events,
            "expected_qa_records_after_geometry_gate": [1680, 2320],
        },
    )
    bank = _write_json(
        tmp_path / "formal_bank.json",
        {
            "schema": "orion.stage2_l.formal_event_bank.v1",
            "status": "formal_event_bank_complete_reviewed",
            "checks": {"all_24_planned_routes_present": True},
            "events": events,
            "provenance": {
                "formal_route_plan": {
                    "path": str(plan),
                    "sha256": sha256_file(plan),
                }
            },
        },
    )
    review = _write_json(
        tmp_path / "review.json",
        {
            "schema": "orion.stage2_l.qa_geometry_review_bank.v1",
            "status": "frozen_human_qa_geometry_review",
            "accepted": accepted,
            "rejected": [],
        },
    )
    protocol = _write_json(
        tmp_path / "data_protocol.json",
        {
            "schema": "orion.stage2_l.formal_data_and_corruption_protocol.v1",
            "status": "frozen_data_and_corruption_isolation_training_locked",
            "stage1_signal_role": {
                "checkpoint": {"sha256": stage1_checkpoint_sha256},
                "checkpoint_update_during_stage2_l": False,
            },
            "visual_context_cache": {
                "orion_checkpoint": {"sha256": "4" * 64},
                "stage1_uq_inputs_used": False,
                "task_relevance_targets_used": False,
                "qa_answers_used": False,
                "privileged_safety_inputs_used": False,
                "llm_run_during_cache": False,
                "trajectory_decoder_run_during_cache": False,
            },
            "frozen_event_split": {
                "formal_route_plan": {
                    "path": str(plan),
                    "sha256": sha256_file(plan),
                }
            },
            "fixed_qa_construction": {
                "source_condition": "clean_off closed-loop traces",
                "formal_qa_factory": {
                    "path": str(qa_config),
                    "sha256": sha256_file(qa_config),
                }
            },
            "validated_sources": {
                "stage2l_schedule": {
                    "path": str(schedule),
                    "sha256": sha256_file(schedule),
                }
            },
            "corruption_family_isolation": {
                "stage2_l_image_training_corruptions": [],
                "stage2_l_training_statement": (
                    "Formal Stage2-L semantic training uses clean visual observations"
                ),
                "formal_unseen_family_primary": {
                    "adapter_training_allowed": False,
                    "stage2_l_training_allowed": False,
                    "checkpoint_selection_allowed": False,
                },
            },
            "launch_locks": {
                "formal_stage2_l_allowed": False,
                "stage2_p_allowed": False,
                "closed_loop_matrix_allowed": False,
            },
        },
    )
    if incomplete_protocol:
        payload = json.loads(protocol.read_text(encoding="utf-8"))
        payload.pop("corruption_family_isolation")
        protocol.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "formal_bank_path": bank,
        "schedule_path": schedule,
        "formal_data_protocol_path": protocol,
        "qa_factory_config_path": qa_config,
        "qa_review_bank_path": review,
        "factory_reports": reports,
        "visual_cache_manifests": caches,
    }


def test_assembles_frozen_formal_24_event_dataset(tmp_path):
    result = assemble_formal_dataset(**_fixture(tmp_path))
    assert result["status"] == "assembled_formal_data_training_launch_locked"
    assert result["formal_training_ready"] is False
    assert result["stage2p_allowed"] is False
    assert result["event_count"] == 24
    assert result["qa_record_count"] == 1920
    assert result["qa_split_counts"] == {
        "dev": 320,
        "test": 320,
        "train": 1280,
    }
    assert result["town_count"] == 6
    assert result["scenario_family_count"] == 9
    assert result["checks"]["frozen_16_4_4_split_preserved"] is True


def test_rejects_formal_split_drift(tmp_path):
    with pytest.raises(ValueError, match="16/4/4 split"):
        assemble_formal_dataset(**_fixture(tmp_path, altered_split=True))


def test_rejects_nonfrozen_qa_factory_config(tmp_path):
    with pytest.raises(ValueError, match="different QA factory config"):
        assemble_formal_dataset(**_fixture(tmp_path, qa_config_mismatch=True))


def test_rejects_protocol_without_corruption_isolation_contract(tmp_path):
    with pytest.raises(ValueError, match="forbid Stage2-L image corruptions"):
        assemble_formal_dataset(**_fixture(tmp_path, incomplete_protocol=True))


def test_rejects_native_glare_source_run(tmp_path):
    with pytest.raises(ValueError, match="clean render condition"):
        assemble_formal_dataset(**_fixture(tmp_path, native_glare=True))


def test_rejects_stage1_checkpoint_drift(tmp_path):
    with pytest.raises(ValueError, match="wrong frozen checkpoint"):
        assemble_formal_dataset(
            **_fixture(tmp_path, stage1_checkpoint_mismatch=True)
        )


def test_rejects_legacy_qa_records_under_v5_config(tmp_path):
    with pytest.raises(ValueError, match="legacy or non-v5 QA records"):
        assemble_formal_dataset(**_fixture(tmp_path, legacy_qa_records=True))
