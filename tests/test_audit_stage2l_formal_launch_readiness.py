import hashlib
import json
from pathlib import Path

from scripts.audit_stage2l_formal_launch_readiness import audit_readiness


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _data_protocol():
    return {
        "schema": "orion.stage2_l.formal_data_and_corruption_protocol.v1",
        "status": "frozen_data_and_corruption_isolation_training_locked",
        "corruption_family_isolation": {
            "stage2_l_image_training_corruptions": [],
            "stage2_l_training_statement": "Formal Stage2-L uses clean visual observations",
            "formal_unseen_family_primary": {
                "adapter_training_allowed": False,
                "stage2_l_training_allowed": False,
                "checkpoint_selection_allowed": False,
            },
        },
        "stage1_signal_role": {
            "checkpoint": {"sha256": "1" * 64},
            "checkpoint_update_during_stage2_l": False,
        },
        "visual_context_cache": {
            "orion_checkpoint": {"sha256": "2" * 64},
            "stage1_uq_inputs_used": False,
            "task_relevance_targets_used": False,
            "qa_answers_used": False,
            "privileged_safety_inputs_used": False,
            "llm_run_during_cache": False,
            "trajectory_decoder_run_during_cache": False,
        },
        "fixed_qa_construction": {"source_condition": "clean_off traces"},
        "launch_locks": {
            "formal_stage2_l_allowed": False,
            "stage2_p_allowed": False,
            "closed_loop_matrix_allowed": False,
        },
    }


def _comparison(decision="duration_overfit_stop"):
    return {
        "schema": "orion.stage2l_mr1_duration_comparison.v1",
        "status": "mr1_duration_compared_no_training_launched",
        "decision": decision,
        "controlled_comparison": {"valid": True},
        "overfit_diagnostic": {
            "gate_count_overfit_established": decision == "duration_overfit_stop"
        },
        "next_action": "increase event/class coverage",
    }


def test_current_subset_and_duration_overfit_fail_closed(tmp_path):
    protocol_path = _write(tmp_path / "protocol.json", _data_protocol())
    comparison_path = _write(tmp_path / "comparison.json", _comparison())
    inventory = {
        "schema": "orion.stage2_l.formal_inventory_audit.v1",
        "audited_event_count": 17,
        "qa_record_count": 1600,
        "events": [
            {"event_id": "route%d" % i, "split": "train"}
            for i in range(13)
        ]
        + [
            {"event_id": "dev%d" % i, "split": "dev"}
            for i in range(4)
        ],
        "frozen_plan_routes_without_audited_v5": [
            {"route_index": value} for value in (145, 173, 177, 201, 206, 208, 212)
        ],
        "reviewed_bank_events_without_usable_v5": ["route177_step276"],
        "remaining_gates": {
            "formal_v5_qa_geometry_review_bank_complete": False
        },
        "provenance": {
            "formal_data_protocol": {"sha256": _sha(protocol_path)}
        },
    }
    inventory_path = _write(tmp_path / "inventory.json", inventory)
    result = audit_readiness(
        data_source_path=inventory_path,
        formal_data_protocol_path=protocol_path,
        mr1_duration_comparison_path=comparison_path,
    )
    assert result["formal_training_allowed"] is False
    assert result["stage2p_allowed"] is False
    assert result["training_started"] is False
    assert result["data"]["event_count"] == 17
    assert result["data"]["event_split_counts"] == {"dev": 4, "train": 13}
    assert result["mr1_duration_diagnostic"]["overfit_established"] is True
    assert "exactly_24_events" in result["failed_gates"]
    assert (
        "mr1_e_supports_multievent_learning_without_duration_overfit"
        in result["failed_gates"]
    )
    assert "formal_training_protocol_frozen" in result["failed_gates"]
    assert "immutable_single_run_launch_amendment_valid" in result["failed_gates"]


def test_protocol_hash_drift_is_reported_without_unlock(tmp_path):
    protocol_path = _write(tmp_path / "protocol.json", _data_protocol())
    comparison_path = _write(
        tmp_path / "comparison.json",
        _comparison("engineering_multievent_paradigm_passes"),
    )
    inventory_path = _write(
        tmp_path / "inventory.json",
        {
            "schema": "orion.stage2_l.formal_inventory_audit.v1",
            "audited_event_count": 24,
            "qa_record_count": 2000,
            "events": (
                [{"event_id": "t%d" % i, "split": "train"} for i in range(16)]
                + [{"event_id": "d%d" % i, "split": "dev"} for i in range(4)]
                + [{"event_id": "x%d" % i, "split": "test"} for i in range(4)]
            ),
            "frozen_plan_routes_without_audited_v5": [],
            "reviewed_bank_events_without_usable_v5": [],
            "remaining_gates": {
                "formal_v5_qa_geometry_review_bank_complete": True
            },
            "provenance": {"formal_data_protocol": {"sha256": "0" * 64}},
        },
    )
    result = audit_readiness(
        data_source_path=inventory_path,
        formal_data_protocol_path=protocol_path,
        mr1_duration_comparison_path=comparison_path,
    )
    assert result["checks"]["exactly_24_events"] is True
    assert result["checks"]["frozen_16_4_4_split_complete"] is True
    assert result["checks"]["source_protocol_hash_matches"] is False
    assert result["formal_training_allowed"] is False
