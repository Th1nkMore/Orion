import json

import pytest

from scripts.audit_stage2l_v12_train_coverage_candidates import audit


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _inputs(tmp_path):
    plan = {
        "schema": "orion.stage2_l.formal_route_plan.v1",
        "events": [
            {"route_index": 177, "formal_split": "train"},
            {"route_index": 201, "formal_split": "train"},
            {"route_index": 208, "formal_split": "train"},
            {"route_index": 147, "formal_split": "dev"},
            {"route_index": 145, "formal_split": "test"},
        ],
    }
    bank = {
        "schema": "orion.stage2_l.formal_event_bank.v1",
        "events": [{"route_index": 177, "formal_split": "train"}],
    }
    reviewed = {"events": [{"route_index": 177, "formal_split": "train"}]}
    gate177 = {
        "status": "formal_event_qa_ineligible_under_frozen_geometry_gate",
        "source": {"formal_split": "train"},
        "validation": {
            "retained_geometry_valid_keyframes": 2,
            "minimum_required_keyframes": 3,
            "failure": "too few valid keyframes",
        },
    }
    hold201 = {
        "status": "route201_runtime_retry_exhausted_formal_training_hold",
        "route": {"formal_split": "train"},
        "decision": {"resolution_requires": "new clean runtime"},
    }
    hold208 = {
        "status": "route208_clean_liveness_fast_screen_triggered_stage_b_ineligible",
        "route": {"formal_split": "dev"},
    }
    return {
        "formal_plan_path": _write(tmp_path / "plan.json", plan),
        "accepted_bank_path": _write(tmp_path / "bank.json", bank),
        "route177_reviewed_path": _write(tmp_path / "reviewed.json", reviewed),
        "route177_geometry_gate_path": _write(tmp_path / "gate177.json", gate177),
        "route201_hold_path": _write(tmp_path / "hold201.json", hold201),
        "route208_hold_path": _write(tmp_path / "hold208.json", hold208),
    }


def test_audit_closes_gpu_and_never_reads_locked_test_results(tmp_path):
    result = audit(**_inputs(tmp_path))
    assert result["passed"] is True
    assert result["formal_identity_inventory"]["missing_train_routes"] == [201, 208]
    assert result["coverage_repair"]["current_candidates_with_eligible_r_geometry"] == []
    assert result["inspection_boundary"]["locked_test_result_files_read"] == []
    assert result["lineage_discrepancies"] == [
        {
            "route_index": 208,
            "authoritative_formal_plan_split": "train",
            "legacy_amendment_split_label": "dev",
            "disposition": (
                "recorded_clerical_lineage_mismatch; formal plan owns split and "
                "the route-specific technical exclusion remains applicable"
            ),
        }
    ]
    assert result["next_authorized_work"]["gpu_r_only_smoke"] is False


def test_audit_rejects_changed_missing_train_set(tmp_path):
    inputs = _inputs(tmp_path)
    bank_path = inputs["accepted_bank_path"]
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    bank["events"].append({"route_index": 201, "formal_split": "train"})
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    with pytest.raises(ValueError, match="missing-train set differs"):
        audit(**inputs)
