import hashlib
import json

from scripts.preflight_stage2l_v8_route151 import preflight


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value)
    else:
        path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preflight_keeps_training_locked_after_all_cpu_checks(tmp_path):
    project = tmp_path / "project"
    source = project / "source.py"
    source_hash = _write(source, "pass\n")
    qa = (
        project
        / "configs"
        / "scenario_factory"
        / "qa_factory_v4_structured_semantics.json"
    )
    qa_hash = _write(qa, {"schema": "orion.uq_relevance_qa_factory_config.v4"})
    records = tmp_path / "dataset" / "records.jsonl"
    row = {
        "schema": "orion.uq_relevance_qa_record.v4",
        "target": {"qa_contract_schema": "orion.stage2l_qa_contract.v4"},
    }
    records.parent.mkdir(parents=True)
    records.write_text("".join(json.dumps(row) + "\n" for _ in range(100)))
    records_hash = hashlib.sha256(records.read_bytes()).hexdigest()
    audit_value = {
        "passed": True,
        "matched_group_count": 5,
        "hard_language_record_count": 90,
        "hard_stance_record_count": 15,
        "same_family_preference_anchor_count": 90,
        "distinct_counterfactual_negative_count": 270,
    }
    audit = tmp_path / "audit.json"
    audit_hash = _write(audit, audit_value)
    refs = tmp_path / "refs.json"
    refs_hash = _write(refs, {"passed": True})
    v7_report = tmp_path / "report.json"
    report_hash = _write(v7_report, {"status": "engineering_v7_calibrated_smoke_failed_gate"})
    v7_validation = tmp_path / "validation.json"
    validation_hash = _write(v7_validation, {
        "status": "validated_failed_gate", "integrity_valid": True, "smoke_passed": False
    })
    v7_diagnosis = tmp_path / "diagnosis.json"
    diagnosis_hash = _write(v7_diagnosis, {"status": "diagnosed_v7_gate_failure"})
    protocol = {
        "schema": "orion.stage2l_gradient_routed_training_protocol.v1",
        "implementation_sources": {"source.py": source_hash, "configs/scenario_factory/qa_factory_v4_structured_semantics.json": qa_hash},
        "route151_v8_dataset": {
            "records_sha256": records_hash, "audit_sha256": audit_hash,
            "reference_audit_sha256": refs_hash, "record_count": 100,
            "matched_group_count": 5, "hard_language_record_count": 90,
            "hard_stance_record_count": 15, "same_family_preference_anchor_count": 90,
            "distinct_counterfactual_negative_count": 270,
        },
        "triggering_v7_failure": {
            "report_sha256": report_hash,
            "independent_validation_sha256": validation_hash,
            "diagnosis_sha256": diagnosis_hash,
        },
        "gradient_ownership": {
            "qa_language_loss_to_relevance_logits": False,
            "qa_language_loss_to_stance_classifier": False,
            "ground_truth_stance_enters_forward": False,
        },
        "architecture": {
            "trajectory_training_enabled": False,
            "direct_control_training_enabled": False,
            "legacy_density_uq_used": False,
            "hard_governor_used": False,
        },
        "launch_locks": {
            "real_orion_smoke_allowed": False,
            "stage2l_pilot_training_allowed": False,
            "stage2p_allowed": False,
            "new_immutable_amendment_required": True,
        },
        "future_probe_bound_not_authorized": {"automatic_retry_or_extension": False},
        "cpu_preflight_tests": {"selected_test_result": "19 passed"},
    }
    protocol_path = tmp_path / "protocol.json"
    _write(protocol_path, protocol)
    result = preflight(
        protocol=protocol,
        qa_config={"schema": "orion.uq_relevance_qa_factory_config.v4"},
        dataset_audit=audit_value,
        reference_audit={"passed": True},
        v7_report={"status": "engineering_v7_calibrated_smoke_failed_gate"},
        v7_validation={"status": "validated_failed_gate", "integrity_valid": True, "smoke_passed": False},
        v7_diagnosis={"status": "diagnosed_v7_gate_failure"},
        project_root=project,
        records_path=records,
        protocol_path=protocol_path,
        qa_config_path=qa,
        dataset_audit_path=audit,
        reference_audit_path=refs,
        v7_report_path=v7_report,
        v7_validation_path=v7_validation,
        v7_diagnosis_path=v7_diagnosis,
    )
    assert result["passed"] is True
    assert result["training_started"] is False
    assert result["real_orion_smoke_authorized"] is False
