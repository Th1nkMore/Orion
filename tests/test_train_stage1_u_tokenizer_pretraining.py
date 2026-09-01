import json

import pytest

from scripts.train_stage1_u_tokenizer_pretraining import _validate_protocol


def _protocol(audit_sha):
    return {
        "schema": "orion.stage1_u_tokenizer_pretraining_protocol.v1",
        "training_inputs": ["normalized_frozen_stage1_uncertainty_components"],
        "forbidden_inputs": {
            "route_context": True,
            "task_relevance": True,
            "qa_text_or_fields": True,
            "ttc_collision_or_control": True,
            "corruption_metadata": True,
        },
        "source_audit": {"sha256": audit_sha},
        "launch_locks": {"stage2l_v10_training_allowed": False},
    }


def test_protocol_validation_binds_source_and_forbids_task_inputs(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"passed": True}), encoding="utf-8")
    import hashlib
    sha = hashlib.sha256(audit.read_bytes()).hexdigest()
    _validate_protocol(_protocol(sha), audit)


def test_protocol_validation_rejects_task_input_or_unlocked_stage2l(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text("{}", encoding="utf-8")
    import hashlib
    sha = hashlib.sha256(audit.read_bytes()).hexdigest()
    protocol = _protocol(sha)
    protocol["forbidden_inputs"]["task_relevance"] = False
    with pytest.raises(ValueError, match="prohibit"):
        _validate_protocol(protocol, audit)
    protocol = _protocol(sha)
    protocol["launch_locks"]["stage2l_v10_training_allowed"] = True
    with pytest.raises(ValueError, match="unlock"):
        _validate_protocol(protocol, audit)
