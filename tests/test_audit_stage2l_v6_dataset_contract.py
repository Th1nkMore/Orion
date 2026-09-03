import json

import pytest

from scripts.audit_stage2l_v6_dataset_contract import audit_dataset
from uq_estimator.stage2l_matched_objective import (
    MATCHED_VARIANTS,
    QUESTION_FAMILIES,
)


def _write_inputs(tmp_path, *, omit_last=False, training_allowed=False):
    rows = []
    for group_index in range(2):
        group_id = "event/frame%d" % group_index
        for variant in MATCHED_VARIANTS:
            for family in QUESTION_FAMILIES:
                rows.append({
                    "sample_id": "%s/%s/%s" % (group_id, variant, family),
                    "event_id": "event",
                    "counterfactual": {"group_id": group_id, "variant": variant},
                    "question_family": family,
                })
    if omit_last:
        rows.pop()
    records = tmp_path / "records.jsonl"
    records.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "schema": "orion.stage2l_uq_language_grounding_protocol.v2",
        "launch_locks": {
            "stage2l_pilot_training_allowed": training_allowed,
        },
    }), encoding="utf-8")
    return records, protocol


def test_offline_audit_counts_complete_optimizer_groups(tmp_path):
    records, protocol = _write_inputs(tmp_path)
    report = audit_dataset(records, protocol)
    assert report["status"] == "passed_offline_dataset_contract_training_still_locked"
    assert report["pilot_training_allowed"] is False
    assert report["counts"] == {
        "record_count": 40,
        "matched_group_count": 2,
        "optimizer_step_count_per_epoch": 2,
        "optimizer_steps_inside_group": 0,
        "hard_language_record_count": 36,
        "hard_stance_record_count": 6,
        "diagnostic_only_driving_record_count": 4,
        "cross_family_pairwise_comparison_count": 108,
    }


def test_offline_audit_rejects_partial_group_or_unlocked_protocol(tmp_path):
    records, protocol = _write_inputs(tmp_path, omit_last=True)
    with pytest.raises(ValueError, match="exactly 5x4"):
        audit_dataset(records, protocol)
    records, protocol = _write_inputs(tmp_path, training_allowed=True)
    with pytest.raises(ValueError, match="remain locked"):
        audit_dataset(records, protocol)
