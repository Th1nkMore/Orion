import pytest

from scripts.prepare_stage2l_expanded_coverage_dataset import (
    _validate_inventory_and_bank,
)


def _event(index, split):
    return {
        "event_id": "route%d_step10" % index,
        "route_index": index,
        "split": split,
    }


def _bank_event(index, split):
    return {
        "event_id": "route%d_step10" % index,
        "route_index": index,
        "formal_split": split,
        "qa_input_ready": True,
        "human_review": {"decision": "accept"},
    }


def _inventory(events):
    return {
        "schema": "orion.stage2_l.formal_inventory_audit.v1",
        "status": "audited_available_subset_training_locked",
        "formal_training_ready": False,
        "stage2p_allowed": False,
        "audited_event_count": len(events),
        "events": events,
        "checks": {
            "all_available_factories_pass_v5_qa_contract": True,
            "all_available_caches_cover_their_qa_groups": True,
            "all_available_cache_lineage_is_current_or_reattested": True,
            "all_available_source_runs_are_clean_off": True,
            "all_available_stage1_sequences_use_frozen_checkpoint": True,
            "training_remains_fail_closed": True,
        },
    }


def test_expanded_dataset_preserves_reviewed_train_dev_only():
    events = [_event(i, "train" if i < 6 else "dev") for i in range(8)]
    bank = {
        "schema": "orion.stage2_l.formal_event_bank.v1",
        "events": [_bank_event(i, "train" if i < 6 else "dev") for i in range(8)],
    }
    indexed, bank_indexed = _validate_inventory_and_bank(_inventory(events), bank)
    assert len(indexed) == len(bank_indexed) == 8


def test_expanded_dataset_rejects_test_or_unreviewed_event():
    events = [_event(i, "train" if i < 6 else "dev") for i in range(8)]
    bank_events = [_bank_event(i, "train" if i < 6 else "dev") for i in range(8)]
    bank = {"schema": "orion.stage2_l.formal_event_bank.v1", "events": bank_events}
    events[7]["split"] = "test"
    bank_events[7]["formal_split"] = "test"
    with pytest.raises(ValueError, match="unreviewed/test/mismatched"):
        _validate_inventory_and_bank(_inventory(events), bank)
    events[7]["split"] = "dev"
    bank_events[7]["formal_split"] = "dev"
    bank_events[7]["human_review"]["decision"] = "pending"
    with pytest.raises(ValueError, match="unreviewed/test/mismatched"):
        _validate_inventory_and_bank(_inventory(events), bank)
