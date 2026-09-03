import json

import pytest

from scripts.merge_scenario_event_banks import merge_event_banks


def _bank(path, event_id, route_id, town, family):
    path.write_text(json.dumps({
        "schema": "orion.scenario_event_bank.v1",
        "events": [{
            "event_id": event_id,
            "route_index": route_id,
            "town": town,
            "scenario_family": family,
            "split_origin": "development_screen",
            "stage2_split": "train",
            "human_review": {"decision": "accept"},
        }],
        "rejected_events": [],
    }))
    return path


def test_merge_strips_stale_splits_and_records_source_hashes(tmp_path):
    first = _bank(tmp_path / "a.json", "route1_step10", 1, "Town01", "A")
    second = _bank(tmp_path / "b.json", "route2_step20", 2, "Town02", "B")

    merged = merge_event_banks([first, second])

    assert merged["counts"]["accepted_events"] == 2
    assert merged["counts"]["towns"] == 2
    assert all("stage2_split" not in row for row in merged["events"])
    assert len(merged["provenance"]["source_event_banks"]) == 2


def test_merge_rejects_duplicate_routes(tmp_path):
    first = _bank(tmp_path / "a.json", "route1_step10", 1, "Town01", "A")
    second = _bank(tmp_path / "b.json", "route1_step20", 1, "Town02", "B")

    with pytest.raises(ValueError, match="route id"):
        merge_event_banks([first, second])
