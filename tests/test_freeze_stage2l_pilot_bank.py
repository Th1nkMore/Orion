import json
from pathlib import Path

from scripts.freeze_stage2l_pilot_bank import freeze_pilot_bank
from scripts.scenario_factory_lib import sha256_file


def test_freezes_diverse_eight_event_six_two_pilot(tmp_path):
    project = tmp_path / "project"
    protocol = project / "configs" / "scenario_factory" / "protocol_v1.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("{}")
    schedule = protocol.parent / "stage2l_schedule_v1.json"
    schedule.write_text(json.dumps({
        "schema": "orion.stage2_l.schedule.v1",
        "base_scenario_factory_protocol": {
            "path": "configs/scenario_factory/protocol_v1.json",
            "sha256": sha256_file(protocol),
        },
        "pilot_gate": {
            "minimum_independent_events": 8,
            "minimum_towns": 3,
            "minimum_scenario_families": 4,
            "event_level_split": {"train": 6, "dev": 2, "test": 0},
        },
    }))
    events = []
    for index in range(8):
        events.append({
            "event_id": "route%d_step10" % index,
            "route_index": index,
            "town": "Town%02d" % (1 + index % 3),
            "scenario_family": "Family%d" % (index % 4),
            "split_origin": "development_screen",
        })
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({
        "schema": "orion.scenario_event_bank.v1",
        "events": events,
    }))
    result = freeze_pilot_bank(
        event_bank_path=bank,
        schedule_path=schedule,
        event_ids=[],
    )
    assert result["counts"] == {
        "events": 8,
        "towns": 3,
        "scenario_families": 4,
        "splits": {"train": 6, "dev": 2},
    }
    assert sum(row["pilot_split"] == "dev" for row in result["events"]) == 2
