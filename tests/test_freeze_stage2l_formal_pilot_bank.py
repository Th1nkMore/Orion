import json

import pytest

from scripts.freeze_stage2l_formal_pilot_bank import freeze_formal_pilot_bank
from scripts.scenario_factory_lib import sha256_file


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _event(route, split, town, family, *, collision_count=0):
    return {
        "event_id": "route%d_step%d" % (route, route + 10),
        "route_index": route,
        "town": town,
        "scenario_family": family,
        "formal_split": split,
        "split_origin": "development_screen",
        "qa_input_ready": True,
        "human_review": {"decision": "accept"},
        "event_package": {"sha256": ("%064x" % route)},
        "official_endpoint": {"collision_count": collision_count},
    }


def _fixture(tmp_path):
    project = tmp_path / "project"
    protocol = _write(
        project / "configs/scenario_factory/protocol_v1.json", {"version": 1}
    )
    rows = [
        (1, "dev", "Town01", "Pedestrian"),
        (2, "dev", "Town02", "RedLight"),
        (10, "train", "Town01", "CutIn"),
        (11, "train", "Town02", "HardBrake"),
        (12, "train", "Town03", "SideLane"),
        (13, "train", "Town04", "Crossing"),
        (14, "train", "Town05", "SideLane"),
        (15, "train", "Town05", "TwoWays"),
        (16, "train", "Town04", "RedLight"),
    ]
    plan = _write(
        project / "results/formal_plan.json",
        {
            "schema": "orion.stage2_l.formal_route_plan.v1",
            "events": [
                {
                    "route_index": route,
                    "formal_split": split,
                    "town": town,
                    "scenario_family": family,
                    "split_origin": "development_screen",
                }
                for route, split, town, family in rows
            ],
        },
    )
    schedule = _write(
        project / "configs/scenario_factory/schedule.json",
        {
            "schema": "orion.stage2_l.schedule.v2",
            "base_scenario_factory_protocol": {
                "path": str(protocol),
                "sha256": sha256_file(protocol),
            },
            "formal_route_plan": {
                "path": str(plan),
                "sha256": sha256_file(plan),
            },
            "pilot_gate": {
                "minimum_independent_events": 8,
                "minimum_towns": 3,
                "minimum_scenario_families": 4,
                "preserve_formal_route_plan_splits": True,
                "geometry_eligibility": {"minimum_retained_keyframes": 3},
                "event_level_split": {"train": 6, "dev": 2, "test": 0},
            },
        },
    )
    events = [
        _event(route, split, town, family, collision_count=int(route == 16))
        for route, split, town, family in rows
    ]
    waves = []
    for index, shard in enumerate((events[:3], events[3:6], events[6:])):
        waves.append(_write(
            project / ("wave%d.json" % index),
            {
                "schema": "orion.stage2_l.formal_reviewed_wave.v1",
                "events": shard,
                "provenance": {
                    "formal_route_plan": {"sha256": sha256_file(plan)}
                },
            },
        ))
    preflights = []
    for event in events:
        preflights.append(_write(
            project / (event["event_id"] + ".geometry.json"),
            {
                "schema": "orion.stage2l_event_geometry_preflight.v1",
                "status": "eligible_before_stage1_extraction",
                "eligible": True,
                "event_id": event["event_id"],
                "retained_keyframe_count": 5,
                "minimum_retained_keyframes": 3,
                "provenance": {
                    "event_package": event["event_package"],
                },
            },
        ))
    return plan, schedule, waves, preflights


def test_preserves_formal_splits_and_uses_plan_order_not_outcomes(tmp_path):
    plan, schedule, waves, preflights = _fixture(tmp_path)

    result = freeze_formal_pilot_bank(
        formal_plan_path=plan,
        schedule_path=schedule,
        reviewed_wave_paths=waves,
        geometry_preflight_paths=preflights,
    )

    assert result["counts"] == {
        "events": 8,
        "towns": 5,
        "scenario_families": 7,
        "splits": {"train": 6, "dev": 2},
        "reviewed_reserves": 1,
        "technical_geometry_exclusions": 0,
    }
    assert [row["route_index"] for row in result["events"]] == [
        1, 2, 10, 11, 12, 13, 14, 15
    ]
    assert result["reviewed_reserve_events"][0]["route_index"] == 16
    assert result["reviewed_reserve_events"][0]["official_endpoint"][
        "collision_count"
    ] == 1
    assert all(row["pilot_split"] == row["formal_split"] for row in result["events"])
    assert result["pilot_training_ready"] is False
    assert result["selection_policy"]["uses_collision_or_ttc"] is False


def test_rejects_reviewed_wave_bound_to_another_plan(tmp_path):
    plan, schedule, waves, preflights = _fixture(tmp_path)
    value = json.loads(waves[0].read_text())
    value["provenance"]["formal_route_plan"]["sha256"] = "0" * 64
    waves[0].write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="different formal plan"):
        freeze_formal_pilot_bank(
            formal_plan_path=plan,
            schedule_path=schedule,
            reviewed_wave_paths=waves,
            geometry_preflight_paths=preflights,
        )


def test_rejects_locked_test_event_in_pilot_sources(tmp_path):
    plan, schedule, waves, preflights = _fixture(tmp_path)
    plan_value = json.loads(plan.read_text())
    plan_value["events"][0]["formal_split"] = "test"
    plan.write_text(json.dumps(plan_value), encoding="utf-8")
    schedule_value = json.loads(schedule.read_text())
    schedule_value["formal_route_plan"]["sha256"] = sha256_file(plan)
    schedule.write_text(json.dumps(schedule_value), encoding="utf-8")
    for wave in waves:
        value = json.loads(wave.read_text())
        value["provenance"]["formal_route_plan"]["sha256"] = sha256_file(plan)
        if value["events"][0]["route_index"] == 1:
            value["events"][0]["formal_split"] = "test"
        wave.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="locked test"):
        freeze_formal_pilot_bank(
            formal_plan_path=plan,
            schedule_path=schedule,
            reviewed_wave_paths=waves,
            geometry_preflight_paths=preflights,
        )


def test_rejects_insufficient_preserved_split_quota(tmp_path):
    plan, schedule, waves, preflights = _fixture(tmp_path)
    value = json.loads(waves[-1].read_text())
    value["events"] = value["events"][:1]
    waves[-1].write_text(json.dumps(value), encoding="utf-8")
    preflights = preflights[:-2]

    with pytest.raises(ValueError, match="split quotas"):
        freeze_formal_pilot_bank(
            formal_plan_path=plan,
            schedule_path=schedule,
            reviewed_wave_paths=waves,
            geometry_preflight_paths=preflights,
        )


def test_skips_only_hash_bound_geometry_failure_then_uses_next_plan_event(tmp_path):
    plan, schedule, waves, preflights = _fixture(tmp_path)
    value = json.loads(preflights[2].read_text())
    value.update({
        "status": "ineligible_before_stage1_extraction",
        "eligible": False,
        "retained_keyframe_count": 2,
    })
    preflights[2].write_text(json.dumps(value), encoding="utf-8")

    result = freeze_formal_pilot_bank(
        formal_plan_path=plan,
        schedule_path=schedule,
        reviewed_wave_paths=waves,
        geometry_preflight_paths=preflights,
    )

    assert [row["route_index"] for row in result["events"]] == [
        1, 2, 11, 12, 13, 14, 15, 16
    ]
    assert result["counts"]["technical_geometry_exclusions"] == 1
    assert result["technical_geometry_exclusions"][0]["route_index"] == 10


def test_requires_geometry_preflight_for_every_reviewed_event(tmp_path):
    plan, schedule, waves, preflights = _fixture(tmp_path)
    with pytest.raises(ValueError, match="exactly cover"):
        freeze_formal_pilot_bank(
            formal_plan_path=plan,
            schedule_path=schedule,
            reviewed_wave_paths=waves,
            geometry_preflight_paths=preflights[:-1],
        )
