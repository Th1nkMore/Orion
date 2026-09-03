import json
import xml.etree.ElementTree as ET

import pytest

from scripts.freeze_stage2l_formal_route_plan import (
    CONFIG_SCHEMA,
    PILOT_BANK_SCHEMA,
    PILOT_DATASET_SCHEMA,
    freeze_formal_route_plan,
)


SCENARIOS = (
    "DynamicObjectCrossing",
    "ParkingCutIn",
    "HazardAtSideLane",
    "OppositeVehicleRunningRedLight",
)


def _write_route(path, route_id, town, scenario_type):
    root = ET.Element("routes")
    route = ET.SubElement(root, "route", id=str(route_id), town=town)
    waypoints = ET.SubElement(route, "waypoints")
    ET.SubElement(waypoints, "position", x="0", y="0", z="0")
    ET.SubElement(waypoints, "position", x="30", y="0", z="0")
    scenarios = ET.SubElement(route, "scenarios")
    scenario = ET.SubElement(
        scenarios, "scenario", name=scenario_type + "_1", type=scenario_type
    )
    ET.SubElement(scenario, "trigger_point", x="8", y="0", z="0", yaw="0")
    ET.ElementTree(root).write(path)


def _inputs(tmp_path):
    pilot_events = []
    pilot_families = (
        "HardBreakRoute", "Accident", "ConstructionObstacle",
        "ParkedObstacle", "StaticCutIn", "T_Junction",
        "HardBreakRoute", "Accident",
    )
    towns = ("Town01", "Town02", "Town03", "Town04", "Town05")
    for index in range(1, 9):
        pilot_events.append({
            "event_id": "route%d_step1" % index,
            "route_index": index,
            "pilot_split": "train" if index <= 6 else "dev",
            "town": towns[(index - 1) % len(towns)],
            "scenario_family": pilot_families[index - 1],
            "split_origin": "development_screen",
        })
    pilot_bank = {
        "schema": PILOT_BANK_SCHEMA,
        "status": "frozen_before_stage2l_pilot_training",
        "events": pilot_events,
    }
    pilot_dataset = {
        "schema": PILOT_DATASET_SCHEMA,
        "status": "assembled_ready_for_stage2l_pilot_training",
        "event_count": 8,
        "qa_record_count": 720,
        "events": [{"event_id": row["event_id"]} for row in pilot_events],
    }

    routes = tmp_path / "routes"
    routes.mkdir()
    development = []
    locked_test = []
    baseline = {}
    for offset, index in enumerate(range(100, 116)):
        scenario = SCENARIOS[offset % len(SCENARIOS)]
        town = towns[offset % len(towns)]
        _write_route(
            routes / ("bench2drive220_%d_orion_traj.xml" % index),
            index,
            town,
            scenario,
        )
        if offset < 12:
            role = (
                "published_failure_hard_case"
                if index == 111 else "published_clean_valid"
            )
            development.append({
                "route_index": index,
                "scenario_type": scenario,
                "formal_split": "dev" if offset < 2 else "train",
                "development_selection_role": role,
            })
            baseline[str(index)] = {
                "status": "Completed",
                "scores": {
                    "score_route": 100,
                    "score_penalty": 0.6 if index == 111 else 1,
                },
                "infractions": {
                    "collisions_vehicle": ["collision"] if index == 111 else []
                },
            }
        else:
            locked_test.append({
                "route_index": index,
                "scenario_type": scenario,
                "formal_split": "test",
            })
    config = {
        "schema": CONFIG_SCHEMA,
        "additions": {"development": development, "locked_test": locked_test},
        "formal_gate": {
            "events": 24,
            "minimum_towns": 5,
            "minimum_scenario_families": 8,
            "split": {"train": 16, "dev": 4, "test": 4},
            "qa_record_range": [1500, 2500],
        },
    }
    amendment = {
        "schema": "orion.scenario_factory.amendment.v1",
        "allowed_development_failure_routes": [111],
        "launch_locks": {"formal_stage2l_training_allowed": False},
    }
    return config, pilot_bank, pilot_dataset, routes, baseline, amendment


def test_freeze_formal_plan_meets_split_diversity_and_qa_contract(tmp_path):
    values = _inputs(tmp_path)

    result = freeze_formal_route_plan(
        config=values[0], pilot_bank=values[1], pilot_dataset=values[2],
        routes_dir=values[3], baseline=values[4], failure_amendment=values[5],
    )

    plan = result["formal_plan"]
    assert plan["counts"]["events"] == 24
    assert plan["counts"]["splits"] == {"dev": 4, "test": 4, "train": 16}
    assert plan["expected_qa_records_after_geometry_gate"] == [1680, 2320]
    assert plan["formal_training_ready"] is False
    assert result["development_candidates"]["candidate_count"] == 12
    assert result["development_candidates"][
        "development_failure_candidates_allowed"
    ] is True
    assert result["locked_test_candidates"]["candidate_count"] == 4
    assert all(
        row["clean_baseline"]["available"] is False
        for row in result["locked_test_candidates"]["candidates"]
    )


def test_unamended_development_failure_is_rejected(tmp_path):
    values = list(_inputs(tmp_path))
    values[5] = {
        "schema": "orion.scenario_factory.amendment.v1",
        "allowed_development_failure_routes": [],
        "launch_locks": {"formal_stage2l_training_allowed": False},
    }

    with pytest.raises(ValueError, match="not explicitly amended"):
        freeze_formal_route_plan(
            config=values[0], pilot_bank=values[1], pilot_dataset=values[2],
            routes_dir=values[3], baseline=values[4], failure_amendment=values[5],
        )
