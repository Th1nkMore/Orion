import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "select_closedloop_heldout_routes.py"
SPEC = importlib.util.spec_from_file_location("route_selector", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_route(path: Path, scenario_type: str, town: str = "Town04"):
    root = ET.Element("routes")
    route = ET.SubElement(root, "route", id="42", town=town)
    waypoints = ET.SubElement(route, "waypoints")
    ET.SubElement(waypoints, "position", x="0", y="0", z="0")
    ET.SubElement(waypoints, "position", x="20", y="0", z="0")
    scenarios = ET.SubElement(route, "scenarios")
    scenario = ET.SubElement(
        scenarios, "scenario", name=scenario_type + "_1", type=scenario_type
    )
    ET.SubElement(scenario, "trigger_point", x="5", y="0", z="0", yaw="0")
    ET.ElementTree(root).write(path)


def test_secondary_static_and_negative_scenarios_are_explicitly_screenable(tmp_path):
    route_path = tmp_path / "bench2drive220_12_orion_traj.xml"
    _write_route(route_path, "ConstructionObstacle")
    baseline = {
        "42": {
            "status": "Completed",
            "scores": {"score_route": 100, "score_penalty": 1},
            "infractions": {},
        }
    }

    rows = MODULE.parse_route(route_path, baseline)

    assert len(rows) == 1
    assert rows[0]["scenario_type"] == "ConstructionObstacle"
    assert rows[0]["screen_role"] == "static_visual_hazard"
    assert rows[0]["clean_baseline"]["valid"] is True


def test_control_loss_is_labeled_non_perceptual_hard_negative(tmp_path):
    route_path = tmp_path / "bench2drive220_13_orion_traj.xml"
    _write_route(route_path, "ControlLoss")

    rows = MODULE.parse_route(route_path, {})

    assert rows[0]["screen_role"] == "non_perceptual_control_hard_negative"


def test_town03_native_scenarios_are_screenable(tmp_path):
    expected = {
        "PedestrianCrossing": "dynamic_path_conflict",
        "Accident": "static_visual_hazard",
        "ParkedObstacleTwoWays": "static_visual_hazard_with_opposing_flow",
    }
    for index, (scenario, role) in enumerate(expected.items(), start=20):
        route_path = tmp_path / ("bench2drive220_%d_orion_traj.xml" % index)
        _write_route(route_path, scenario, town="Town03")
        rows = MODULE.parse_route(route_path, {})
        assert len(rows) == 1
        assert rows[0]["town"] == "Town03"
        assert rows[0]["screen_role"] == role


def test_cli_without_baseline_attests_locked_test_selection_inputs(tmp_path, capsys):
    _write_route(tmp_path / "bench2drive220_12_orion_traj.xml", "ConstructionObstacle")

    assert MODULE.main(["--routes-dir", str(tmp_path), "--limit", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["locked_test_selection_eligible"] is True
    assert payload["selection_inputs"]["published_orion_outcomes_used"] is False
    assert payload["selection_inputs"]["learned_uq_outcomes_used"] is False
    assert payload["selection_inputs"]["stage2_outcomes_used"] is False
