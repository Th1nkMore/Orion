import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_scenario_factory_batch.py"
SPEC = importlib.util.spec_from_file_location("scenario_batch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _route(path: Path):
    root = ET.Element("routes")
    route = ET.SubElement(root, "route", id="42", town="Town04")
    waypoints = ET.SubElement(route, "waypoints")
    ET.SubElement(waypoints, "position", x="0", y="0", z="0")
    ET.SubElement(waypoints, "position", x="20", y="0", z="0")
    scenarios = ET.SubElement(route, "scenarios")
    scenario = ET.SubElement(
        scenarios,
        "scenario",
        name="ConstructionObstacle_1",
        type="ConstructionObstacle",
    )
    ET.SubElement(scenario, "trigger_point", x="5", y="0", z="0", yaw="0")
    ET.ElementTree(root).write(path)


def _sources(tmp_path: Path):
    routes = tmp_path / "routes"
    routes.mkdir()
    source = routes / "bench2drive220_12_orion_traj.xml"
    _route(source)
    candidate = tmp_path / "candidates.json"
    candidate.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "route_index": 12,
                        "xml_route_id": "42",
                        "town": "Town04",
                        "scenario_type": "ConstructionObstacle",
                        "scenario_name": "ConstructionObstacle_1",
                        "screen_role": "static_visual_hazard",
                        "priority": 76,
                        "trigger_progress": 0.25,
                        "route_length_m": 20.0,
                        "source_xml": source.name,
                        "clean_baseline": {"valid": True, "status": "Completed"},
                    }
                ]
            }
        )
    )
    baseline = tmp_path / "ORION.json"
    baseline.write_text("{}")
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    return routes, candidate, baseline, protocol


def test_dry_run_builds_auditable_manifest_without_writing(tmp_path):
    routes, candidate, baseline, protocol = _sources(tmp_path)
    out = tmp_path / "batch"
    payload = MODULE.build_batch(
        candidate_manifest=candidate,
        source_routes_dir=routes,
        baseline_source=baseline,
        protocol=protocol,
        out_dir=out,
        run_id="scenario_factory_wave0",
        route_indices=[12],
        limit=None,
        writes_performed=False,
    )
    assert not out.exists()
    assert payload["route_count"] == 1
    assert payload["routes"][0]["screen_role"] == "static_visual_hazard"
    assert payload["runtime_contract"]["stage1_adapter_control_influence"] is False
    assert payload["audit"]["jobs_submitted"] is False


def test_real_build_writes_hazard_nohazard_and_refuses_overwrite(tmp_path):
    routes, candidate, baseline, protocol = _sources(tmp_path)
    out = tmp_path / "batch"
    MODULE.build_batch(
        candidate_manifest=candidate,
        source_routes_dir=routes,
        baseline_source=baseline,
        protocol=protocol,
        out_dir=out,
        run_id="scenario_factory_wave0",
        route_indices=[12],
        limit=None,
        writes_performed=True,
    )
    assert (out / "batch_manifest.json").is_file()
    assert len(ET.parse(out / "route_12_hazard.xml").getroot().findall(".//scenario")) == 1
    assert len(ET.parse(out / "route_12_nohazard.xml").getroot().findall(".//scenario")) == 0
    try:
        MODULE.build_batch(
            candidate_manifest=candidate,
            source_routes_dir=routes,
            baseline_source=baseline,
            protocol=protocol,
            out_dir=out,
            run_id="scenario_factory_wave0",
            route_indices=[12],
            limit=None,
            writes_performed=True,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("non-empty batch overwrite was not rejected")


def test_locked_test_requires_outcome_independent_selection(tmp_path):
    routes, candidate, _, protocol = _sources(tmp_path)
    payload = json.loads(candidate.read_text())
    payload["selection_inputs"] = {
        "published_orion_outcomes_used": True,
        "learned_uq_outcomes_used": False,
        "stage2_outcomes_used": False,
    }
    candidate.write_text(json.dumps(payload))

    try:
        MODULE.build_batch(
            candidate_manifest=candidate,
            source_routes_dir=routes,
            baseline_source=None,
            protocol=protocol,
            out_dir=tmp_path / "locked",
            run_id="locked_wave0",
            route_indices=[12],
            limit=None,
            writes_performed=False,
            split="locked_test",
        )
    except ValueError as error:
        assert "must not use" in str(error)
    else:
        raise AssertionError("outcome-informed locked-test selection was accepted")


def test_locked_test_can_be_frozen_without_prior_outcomes(tmp_path):
    routes, candidate, _, protocol = _sources(tmp_path)
    payload = json.loads(candidate.read_text())
    payload["selection_inputs"] = {
        "published_orion_outcomes_used": False,
        "learned_uq_outcomes_used": False,
        "stage2_outcomes_used": False,
    }
    payload["candidates"][0]["clean_baseline"] = {
        "available": False,
        "valid": False,
    }
    candidate.write_text(json.dumps(payload))

    result = MODULE.build_batch(
        candidate_manifest=candidate,
        source_routes_dir=routes,
        baseline_source=None,
        protocol=protocol,
        out_dir=tmp_path / "locked",
        run_id="locked_wave0",
        route_indices=[12],
        limit=None,
        writes_performed=False,
        split="locked_test",
    )

    assert result["split"] == "locked_test"
    assert result["audit"]["eligible_for_locked_test_claim"] is True
    assert result["audit"]["selection_uses_published_orion_outcomes"] is False


def test_explicit_development_failure_candidate_is_allowed_but_audited(tmp_path):
    routes, candidate, baseline, protocol = _sources(tmp_path)
    payload = json.loads(candidate.read_text())
    payload["selection_inputs"] = {
        "published_orion_outcomes_used": True,
        "learned_uq_outcomes_used": False,
        "stage2_outcomes_used": False,
    }
    payload["development_failure_candidates_allowed"] = True
    payload["candidates"][0]["clean_baseline"] = {
        "available": True,
        "valid": False,
        "status": "Completed",
        "collisions": 1,
    }
    payload["candidates"][0]["development_selection_role"] = (
        "published_failure_hard_case"
    )
    candidate.write_text(json.dumps(payload))

    result = MODULE.build_batch(
        candidate_manifest=candidate,
        source_routes_dir=routes,
        baseline_source=baseline,
        protocol=protocol,
        out_dir=tmp_path / "failure_dev",
        run_id="failure_development_wave",
        route_indices=[12],
        limit=None,
        writes_performed=False,
    )

    assert result["routes"][0]["development_selection_role"] == (
        "published_failure_hard_case"
    )
    assert result["audit"]["development_failure_candidates_allowed"] is True


def test_train_coverage_repair_requires_outcome_blind_nonheld_candidate(tmp_path):
    routes, candidate, _, protocol = _sources(tmp_path)
    payload = json.loads(candidate.read_text())
    payload["selection_inputs"] = {
        "published_orion_outcomes_used": False,
        "learned_uq_outcomes_used": False,
        "stage2_outcomes_used": False,
    }
    payload["candidates"][0].update(
        {
            "clean_baseline": {"available": False, "valid": False},
            "coverage_repair_candidate": True,
            "held_out_evidence_eligible": False,
            "formal_plan_member": False,
        }
    )
    candidate.write_text(json.dumps(payload))

    result = MODULE.build_batch(
        candidate_manifest=candidate,
        source_routes_dir=routes,
        baseline_source=None,
        protocol=protocol,
        out_dir=tmp_path / "coverage",
        run_id="coverage_repair_wave0",
        route_indices=[12],
        limit=None,
        writes_performed=False,
        split="train_coverage_repair",
    )

    assert result["split"] == "train_coverage_repair"
    assert result["audit"]["train_coverage_repair_only"] is True
    assert result["audit"]["held_out_evidence_eligible"] is False


def test_train_coverage_repair_rejects_published_outcome_selection(tmp_path):
    routes, candidate, _, protocol = _sources(tmp_path)
    payload = json.loads(candidate.read_text())
    payload["selection_inputs"] = {
        "published_orion_outcomes_used": True,
        "learned_uq_outcomes_used": False,
        "stage2_outcomes_used": False,
    }
    payload["candidates"][0].update(
        {
            "coverage_repair_candidate": True,
            "held_out_evidence_eligible": False,
            "formal_plan_member": False,
        }
    )
    candidate.write_text(json.dumps(payload))
    try:
        MODULE.build_batch(
            candidate_manifest=candidate,
            source_routes_dir=routes,
            baseline_source=None,
            protocol=protocol,
            out_dir=tmp_path / "coverage",
            run_id="coverage_repair_wave0",
            route_indices=[12],
            limit=None,
            writes_performed=False,
            split="train_coverage_repair",
        )
    except ValueError as error:
        assert "must not use model outcomes" in str(error)
    else:
        raise AssertionError("outcome-informed coverage repair was accepted")
