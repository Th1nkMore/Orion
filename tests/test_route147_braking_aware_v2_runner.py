import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_closedloop_uq_pilot.sh"
SUBMITTER = ROOT / "scripts/submit_route147_braking_aware_v2.sh"
EVALUATOR = ROOT / "scripts/evaluate_route147_braking_aware_v2.py"
PREREG = ROOT / "configs/closedloop_scenario_bank/route147_braking_aware_v2.json"
V1_PREREG = ROOT / "configs/closedloop_scenario_bank/route147_bounded_crossing_pair_v1.json"
AMENDMENT = ROOT / "configs/closedloop_scenario_bank/route147_braking_aware_v2_runtime_amendment_20260829.json"
RETRY_SUBMITTER = ROOT / "scripts/submit_route147_braking_aware_v2_retry1.sh"


def test_runner_exposes_density_free_braking_aware_oracle():
    source = RUNNER.read_text(encoding="utf-8")
    case = source.split(
        "native_braking_aware_crossing_oracle)", 1
    )[1].split(";;", 1)[0]
    assert "ORION_CLOSEDLOOP_CORRUPTION=\"\"" in case
    assert "ORION_CLOSEDLOOP_RISK_MODE=off" in case
    assert "ORION_PLANNING_RESPONSE_MODE=privileged_braking_aware_crossing" in case
    assert "ORION_PLANNING_ACTOR_CATEGORIES" in case
    assert "ORION_PLANNING_CERTIFIED_DECELERATION_MPS2" in case
    assert "DENSITY" not in case
    assert '"scripts/evaluate_route147_braking_aware_v2.py"' in source
    assert '"scripts/submit_route147_braking_aware_v2.sh"' in source


def test_submitter_authorizes_only_one_oracle_and_no_clean_rerun():
    source = SUBMITTER.read_text(encoding="utf-8")
    assert "maximum_oracle_submissions" in source
    assert "maximum_clean_submissions" in source
    assert "native_braking_aware_crossing_oracle" in source
    assert "clean_off" not in source
    assert "ORION_ENABLE_LEGACY_DENSITY_UQ=0" in source
    assert "ORION_OBSERVATION_UQ_CHECKPOINT=" in source
    assert "SLURM_CPUS_PER_TASK=2" in source
    assert "SLURM_MEM=192G" in source


def test_evaluator_recomputes_new_trajectory_and_keeps_stage2_locked_on_failure():
    source = EVALUATOR.read_text(encoding="utf-8")
    assert "build_braking_aware_crossing_trajectory" in source
    assert "braking_profile_recomputed_every_frame" in source
    assert "first_hold_commands_immediate_braking" in source
    assert "minimum_walker_ttc_gain_seconds" in source
    assert "keep_stage2_locked_and_redesign_mechanism_or_route" in source
    assert "Density and the learned spatial adapter are absent" in source


def test_v2_prereg_freezes_current_sources_and_preserves_v1_thresholds():
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    v1 = json.loads(V1_PREREG.read_text(encoding="utf-8"))
    assert prereg["success_thresholds"] == {
        **v1["success_thresholds"],
        "logic": prereg["success_thresholds"]["logic"],
    }
    for key, value in v1["success_thresholds"].items():
        if key != "logic":
            assert prereg["success_thresholds"][key] == value
    for relative, expected in prereg["frozen_hashes"].items():
        path = (
            ROOT / "configs/closedloop_scenario_bank/routes/route_147_hazard.xml"
            if relative == "route_147_hazard.xml"
            else ROOT / relative
        )
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    assert prereg["single_oracle_rule"]["maximum_clean_submissions"] == 0
    assert prereg["single_oracle_rule"]["maximum_oracle_submissions"] == 1
    assert prereg["stage2_unlock_rule"]["required"] == {
        "primary_success": True,
        "stage2_eligible": True,
    }


def test_runtime_amendment_freezes_invalid_run_and_allows_one_identical_retry():
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    assert amendment["base_preregistration_sha256"] == hashlib.sha256(
        PREREG.read_bytes()
    ).hexdigest()
    assert amendment["invalid_run"]["scientific_classification"] == (
        "runtime_environment_invalid"
    )
    assert amendment["invalid_run"]["terminal_route_result_exists"] is False
    for artifact in amendment["invalid_run"]["frozen_artifacts"]:
        assert hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest() == (
            artifact["sha256"]
        )
    rule = amendment["one_retry_rule"]
    assert rule["maximum_retry_submissions"] == 1
    assert rule["source_or_parameter_changes_allowed"] is False
    assert rule["retry_submitter_sha256"] == hashlib.sha256(
        RETRY_SUBMITTER.read_bytes()
    ).hexdigest()
    source = RETRY_SUBMITTER.read_text(encoding="utf-8")
    assert "native_braking_aware_crossing_oracle" in source
    assert "ORION_ENABLE_LEGACY_DENSITY_UQ=0" in source
    assert "ORION_OBSERVATION_UQ_CHECKPOINT=" in source
    assert "clean_off" not in source
