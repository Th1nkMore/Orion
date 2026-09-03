from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_closedloop_uq_pilot.sh"
AGENT = ROOT / "team_code/orion_b2d_agent.py"


def test_runner_rejects_failed_v3_and_exposes_dynamics_aware_oracle():
    source = RUNNER.read_text(encoding="utf-8")
    assert "export ORION_CLOSEDLOOP_CONDITIONING=none" in source
    assert "export ORION_CLOSEDLOOP_CONDITIONING=vision_adapter" not in source
    rejected = source.split("native_dynamic_yield_oracle)", 1)[1].split(";;", 1)[0]
    assert "rejected v3 path-clamp oracle cannot be rerun" in rejected
    case = source.split(
        "native_dynamics_aware_yield_oracle)", 1
    )[1].split(";;", 1)[0]
    assert "ORION_CLOSEDLOOP_RISK_MODE=off" in case
    assert (
        "ORION_PLANNING_RESPONSE_MODE=privileged_dynamics_aware_yield"
        in case
    )
    assert "ORION_CLOSEDLOOP_CORRUPTION=\"\"" in case
    assert "ORION_OBSERVATION_UQ_CHECKPOINT" not in case
    assert "DENSITY" not in case


def test_manifest_separates_legacy_and_effective_conditioning():
    source = RUNNER.read_text(encoding="utf-8")
    assert '"ORION_EFFECTIVE_CONDITIONING"' in source
    assert "frozen_spatial_observation_uq_sidecar" in source
    assert "privileged_planning_response:" in source
    assert "do not mix the diagnostic observation-UQ sidecar" in source


def test_bounded_crossing_oracle_is_spatially_filtered_and_density_free():
    source = RUNNER.read_text(encoding="utf-8")
    case = source.split(
        "native_bounded_crossing_oracle)", 1
    )[1].split(";;", 1)[0]
    assert "ORION_CLOSEDLOOP_CORRUPTION=\"\"" in case
    assert "ORION_CLOSEDLOOP_RISK_MODE=off" in case
    assert "ORION_PLANNING_RESPONSE_MODE=privileged_bounded_crossing" in case
    assert "ORION_PLANNING_ACTOR_CATEGORIES" in case
    assert "walker" in case
    assert "DENSITY" not in case
    assert '"ORION_PLANNING_ACTOR_CATEGORIES"' in source


def test_braking_aware_crossing_oracle_freezes_near_horizon_retiming():
    source = RUNNER.read_text(encoding="utf-8")
    case = source.split(
        "native_braking_aware_crossing_oracle)", 1
    )[1].split(";;", 1)[0]
    assert "ORION_CLOSEDLOOP_CORRUPTION=\"\"" in case
    assert "ORION_CLOSEDLOOP_RISK_MODE=off" in case
    assert (
        "ORION_PLANNING_RESPONSE_MODE=privileged_braking_aware_crossing"
        in case
    )
    assert "ORION_PLANNING_CERTIFIED_DECELERATION_MPS2" in case
    assert "ORION_PLANNING_ACTOR_CATEGORIES" in case
    assert "DENSITY" not in case
    assert '"uq_estimator/bounded_crossing_expert.py"' in source
    assert '"scripts/audit_route147_braking_aware_v2.py"' in source


def test_manifest_freezes_dynamic_yield_parameters_and_source():
    source = RUNNER.read_text(encoding="utf-8")
    for key in (
        "ORION_PLANNING_RESPONSE_MODE",
        "ORION_PLANNING_INTERPOLATION_STEP_SECONDS",
        "ORION_PLANNING_SAFETY_MARGIN_M",
        "ORION_PLANNING_IMMINENT_HORIZON_SECONDS",
        "ORION_PLANNING_CERTIFIED_DECELERATION_MPS2",
        "ORION_PLANNING_REACTION_SECONDS",
        "ORION_PLANNING_JUNCTION_FRONT_CLEARANCE_M",
        "ORION_PLANNING_MAP_RESOLUTION_M",
        "ORION_PLANNING_CLEARANCE_SECONDS",
        "ORION_PLANNING_RELEASE_SECONDS",
        "ORION_PLANNING_PREPARE_CREEP_SPEED_MPS",
        "ORION_PLANNING_RELEASE_CREEP_SPEED_MPS",
        "ORION_PLANNING_STOP_BUFFER_M",
        "ORION_PLANNING_RELEASE_CREEP_DISTANCE_M",
    ):
        assert f'"{key}"' in source
    assert '"uq_estimator/privileged_yield_labels.py"' in source
    assert '"uq_estimator/dynamic_yield_expert.py"' in source


def test_agent_applies_target_trajectory_before_pid_and_records_both_plans():
    source = AGENT.read_text(encoding="utf-8")
    target_index = source.index("out_truck = np.asarray(target_plan")
    pid_index = source.index("self.pidcontroller.control_pid(out_truck")
    assert target_index < pid_index
    assert "self.pid_metadata['base_plan'] = base_out_truck.tolist()" in source
    assert "self.pid_metadata['plan'] = out_truck.tolist()" in source
    assert "planning-level oracle requires scalar risk governor off" in source
    assert "first_path_junction_entry" in source
    assert "resolve_junction_scoped_conflict" in source
