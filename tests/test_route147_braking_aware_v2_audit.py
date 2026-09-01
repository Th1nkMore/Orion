from pathlib import Path

import pytest

from scripts.audit_route147_braking_aware_v2 import audit


ROOT = Path(__file__).resolve().parents[1]
PAIR = ROOT / "results/closedloop_scenario_bank/route147_bounded_crossing_pair_v1"


def test_real_v1_failure_authorizes_only_braking_aware_v2():
    report = audit(
        v1_result_path=PAIR / "route147_bounded_crossing_pair_result.json",
        v1_oracle_trace_path=PAIR / "oracle_1087791/control_trace.jsonl",
    )
    assert report["offline_gate_pass"] is True
    assert report["v1_failure"]["failed_outcome_checks"] == [
        "walker_ttc_improved_or_horizon_censored"
    ]
    first = report["first_hold_counterfactual"]
    assert first["sim_time_seconds"] == pytest.approx(10.55)
    assert first["v1_target_pid_desired_speed_mps"] > first["speed_mps"]
    assert first["v2_target_pid_desired_speed_mps"] < first["speed_mps"]
    assert first["v2_immediate_brake_ratio"] > 1.1


def test_audit_rejects_a_stricter_unmet_immediate_brake_ratio():
    report = audit(
        v1_result_path=PAIR / "route147_bounded_crossing_pair_result.json",
        v1_oracle_trace_path=PAIR / "oracle_1087791/control_trace.jsonl",
        immediate_brake_ratio_threshold=2.0,
    )
    assert report["offline_gate_pass"] is False
    assert report["checks"]["v2_first_hold_commands_immediate_braking"] is False
