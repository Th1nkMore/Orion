from pathlib import Path

import pytest

from scripts.audit_route147_bounded_crossing import audit


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/closedloop_scenario_bank/route147_clean_v2_1082515"


def test_real_route147_offline_gate_has_finite_braking_and_release():
    report = audit(
        clean_gate_path=RUN / "clean_safety_gate.json",
        trace_path=RUN / "control_trace.jsonl",
        labels_path=RUN / "privileged_yield_v1/privileged_yield_labels.jsonl",
        meta_dir=RUN / "records/meta",
    )
    assert report["offline_gate_pass"] is True
    assert report["decision"] == (
        "eligible_for_one_preregistered_route147_clean_oracle_pair"
    )
    evidence = report["evidence"]
    assert evidence["conflict_actor_categories"] == ["walker"]
    assert evidence["baseline_min_walker_obb_ttc_seconds"] == pytest.approx(
        0.6807768416126619
    )
    assert evidence["first_hold_time_seconds"] == pytest.approx(10.5)
    assert evidence["certified_braking_margin_m"] > 3.8
    assert evidence["response_duration_seconds"] == pytest.approx(2.0)
    assert evidence["historical_trace_density_score_nonnull_frames"] == 627
    assert report["checks"]["historical_density_was_passive_only"] is True


def test_gate_rejects_an_uncertified_braking_margin():
    report = audit(
        clean_gate_path=RUN / "clean_safety_gate.json",
        trace_path=RUN / "control_trace.jsonl",
        labels_path=RUN / "privileged_yield_v1/privileged_yield_labels.jsonl",
        meta_dir=RUN / "records/meta",
        minimum_braking_margin_m=4.0,
    )
    assert report["offline_gate_pass"] is False
    assert report["checks"]["certified_braking_margin_sufficient"] is False
