import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "scripts/submit_route147_bounded_crossing_pair_v1.sh"
RUNNER = ROOT / "scripts/run_closedloop_uq_pilot.sh"
EVALUATOR = ROOT / "scripts/evaluate_route147_bounded_crossing_pair.py"
PREREG = ROOT / "configs/closedloop_scenario_bank/route147_bounded_crossing_pair_v1.json"


def test_submitter_freezes_one_density_free_clean_oracle_pair():
    source = SUBMITTER.read_text(encoding="utf-8")
    assert "maximum_clean_submissions" in source
    assert "maximum_oracle_submissions" in source
    assert "ORION_ENABLE_LEGACY_DENSITY_UQ=0" in source
    assert source.count("ORION_OBSERVATION_UQ_CHECKPOINT=") == 2
    assert "147 clean_off hazard" in source
    assert "147 native_bounded_crossing_oracle hazard" in source
    assert "ORION_PLANNING_ACTOR_CATEGORIES=walker" in source
    assert "SLURM_MEM=192G" in source


def test_run_manifest_hashes_bounded_crossing_audit_and_evaluator():
    source = RUNNER.read_text(encoding="utf-8")
    for relative in (
        "scripts/audit_route147_bounded_crossing.py",
        "scripts/evaluate_route147_bounded_crossing_pair.py",
        "scripts/submit_route147_bounded_crossing_pair_v1.sh",
        "scripts/submit_closedloop_uq_pilot.sh",
    ):
        assert f'"{relative}"' in source


def test_evaluator_requires_density_absence_and_exact_go_passthrough():
    source = EVALUATOR.read_text(encoding="utf-8")
    assert "density_absent_every_frame" in source
    assert "new_adapter_absent_every_frame" in source
    assert "go_state_exactly_preserves_orion_plan" in source
    assert "primary_success" in source
    assert "stage2_eligible" in source


def test_completed_v1_preserves_frozen_source_attestation_and_unlock_rule():
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    result = json.loads((
        ROOT
        / "results/closedloop_scenario_bank/route147_bounded_crossing_pair_v1"
        / "route147_bounded_crossing_pair_result.json"
    ).read_text(encoding="utf-8"))
    assert result["artifacts"]["preregistration_sha256"] == hashlib.sha256(
        PREREG.read_bytes()
    ).hexdigest()
    assert all(result["manifest_checks"]["clean"].values())
    assert all(result["manifest_checks"]["oracle"].values())
    assert all(
        value
        for condition in result["trace_contract"].values()
        for value in condition["checks"].values()
    )
    assert prereg["pipeline_contract"]["legacy_density_uq"].startswith(
        "hard_disabled"
    )
    assert prereg["stage2_unlock_rule"]["required"] == {
        "primary_success": True,
        "stage2_eligible": True,
    }
