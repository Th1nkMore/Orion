from scripts.analyze_stage2l_v10_staged_smoke import analyze


def _report(status="all_bounded_v10_phases_pass"):
    return {
        "schema": "orion.stage2l_v10_staged_smoke.v1",
        "status": status,
        "engineering_preexperiment_only": True,
        "formal_stage2l_ready": False,
        "stage2p_ready": False,
        "completed_phases": ["A_map_pretrain", "B_risk_alignment", "C_language_grounding"],
        "phases": {
            "A_map_pretrain": {
                "status": "pass",
                "checks": {"map": True},
                "history": [
                    {"loss": 2.0, "map_loss": 2.0, "ranking_loss": 0.0, "finite": True},
                    {"loss": 1.0, "map_loss": 1.0, "ranking_loss": 0.0, "finite": True},
                ],
            },
            "B_risk_alignment": {
                "status": "pass",
                "checks": {"rank": True},
                "history": [
                    {"loss": 1.0, "map_loss": 0.8, "ranking_loss": 0.2, "finite": True},
                    {"loss": 0.5, "map_loss": 0.4, "ranking_loss": 0.1, "finite": True},
                ],
            },
            "C_language_grounding": {
                "status": "pass",
                "checks": {"nll": True, "map_retained": True},
                "history": [
                    {"mean_target_nll": 4.0, "finite": True},
                    {"mean_target_nll": 3.0, "finite": True},
                ],
            },
        },
        "provenance": {
            "trainer": {"sha256": "trainer"},
            "protocol": {"sha256": "protocol"},
            "frozen_u_tokenizer": {"sha256": "u"},
        },
        "locks": {
            "density_uq_or_governor_used": False,
            "trajectory_or_control_loss_used": False,
        },
    }


def test_analysis_preserves_gate_scope_and_trends():
    result = analyze(_report(), report_sha256="report")
    assert result["unlocks"]["one_clean_corrupt_engineering_interface_smoke"] is True
    assert result["unlocks"]["formal_stage2l"] is False
    assert result["phases"]["A_map_pretrain"]["trends"]["map_loss"]["relative_change"] == -0.5


def test_failed_phase_a_stops_at_R_objective():
    report = _report("stopped_after_phase_a_failed_gate")
    report["completed_phases"] = []
    report["phases"] = {"A_map_pretrain": report["phases"]["A_map_pretrain"]}
    result = analyze(report, report_sha256="report")
    assert result["decision"] == "stop_and_revise_spatial_R_objective_or_support_labels"
    assert result["unlocks"]["one_clean_corrupt_engineering_interface_smoke"] is False
