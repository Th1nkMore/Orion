from scripts.diagnose_stage2l_v10_phase_a_failure import diagnose


def _split(recall, fpr, gap, event_orders):
    per_group = {}
    for index, order in enumerate(event_orders):
        per_group["g%d" % index] = {
            "event_id": "e%d" % index,
            "map_loss": 0.5,
            "learned_gap": 0.1 if order else -0.1,
            "attained_fraction": 0.5 if order else -0.5,
            "positive_order": order,
        }
    return {
        "relevance_support": {
            "foreground_recall": recall,
            "background_false_positive_rate": fpr,
            "foreground_background_probability_gap": gap,
        },
        "per_group": per_group,
    }


def test_diagnosis_does_not_convert_underactivation_to_capacity_claim():
    report = {
        "schema": "orion.stage2l_v10_staged_smoke.v1",
        "status": "stopped_after_phase_a_failed_gate",
        "completed_phases": [],
        "phases": {"A_map_pretrain": {
            "history": [
                {"loss": 1.4, "finite": True},
                {"loss": 0.6, "finite": True},
            ],
            "metrics": {
                "train": _split(0.63, 0.01, 0.46, [True, True]),
                "dev": _split(0.34, 0.008, 0.22, [True, False]),
            },
        }},
    }
    result = diagnose(report, report_sha256="a" * 64)
    assert result["evidence"]["optimization_stable"]
    assert result["evidence"]["background_suppression_healthy"]
    assert result["diagnosis"]["foreground_underactivation_supported"]
    assert result["diagnosis"]["heldout_spatial_transfer_gap_supported"]
    assert not result["diagnosis"]["insufficient_parameter_capacity_supported"]
    assert not result["locks"]["v10_reclassified_as_pass"]
