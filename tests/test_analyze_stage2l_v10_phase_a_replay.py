from scripts.analyze_stage2l_v10_phase_a_replay import threshold_feasibility


def test_threshold_feasibility_rejects_tradeoff_without_joint_row():
    rows = [
        {"threshold": 0.0, "recall": 1.0, "precision": 0.1, "background_fpr": 1.0},
        {"threshold": 0.5, "recall": 0.6, "precision": 0.8, "background_fpr": 0.05},
        {"threshold": 1.0, "recall": 0.0, "precision": 1.0, "background_fpr": 0.0},
    ]
    result = threshold_feasibility(
        rows, minimum_recall=0.8, maximum_background_fpr=0.1
    )
    assert result["feasible"] is False
    assert result["maximum_recall_under_fpr_ceiling"] == 0.6
    assert result["minimum_background_fpr_at_recall_floor"] == 1.0


def test_threshold_feasibility_accepts_joint_row():
    rows = [
        {"threshold": 0.2, "recall": 0.9, "precision": 0.7, "background_fpr": 0.08},
        {"threshold": 0.8, "recall": 0.4, "precision": 0.9, "background_fpr": 0.01},
    ]
    result = threshold_feasibility(
        rows, minimum_recall=0.8, maximum_background_fpr=0.1
    )
    assert result["feasible"] is True
    assert result["feasible_rows"] == [rows[0]]
