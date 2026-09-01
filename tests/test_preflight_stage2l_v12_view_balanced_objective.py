import torch

from scripts.preflight_stage2l_v12_view_balanced_objective import (
    summarize_weight_redistribution,
)


def test_redistribution_summary_balances_active_views_and_keeps_unit_mass():
    targets = torch.zeros(2, 6, 10, 10)
    targets[0, 0, 4, 5] = 1.0
    targets[0, 1, 2:4, 2:4] = 1.0
    targets[1, 0, 4, 5] = 1.0
    result = summarize_weight_redistribution(targets, support_fraction=0.1)
    assert result["foreground_weight_sum_minimum"] == 1.0
    assert result["foreground_weight_sum_maximum"] == 1.0
    assert result["per_view"]["CAM_FRONT"]["positive_group_count"] == 2
    assert result["per_view"]["CAM_FRONT_LEFT"]["positive_group_count"] == 1
    assert result["per_view"]["CAM_FRONT_LEFT"][
        "proposed_mean_share_when_active"
    ] == 0.5
