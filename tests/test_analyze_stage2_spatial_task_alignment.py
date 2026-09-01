from scripts.analyze_stage2_spatial_task_alignment import _binary_auroc, _delta


def test_binary_auroc_handles_ordering_and_ties():
    assert _binary_auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert _binary_auroc([0.5, 0.5], [True, False]) == 0.5
    assert _binary_auroc([0.5], [True]) is None


def test_delta_reports_direction_and_ratio():
    result = _delta([2.0, 4.0], [1.0, 2.0])
    assert result["response_minus_go_mean"] == 1.5
    assert result["response_over_go_mean"] == 2.0
