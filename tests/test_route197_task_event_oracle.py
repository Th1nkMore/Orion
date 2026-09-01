from scripts.evaluate_route197_task_event_oracle import rising_edges


def test_task_oracle_window_has_one_rising_edge_only_when_one_shot():
    assert rising_edges([False, False, True, True, False, False]) == 1
    assert rising_edges([False, True, False, True]) == 2
    assert rising_edges([False, False]) == 0
