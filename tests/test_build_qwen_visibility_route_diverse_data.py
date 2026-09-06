import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_qwen_visibility_route_diverse_data.py"
)
SPEC = importlib.util.spec_from_file_location("qwen_route_diverse_data", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_balanced_route_label_assignments_are_seeded_and_exact():
    folders = ["route-%02d" % index for index in range(10)]
    first = MODULE.balanced_route_label_assignments(folders, 17)
    second = MODULE.balanced_route_label_assignments(list(reversed(folders)), 17)
    assert first == second
    assert list(first.values()).count("ON_ROUTE") == 5
    assert list(first.values()).count("OFF_ROUTE") == 5


def test_choose_candidate_index_uses_only_requested_natural_rows():
    labels = [False, True, False, True, False]
    on_index = MODULE.choose_candidate_index(labels, "ON_ROUTE", 31)
    off_index = MODULE.choose_candidate_index(labels, "OFF_ROUTE", 31)
    assert labels[on_index] is True
    assert labels[off_index] is False
    assert on_index == MODULE.choose_candidate_index(labels, "ON_ROUTE", 31)
    assert off_index == MODULE.choose_candidate_index(labels, "OFF_ROUTE", 31)


def test_balanced_assignment_rejects_odd_route_count():
    try:
        MODULE.balanced_route_label_assignments(["a", "b", "c"], 1)
    except ValueError as error:
        assert "even route count" in str(error)
    else:
        raise AssertionError("odd route count must fail closed")


def test_feasible_balanced_assignment_respects_forced_routes():
    availability = {
        "forced-on": {"ON_ROUTE": True, "OFF_ROUTE": False},
        "forced-off": {"ON_ROUTE": False, "OFF_ROUTE": True},
        "flex-a": {"ON_ROUTE": True, "OFF_ROUTE": True},
        "flex-b": {"ON_ROUTE": True, "OFF_ROUTE": True},
    }
    assignments = MODULE.feasible_balanced_route_label_assignments(
        availability, 19
    )
    assert assignments["forced-on"] == "ON_ROUTE"
    assert assignments["forced-off"] == "OFF_ROUTE"
    assert list(assignments.values()).count("ON_ROUTE") == 2
    assert list(assignments.values()).count("OFF_ROUTE") == 2


def test_feasible_balanced_assignment_rejects_impossible_split():
    availability = {
        "a": {"ON_ROUTE": False, "OFF_ROUTE": True},
        "b": {"ON_ROUTE": False, "OFF_ROUTE": True},
        "c": {"ON_ROUTE": False, "OFF_ROUTE": True},
        "d": {"ON_ROUTE": True, "OFF_ROUTE": True},
    }
    try:
        MODULE.feasible_balanced_route_label_assignments(availability, 2)
    except ValueError as error:
        assert "cannot support" in str(error)
    else:
        raise AssertionError("infeasible balanced split must fail closed")


def test_full_row_candidates_exclude_incomplete_frames():
    feature_names = ("route_weight_mean",)
    complete = SimpleNamespace(
        feature_names=feature_names,
        frontier_tokens=np.concatenate(
            [np.ones((1, 1), dtype=np.float32), np.zeros((31, 1), dtype=np.float32)]
        ),
        frontier_mask=np.ones(32, dtype=bool),
    )
    incomplete = SimpleNamespace(
        feature_names=feature_names,
        frontier_tokens=np.ones((32, 1), dtype=np.float32),
        frontier_mask=np.asarray([True] * 31 + [False]),
    )
    flat, labels, valid_counts = MODULE.full_row_candidates(
        [(10, incomplete), (20, complete)], 0.2
    )
    assert valid_counts == [31, 32]
    assert len(flat) == 32
    assert {frame for frame, _, _ in flat} == {20}
    assert labels.count(True) == 1
