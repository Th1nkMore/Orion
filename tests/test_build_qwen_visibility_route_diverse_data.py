import importlib.util
from pathlib import Path
import sys


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
