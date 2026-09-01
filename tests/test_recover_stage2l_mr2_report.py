import json
from pathlib import Path

import pytest

from scripts.recover_stage2l_mr2_report import parse_history, validate_history


def _row(step, events):
    return {
        "optimizer_step": step,
        "primary_group_count": len(events),
        "primary_event_ids": events,
        "primary_group_ids": ["%s_g%d" % (event, step) for event in events],
        "language_group_id": "%s_g%d" % (events[0], step),
        "finite_loss": True,
        "finite_gradient_norm": True,
        "finite_gradients": True,
        "loss": 1.0 / step,
    }


def test_parse_and_validate_balanced_finite_history(tmp_path: Path):
    events = ["route_a", "route_b"]
    rows = [_row(1, events), _row(2, list(reversed(events)))]
    log = tmp_path / "job.out"
    log.write_text(
        "noise\n"
        + "\n".join("[Stage2LMR1] " + json.dumps(row) for row in rows)
        + "\nKeyError: 'known_coverage_gaps'\n",
        encoding="utf-8",
    )
    parsed = parse_history(log)
    assert parsed == rows
    assert validate_history(parsed, expected_steps=2, expected_events=events) == {
        "route_a": 2,
        "route_b": 2,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.__setitem__(1, {**rows[1], "optimizer_step": 3}),
        lambda rows: rows[0].__setitem__("finite_loss", False),
        lambda rows: rows[0].__setitem__("loss", float("nan")),
        lambda rows: rows[0].__setitem__("primary_event_ids", ["route_a"]),
    ],
)
def test_validate_history_rejects_incomplete_or_nonfinite(mutation):
    events = ["route_a", "route_b"]
    rows = [_row(1, events), _row(2, events)]
    mutation(rows)
    with pytest.raises(ValueError):
        validate_history(rows, expected_steps=2, expected_events=events)
