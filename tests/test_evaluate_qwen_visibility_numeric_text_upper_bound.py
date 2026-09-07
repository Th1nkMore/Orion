import importlib.util
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_qwen_visibility_numeric_text_upper_bound.py"
)
SPEC = importlib.util.spec_from_file_location("qwen_u_text_upper_bound", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_render_question_exposes_exact_field_and_rule():
    prompt = MODULE.render_question("F03", 0.375)
    assert "F03.route_weight_mean = 0.375000" in prompt
    assert "at least 0.2" in prompt
    assert "exactly ON_ROUTE or OFF_ROUTE" in prompt


def test_summarize_keeps_modes_and_route_splits_separate():
    rows = []
    for mode in ("text_only", "rgb_plus_text"):
        for split, expected in (
            ("validation", "ON_ROUTE"),
            ("held_out", "OFF_ROUTE"),
        ):
            rows.append(
                {
                    "mode": mode,
                    "sample_id": split,
                    "split": split,
                    "expected": expected,
                    "predicted": expected,
                    "correct": True,
                }
            )
    summary = MODULE.summarize(rows)
    assert summary["text_only"]["accuracy"] == 1.0
    assert summary["rgb_plus_text"]["accuracy"] == 1.0
    assert summary["mode_prediction_agreement"] == 1.0
