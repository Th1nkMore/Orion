import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_qwen_visibility_field_language_result.py"
SPEC = importlib.util.spec_from_file_location("audit_qwen_field_language", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_summary_uses_real_control_accuracy_without_forced_inequality():
    rows = []
    for control, answers in {
        "true_u": ("BELOW", "AT_OR_ABOVE"),
        "zero_u": ("BELOW", "AT_OR_ABOVE"),
        "spatial_shuffle": ("AT_OR_ABOVE", "BELOW"),
    }.items():
        for split, expected, parsed in zip(
            ("validation", "held_out"),
            ("BELOW", "AT_OR_ABOVE"),
            answers,
        ):
            rows.append(
                {
                    "control": control,
                    "split": split,
                    "task_field": "field",
                    "expected_answer": expected,
                    "parsed": parsed,
                    "canonical_exact": parsed == expected,
                    "control_semantic_exact": True,
                }
            )
    result = MODULE.summarize(rows)
    assert result["accuracy_by_control_against_true_target"]["true_u"] == 1.0
    assert result["accuracy_by_control_against_true_target"]["zero_u"] == 1.0
    assert result["true_minus_zero_accuracy_pp"] == 0.0
    assert result["true_minus_shuffle_accuracy_pp"] == 100.0
