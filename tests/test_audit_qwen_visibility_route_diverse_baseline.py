import importlib.util
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_qwen_visibility_route_diverse_baseline.py"
)
SPEC = importlib.util.spec_from_file_location("qwen_route_diverse_baseline_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_summarize_evaluations_uses_original_target_for_causal_gap():
    rows = []
    answers = {
        "true_u": ["ON_ROUTE", "OFF_ROUTE"],
        "zero_u": ["OFF_ROUTE", "OFF_ROUTE"],
        "spatial_shuffle": ["ON_ROUTE", "ON_ROUTE"],
    }
    originals = ["ON_ROUTE", "OFF_ROUTE"]
    control_targets = {
        "true_u": originals,
        "zero_u": ["OFF_ROUTE", "OFF_ROUTE"],
        "spatial_shuffle": ["OFF_ROUTE", "ON_ROUTE"],
    }
    for control, outputs in answers.items():
        for index, output in enumerate(outputs):
            rows.append(
                {
                    "control": control,
                    "split": "validation" if index == 0 else "held_out",
                    "parsed": output,
                    "expected_answer": originals[index],
                    "control_expected_answer": control_targets[control][index],
                }
            )
    summary = MODULE.summarize_evaluations(rows)
    assert summary["true_u"]["original_target_accuracy"] == 1.0
    assert summary["zero_u"]["original_target_accuracy"] == 0.5
    assert summary["spatial_shuffle"]["original_target_accuracy"] == 0.5
    assert summary["causal_gap_vs_stronger_control"] == 0.5
