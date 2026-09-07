import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_qwen_visibility_field_alignment_corpus.py"
SPEC = importlib.util.spec_from_file_location("qwen_field_corpus_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_describe_reports_distribution_without_rounding_away_range():
    result = MODULE.describe([0.0, 0.25, 0.5, 0.75, 1.0])
    assert result["count"] == 5
    assert result["finite"] is True
    assert result["minimum"] == 0.0
    assert result["maximum"] == 1.0
    assert result["rounded_unique_values"] == 5
    assert result["quantiles"]["q50"] == 0.5
