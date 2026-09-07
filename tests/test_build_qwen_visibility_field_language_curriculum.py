import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_qwen_visibility_field_language_curriculum.py"
SPEC = importlib.util.spec_from_file_location("qwen_field_language_curriculum", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_field_language_schema_is_explicit_and_threshold_is_literal():
    assert len(MODULE.FIELDS) == 8
    for field, (_, meaning) in MODULE.FIELDS.items():
        assert field in MODULE.SYSTEM_PROMPT
        assert meaning in MODULE.SYSTEM_PROMPT
    assert MODULE.answer(0.2, 0.2) == "AT_OR_ABOVE"
    assert MODULE.answer(0.199, 0.2) == "BELOW"
    assert "hidden-actor" in MODULE.SYSTEM_PROMPT
