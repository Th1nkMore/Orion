import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_qwen_visibility_route_diverse_data.py"
)
SPEC = importlib.util.spec_from_file_location("qwen_route_diverse_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_route_label_reads_the_addressed_route_weight():
    names = ("x", "route_weight_mean", "y")
    rows = np.asarray([[0.0, 0.199, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32)
    assert MODULE.route_label(rows, names, 0) == "OFF_ROUTE"
    assert MODULE.route_label(rows, names, 1) == "ON_ROUTE"
