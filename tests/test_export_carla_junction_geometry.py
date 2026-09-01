"""Unit tests for versioned control-trace geometry extraction."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


def _load_exporter(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "carla",
        types.SimpleNamespace(LaneType=types.SimpleNamespace(Driving=1)),
    )
    path = Path(__file__).resolve().parents[1] / "scripts" / "export_carla_junction_geometry.py"
    spec = importlib.util.spec_from_file_location("junction_geometry_exporter", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_raw_plan_prefers_v4_provenance(monkeypatch):
    exporter = _load_exporter(monkeypatch)
    row = {
        "planning_response": {
            "raw_conflict": {"base_plan_world_xy": [[1.0, 2.0]]},
            "conflict": {"base_plan_world_xy": [[9.0, 9.0]]},
        }
    }
    assert exporter.raw_plan_world_xy(row) == ([[1.0, 2.0]], "raw_conflict")


def test_raw_plan_supports_v3_and_rejects_effective_target(monkeypatch):
    exporter = _load_exporter(monkeypatch)
    v3 = {
        "planning_response": {
            "conflict": {"base_plan_world_xy": [[3.0, 4.0]]},
        }
    }
    assert exporter.raw_plan_world_xy(v3) == ([[3.0, 4.0]], "conflict")

    intervened_only = {
        "planning_response": {
            "effective_conflict": {"base_plan_world_xy": [[5.0, 6.0]]},
            "target_plan_world_xy": [[7.0, 8.0]],
        }
    }
    assert exporter.raw_plan_world_xy(intervened_only) == (None, None)
