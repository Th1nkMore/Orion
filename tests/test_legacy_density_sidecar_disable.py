import os
from pathlib import Path
import runpy

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "adzoo/orion/configs/orion_stage3_agent_uq.py"
RUNNER = PROJECT_ROOT / "scripts/run_closedloop_uq_pilot.sh"


def _load_config(monkeypatch, value):
    monkeypatch.setenv("ORION_ENABLE_LEGACY_DENSITY_UQ", value)
    return runpy.run_path(str(CONFIG))["model"]["pts_bbox_head"]


def test_config_does_not_construct_density_estimator_when_disabled(monkeypatch):
    head = _load_config(monkeypatch, "0")
    assert head["use_uncertainty"] is False
    assert head["uq_checkpoint"] == ""


def test_config_rejects_explicit_legacy_density_enable(monkeypatch):
    with pytest.raises(RuntimeError, match="retired"):
        _load_config(monkeypatch, "1")


def test_config_defaults_to_legacy_density_disabled(monkeypatch):
    monkeypatch.delenv("ORION_ENABLE_LEGACY_DENSITY_UQ", raising=False)
    head = runpy.run_path(str(CONFIG))["model"]["pts_bbox_head"]
    assert head["use_uncertainty"] is False
    assert head["uq_checkpoint"] == ""


def test_current_runner_globally_hard_disables_legacy_density():
    source = RUNNER.read_text(encoding="utf-8")
    assert "export ORION_ENABLE_LEGACY_DENSITY_UQ=0" in source
    assert "legacy Density UQ is retired" in source
    assert "export DENSITY_UQ_CHECKPOINT=" not in source


def test_current_agent_rejects_legacy_density_even_without_runner():
    source = (PROJECT_ROOT / "team_code/orion_b2d_agent.py").read_text(
        encoding="utf-8"
    )
    assert "requested_legacy_density_uq" in source
    assert "Legacy Density UQ is retired from the current closed-loop" in source
    assert "Legacy Density token/vision-adapter conditioning is retired" in source
    assert "ORION_CLOSEDLOOP_CONDITIONING must be none" in source


def test_rejected_scalar_learned_conditions_cannot_run():
    source = RUNNER.read_text(encoding="utf-8")
    assert "scalar learned-UQ governor is retired" in source
    assert "scalar learned-UQ stop governor is rejected by Route197" in source
