import os
from pathlib import Path
import runpy

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    PROJECT_ROOT
    / "adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py"
)


def _config(monkeypatch, source="disabled", checkpoint=""):
    monkeypatch.setenv("ORION_ENABLE_LEGACY_DENSITY_UQ", "0")
    monkeypatch.setenv("ORION_CLOSEDLOOP_CONDITIONING", "none")
    monkeypatch.setenv("ORION_STAGE2_SPATIAL_UQ_SOURCE", source)
    monkeypatch.setenv("ORION_STAGE1_SPATIAL_UQ_CHECKPOINT", checkpoint)
    return runpy.run_path(str(CONFIG))["model"]


def test_disabled_config_has_no_legacy_or_spatial_conditioning(monkeypatch):
    model = _config(monkeypatch)
    assert model["use_uq_token"] is False
    assert model["use_uq_vision_adapter"] is False
    assert model["use_uncertainty_l2"] is False
    assert model["use_bev_uncertainty"] is False
    assert model["use_stage2_spatial_uq"] is False
    assert model["pts_bbox_head"]["use_uncertainty"] is False
    assert model["pts_bbox_head"]["use_stage2_spatial_uq"] is False


def test_learned_adapter_enables_only_new_spatial_path(monkeypatch):
    model = _config(monkeypatch, "learned_adapter", "/tmp/stage1.pt")
    assert model["use_stage2_spatial_uq"] is True
    assert model["stage2_spatial_uq_source"] == "learned_adapter"
    assert model["pts_bbox_head"]["use_stage2_spatial_uq"] is True
    assert model["pts_bbox_head"]["use_uncertainty"] is False
    assert model["use_uq_token"] is False


def test_learned_adapter_requires_stage1_checkpoint(monkeypatch):
    with pytest.raises(RuntimeError, match="requires"):
        _config(monkeypatch, "learned_adapter", "")


def test_legacy_conditioning_is_rejected(monkeypatch):
    monkeypatch.setenv("ORION_CLOSEDLOOP_CONDITIONING", "token")
    monkeypatch.setenv("ORION_ENABLE_LEGACY_DENSITY_UQ", "0")
    with pytest.raises(RuntimeError, match="retired"):
        runpy.run_path(str(CONFIG))
