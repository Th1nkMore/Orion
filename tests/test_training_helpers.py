"""Tests for training script helper functions.

Tests warmup_lambda schedule, compute_separation metric, and
smoke-trains with all ablation configs to catch config/model mismatches
before expensive server runs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml


# ── Import helpers directly from training script ──────────────────────────────

# The script isn't a module, so we add scripts/ to path temporarily
import importlib
import sys as _sys
import types


def _load_train_helpers():
    """Import warmup_lambda and compute_separation from train_uq.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_uq", "scripts/train_uq.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    _train_mod = _load_train_helpers()
    warmup_lambda = _train_mod.warmup_lambda
    compute_separation = _train_mod.compute_separation
    _HELPERS_AVAILABLE = True
except Exception:
    _HELPERS_AVAILABLE = False


# ── warmup_lambda ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HELPERS_AVAILABLE, reason="Could not import train_uq.py helpers")
class TestWarmupLambda:
    def test_epoch_zero_returns_point_one(self):
        """At epoch 0, lambda = 0.1 (start of warmup)."""
        val = warmup_lambda(0, warmup_epochs=3)
        assert abs(val - 0.1) < 1e-6, f"Expected 0.1, got {val}"

    def test_epoch_equals_warmup_returns_one(self):
        """At epoch == warmup_epochs, lambda = 1.0."""
        val = warmup_lambda(3, warmup_epochs=3)
        assert abs(val - 1.0) < 1e-6, f"Expected 1.0, got {val}"

    def test_epoch_after_warmup_returns_one(self):
        """After warmup, lambda stays at 1.0."""
        for ep in [4, 10, 100]:
            val = warmup_lambda(ep, warmup_epochs=3)
            assert abs(val - 1.0) < 1e-6, f"Epoch {ep}: expected 1.0, got {val}"

    def test_linear_increase_during_warmup(self):
        """Lambda increases linearly from 0.1 to 1.0 during warmup."""
        warmup_epochs = 5
        vals = [warmup_lambda(ep, warmup_epochs) for ep in range(warmup_epochs)]
        for i in range(len(vals) - 1):
            assert vals[i + 1] > vals[i], f"Lambda should increase: {vals}"

    def test_warmup_zero_epochs_always_one(self):
        """warmup_epochs=0 → lambda is always 1.0."""
        val = warmup_lambda(0, warmup_epochs=0)
        assert val == 1.0


# ── compute_separation ───────────────────────────────────────────────────────

@pytest.mark.skipif(not _HELPERS_AVAILABLE, reason="Could not import train_uq.py helpers")
class TestComputeSeparation:
    def test_perfect_separation(self):
        """High-UQ samples score 1.0, low-UQ score 0.0 → separation = 1.0."""
        labels = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        preds  = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        sep = compute_separation(preds, labels, low_thresh=0.4, high_thresh=0.6)
        assert abs(sep - 1.0) < 1e-6, f"Expected 1.0, got {sep}"

    def test_no_low_samples_returns_nan(self):
        """No samples with label < low_thresh → returns nan."""
        import math
        labels = [0.7, 0.8, 0.9]
        preds  = [0.5, 0.6, 0.7]
        sep = compute_separation(preds, labels, low_thresh=0.4, high_thresh=0.6)
        assert math.isnan(sep), f"Expected nan, got {sep}"

    def test_no_high_samples_returns_nan(self):
        """No samples with label > high_thresh → returns nan."""
        import math
        labels = [0.1, 0.2, 0.3]
        preds  = [0.5, 0.4, 0.3]
        sep = compute_separation(preds, labels, low_thresh=0.4, high_thresh=0.6)
        assert math.isnan(sep), f"Expected nan, got {sep}"

    def test_zero_separation_when_uniform_preds(self):
        """Identical predictions → gap = 0."""
        labels = [0.1, 0.9]
        preds  = [0.5, 0.5]
        sep = compute_separation(preds, labels, low_thresh=0.4, high_thresh=0.6)
        assert abs(sep) < 1e-6, f"Expected 0.0, got {sep}"


# ── Ablation config smoke tests ──────────────────────────────────────────────

ABLATION_CONFIGS = [
    "configs/uq_train.yaml",
    "configs/uq_ablation_no_stat.yaml",
    "configs/uq_ablation_no_decoder.yaml",
    "configs/uq_ablation_no_ranking.yaml",
    "configs/uq_ablation_no_cal.yaml",
]


def _run_smoke(config_path: str, tmp_path: Path, timeout: int = 180):
    import os
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    save_dir = tmp_path / "ckpt"
    cfg["logging"]["save_dir"] = str(save_dir)
    cfg["logging"]["save_interval"] = 1
    tmp_cfg = tmp_path / "cfg.yaml"
    tmp_cfg.write_text(yaml.dump(cfg))

    # Ensure project root is on PYTHONPATH so uq_estimator is importable
    env = os.environ.copy()
    project_root = str(Path(__file__).parent.parent)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}:{existing_pp}" if existing_pp else project_root

    result = subprocess.run(
        [sys.executable, "scripts/train_uq.py",
         "--config", str(tmp_cfg),
         "--mock", "--smoke"],
        capture_output=True, text=True, timeout=timeout,
        cwd=project_root, env=env,
    )
    return result, save_dir


@pytest.mark.parametrize("config_path", ABLATION_CONFIGS)
def test_ablation_smoke_train(config_path: str, tmp_path: Path):
    """Each ablation config completes a 2-epoch mock smoke train without errors."""
    result, save_dir = _run_smoke(config_path, tmp_path)
    assert result.returncode == 0, (
        f"{config_path} smoke train failed:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert (save_dir / "best.pt").exists(), \
        f"{config_path}: best.pt not created"


@pytest.mark.parametrize("config_path", ABLATION_CONFIGS)
def test_ablation_smoke_no_nan(config_path: str, tmp_path: Path):
    """Loss values must not be NaN or Inf for any ablation config.

    Checks train_loss and val_loss fields specifically — not the full output
    (which contains 'n/a' for Spearman when scipy is absent).
    """
    import re
    import math

    result, _ = _run_smoke(config_path, tmp_path)
    assert result.returncode == 0, (
        f"{config_path} failed:\n{result.stdout}\n{result.stderr}"
    )

    # Extract all train_loss / val_loss values from epoch log lines
    # Format: "Epoch NN/MM | train_loss: X.XXXX | val_loss: Y.YYYY | ..."
    losses = re.findall(
        r"(?:train_loss|val_loss):\s+([0-9eE+\-.nan]+)",
        result.stdout, re.IGNORECASE,
    )
    assert len(losses) >= 2, (
        f"{config_path}: Expected at least 2 loss values in output, found: {losses}\n"
        f"stdout:\n{result.stdout}"
    )
    for raw in losses:
        val = float(raw)
        assert not math.isnan(val), f"{config_path}: NaN loss detected: {raw}"
        assert not math.isinf(val), f"{config_path}: Inf loss detected: {raw}"
