"""Tests for training and validation scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch


TRAIN_CMD = [sys.executable, "scripts/train_uq.py"]
SMOKE_ARGS = ["--mock", "--smoke"]

# Project root — ensure uq_estimator is importable in subprocess
import os as _os
_PROJECT_ROOT = str(Path(__file__).parent.parent)


def _run_train(extra_args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    env = _os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_PROJECT_ROOT}:{existing_pp}" if existing_pp else _PROJECT_ROOT
    return subprocess.run(
        TRAIN_CMD + extra_args,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=_PROJECT_ROOT,
        env=env,
    )


def test_smoke_train(tmp_path: Path) -> None:
    """Mock smoke training completes without errors and creates checkpoints."""
    save_dir = tmp_path / "ckpt"
    # Override save_dir via a temp config
    import yaml

    with open("configs/uq_train.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["logging"]["save_dir"] = str(save_dir)
    cfg["logging"]["save_interval"] = 1
    tmp_cfg = tmp_path / "cfg.yaml"
    tmp_cfg.write_text(yaml.dump(cfg))

    result = _run_train(["--config", str(tmp_cfg)] + SMOKE_ARGS)
    assert result.returncode == 0, f"Train failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    # Checkpoint files should exist
    assert (save_dir / "best.pt").exists(), "best.pt not created"

    # Check for nan/inf in output
    for bad in ["nan", "inf"]:
        assert bad not in result.stdout.lower(), f"Found {bad} in output:\n{result.stdout}"


def test_resume_training(tmp_path: Path) -> None:
    """Resume continues from the saved epoch."""
    import yaml

    save_dir = tmp_path / "ckpt"
    with open("configs/uq_train.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["logging"]["save_dir"] = str(save_dir)
    cfg["logging"]["save_interval"] = 1
    tmp_cfg = tmp_path / "cfg.yaml"
    tmp_cfg.write_text(yaml.dump(cfg))

    # Phase 1: train 2 epochs
    r1 = _run_train(["--config", str(tmp_cfg)] + SMOKE_ARGS)
    assert r1.returncode == 0, f"Phase 1 failed:\n{r1.stdout}\n{r1.stderr}"

    best_ckpt = save_dir / "best.pt"
    assert best_ckpt.exists()

    # Phase 2: resume
    r2 = _run_train(["--config", str(tmp_cfg), "--resume", str(best_ckpt)] + SMOKE_ARGS)
    assert r2.returncode == 0, f"Phase 2 failed:\n{r2.stdout}\n{r2.stderr}"

    # The resumed run should log that it started from the saved epoch.
    # With smoke=2 epochs, after resuming from epoch 2 the loop range is exhausted
    # immediately — so we just verify the resume message appeared.
    assert "Resumed from epoch" in r2.stdout or "resumed from epoch" in r2.stdout.lower(), (
        f"Expected 'Resumed from epoch' in output, got:\n{r2.stdout}"
    )


def test_loss_decreases(tmp_path: Path) -> None:
    """Train 5 epochs on mock data; loss should not be nan and ideally decreases.

    Because mock data is random this test only asserts loss is finite.
    """
    import yaml

    save_dir = tmp_path / "ckpt"
    with open("configs/uq_train.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["logging"]["save_dir"] = str(save_dir)
    cfg["logging"]["save_interval"] = 1
    cfg["training"]["epochs"] = 5
    tmp_cfg = tmp_path / "cfg.yaml"
    tmp_cfg.write_text(yaml.dump(cfg))

    result = _run_train(["--config", str(tmp_cfg), "--mock"])
    assert result.returncode == 0, f"Train failed:\n{result.stdout}\n{result.stderr}"

    # Extract train losses from output lines
    losses = []
    for line in result.stdout.splitlines():
        if "train_loss:" in line:
            # e.g. "... train_loss: 0.1234 | ..."
            part = line.split("train_loss:")[1].split("|")[0].strip()
            losses.append(float(part))

    assert len(losses) >= 2, f"Expected >=2 loss values, got {len(losses)}"
    for l in losses:
        assert not (torch.tensor(l).isnan() or torch.tensor(l).isinf()), f"Bad loss: {l}"
