"""Tests for scripts/generate_labels.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.generate_labels import compute_uq_score
from tests.fixtures import create_mock_feature_dir


@pytest.fixture
def normal_feature(tmp_path: Path) -> Path:
    """Create a single normal-scene feature file."""
    from tests.fixtures import _make_normal_sample

    data = _make_normal_sample(n_views=6, n_patches=256, d_patch=1152, img_h=64, img_w=64)
    p = tmp_path / "normal.pt"
    torch.save(data, str(p))
    return p


@pytest.fixture
def adverse_feature(tmp_path: Path) -> Path:
    """Create a single adverse-scene feature file."""
    from tests.fixtures import _make_adverse_sample

    data = _make_adverse_sample(n_views=6, n_patches=256, d_patch=1152, img_h=64, img_w=64)
    p = tmp_path / "adverse.pt"
    torch.save(data, str(p))
    return p


def test_compute_uq_score_normal(normal_feature: Path) -> None:
    """Normal scene should produce score <= 0.45 after calibration."""
    fname, score, scene_type = compute_uq_score(normal_feature)
    assert score is not None
    assert scene_type == "normal"
    assert score <= 0.45, f"Normal scene score {score:.4f} should be <= 0.45"


def test_compute_uq_score_adverse(adverse_feature: Path) -> None:
    """Adverse scene should produce score >= 0.55 after calibration."""
    fname, score, scene_type = compute_uq_score(adverse_feature)
    assert score is not None
    assert scene_type == "adverse"
    assert score >= 0.55, f"Adverse scene score {score:.4f} should be >= 0.55"


def test_score_range(normal_feature: Path, adverse_feature: Path) -> None:
    """All scores must be in [0, 1]."""
    for path in [normal_feature, adverse_feature]:
        _, score, _ = compute_uq_score(path)
        assert score is not None
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1]"


def test_missing_image_field(tmp_path: Path) -> None:
    """Feature file without 'image' key should not error."""
    tokens = torch.randn(6, 256, 1152)  # [N_views, N_patches, D]
    p = tmp_path / "no_image.pt"
    torch.save({"tokens": tokens}, str(p))

    fname, score, scene_type = compute_uq_score(p)
    assert score is not None
    assert 0.0 <= score <= 1.0
    assert scene_type == "unknown"


def test_dry_run(tmp_path: Path) -> None:
    """--dry_run should exit 0 and not create an output file."""
    feature_dir = create_mock_feature_dir(tmp_path, n_files=12)
    output_file = tmp_path / "labels" / "uq_labels.pt"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_labels.py",
            "--feature_dir", str(feature_dir),
            "--output_file", str(output_file),
            "--dry_run",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"dry_run failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert not output_file.exists(), "dry_run should not create output file"
