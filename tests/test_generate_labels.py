"""Tests for scripts/generate_labels.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.generate_labels import compute_uq_score
from tests.fixtures import create_mock_feature_dir


def _make_tokens_in_empirical_range(
    n_views: int, n_patches: int, d_patch: int,
    max_mean_target: float,
    consistent: bool,
) -> torch.Tensor:
    """Create tokens whose amax(dim=-1).mean() is near max_mean_target.

    _compute_max_mean_score uses empirical range [13, 16]. We scale standard
    normal tokens so E[max over D=d_patch dims] ≈ max_mean_target.

    For N(0,σ) with d_patch dims, E[max] ≈ σ * sqrt(2 * ln(d_patch)).
    Solve for σ: σ = max_mean_target / sqrt(2 * ln(d_patch)).
    """
    import math
    scale = max_mean_target / math.sqrt(2.0 * math.log(d_patch))
    if consistent:
        base = torch.randn(1, n_patches, d_patch) * scale
        noise = torch.randn(n_views, n_patches, d_patch) * (scale * 0.02)
        tokens = base.expand(n_views, -1, -1) + noise
    else:
        tokens = torch.randn(n_views, n_patches, d_patch) * scale
    return tokens


@pytest.fixture
def normal_feature(tmp_path: Path) -> Path:
    """Create a normal-scene feature file with tokens in the expected empirical range.

    Normal = high max_mean (≈15.5, clear features) + high cross-view consistency.
    """
    n_views, n_patches, d_patch = 6, 256, 1152
    tokens = _make_tokens_in_empirical_range(
        n_views, n_patches, d_patch,
        max_mean_target=16.0,  # top of empirical range → max_mean_score≈0 → lowest uncertainty
        consistent=True,
    )
    p = tmp_path / "normal.pt"
    torch.save({"tokens": tokens, "scene_type": "normal"}, str(p))
    return p


@pytest.fixture
def adverse_feature(tmp_path: Path) -> Path:
    """Create an adverse-scene feature file with tokens in the expected empirical range.

    Adverse = low max_mean (≈13.1, degraded features) + low cross-view consistency.
    """
    n_views, n_patches, d_patch = 6, 256, 1152
    tokens = _make_tokens_in_empirical_range(
        n_views, n_patches, d_patch,
        max_mean_target=13.1,  # low end of range → high uncertainty from max_mean
        consistent=False,
    )
    p = tmp_path / "adverse.pt"
    torch.save({"tokens": tokens, "scene_type": "adverse"}, str(p))
    return p


def test_compute_uq_score_normal(normal_feature: Path) -> None:
    """Normal scene should produce score below the midpoint (< 0.5).

    With max_mean at the top of the empirical range and high cross-view
    consistency, the score should stay clearly in the lower half.
    """
    fname, score, scene_type = compute_uq_score(normal_feature)
    assert score is not None
    assert scene_type == "normal"
    assert score < 0.50, f"Normal scene score {score:.4f} should be < 0.50"


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
