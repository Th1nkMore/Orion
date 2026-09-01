"""Scene-content-safe waterdrop template extracted from a published pair.

The source pair is published synthetic data.  The reusable template retains
only a scalar alpha field, a two-channel displacement field, a binary support,
and a per-component zero-mean scalar luminance residual.  Source RGB is never
copied into target scenes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


SCHEMA = "orion.lens_waterdrop_paired_template.v1"
REQUIRED_RESOLUTION = (1600, 900)
PROFILES = {"light": 0.58, "medium": 0.80, "heavy": 1.0}


@dataclass(frozen=True)
class WaterdropAppearanceTemplate:
    support: np.ndarray
    alpha: np.ndarray
    displacement_px: np.ndarray
    luminance_residual: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PairedWaterdropResult:
    image: np.ndarray
    support: np.ndarray
    alpha: np.ndarray
    displacement_px: np.ndarray
    luminance_residual: np.ndarray
    refracted: np.ndarray
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_asset(path: Path, metadata: dict[str, Any], expected_role: str) -> dict[str, Any]:
    matches = [row for row in metadata.get("assets", []) if row.get("file") == path.name]
    if len(matches) != 1:
        raise ValueError("source asset is not uniquely listed in the reference bank")
    row = matches[0]
    if row.get("role") != expected_role:
        raise ValueError("source asset has the wrong frozen role")
    if _sha256(path) != row.get("sha256"):
        raise ValueError("source asset SHA-256 differs from frozen metadata")
    return row


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _component_balance(
    residual: np.ndarray, alpha: np.ndarray
) -> tuple[np.ndarray, list[dict[str, float]]]:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (alpha > 0.08).astype(np.uint8), 8
    )
    balanced = residual.copy()
    audit = []
    for component in range(1, component_count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        mask = labels == component
        weights = alpha[mask]
        before = float((residual[mask] * weights).sum() / (weights.sum() + 1e-6))
        coefficient = float(
            (residual[mask] * weights).sum() / ((weights * weights).sum() + 1e-6)
        )
        balanced[mask] = np.clip(
            residual[mask] - coefficient * alpha[mask], -0.28, 0.28
        )
        after = float((balanced[mask] * weights).sum() / (weights.sum() + 1e-6))
        audit.append(
            {
                "area": area,
                "weighted_mean_before": before,
                "weighted_mean_after": after,
            }
        )
    balanced = cv2.GaussianBlur(balanced, (0, 0), 0.35)
    balanced *= np.clip(alpha * 1.4, 0.0, 1.0)
    # Smoothing and tapering can reintroduce a small component DC term. Remove
    # it once more along the same alpha basis without retaining chroma.
    for component in range(1, component_count):
        mask = labels == component
        if int(stats[component, cv2.CC_STAT_AREA]) < 4:
            continue
        weights = alpha[mask]
        coefficient = float(
            (balanced[mask] * weights).sum() / ((weights * weights).sum() + 1e-6)
        )
        balanced[mask] = np.clip(
            balanced[mask] - coefficient * alpha[mask], -0.28, 0.28
        )
    balanced[alpha <= 0.04] = 0.0
    return balanced.astype(np.float32), audit


def extract_paired_waterdrop_template(
    *, clean_path: str | Path, rainy_path: str | Path, metadata_path: str | Path
) -> WaterdropAppearanceTemplate:
    clean_file, rainy_file, metadata_file = (
        Path(clean_path).resolve(),
        Path(rainy_path).resolve(),
        Path(metadata_path).resolve(),
    )
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("schema") != "orion.icra2023_waterdrop_reference_bank.v1":
        raise ValueError("reference-bank metadata schema differs")
    clean_source = _verified_asset(clean_file, metadata, "paired_template_source")
    rainy_source = _verified_asset(rainy_file, metadata, "paired_template_source")
    if "/clean_vid/" not in clean_source["member"] or "/rainy_vid/" not in rainy_source["member"]:
        raise ValueError("paired source roles do not identify clean/rainy members")
    if clean_source["member"].replace("/clean_vid/", "/rainy_vid/") != rainy_source["member"]:
        raise ValueError("clean and rainy source members are not an exact pair")
    clean, rainy = _read_rgb(clean_file), _read_rgb(rainy_file)
    if clean.shape != rainy.shape or clean.shape != (540, 960, 3):
        raise ValueError("paired source must be aligned 960x540 RGB")

    clean_float, rainy_float = clean.astype(np.float32), rainy.astype(np.float32)
    difference = np.sqrt(np.mean((rainy_float - clean_float) ** 2, axis=2))
    alpha = np.clip((difference - 2.0) / 24.0, 0.0, 1.0).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)
    alpha = np.where(alpha > 0.04, alpha, 0.0).astype(np.float32)
    support = alpha > 0.04
    if not support.any():
        raise ValueError("paired source yielded an empty waterdrop support")

    clean_gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
    rainy_gray = cv2.cvtColor(rainy, cv2.COLOR_RGB2GRAY)
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    estimator.setUseSpatialPropagation(True)
    displacement = estimator.calc(rainy_gray, clean_gray, None).astype(np.float32)
    magnitude = np.linalg.norm(displacement, axis=2)
    displacement *= np.minimum(1.0, 18.0 / (magnitude + 1e-6))[..., None]
    displacement = cv2.GaussianBlur(displacement, (0, 0), 1.4)
    displacement *= alpha[..., None]
    displacement[~support] = 0.0

    height, width = clean.shape[:2]
    map_x, map_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    refracted_clean = cv2.remap(
        clean,
        map_x + displacement[..., 0],
        map_y + displacement[..., 1],
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    raw_luminance = (
        rainy_gray.astype(np.float32)
        - cv2.cvtColor(refracted_clean, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ) / 255.0
    raw_luminance = np.clip(cv2.GaussianBlur(raw_luminance, (0, 0), 0.5), -0.35, 0.35)
    raw_luminance *= alpha
    luminance_residual, component_audit = _component_balance(raw_luminance, alpha)

    reconstruction_float = clean_float / 255.0
    reconstruction_float = (
        reconstruction_float * (1.0 - alpha[..., None])
        + refracted_clean.astype(np.float32) / 255.0 * alpha[..., None]
        + luminance_residual[..., None]
    )
    reconstruction = np.clip(np.round(reconstruction_float * 255.0), 0, 255).astype(np.uint8)
    reconstruction[~support] = clean[~support]
    reconstruction_mae = float(np.abs(reconstruction.astype(np.float32) - rainy_float).mean())
    displacement_norm = np.linalg.norm(displacement, axis=2)
    weighted_component_bias_max = max(
        (abs(row["weighted_mean_after"]) for row in component_audit), default=0.0
    )
    return WaterdropAppearanceTemplate(
        support=support,
        alpha=alpha,
        displacement_px=displacement,
        luminance_residual=luminance_residual,
        metadata={
            "schema": SCHEMA,
            "source_kind": "published_synthetic_clean_rainy_pair",
            "source_license_as_declared": metadata["source"]["license_as_declared_by_dataset_card"],
            "source_clean": clean_source,
            "source_rainy": rainy_source,
            "source_resolution": [width, height],
            "support_fraction": float(support.mean()),
            "alpha_mean_inside": float(alpha[support].mean()),
            "alpha_max": float(alpha.max()),
            "displacement_px_max": float(displacement_norm.max()),
            "displacement_px_mean_inside": float(displacement_norm[support].mean()),
            "luminance_residual_min": float(luminance_residual.min()),
            "luminance_residual_max": float(luminance_residual.max()),
            "component_count": len(component_audit),
            "component_weighted_luminance_bias_after_max_abs": weighted_component_bias_max,
            "source_pair_reconstruction_mae_rgb": reconstruction_mae,
            "retained_template_fields": [
                "binary_support",
                "scalar_alpha",
                "two_channel_displacement_px",
                "scalar_component_balanced_luminance_residual",
            ],
            "source_rgb_retained_in_template": False,
            "source_chromatic_residual_retained": False,
            "real_reference_frames_used_in_extraction": False,
            "placement_policy": "fixed_lens_coordinates_actor_independent",
            "temporal_policy": "constant_template_for_fixed_source_pair",
            "claim_boundary": metadata["claim_boundary"],
        },
    )


def apply_paired_waterdrop_template(
    image: np.ndarray,
    *,
    template: WaterdropAppearanceTemplate,
    profile: str,
    require_resolution: bool = True,
) -> PairedWaterdropResult:
    if profile not in PROFILES:
        raise ValueError("profile must be light, medium, or heavy")
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise TypeError("image must be an HxWx3 numpy array")
    if image.dtype != np.uint8:
        raise TypeError("image must use uint8 RGB values")
    height, width = image.shape[:2]
    if require_resolution and (width, height) != REQUIRED_RESOLUTION:
        raise ValueError("paired waterdrop template must run at 1600x900")
    source_height, source_width = template.alpha.shape
    strength = float(PROFILES[profile])
    support = cv2.resize(
        template.support.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    alpha = cv2.resize(template.alpha, (width, height), interpolation=cv2.INTER_CUBIC)
    alpha = np.clip(alpha * strength, 0.0, 1.0).astype(np.float32)
    alpha[~support] = 0.0
    displacement = cv2.resize(
        template.displacement_px, (width, height), interpolation=cv2.INTER_CUBIC
    ).astype(np.float32)
    displacement[..., 0] *= strength * width / float(source_width)
    displacement[..., 1] *= strength * height / float(source_height)
    displacement[~support] = 0.0
    luminance = cv2.resize(
        template.luminance_residual, (width, height), interpolation=cv2.INTER_CUBIC
    ).astype(np.float32)
    luminance *= strength
    luminance[~support] = 0.0

    map_x, map_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    refracted = cv2.remap(
        image,
        map_x + displacement[..., 0],
        map_y + displacement[..., 1],
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    source_float = image.astype(np.float32) / 255.0
    output = (
        source_float * (1.0 - alpha[..., None])
        + refracted.astype(np.float32) / 255.0 * alpha[..., None]
        + luminance[..., None]
    )
    output = np.clip(np.round(output * 255.0), 0, 255).astype(np.uint8)
    output[~support] = image[~support]
    displacement_norm = np.linalg.norm(displacement, axis=2)
    return PairedWaterdropResult(
        image=output,
        support=support,
        alpha=alpha,
        displacement_px=displacement,
        luminance_residual=luminance,
        refracted=refracted,
        metadata={
            "schema": SCHEMA,
            "profile": profile,
            "strength": strength,
            "resolution": [width, height],
            "source_template": template.metadata,
            "source_rgb_copied_to_target": False,
            "retained_application_fields": [
                "scalar_alpha",
                "two_channel_displacement_px",
                "scalar_luminance_residual",
            ],
            "support_fraction": float(support.mean()),
            "changed_pixel_fraction_inside": float(
                np.any(output != image, axis=2)[support].mean()
            ),
            "displacement_px_max": float(displacement_norm.max()),
            "placement_policy": template.metadata["placement_policy"],
            "temporal_policy": template.metadata["temporal_policy"],
        },
    )
