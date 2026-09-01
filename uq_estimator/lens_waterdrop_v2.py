"""Auditable real-silhouette-derived windshield waterdrop renderer.

The only real-data field used by this module is a binary waterdrop silhouette.
Alpha, displacement, refraction, crescent shading, and highlights are analytic
simulation fields.  They must not be described as real optical ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCHEMA = "orion.lens_waterdrop.v2"
REQUIRED_RESOLUTION = (1600, 900)  # width, height
PROFILES: dict[str, dict[str, float]] = {
    "light": {
        "body_alpha": 0.62,
        "inversion_strength": 0.42,
        "max_displacement_px": 42.0,
        "normal_displacement_px": 4.0,
        "interior_blur_sigma": 0.75,
        "dark_crescent": 0.12,
        "highlight": 0.24,
    },
    "medium": {
        "body_alpha": 0.78,
        "inversion_strength": 0.64,
        "max_displacement_px": 68.0,
        "normal_displacement_px": 7.0,
        "interior_blur_sigma": 1.05,
        "dark_crescent": 0.18,
        "highlight": 0.34,
    },
    "heavy": {
        "body_alpha": 0.90,
        "inversion_strength": 0.86,
        "max_displacement_px": 96.0,
        "normal_displacement_px": 11.0,
        "interior_blur_sigma": 1.35,
        "dark_crescent": 0.24,
        "highlight": 0.44,
    },
}


@dataclass(frozen=True)
class LensWaterdropResultV2:
    image: np.ndarray
    silhouette: np.ndarray
    alpha: np.ndarray
    displacement_px: np.ndarray
    refracted: np.ndarray
    edge_contribution: np.ndarray
    highlight_contribution: np.ndarray
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_verified_silhouette(
    mask_path: Path, metadata_path: Path, width: int, height: int
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "orion.real_waterdrop_mask_bank.v1":
        raise ValueError("mask-bank metadata schema differs")
    matches = [item for item in metadata.get("assets", []) if item.get("file") == mask_path.name]
    if len(matches) != 1:
        raise ValueError("mask is not uniquely listed in the frozen mask bank")
    expected = str(matches[0]["sha256"])
    observed = _sha256(mask_path)
    if observed != expected:
        raise ValueError("waterdrop silhouette SHA-256 differs from frozen source")
    raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError("unable to read waterdrop silhouette")
    if set(np.unique(raw).tolist()) - {0, 255}:
        raise ValueError("waterdrop source mask is not binary")
    resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST)
    silhouette = resized >= 128
    if not silhouette.any():
        raise ValueError("waterdrop source mask is empty")
    return silhouette, {
        "file": mask_path.name,
        "member": matches[0]["member"],
        "sha256": observed,
        "source_width": int(matches[0]["width"]),
        "source_height": int(matches[0]["height"]),
        "source_record": metadata["source"]["record"],
        "source_license": metadata["source"]["license"],
    }


def _component_optics(
    silhouette: np.ndarray, profile: dict[str, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive alpha, displacement, signed edge, and highlight fields."""
    height, width = silhouette.shape
    binary = silhouette.astype(np.uint8)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    alpha = np.zeros((height, width), dtype=np.float32)
    displacement = np.zeros((height, width, 2), dtype=np.float32)
    edge = np.zeros((height, width), dtype=np.float32)
    highlight = np.zeros((height, width), dtype=np.float32)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    for component in range(1, component_count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < 8:
            continue
        component_mask = labels == component
        local = component_mask.astype(np.uint8)
        distance = cv2.distanceTransform(local, cv2.DIST_L2, 5)
        maximum = float(distance.max())
        if maximum <= 0.0:
            continue
        depth = np.clip(distance / maximum, 0.0, 1.0).astype(np.float32)
        # The inward ramp keeps every derived field exactly inside the source
        # silhouette while avoiding a pasted alpha edge.
        inward = np.clip(distance / max(2.0, 0.065 * maximum), 0.0, 1.0)
        component_alpha = profile["body_alpha"] * (0.58 + 0.42 * np.sqrt(depth)) * inward
        alpha = np.maximum(alpha, component_alpha.astype(np.float32))

        center_x, center_y = (float(value) for value in centroids[component])
        dx = xx - center_x
        dy = yy - center_y
        target_dx = -profile["inversion_strength"] * dx
        target_dy = -profile["inversion_strength"] * dy
        inverse_offset_x = target_dx - dx
        inverse_offset_y = target_dy - dy
        inverse_norm = np.sqrt(inverse_offset_x**2 + inverse_offset_y**2) + 1e-6
        cap = np.minimum(1.0, profile["max_displacement_px"] / inverse_norm)
        core_weight = np.power(depth, 0.82) * inward

        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5)
        grad_norm = np.sqrt(grad_x**2 + grad_y**2) + 1e-6
        normal_x = grad_x / grad_norm
        normal_y = grad_y / grad_norm
        rim_weight = np.exp(-((depth - 0.22) / 0.16) ** 2).astype(np.float32)
        offset_x = core_weight * inverse_offset_x * cap + (
            profile["normal_displacement_px"] * rim_weight * normal_x
        )
        offset_y = core_weight * inverse_offset_y * cap + (
            profile["normal_displacement_px"] * rim_weight * normal_y
        )
        displacement[..., 0][component_mask] = offset_x[component_mask]
        displacement[..., 1][component_mask] = offset_y[component_mask]

        half_width = max(1.0, 0.5 * float(stats[component, cv2.CC_STAT_WIDTH]))
        half_height = max(1.0, 0.5 * float(stats[component, cv2.CC_STAT_HEIGHT]))
        nx = dx / half_width
        ny = dy / half_height
        ring = np.exp(-((depth - 0.10) / 0.085) ** 2).astype(np.float32) * inward
        lower_right = np.clip(0.70 * nx + 0.72 * ny, 0.0, 1.0)
        upper_left = np.clip(-0.76 * nx - 0.65 * ny, 0.0, 1.0)
        component_edge = -profile["dark_crescent"] * ring * (0.35 + 0.65 * lower_right)
        # A narrow directional rim plus a bounded, elongated specular lobe.
        spec_x = (nx + 0.33) / 0.20
        spec_y = (ny + 0.36) / 0.13
        specular_lobe = np.exp(-(spec_x**2 + spec_y**2)).astype(np.float32)
        component_highlight = profile["highlight"] * (
            0.56 * ring * upper_left + 0.44 * specular_lobe * inward
        )
        edge[component_mask] = np.minimum(edge[component_mask], component_edge[component_mask])
        highlight[component_mask] = np.maximum(
            highlight[component_mask], component_highlight[component_mask]
        )

    outside = ~silhouette
    alpha[outside] = 0.0
    displacement[outside] = 0.0
    edge[outside] = 0.0
    highlight[outside] = 0.0
    return alpha, displacement, edge, highlight


def apply_lens_waterdrop_v2(
    image: np.ndarray,
    *,
    mask_path: str | Path,
    metadata_path: str | Path,
    profile: str,
    require_resolution: bool = True,
) -> LensWaterdropResultV2:
    """Apply fixed lens-space waterdrop optics before any review downsample.

    No actor state, boxes, semantic labels, route progress, or per-frame random
    values enter this function. Reusing the same source mask and profile across
    frames therefore gives a temporally stable lens-space corruption.
    """
    if profile not in PROFILES:
        raise ValueError("profile must be light, medium, or heavy")
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise TypeError("image must be an HxWx3 numpy array")
    if image.dtype != np.uint8:
        raise TypeError("image must use uint8 RGB values")
    height, width = image.shape[:2]
    if require_resolution and (width, height) != REQUIRED_RESOLUTION:
        raise ValueError("lens_waterdrop_v2 must run at 1600x900 before review downsample")

    mask_file = Path(mask_path).resolve()
    metadata_file = Path(metadata_path).resolve()
    silhouette, source = _load_verified_silhouette(
        mask_file, metadata_file, width, height
    )
    settings = PROFILES[profile]
    alpha, displacement, edge, highlight = _component_optics(silhouette, settings)
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
    sigma = float(settings["interior_blur_sigma"])
    refracted = cv2.GaussianBlur(refracted, (0, 0), sigmaX=sigma, sigmaY=sigma)
    source_float = image.astype(np.float32) / 255.0
    refracted_float = refracted.astype(np.float32) / 255.0
    composed = source_float * (1.0 - alpha[..., None]) + refracted_float * alpha[..., None]
    composed += edge[..., None] + highlight[..., None]
    composed = np.clip(np.round(composed * 255.0), 0, 255).astype(np.uint8)
    # The source silhouette is a hard provenance boundary.  This also makes
    # the "no effect outside the annotated drop" property exactly testable.
    composed[~silhouette] = image[~silhouette]

    displacement_norm = np.linalg.norm(displacement, axis=2)
    return LensWaterdropResultV2(
        image=composed,
        silhouette=silhouette,
        alpha=alpha,
        displacement_px=displacement,
        refracted=refracted,
        edge_contribution=edge,
        highlight_contribution=highlight,
        metadata={
            "schema": SCHEMA,
            "profile": profile,
            "resolution": [width, height],
            "source_silhouette": source,
            "real_data_scope": "binary_silhouette_only",
            "derived_fields": [
                "soft_alpha",
                "two_channel_displacement_px",
                "refracted_rgb",
                "signed_edge_crescent",
                "bounded_highlight",
            ],
            "optics_description": "real-mask-derived analytic optics, not physical ground truth",
            "placement_policy": "fixed_normalized_lens_coordinates_actor_independent",
            "temporal_policy": "constant_field_for_fixed_mask_and_profile",
            "source_mask_fraction": float(silhouette.mean()),
            "alpha_max": float(alpha.max()),
            "displacement_px_max": float(displacement_norm.max()),
            "displacement_px_mean_inside": float(displacement_norm[silhouette].mean()),
            "edge_min": float(edge.min()),
            "highlight_max": float(highlight.max()),
            "profile_parameters": settings,
        },
    )
