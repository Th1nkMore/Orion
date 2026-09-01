#!/usr/bin/env python3
"""Render legacy, external analytic, and CARLA-native glare on one image stage.

AlbumentationsX is an optional, isolated visual-development comparator.  It is
never imported by the ORION runtime or training pipeline and is not required
for the CARLA-native held-out glare gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

try:
    import albumentations as A
except ModuleNotFoundError:
    A = None

try:
    from analyze_native_glare_bakeoff import (
        PROFILES,
        _frame_metrics,
        _load_rows,
        _match_rows,
        _mean,
        _read_image,
        _save_gif,
        _select_event_rows,
        _select_progress_rows,
        _tile,
    )
except ModuleNotFoundError:
    from scripts.analyze_native_glare_bakeoff import (
        PROFILES,
        _frame_metrics,
        _load_rows,
        _match_rows,
        _mean,
        _read_image,
        _save_gif,
        _select_event_rows,
        _select_progress_rows,
        _tile,
    )


AUTOMOLD_SHA256 = "4f75a9ef25dcd539833f2d276c6bc40553e17456f7ffba8f2094ad6065863eca"
SCHEMA = "orion.glare_method_visual_bakeoff.v1"
SELECTED_IMAGES_SCHEMA = "orion.glare_method_selected_images.v1"
SEVERITIES = ("light", "medium", "heavy")
ALBUMENTATIONSX_PARAMETERS = {
    "light": {
        "intensity_range": (0.20, 0.20),
        "num_ghosts_range": (3, 3),
        "num_rays_range": (4, 4),
        "bloom_range": (0.01, 0.01),
    },
    "medium": {
        "intensity_range": (0.45, 0.45),
        "num_ghosts_range": (5, 5),
        "num_rays_range": (6, 6),
        "bloom_range": (0.025, 0.025),
    },
    "heavy": {
        "intensity_range": (0.70, 0.70),
        "num_ghosts_range": (7, 7),
        "num_rays_range": (8, 8),
        "bloom_range": (0.05, 0.05),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_automold(path: Path):
    if _sha256(path) != AUTOMOLD_SHA256:
        raise RuntimeError("CVPR 2023 Automold source hash differs from audited commit")
    # The frozen upstream file predates NumPy 1.24.  These aliases are a local
    # runtime compatibility shim; the upstream source remains byte-identical.
    if "float" not in np.__dict__:
        np.float = float
    if "int" not in np.__dict__:
        np.int = int
    spec = importlib.util.spec_from_file_location("cvpr2023_automold", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_selected_image(reference: dict, base: Path) -> Path:
    path = Path(str(reference.get("path", "")))
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.is_file() or _sha256(path) != reference.get("sha256"):
        raise RuntimeError("selected glare image is absent or has a SHA-256 mismatch")
    return path


def _load_selected_images(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SELECTED_IMAGES_SCHEMA:
        raise RuntimeError("unsupported selected glare image manifest")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != set(PROFILES):
        raise RuntimeError("selected glare image manifest lacks the four profiles")
    lengths = {profile: len(rows) for profile, rows in profiles.items()}
    if len(set(lengths.values())) != 1 or not 3 <= next(iter(lengths.values())) <= 5:
        raise RuntimeError("selected glare profiles must contain the same three to five poses")
    resolved = {
        profile: [
            {
                "front": str(_resolve_selected_image(row, path.parent)),
                "capture_index": int(row["capture_index"]),
            }
            for row in rows
        ]
        for profile, rows in profiles.items()
    }
    clean_indices = [row["capture_index"] for row in resolved["clean"]]
    if any(
        [row["capture_index"] for row in resolved[profile]] != clean_indices
        for profile in SEVERITIES
    ):
        raise RuntimeError("selected glare profiles are not aligned by capture index")
    native_matches = {
        profile: [{"candidate": row} for row in resolved[profile]]
        for profile in SEVERITIES
    }
    return resolved["clean"], native_matches


def _legacy_white_patch(image: np.ndarray, roi: Sequence[float], alpha: float) -> np.ndarray:
    result = image.copy()
    height, width = result.shape[:2]
    left, top, right, bottom = roi
    x0, y0 = int(round(left * width)), int(round(top * height))
    x1, y1 = int(round(right * width)), int(round(bottom * height))
    patch = result[y0:y1, x0:x1].astype(np.float32)
    result[y0:y1, x0:x1] = np.clip(
        patch * (1.0 - alpha) + 255.0 * alpha, 0, 255
    ).astype(np.uint8)
    return result


def _cvpr_sun(image: np.ndarray, automold, severity_index: int) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    center = np.array([0.50 * width, 0.35 * height])
    toward_center = np.array([width / 2.0, height / 2.0]) - center
    angle = math.atan2(float(toward_center[1]), float(toward_center[0]))
    # Replay the same optics parameters on every event frame.  This removes
    # framewise random flicker while preserving the published primitive.
    np.random.seed(20260831)
    rendered, _ = automold.add_sun_flare(
        rgb,
        flare_center=center,
        angle=angle,
        no_of_flare_circles=1,
        src_radius=(30, 50, 70)[severity_index],
        src_color=(255, 255, 255),
    )
    return cv2.cvtColor(np.clip(rendered, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _albumentationsx_lensflare(image: np.ndarray, severity: str) -> np.ndarray:
    if A is None or not hasattr(A, "LensFlare"):
        raise RuntimeError("AlbumentationsX LensFlare is unavailable")
    parameters = ALBUMENTATIONSX_PARAMETERS[severity]
    # Recreate the transform with the same seed for every event frame.  This
    # deliberately holds its screen-space optics sample fixed for the visual
    # bake-off; it is not the temporally coherent formal corruption mechanism.
    transform = A.Compose(
        [
            A.LensFlare(
                flare_roi=(0.45, 0.25, 0.55, 0.35),
                p=1.0,
                **parameters,
            )
        ],
        seed=20260831,
    )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rendered = transform(image=rgb)["image"]
    return cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)


def render(
    root: Path,
    protocol: Path,
    automold_path: Path,
    output: Path,
    *,
    require_albumentationsx: bool = False,
    selected_images_manifest: Optional[Path] = None,
) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite %s" % output)
    output.mkdir(parents=True)
    spec = json.loads(protocol.read_text(encoding="utf-8"))
    roi = spec["existing_clean_source"]["hazard_roi_normalized"]
    alphas = spec["methods"]["legacy_local_glare"]["alpha_by_severity"]
    automold = _load_automold(automold_path)
    has_albumentationsx = A is not None and hasattr(A, "LensFlare")
    if require_albumentationsx and not has_albumentationsx:
        raise RuntimeError("--require-albumentationsx was set but LensFlare is unavailable")
    if selected_images_manifest is not None:
        references, native_matches = _load_selected_images(selected_images_manifest)
    else:
        rows = {profile: _load_rows(root / "captures" / profile) for profile in PROFILES}
        targets = spec["existing_clean_source"].get("native_capture_target_route_progress")
        references = (
            _select_progress_rows(rows["clean"], targets)
            if targets else _select_event_rows(rows["clean"], count=5)
        )
        native_matches = {
            profile: _match_rows(references, rows[profile]) for profile in SEVERITIES
        }
    if any(len(matches) != len(references) for matches in native_matches.values()):
        raise RuntimeError("native profiles do not all pose-match the five event frames")

    method_metrics = {}
    visuals = {}
    for severity_index, severity in enumerate(SEVERITIES):
        gif_frames = []
        sheet_rows = []
        for frame_index, reference in enumerate(references):
            clean = _read_image(reference["front"])
            legacy = _legacy_white_patch(clean, roi, float(alphas[severity]))
            cvpr = _cvpr_sun(clean, automold, severity_index)
            native = _read_image(native_matches[severity][frame_index]["candidate"]["front"])
            methods = {
                "legacy_white_patch": legacy,
                "cvpr2023_sun": cvpr,
            }
            if has_albumentationsx:
                methods["albumentationsx_lensflare"] = _albumentationsx_lensflare(
                    clean, severity
                )
            methods["carla_native"] = native
            for method, image in methods.items():
                values = _frame_metrics(clean, image, roi)
                values["clean_capture_index"] = reference["capture_index"]
                method_metrics.setdefault(severity, {}).setdefault(method, []).append(values)
            ordered = [clean, legacy, cvpr]
            labels = ["clean low-sun", "legacy white patch", "CVPR 2023 sun"]
            if has_albumentationsx:
                ordered.append(methods["albumentationsx_lensflare"])
                labels.append("AlbumentationsX LensFlare")
            ordered.append(native)
            labels.append("CARLA native")
            tiled = _tile(ordered, labels, cell=(480, 270))
            gif_frames.append(tiled)
            sheet_rows.append(tiled)
        gif_path = output / ("route151_glare_methods_%s.gif" % severity)
        sheet_path = output / ("route151_glare_methods_%s.png" % severity)
        _save_gif(gif_frames, gif_path)
        cv2.imwrite(str(sheet_path), np.concatenate(sheet_rows, axis=0))
        visuals[severity] = {
            "gif": str(gif_path.resolve()),
            "contact_sheet": str(sheet_path.resolve()),
        }

    summary = {}
    metric_names = (
        "mean_absolute_pixel_delta",
        "saturated_pixel_fraction_changed",
        "hazard_roi_contrast_ratio",
        "hazard_roi_edge_visibility_ratio",
        "rectangular_boundary_artifact_score",
    )
    for severity in SEVERITIES:
        summary[severity] = {}
        for method, values in method_metrics[severity].items():
            summary[severity][method] = {
                metric: _mean([row[metric] for row in values]) for metric in metric_names
            }

    payload = {
        "schema": SCHEMA,
        "comparison_image_stage": "raw CARLA RGB sensor PNG before ORION JPEG/preprocessing",
        "clean_base": "CARLA native low-sun with front lens_flare_intensity=0 and bloom_intensity=0",
        "matched_event_frame_count": len(references),
        "hazard_roi_normalized": roi,
        "selected_images_manifest": (
            {
                "path": str(selected_images_manifest.resolve()),
                "sha256": _sha256(selected_images_manifest.resolve()),
            }
            if selected_images_manifest is not None
            else None
        ),
        "cvpr2023": {
            "upstream_commit": "48c23f77fe82beab599f8248b7794928334a3fb5",
            "automold_path": str(automold_path.resolve()),
            "automold_sha256": _sha256(automold_path),
            "license": "MIT",
            "temporal_policy": "fixed screen-space flare center and replayed NumPy random state per frame",
            "formal_role": "published analytic comparator, not unique formal glare mechanism",
        },
        "albumentationsx": {
            "included": has_albumentationsx,
            "distribution_version": (
                importlib.metadata.version("albumentationsx")
                if has_albumentationsx
                else None
            ),
            "license": "AGPL-3.0-only or separately negotiated commercial terms",
            "parameters_by_severity": ALBUMENTATIONSX_PARAMETERS,
            "temporal_policy": "fixed screen-space optics sample by recreating Compose(seed=20260831) per frame",
            "formal_role": "isolated visual-development comparator only; not vendored, not an ORION dependency and not a formal held-out family",
            "reason_if_absent": (
                None
                if has_albumentationsx
                else "optional package unavailable in the executing environment"
            ),
        },
        "summary": summary,
        "per_frame": method_metrics,
        "visuals": visuals,
        "outcome_fields_read": [],
        "claim_boundary": "visual implementation comparison only; no UQ, planning, collision, TTC, or safety claim",
    }
    report = output / "glare_method_visual_bakeoff.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report), "visuals": visuals}, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--automold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-albumentationsx", action="store_true")
    parser.add_argument("--selected-images-manifest", type=Path)
    args = parser.parse_args()
    render(
        args.root.resolve(),
        args.protocol.resolve(),
        args.automold.resolve(),
        args.output.resolve(),
        require_albumentationsx=args.require_albumentationsx,
        selected_images_manifest=(
            args.selected_images_manifest.resolve()
            if args.selected_images_manifest is not None
            else None
        ),
    )


if __name__ == "__main__":
    main()
