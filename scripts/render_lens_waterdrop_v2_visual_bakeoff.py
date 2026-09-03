#!/usr/bin/env python3
"""Render the preregistered full-resolution lens_waterdrop_v2 review package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from uq_estimator.lens_waterdrop_v2 import apply_lens_waterdrop_v2


SCHEMA = "orion.lens_waterdrop_v2_visual_bakeoff_result.v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _label(image: Image.Image, text: str, bar: int = 28) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, result.width, bar), fill=(0, 0, 0))
    draw.text((8, 7), text, fill=(255, 255, 255))
    return result


def _horizontal(images: list[Image.Image], labels: list[str]) -> Image.Image:
    labeled = [_label(image, label) for image, label in zip(images, labels)]
    result = Image.new("RGB", (sum(item.width for item in labeled), labeled[0].height))
    left = 0
    for item in labeled:
        result.paste(item, (left, 0))
        left += item.width
    return result


def _save_gif(path: Path, frames: list[Image.Image], fps: int) -> None:
    if not frames:
        raise ValueError("refusing to save empty GIF")
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000.0 / fps)),
        loop=0,
        optimize=False,
    )


def _field_image(values: np.ndarray, *, signed: bool = False) -> Image.Image:
    if signed:
        maximum = max(float(np.abs(values).max()), 1e-6)
        normalized = np.clip(0.5 + 0.5 * values / maximum, 0.0, 1.0)
    else:
        maximum = max(float(values.max()), 1e-6)
        normalized = np.clip(values / maximum, 0.0, 1.0)
    gray = np.round(normalized * 255.0).astype(np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGB")


def _displacement_image(displacement: np.ndarray) -> Image.Image:
    dx, dy = displacement[..., 0], displacement[..., 1]
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = (np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)
    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.round(angle * 179.0).astype(np.uint8)
    hsv[..., 1] = np.where(magnitude > 1e-4, 235, 0).astype(np.uint8)
    hsv[..., 2] = np.round(255.0 * magnitude / max(float(magnitude.max()), 1e-6)).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), mode="RGB")


def _crop_box(mask: np.ndarray, padding: int) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError("empty silhouette")
    height, width = mask.shape
    return (
        max(0, int(columns.min()) - padding),
        max(0, int(rows.min()) - padding),
        min(width, int(columns.max()) + padding + 1),
        min(height, int(rows.max()) + padding + 1),
    )


def render(
    source_root: Path,
    protocol_path: Path,
    clean_audit_path: Path,
    repository_root: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("refusing to overwrite visual bakeoff output")
    protocol = _read_json(protocol_path)
    if protocol.get("schema") != "orion.lens_waterdrop_v2_visual_bakeoff_protocol.v1":
        raise ValueError("visual protocol schema differs")
    if protocol.get("locks", {}).get("orion_loading") is not False:
        raise RuntimeError("protocol does not explicitly lock ORION loading")
    clean_audit = _read_json(clean_audit_path)
    if clean_audit.get("status") != "passed_clean_render_artifact_gate":
        raise RuntimeError("clean source did not pass the frozen artifact gate")
    if not (clean_audit.get("gate") or {}).get("passed"):
        raise RuntimeError("clean source gate is not passed")

    record_roots = tuple(source_root.glob("records_*"))
    if len(record_roots) != 1:
        raise ValueError("expected one records directory below clean source")
    record_root = record_roots[0]
    front_paths = sorted((record_root / "rgb_front").glob("*.png"))
    bev_paths = sorted((record_root / "bev").glob("*.png"))
    if not front_paths or len(front_paths) != len(bev_paths):
        raise ValueError("front/BEV source frame count differs or is empty")
    maximum_frames = int(protocol["review"]["maximum_frames"])
    front_paths = front_paths[:maximum_frames]
    bev_paths = bev_paths[:maximum_frames]
    clean_frames = [_load_rgb(path) for path in front_paths]
    if any(frame.shape != (900, 1600, 3) for frame in clean_frames):
        raise ValueError("clean source is not full-resolution 1600x900 RGB")

    mask_root = repository_root / "assets/waterdrop_patterns/evocargo_ccby4_v1"
    metadata_path = mask_root / "metadata.json"
    mask_path = mask_root / protocol["renderer"]["primary_visual_mask"]
    profiles = [str(value) for value in protocol["renderer"]["profiles"]]
    rendered: dict[str, list[Any]] = {profile: [] for profile in profiles}
    for frame in clean_frames:
        for profile in profiles:
            rendered[profile].append(
                apply_lens_waterdrop_v2(
                    frame,
                    mask_path=mask_path,
                    metadata_path=metadata_path,
                    profile=profile,
                )
            )

    output.mkdir(parents=True)
    fps = int(protocol["review"]["output_fps"])
    review_width, review_height = (int(value) for value in protocol["review"]["review_size"])
    clean_pil = [Image.fromarray(frame, mode="RGB") for frame in clean_frames]
    profile_pil = {
        profile: [Image.fromarray(item.image, mode="RGB") for item in rendered[profile]]
        for profile in profiles
    }
    _save_gif(output / "front_clean_fullres.gif", clean_pil, fps)
    for profile in profiles:
        _save_gif(output / ("front_waterdrop_%s_fullres.gif" % profile), profile_pil[profile], fps)
    comparison_frames = []
    for frame_index in range(len(clean_frames)):
        images = [clean_pil[frame_index]] + [profile_pil[profile][frame_index] for profile in profiles]
        resized = [image.resize((review_width, review_height), Image.Resampling.LANCZOS) for image in images]
        comparison_frames.append(_horizontal(resized, ["clean"] + profiles))
    _save_gif(output / "front_clean_vs_waterdrop_profiles.gif", comparison_frames, fps)

    bev_frames = [
        _label(Image.open(path).convert("RGB").resize((review_width, review_height), Image.Resampling.LANCZOS),
               "clean BEV context; corruption is front-camera only")
        for path in bev_paths
    ]
    _save_gif(output / "clean_bev_context.gif", bev_frames, fps)

    contact_index = int(protocol["review"]["contact_frame_index"])
    if contact_index >= len(clean_frames):
        raise ValueError("contact frame index is outside captured sequence")
    clean_contact = clean_pil[contact_index]
    clean_contact.save(output / "contact_clean_fullres.png")
    comparison_frames[contact_index].save(output / "contact_clean_vs_profiles.png")
    audit_item = rendered["medium"][contact_index]
    Image.fromarray((audit_item.silhouette.astype(np.uint8) * 255), mode="L").save(
        output / "contact_source_silhouette.png"
    )
    _field_image(audit_item.alpha).save(output / "contact_derived_alpha.png")
    _displacement_image(audit_item.displacement_px).save(
        output / "contact_derived_displacement.png"
    )
    Image.fromarray(audit_item.refracted, mode="RGB").save(output / "contact_refracted_rgb.png")
    _field_image(audit_item.edge_contribution, signed=True).save(
        output / "contact_edge_crescent.png"
    )
    _field_image(audit_item.highlight_contribution).save(output / "contact_highlight.png")

    crop = _crop_box(audit_item.silhouette, int(protocol["review"]["crop_padding_px"]))
    crop_images = [clean_contact.crop(crop)] + [profile_pil[profile][contact_index].crop(crop) for profile in profiles]
    _horizontal(crop_images, ["clean crop"] + [profile + " crop" for profile in profiles]).save(
        output / "contact_high_resolution_crops.png"
    )

    metrics = {}
    for profile in profiles:
        item = rendered[profile][contact_index]
        difference = np.abs(item.image.astype(np.float32) - clean_frames[contact_index].astype(np.float32))
        metrics[profile] = {
            "mean_absolute_rgb_difference_inside": float(difference[item.silhouette].mean()),
            "changed_pixel_fraction_inside": float(np.any(item.image != clean_frames[contact_index], axis=2)[item.silhouette].mean()),
            "source_mask_fraction": item.metadata["source_mask_fraction"],
            "displacement_px_max": item.metadata["displacement_px_max"],
            "displacement_px_mean_inside": item.metadata["displacement_px_mean_inside"],
            "alpha_max": item.metadata["alpha_max"],
            "edge_min": item.metadata["edge_min"],
            "highlight_max": item.metadata["highlight_max"],
        }

    result = {
        "schema": SCHEMA,
        "status": "rendered_pending_human_visual_review_orion_locked",
        "orion_loaded": False,
        "source_frame_count": len(clean_frames),
        "source_resolution": [1600, 900],
        "clean_artifact_gate": {
            "passed": True,
            "suspicious_frame_count": clean_audit["gate"]["suspicious_frame_count"],
            "frame_count": clean_audit["gate"]["frame_count"],
            "audit_sha256": _sha256(clean_audit_path),
        },
        "renderer": {
            "implementation": str((repository_root / "uq_estimator/lens_waterdrop_v2.py").resolve()),
            "implementation_sha256": _sha256(repository_root / "uq_estimator/lens_waterdrop_v2.py"),
            "mask": audit_item.metadata["source_silhouette"],
            "real_data_scope": audit_item.metadata["real_data_scope"],
            "optics_description": audit_item.metadata["optics_description"],
            "placement_policy": audit_item.metadata["placement_policy"],
            "temporal_policy": audit_item.metadata["temporal_policy"],
        },
        "contact_frame_index": contact_index,
        "crop_box_xyxy": list(crop),
        "profile_metrics_at_contact": metrics,
        "provenance": {
            "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
            "clean_audit": {"path": str(clean_audit_path), "sha256": _sha256(clean_audit_path)},
            "capture_trace": {
                "path": str((record_root / "capture_trace.jsonl").resolve()),
                "sha256": _sha256(record_root / "capture_trace.jsonl"),
            },
            "source_first_front_sha256": _sha256(front_paths[0]),
            "source_last_front_sha256": _sha256(front_paths[-1]),
        },
        "review_decision": None,
        "locks": protocol["locks"],
        "claim_boundary": protocol["claim_boundary"],
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["artifacts"] = {
        path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(output.iterdir())
        if path.is_file() and path != result_path
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--clean-audit", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = render(
        args.source_root.resolve(),
        args.protocol.resolve(),
        args.clean_audit.resolve(),
        args.repository_root.resolve(),
        args.output.resolve(),
    )
    print(json.dumps({
        "status": result["status"],
        "source_frame_count": result["source_frame_count"],
        "profiles": sorted(result["profile_metrics_at_contact"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
