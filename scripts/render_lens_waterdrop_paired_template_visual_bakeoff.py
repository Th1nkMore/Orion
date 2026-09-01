#!/usr/bin/env python3
"""Render the preregistered paired-template waterdrop visual review package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from uq_estimator.lens_waterdrop_paired_template import (
    apply_paired_waterdrop_template,
    extract_paired_waterdrop_template,
)


SCHEMA = "orion.lens_waterdrop_paired_template_visual_bakeoff_result.v1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _label(image: Image.Image, text: str) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, result.width, 28), fill=(0, 0, 0))
    draw.text((8, 7), text, fill=(255, 255, 255))
    return result


def _row(images: list[Image.Image], labels: list[str]) -> Image.Image:
    labeled = [_label(image, label) for image, label in zip(images, labels)]
    result = Image.new("RGB", (sum(image.width for image in labeled), labeled[0].height))
    left = 0
    for image in labeled:
        result.paste(image, (left, 0))
        left += image.width
    return result


def _gif(path: Path, frames: list[Image.Image], fps: int) -> None:
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000.0 / fps)),
        loop=0,
        optimize=False,
    )


def _gray(values: np.ndarray, signed: bool = False) -> Image.Image:
    if signed:
        scale = max(float(np.abs(values).max()), 1e-6)
        normalized = np.clip(0.5 + 0.5 * values / scale, 0.0, 1.0)
    else:
        normalized = values / max(float(values.max()), 1e-6)
    return Image.fromarray(np.round(normalized * 255).astype(np.uint8)).convert("RGB")


def _flow(displacement: np.ndarray) -> Image.Image:
    dx, dy = displacement[..., 0], displacement[..., 1]
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)
    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.round(angle * 179).astype(np.uint8)
    hsv[..., 1] = np.where(magnitude > 1e-4, 235, 0).astype(np.uint8)
    hsv[..., 2] = np.round(magnitude / max(float(magnitude.max()), 1e-6) * 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))


def _densest_crop(support: np.ndarray, crop_width: int, crop_height: int) -> tuple[int, int, int, int]:
    # Select by template support only, never by target RGB, actor, or semantics.
    reduced = cv2.resize(support.astype(np.float32), (80, 45), interpolation=cv2.INTER_AREA)
    target_w = max(1, int(round(crop_width * 80 / support.shape[1])))
    target_h = max(1, int(round(crop_height * 45 / support.shape[0])))
    sums = cv2.boxFilter(reduced, cv2.CV_32F, (target_w, target_h), normalize=False)
    _, _, _, maximum_location = cv2.minMaxLoc(sums)
    center_x = int(round((maximum_location[0] + 0.5) * support.shape[1] / 80))
    center_y = int(round((maximum_location[1] + 0.5) * support.shape[0] / 45))
    left = min(max(0, center_x - crop_width // 2), support.shape[1] - crop_width)
    top = min(max(0, center_y - crop_height // 2), support.shape[0] - crop_height)
    return left, top, left + crop_width, top + crop_height


def render(source_root: Path, protocol_path: Path, audit_path: Path, repository: Path, output: Path):
    if output.exists():
        raise FileExistsError("refusing to overwrite paired-template visual output")
    protocol, audit = _json(protocol_path), _json(audit_path)
    if protocol.get("schema") != "orion.lens_waterdrop_paired_template_visual_bakeoff_protocol.v1":
        raise ValueError("protocol schema differs")
    if protocol["locks"].get("orion_loading") is not False:
        raise RuntimeError("ORION is not locked")
    if audit.get("status") != "passed_clean_render_artifact_gate" or not audit["gate"]["passed"]:
        raise RuntimeError("clean artifact gate did not pass")
    records = tuple(source_root.glob("records_*"))
    if len(records) != 1:
        raise ValueError("expected one clean records directory")
    front_paths = sorted((records[0] / "rgb_front").glob("*.png"))[: int(protocol["review"]["maximum_frames"])]
    bev_paths = sorted((records[0] / "bev").glob("*.png"))[: len(front_paths)]
    if not front_paths or len(front_paths) != len(bev_paths):
        raise ValueError("front and BEV inputs differ")
    clean_frames = [_rgb(path) for path in front_paths]
    if any(frame.shape != (900, 1600, 3) for frame in clean_frames):
        raise ValueError("clean source must be 1600x900 RGB")

    bank = repository / protocol["template_source"]["bank"]
    template = extract_paired_waterdrop_template(
        clean_path=bank / protocol["template_source"]["paired_clean"],
        rainy_path=bank / protocol["template_source"]["paired_rainy"],
        metadata_path=bank / "metadata.json",
    )
    if template.metadata["source_rgb_retained_in_template"]:
        raise RuntimeError("source RGB unexpectedly retained")
    profiles = protocol["renderer"]["profiles"]
    rendered = {
        profile: [
            apply_paired_waterdrop_template(frame, template=template, profile=profile)
            for frame in clean_frames
        ]
        for profile in profiles
    }
    output.mkdir(parents=True)
    fps = int(protocol["review"]["output_fps"])
    review_size = tuple(int(value) for value in protocol["review"]["review_size"])
    clean_pil = [Image.fromarray(frame) for frame in clean_frames]
    rendered_pil = {
        profile: [Image.fromarray(item.image) for item in rendered[profile]]
        for profile in profiles
    }
    _gif(output / "front_clean_fullres.gif", clean_pil, fps)
    for profile in profiles:
        _gif(output / ("front_%s_fullres.gif" % profile), rendered_pil[profile], fps)
    comparison = []
    for index in range(len(clean_frames)):
        images = [clean_pil[index]] + [rendered_pil[profile][index] for profile in profiles]
        images = [image.resize(review_size, Image.Resampling.LANCZOS) for image in images]
        comparison.append(_row(images, ["clean"] + profiles))
    _gif(output / "front_clean_vs_profiles.gif", comparison, fps)
    bev = [
        _label(Image.open(path).convert("RGB").resize(review_size, Image.Resampling.LANCZOS),
               "clean BEV context; front-camera template only")
        for path in bev_paths
    ]
    _gif(output / "clean_bev_context.gif", bev, fps)

    contact_index = int(protocol["review"]["contact_frame_index"])
    comparison[contact_index].save(output / "contact_clean_vs_profiles.png")
    heavy = rendered["heavy"][contact_index]
    _gray(heavy.alpha).save(output / "contact_alpha.png")
    _flow(heavy.displacement_px).save(output / "contact_displacement.png")
    _gray(heavy.luminance_residual, signed=True).save(output / "contact_balanced_luminance.png")
    Image.fromarray(heavy.refracted).save(output / "contact_refracted.png")
    crop = _densest_crop(heavy.support, 620, 360)
    crop_images = [clean_pil[contact_index].crop(crop)] + [
        rendered_pil[profile][contact_index].crop(crop) for profile in profiles
    ]
    _row(crop_images, ["clean crop"] + [profile + " crop" for profile in profiles]).save(
        output / "contact_high_resolution_crops.png"
    )

    source_clean = _rgb(bank / protocol["template_source"]["paired_clean"])
    source_rainy = _rgb(bank / protocol["template_source"]["paired_rainy"])
    source_reconstruction = apply_paired_waterdrop_template(
        source_clean, template=template, profile="heavy", require_resolution=False
    ).image
    _row(
        [Image.fromarray(source_clean), Image.fromarray(source_rainy), Image.fromarray(source_reconstruction)],
        ["published synthetic clean", "published synthetic rainy", "field-only reconstruction"],
    ).save(output / "source_pair_and_reconstruction.png")
    real_references = [Image.open(bank / name).convert("RGB") for name in protocol["template_source"]["real_visual_references"]]
    real_references = [image.resize((640, 352), Image.Resampling.LANCZOS) for image in real_references]
    _row(real_references, ["real frame 40", "real frame 120", "real frame 200"]).save(
        output / "real_driving_visual_references.png"
    )

    metrics = {}
    for profile in profiles:
        item = rendered[profile][contact_index]
        delta = np.abs(item.image.astype(np.float32) - clean_frames[contact_index].astype(np.float32))
        metrics[profile] = {
            "support_fraction": item.metadata["support_fraction"],
            "changed_pixel_fraction_inside": item.metadata["changed_pixel_fraction_inside"],
            "mean_absolute_rgb_difference_inside": float(delta[item.support].mean()),
            "displacement_px_max": item.metadata["displacement_px_max"],
        }
    result = {
        "schema": SCHEMA,
        "status": "rendered_pending_human_visual_review_orion_locked",
        "orion_loaded": False,
        "source_frame_count": len(clean_frames),
        "source_resolution": [1600, 900],
        "clean_artifact_gate": {
            "passed": True,
            "suspicious_frame_count": audit["gate"]["suspicious_frame_count"],
            "frame_count": audit["gate"]["frame_count"],
            "audit_sha256": _sha(audit_path),
        },
        "template": template.metadata,
        "target_source_rgb_copied": False,
        "target_chromatic_residual_copied": False,
        "real_reference_frames_composited": False,
        "profile_metrics_at_contact": metrics,
        "contact_frame_index": contact_index,
        "crop_box_xyxy": list(crop),
        "crop_selection_policy": "densest template support tile; target RGB, actors, and semantics excluded",
        "provenance": {
            "protocol": {"path": str(protocol_path), "sha256": _sha(protocol_path)},
            "clean_audit": {"path": str(audit_path), "sha256": _sha(audit_path)},
            "capture_trace_sha256": _sha(records[0] / "capture_trace.jsonl"),
            "implementation_sha256": _sha(repository / protocol["renderer"]["implementation"]),
            "bank_metadata_sha256": _sha(bank / "metadata.json"),
        },
        "review_decision": None,
        "locks": protocol["locks"],
        "claim_boundary": protocol["claim_boundary"],
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["artifacts"] = {
        path.name: {"sha256": _sha(path), "bytes": path.stat().st_size}
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
        args.source_root.resolve(), args.protocol.resolve(), args.clean_audit.resolve(),
        args.repository_root.resolve(), args.output.resolve()
    )
    print(json.dumps({"status": result["status"], "frames": result["source_frame_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
