#!/usr/bin/env python3
"""Check CARLA-native glare after the exact ORION image pixel pipeline.

This analysis deliberately stops before loading ORION weights.  It replays the
Q20 JPEG round-trip performed by ``orion_b2d_agent.py`` and the deterministic
inference image transforms frozen in ``orion_stage3_agent.py``.  The resulting
float32 arrays are the normalized HWC values immediately before formatting into
the model's multi-view tensor.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


SCHEMA = "orion.native_glare_orion_input_stage_analysis.v1"
SELECTED_IMAGES_SCHEMA = "orion.glare_method_selected_images.v1"
PROFILES = ("clean", "light", "medium", "heavy")
SEVERITIES = PROFILES[1:]
JPEG_QUALITY = 20
MODEL_IMAGE_SCALE = (640, 640)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: np.ndarray) -> str:
    canonical = np.ascontiguousarray(value, dtype="<f4")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _literal_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    if len(matches) != 1:
        raise ValueError("expected exactly one literal %s assignment" % name)
    value = matches[0]
    try:
        return ast.literal_eval(value)
    except (TypeError, ValueError):
        # ORION uses the equally static ``dict(mean=..., ...)`` spelling for
        # img_norm_cfg.  Accept only keyword-only calls to builtin ``dict``;
        # do not execute the configuration file or arbitrary AST nodes.
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "dict"
            and not value.args
            and all(keyword.arg is not None for keyword in value.keywords)
        ):
            try:
                return {
                    str(keyword.arg): ast.literal_eval(keyword.value)
                    for keyword in value.keywords
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "%s dict() values must remain literal" % name
                ) from exc
        raise ValueError("%s must remain a literal configuration" % name)


def _load_pipeline_config(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ida = _literal_assignment(path, "ida_aug_conf")
    norm = _literal_assignment(path, "img_norm_cfg")
    required_ida = {"H", "W", "final_dim", "bot_pct_lim", "rot_lim", "rand_flip"}
    if not isinstance(ida, dict) or not required_ida.issubset(ida):
        raise ValueError("ORION IDA configuration is incomplete")
    if tuple(ida["rot_lim"]) != (0.0, 0.0) or ida["rand_flip"] is not False:
        raise ValueError("analysis only supports the frozen deterministic inference IDA")
    if not isinstance(norm, dict) or set(("mean", "std", "to_rgb")) - set(norm):
        raise ValueError("ORION normalization configuration is incomplete")
    if norm["to_rgb"] is not True:
        raise ValueError("analysis expects the frozen BGR-to-RGB normalization")
    return dict(ida), dict(norm)


def _q20_round_trip(image_bgr: np.ndarray) -> np.ndarray:
    ok, encoded = cv2.imencode(
        ".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError("failed to encode the ORION Q20 input")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("failed to decode the ORION Q20 input")
    return decoded


def _ida_geometry(ida: Mapping[str, Any]) -> Dict[str, Any]:
    height, width = int(ida["H"]), int(ida["W"])
    final_height, final_width = map(int, ida["final_dim"])
    resize = max(final_height / height, final_width / width)
    resize_dims = (int(width * resize), int(height * resize))
    new_width, new_height = resize_dims
    bottom_fraction = float(np.mean(ida["bot_pct_lim"]))
    crop_y = int((1.0 - bottom_fraction) * new_height) - final_height
    crop_x = int(max(0, new_width - final_width) / 2)
    crop = (crop_x, crop_y, crop_x + final_width, crop_y + final_height)
    return {
        "source_shape_hw": [height, width],
        "resize": float(resize),
        "resize_dims_wh": list(resize_dims),
        "crop_ltrb": list(crop),
        "ida_output_shape_hw": [final_height, final_width],
        "model_image_scale_wh": list(MODEL_IMAGE_SCALE),
    }


def preprocess_orion_front(
    image_bgr: np.ndarray,
    *,
    ida: Mapping[str, Any],
    norm: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return normalized HWC values and their pre-normalization BGR pixels."""

    expected = (int(ida["H"]), int(ida["W"]))
    if image_bgr.shape[:2] != expected or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            "source image shape %r differs from frozen ORION input %r"
            % (image_bgr.shape, expected + (3,))
        )
    q20 = _q20_round_trip(image_bgr)
    geometry = _ida_geometry(ida)
    resized = Image.fromarray(np.uint8(q20)).resize(tuple(geometry["resize_dims_wh"]))
    cropped = resized.crop(tuple(geometry["crop_ltrb"]))
    ida_bgr = np.asarray(cropped, dtype=np.float32)
    model_bgr = cv2.resize(
        ida_bgr,
        MODEL_IMAGE_SCALE,
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32, copy=False)
    rgb = model_bgr[..., ::-1]
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    normalized = np.ascontiguousarray((rgb - mean) / std, dtype=np.float32)
    if normalized.shape[:2] != (MODEL_IMAGE_SCALE[1], MODEL_IMAGE_SCALE[0]):
        raise RuntimeError("unexpected ORION model-input shape")
    # PadMultiViewImage(size_divisor=32) is a no-op for 640 x 640.  Bind that
    # invariant so a future configuration change fails instead of drifting.
    if normalized.shape[0] % 32 or normalized.shape[1] % 32:
        raise RuntimeError("frozen ORION image is not already divisible by 32")
    return normalized, model_bgr, geometry


def _map_normalized_roi(
    roi: Sequence[float], ida: Mapping[str, Any]
) -> Tuple[float, float, float, float]:
    left, top, right, bottom = map(float, roi)
    if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
        raise ValueError("invalid normalized source ROI")
    geometry = _ida_geometry(ida)
    height, width = geometry["source_shape_hw"]
    resize = geometry["resize"]
    crop_left, crop_top, _, _ = geometry["crop_ltrb"]
    ida_height, ida_width = geometry["ida_output_shape_hw"]

    def x(value: float) -> float:
        return (value * width * resize - crop_left) / ida_width

    def y(value: float) -> float:
        return (value * height * resize - crop_top) / ida_height

    mapped = (x(left), y(top), x(right), y(bottom))
    return tuple(max(0.0, min(1.0, value)) for value in mapped)


def _roi(image: np.ndarray, normalized: Sequence[float]) -> np.ndarray:
    height, width = image.shape[:2]
    left, top, right, bottom = normalized
    x0, y0 = int(round(left * width)), int(round(top * height))
    x1, y1 = int(round(right * width)), int(round(bottom * height))
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("mapped model-input ROI is empty")
    return image[y0:y1, x0:x1]


def _contrast(image_bgr: np.ndarray) -> float:
    return float(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).std())


def _edge_energy(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def _frame_metrics(
    clean_tensor: np.ndarray,
    changed_tensor: np.ndarray,
    clean_bgr: np.ndarray,
    changed_bgr: np.ndarray,
    roi: Sequence[float],
) -> Dict[str, float]:
    delta = changed_tensor.astype(np.float64) - clean_tensor.astype(np.float64)
    clean_roi, changed_roi = _roi(clean_bgr, roi), _roi(changed_bgr, roi)
    return {
        "normalized_tensor_mean_abs_delta": float(np.abs(delta).mean()),
        "normalized_tensor_rms_delta": float(np.sqrt(np.square(delta).mean())),
        "model_input_pixel_mean_abs_delta": float(
            np.abs(changed_bgr.astype(np.float64) - clean_bgr.astype(np.float64)).mean()
        ),
        "model_input_saturated_pixel_fraction_clean": float(
            (clean_bgr.max(axis=2) >= 250.0).mean()
        ),
        "model_input_saturated_pixel_fraction_changed": float(
            (changed_bgr.max(axis=2) >= 250.0).mean()
        ),
        "mapped_roi_contrast_ratio": _contrast(changed_roi)
        / max(_contrast(clean_roi), 1e-6),
        "mapped_roi_edge_visibility_ratio": _edge_energy(changed_roi)
        / max(_edge_energy(clean_roi), 1e-6),
    }


def _load_selected(path: Path) -> Tuple[Dict[str, Any], Dict[str, list]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SELECTED_IMAGES_SCHEMA:
        raise ValueError("unsupported selected-image manifest")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != set(PROFILES):
        raise ValueError("selected-image manifest must contain four native profiles")
    resolved: Dict[str, list] = {}
    for profile in PROFILES:
        rows = profiles[profile]
        if not isinstance(rows, list) or not 3 <= len(rows) <= 5:
            raise ValueError("each native profile requires three to five poses")
        resolved[profile] = []
        for row in rows:
            image_path = Path(str(row.get("path", "")))
            if not image_path.is_absolute():
                image_path = (path.parent / image_path).resolve()
            if not image_path.is_file() or _sha256(image_path) != row.get("sha256"):
                raise ValueError("selected native glare image is absent or hash-mismatched")
            resolved[profile].append(
                {"capture_index": int(row["capture_index"]), "path": image_path}
            )
    clean_indices = [row["capture_index"] for row in resolved["clean"]]
    if any(
        [row["capture_index"] for row in resolved[profile]] != clean_indices
        for profile in SEVERITIES
    ):
        raise ValueError("native glare profiles are not capture-index aligned")
    return payload, resolved


def _mean(rows: Sequence[Mapping[str, float]], key: str) -> float:
    return float(sum(float(row[key]) for row in rows) / len(rows))


def _label(image: np.ndarray, text: str) -> np.ndarray:
    result = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    cv2.rectangle(result, (0, 0), (result.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(
        result,
        text,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def analyze(
    *,
    selected_manifest_path: Path,
    config_path: Path,
    agent_path: Path,
    hazard_roi: Sequence[float],
    output_dir: Path,
) -> Dict[str, Any]:
    _, selected = _load_selected(selected_manifest_path)
    ida, norm = _load_pipeline_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_root = output_dir / "model_input_previews"
    preview_root.mkdir(exist_ok=True)
    mapped_roi = _map_normalized_roi(hazard_roi, ida)

    processed: Dict[str, list] = {profile: [] for profile in PROFILES}
    geometry = None
    for profile in PROFILES:
        profile_root = preview_root / profile
        profile_root.mkdir(exist_ok=True)
        for row in selected[profile]:
            source = cv2.imread(str(row["path"]), cv2.IMREAD_COLOR)
            if source is None:
                raise RuntimeError("failed to read %s" % row["path"])
            tensor, bgr, geometry = preprocess_orion_front(source, ida=ida, norm=norm)
            preview_path = profile_root / ("%04d.png" % row["capture_index"])
            if not cv2.imwrite(str(preview_path), np.clip(np.rint(bgr), 0, 255).astype(np.uint8)):
                raise RuntimeError("failed to save exact model-input preview")
            processed[profile].append(
                {
                    "capture_index": row["capture_index"],
                    "tensor": tensor,
                    "bgr": bgr,
                    "tensor_sha256": _tensor_sha256(tensor),
                    "preview": preview_path,
                    "preview_sha256": _sha256(preview_path),
                }
            )

    per_frame: Dict[str, list] = {profile: [] for profile in SEVERITIES}
    for severity in SEVERITIES:
        for clean, changed in zip(processed["clean"], processed[severity]):
            metrics = _frame_metrics(
                clean["tensor"], changed["tensor"], clean["bgr"], changed["bgr"], mapped_roi
            )
            metrics.update(
                {
                    "capture_index": clean["capture_index"],
                    "clean_tensor_sha256": clean["tensor_sha256"],
                    "changed_tensor_sha256": changed["tensor_sha256"],
                    "clean_preview_sha256": clean["preview_sha256"],
                    "changed_preview_sha256": changed["preview_sha256"],
                }
            )
            per_frame[severity].append(metrics)

    metric_keys = (
        "normalized_tensor_mean_abs_delta",
        "normalized_tensor_rms_delta",
        "model_input_pixel_mean_abs_delta",
        "model_input_saturated_pixel_fraction_clean",
        "model_input_saturated_pixel_fraction_changed",
        "mapped_roi_contrast_ratio",
        "mapped_roi_edge_visibility_ratio",
    )
    summary = {
        severity: {key: _mean(per_frame[severity], key) for key in metric_keys}
        for severity in SEVERITIES
    }
    all_pose_tensor_order = all(
        per_frame["light"][index]["normalized_tensor_mean_abs_delta"]
        < per_frame["medium"][index]["normalized_tensor_mean_abs_delta"]
        < per_frame["heavy"][index]["normalized_tensor_mean_abs_delta"]
        for index in range(len(per_frame["light"]))
    )
    aggregate_tensor_order = all(
        summary[left]["normalized_tensor_mean_abs_delta"]
        < summary[right]["normalized_tensor_mean_abs_delta"]
        for left, right in zip(SEVERITIES[:-1], SEVERITIES[1:])
    )
    aggregate_edge_loss_order = all(
        summary[left]["mapped_roi_edge_visibility_ratio"]
        > summary[right]["mapped_roi_edge_visibility_ratio"]
        for left, right in zip(SEVERITIES[:-1], SEVERITIES[1:])
    )

    contact_rows = []
    gif_frames = []
    for index in range(len(processed["clean"])):
        tiled = np.concatenate(
            [
                _label(
                    processed[profile][index]["bgr"],
                    "%s | pose %04d" % (profile, processed[profile][index]["capture_index"]),
                )
                for profile in PROFILES
            ],
            axis=1,
        )
        contact_rows.append(tiled)
        gif_frames.append(cv2.cvtColor(tiled, cv2.COLOR_BGR2RGB))
    contact_path = output_dir / "route151_native_glare_orion_input_contact_sheet.png"
    gif_path = output_dir / "route151_native_glare_orion_input.gif"
    cv2.imwrite(str(contact_path), np.concatenate(contact_rows, axis=0))
    Image.fromarray(gif_frames[0]).save(
        gif_path,
        save_all=True,
        append_images=[Image.fromarray(frame) for frame in gif_frames[1:]],
        duration=500,
        loop=0,
    )

    gates = {
        "all_pose_tensor_impact_strictly_light_medium_heavy": all_pose_tensor_order,
        "aggregate_tensor_impact_strictly_light_medium_heavy": aggregate_tensor_order,
        "aggregate_mapped_roi_edge_loss_strictly_light_medium_heavy": aggregate_edge_loss_order,
        "input_stage_severity_order_preserved": bool(
            all_pose_tensor_order and aggregate_tensor_order and aggregate_edge_loss_order
        ),
    }
    report = {
        "schema": SCHEMA,
        "analysis_stage": (
            "ORION Q20 JPEG round-trip, deterministic inference IDA, 640x640 resize, "
            "RGB normalization and no-op divisor-32 padding; before model execution"
        ),
        "orion_weights_loaded": False,
        "outcome_fields_read": [],
        "profiles": list(PROFILES),
        "matched_pose_count": len(processed["clean"]),
        "source_hazard_roi_normalized_ltrb": list(map(float, hazard_roi)),
        "mapped_model_input_roi_normalized_ltrb": list(mapped_roi),
        "geometry": geometry,
        "normalization": norm,
        "jpeg_quality": JPEG_QUALITY,
        "per_frame": per_frame,
        "summary": summary,
        "gates": gates,
        "visuals": {
            "contact_sheet": str(contact_path.resolve()),
            "contact_sheet_sha256": _sha256(contact_path),
            "gif": str(gif_path.resolve()),
            "gif_sha256": _sha256(gif_path),
            "preview_root": str(preview_root.resolve()),
        },
        "provenance": {
            "selected_images_manifest": {
                "path": str(selected_manifest_path.resolve()),
                "sha256": _sha256(selected_manifest_path),
            },
            "orion_agent_config": {
                "path": str(config_path.resolve()),
                "sha256": _sha256(config_path),
            },
            "orion_closedloop_agent": {
                "path": str(agent_path.resolve()),
                "sha256": _sha256(agent_path),
            },
        },
        "claim_boundary": (
            "pixel-pipeline severity preservation only; no adapter, VLM, UQ, planning, "
            "TTC, collision or safety claim"
        ),
    }
    report_path = output_dir / "orion_input_stage_analysis.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(gates, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("adzoo/orion/configs/orion_stage3_agent.py")
    )
    parser.add_argument(
        "--agent", type=Path, default=Path("team_code/orion_b2d_agent.py")
    )
    parser.add_argument(
        "--hazard-roi",
        default="0.35,0.50,0.90,0.85",
        help="raw sensor normalized left,top,right,bottom",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roi = tuple(float(value) for value in args.hazard_roi.split(","))
    if len(roi) != 4:
        raise ValueError("--hazard-roi requires four values")
    analyze(
        selected_manifest_path=args.selected_manifest,
        config_path=args.config,
        agent_path=args.agent,
        hazard_roi=roi,
        output_dir=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
