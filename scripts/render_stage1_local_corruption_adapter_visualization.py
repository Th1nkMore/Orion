#!/usr/bin/env python3
"""Render paired local-corruption targets and frozen Stage-1 adapter responses.

This diagnostic deliberately avoids ORION loading.  It reuses frozen backbone
features from one direct-FP16 route shard, reconstructs the exact deterministic
image intervention for display, and compares the measured paired feature-loss
target with the adapter's identified within-pair score increment.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uq_estimator.corruptions import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    corrupt_multiview_images_with_metadata,
)
from uq_estimator.online_observation_uq import (  # noqa: E402
    load_frozen_pairwise_adapter,
)


CAMERA_NAME = "CAM_FRONT"
FAMILIES = ("local_dark", "local_blur")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser


def _load_annotation_records(path: Path) -> list[Mapping[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, Mapping):
        records = payload.get("infos", payload.get("data_list"))
    else:
        records = payload
    if not isinstance(records, list):
        raise RuntimeError("annotation payload does not contain a record list")
    return records


def _front_path(
    records: list[Mapping[str, Any]],
    route_id: str,
    frame_idx: int,
    data_root: Path,
) -> Path:
    matches = [
        row
        for row in records
        if str(row.get("folder")) == route_id
        and int(row.get("frame_idx", -1)) == frame_idx
    ]
    if len(matches) != 1:
        raise RuntimeError("route/frame annotation lookup is not unique")
    sensor = matches[0].get("sensors", {}).get(CAMERA_NAME)
    if not isinstance(sensor, Mapping) or not sensor.get("data_path"):
        raise RuntimeError("front camera annotation is missing")
    path = Path(str(sensor["data_path"]))
    return path if path.is_absolute() else data_root / path


def _orion_input_rgb(path: Path) -> np.ndarray:
    """Reproduce the frozen inference geometry before normalization.

    The pipeline first resizes 1600x900 to 640x360, center-crops the lower
    640x320 area, and then resizes to 640x640 without preserving aspect ratio.
    """

    image = Image.open(path).convert("RGB")
    resized = image.resize((640, 360), resample=Image.Resampling.BILINEAR)
    cropped = resized.crop((0, 40, 640, 360))
    return np.asarray(
        cropped.resize((640, 640), resample=Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def _normalized_tensor(rgb: np.ndarray) -> torch.Tensor:
    value = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()
    mean = value.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = value.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (value - mean) / std


def _render_corruption(rgb: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
    clean = _normalized_tensor(rgb)
    multiview = clean.unsqueeze(0).repeat(6, 1, 1, 1)
    result = corrupt_multiview_images_with_metadata(
        multiview,
        corruption=str(metadata["corruption"]),
        severity=int(metadata["severity"]),
        view_indices=[int(metadata["view_indices"][0])],
        seed=int(metadata["seed"]),
        region=tuple(float(value) for value in metadata["normalized_region"]),
    )
    observed = result.images[0]
    mean = observed.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = observed.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (
        (observed * std + mean)
        .permute(1, 2, 0)
        .round()
        .clamp(0, 255)
        .byte()
        .cpu()
        .numpy()
    )


def _find_sample_indices(
    items: list[Mapping[str, Any]], severity: int
) -> dict[str, int]:
    selected: dict[str, int] = {}
    for family in FAMILIES:
        candidates = []
        for index, item in enumerate(items):
            metadata = item.get("corruption_metadata")
            if not isinstance(metadata, Mapping):
                continue
            views = metadata.get("view_indices")
            if (
                str(item.get("family")) == family
                and float(item.get("severity", -1)) == float(severity)
                and list(views or []) == [0]
                and bool(item.get("previous_valid"))
            ):
                candidates.append((int(item["frame_idx"]), index))
        if not candidates:
            raise RuntimeError("no valid front-view %s sample" % family)
        selected[family] = min(candidates)[1]
    return selected


def _previous_indices(
    clean_items: list[Mapping[str, Any]],
    observed_items: list[Mapping[str, Any]],
) -> tuple[dict[tuple[str, int], int], dict[tuple[str, int, str, float], int]]:
    clean = {
        (str(item["route_id"]), int(item["frame_idx"])): index
        for index, item in enumerate(clean_items)
    }
    observed = {
        (
            str(item["route_id"]),
            int(item["frame_idx"]),
            str(item["family"]),
            float(item["severity"]),
        ): index
        for index, item in enumerate(observed_items)
    }
    return clean, observed


def _resize_map(value: np.ndarray, size: int = 640) -> np.ndarray:
    tensor = torch.from_numpy(value).float()[None, None]
    return F.interpolate(
        tensor, size=(size, size), mode="bilinear", align_corners=False
    )[0, 0].numpy()


def _region_metrics(
    value: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    selected = mask >= 0.5
    if not bool(selected.any()) or bool(selected.all()):
        raise RuntimeError("diagnostic region mask is empty or full")
    inside = value[selected]
    outside = value[~selected]
    return {
        "region_mean": float(inside.mean()),
        "outside_mean": float(outside.mean()),
        "region_minus_outside": float(inside.mean() - outside.mean()),
        "positive_fraction_region": float((inside > 0).float().mean()),
        "positive_fraction_outside": float((outside > 0).float().mean()),
    }


def _top_fraction_overlap(
    value: torch.Tensor, mask: torch.Tensor, fraction: float = 0.20
) -> float:
    flat = value.flatten()
    count = max(1, int(math.ceil(flat.numel() * fraction)))
    top = torch.topk(flat, count).indices
    labels = (mask >= 0.5).flatten()
    return float(labels[top].float().mean())


def _draw_region(axis: Any, region: list[float]) -> None:
    top, left, bottom, right = region
    rectangle = plt.Rectangle(
        (left * 640, top * 640),
        (right - left) * 640,
        (bottom - top) * 640,
        fill=False,
        edgecolor="#00f5ff",
        linewidth=2.2,
    )
    axis.add_patch(rectangle)


def main() -> int:
    args = build_parser().parse_args()
    if args.severity not in (1, 2, 3):
        raise SystemExit("severity must be 1, 2, or 3")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise SystemExit("requested CUDA device is unavailable")

    payload = torch.load(
        args.shard, map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(payload, Mapping):
        raise RuntimeError("route shard must be a mapping")
    clean_features = payload["clean_features"]
    observed_features = payload["observed_features"]
    clean_items = payload["clean_items"]
    observed_items = payload["observed_items"]
    target_values = payload["target"]["values"]
    target_valid = payload["target"]["component_valid"]
    selected = _find_sample_indices(observed_items, args.severity)
    clean_lookup, observed_lookup = _previous_indices(clean_items, observed_items)

    model, checkpoint_metadata = load_frozen_pairwise_adapter(
        args.checkpoint,
        expected_sha256=args.checkpoint_sha256,
        device=args.device,
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    scales = checkpoint["component_scales"].float().view(1, 1, 1, 1, 3)
    annotations = _load_annotation_records(args.ann_file)
    device = torch.device(args.device)

    results: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    all_positive_maps = []
    all_absolute_maps = []

    for family in FAMILIES:
        observed_index = selected[family]
        item = observed_items[observed_index]
        route_id = str(item["route_id"])
        frame_idx = int(item["frame_idx"])
        severity = float(item["severity"])
        clean_index = int(item["clean_index"])
        previous_clean_index = clean_lookup[(route_id, frame_idx - 1)]
        previous_observed_index = observed_lookup[
            (route_id, frame_idx - 1, family, severity)
        ]

        reference_current = clean_features[clean_index].float()[None].to(device)
        observed_current = observed_features[observed_index].float()[None].to(device)
        reference_previous = clean_features[previous_clean_index].float()[None].to(device)
        observed_previous = observed_features[previous_observed_index].float()[None].to(device)
        previous_valid = torch.ones(1, dtype=torch.bool, device=device)
        with torch.inference_mode():
            reference_score = model(
                reference_current, reference_previous, previous_valid
            ).cpu()
            observed_score = model(
                observed_current, observed_previous, previous_valid
            ).cpu()
        predicted_delta = observed_score - reference_score
        scaled_target = target_values[observed_index].float()[None] / scales
        valid = target_valid[observed_index][None]
        scaled_target = torch.where(valid, scaled_target, torch.zeros_like(scaled_target))

        front_reference = reference_score[0, 0].mean(dim=-1)
        front_observed = observed_score[0, 0].mean(dim=-1)
        front_delta = predicted_delta[0, 0].mean(dim=-1)
        front_target = scaled_target[0, 0].mean(dim=-1)
        front_mask = item["corruption_mask"][0].float()
        metadata = dict(item["corruption_metadata"])

        image_path = _front_path(annotations, route_id, frame_idx, args.data_root)
        clean_rgb = _orion_input_rgb(image_path)
        corrupt_rgb = _render_corruption(clean_rgb, metadata)
        row = {
            "family": family,
            "frame_idx": frame_idx,
            "clean_rgb": clean_rgb,
            "corrupt_rgb": corrupt_rgb,
            "target": _resize_map(front_target.numpy()),
            "reference": _resize_map(front_reference.numpy()),
            "observed": _resize_map(front_observed.numpy()),
            "delta": _resize_map(front_delta.clamp_min(0).numpy()),
            "region": list(metadata["normalized_region"]),
        }
        visual_rows.append(row)
        all_positive_maps.extend([row["target"], row["delta"]])
        all_absolute_maps.extend([row["reference"], row["observed"]])
        results.append(
            {
                "family": family,
                "severity": severity,
                "route_id": route_id,
                "frame_idx": frame_idx,
                "sample_id": str(item["sample_id"]),
                "front_image_path": str(image_path.resolve()),
                "corruption_metadata": metadata,
                "target_scaled": _region_metrics(front_target, front_mask),
                "adapter_reference": _region_metrics(front_reference, front_mask),
                "adapter_observed": _region_metrics(front_observed, front_mask),
                "adapter_positive_delta": _region_metrics(
                    front_delta.clamp_min(0), front_mask
                ),
                "target_top20_mask_precision": _top_fraction_overlap(
                    front_target, front_mask
                ),
                "adapter_delta_top20_mask_precision": _top_fraction_overlap(
                    front_delta, front_mask
                ),
                "adapter_delta_negative_fraction": float(
                    (front_delta < 0).float().mean()
                ),
            }
        )

    positive_vmax = float(
        np.quantile(np.concatenate([value.ravel() for value in all_positive_maps]), 0.99)
    )
    positive_vmax = max(positive_vmax, 1e-6)
    absolute_vmin = float(
        np.quantile(np.concatenate([value.ravel() for value in all_absolute_maps]), 0.01)
    )
    absolute_vmax = float(
        np.quantile(np.concatenate([value.ravel() for value in all_absolute_maps]), 0.99)
    )
    if absolute_vmax <= absolute_vmin:
        absolute_vmax = absolute_vmin + 1e-6

    fig, axes = plt.subplots(2, 6, figsize=(21, 7.7))
    fig.subplots_adjust(
        left=0.035, right=0.985, top=0.89, bottom=0.15, wspace=0.08, hspace=0.06
    )
    titles = (
        "Clean ORION input",
        "Corrupted ORION input",
        "Paired feature-loss target",
        "Adapter U(clean)",
        "Adapter U(corrupt)",
        "Adapter positive ΔU",
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=11, fontweight="bold")
    positive_image = None
    absolute_image = None
    for row_index, row in enumerate(visual_rows):
        axes[row_index, 0].imshow(row["clean_rgb"])
        axes[row_index, 1].imshow(row["corrupt_rgb"])
        _draw_region(axes[row_index, 1], row["region"])
        positive_image = axes[row_index, 2].imshow(
            row["target"], cmap="inferno", vmin=0.0, vmax=positive_vmax
        )
        _draw_region(axes[row_index, 2], row["region"])
        absolute_image = axes[row_index, 3].imshow(
            row["reference"],
            cmap="magma",
            vmin=absolute_vmin,
            vmax=absolute_vmax,
        )
        axes[row_index, 4].imshow(
            row["observed"],
            cmap="magma",
            vmin=absolute_vmin,
            vmax=absolute_vmax,
        )
        axes[row_index, 5].imshow(
            row["delta"], cmap="inferno", vmin=0.0, vmax=positive_vmax
        )
        _draw_region(axes[row_index, 5], row["region"])
        axes[row_index, 0].set_ylabel(
            "%s, severity 3\nframe %d"
            % (row["family"].replace("local_", ""), row["frame_idx"]),
            fontsize=11,
            fontweight="bold",
        )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
    if positive_image is not None:
        positive_axis = fig.add_axes((0.37, 0.055, 0.22, 0.025))
        fig.colorbar(
            positive_image,
            cax=positive_axis,
            orientation="horizontal",
            label="target / positive ΔU (shared scaled units)",
        )
    if absolute_image is not None:
        absolute_axis = fig.add_axes((0.64, 0.055, 0.22, 0.025))
        fig.colorbar(
            absolute_image,
            cax=absolute_axis,
            orientation="horizontal",
            label="raw adapter U (shared scale; no p95 calibration)",
        )
    fig.suptitle(
        "Frozen Stage-1 adapter on unseen-route validation samples — no p95 calibration",
        fontsize=15,
        fontweight="bold",
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    figure_path = args.output_dir / "local_dark_blur_adapter_comparison.png"
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    report = {
        "schema_version": "orion.stage1-local-corruption-visualization/v1",
        "claim_boundary": {
            "validation_route_not_training_route": True,
            "p95_calibration_applied": False,
            "paired_reference_used_only_for_diagnostic_delta": True,
            "corruption_mask_used_only_for_visual_audit_metrics": True,
            "supports_task_relevance_claim": False,
            "supports_closed_loop_safety_claim": False,
        },
        "checkpoint": checkpoint_metadata,
        "component_scales": [float(value) for value in scales.flatten()],
        "samples": results,
        "figure": str(figure_path.resolve()),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
