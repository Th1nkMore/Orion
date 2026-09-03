"""Visualize Density-UQ embedding sensitivity to local token masking.

This diagnostic is CPU-only. It does not re-run EVAViT. Instead, it uses
cached EVAViT patch tokens shaped [6, 1600, 1024], masks local token blocks,
and measures how Density-UQ score and active embedding change.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from uq_estimator.density import DensityUQEstimator, compute_view_moments


CAMERAS = (
    "rgb_front",
    "rgb_front_left",
    "rgb_front_right",
    "rgb_back",
    "rgb_back_left",
    "rgb_back_right",
)


@dataclass(frozen=True)
class Sample:
    filename: str
    route: str
    frame_idx: int
    weather_id: int
    scene_type: str
    score: float

    @property
    def feature_name(self) -> str:
        return self.filename

    @property
    def short_name(self) -> str:
        return f"{self.route}__{self.frame_idx:05d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--bench-root", required=True, type=Path)
    parser.add_argument("--descriptor-cache", required=True, type=Path)
    parser.add_argument("--density-checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--sample-filenames",
        nargs="*",
        default=None,
        help="Optional explicit feature filenames. Skips dataset-wide ranking.",
    )
    parser.add_argument("--camera", default="rgb_front", choices=CAMERAS)
    parser.add_argument("--grid", type=int, default=10, help="Block grid per side.")
    parser.add_argument("--examples-per-band", type=int, default=2)
    parser.add_argument(
        "--mask-mode",
        default="view_mean",
        choices=("view_mean", "zero", "global_mean"),
        help="Replacement for the masked token block.",
    )
    parser.add_argument("--image-width", type=int, default=760)
    return parser.parse_args()


def parse_filename(name: str) -> tuple[str, int]:
    match = re.match(r"(.+)__(\d+)\.pt$", name)
    if not match:
        raise ValueError(f"Unexpected feature filename: {name}")
    return match.group(1), int(match.group(2))


def image_path(bench_root: Path, sample: Sample, camera: str) -> Path:
    return bench_root / sample.route / "camera" / camera / f"{sample.frame_idx:05d}.jpg"


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_samples(args: argparse.Namespace) -> list[Sample]:
    if args.sample_filenames:
        samples = []
        for filename in args.sample_filenames:
            route, frame_idx = parse_filename(str(filename))
            weather_match = re.search(r"Weather(\d+)", route)
            weather_id = int(weather_match.group(1)) if weather_match else -1
            scene_type = "normal" if weather_id in {0, 1, 2, 3} else "adverse"
            sample = Sample(
                filename=str(filename),
                route=route,
                frame_idx=frame_idx,
                weather_id=weather_id,
                scene_type=scene_type,
                score=float("nan"),
            )
            if not (args.feature_dir / sample.feature_name).exists():
                raise FileNotFoundError(args.feature_dir / sample.feature_name)
            if not image_path(args.bench_root, sample, args.camera).exists():
                raise FileNotFoundError(image_path(args.bench_root, sample, args.camera))
            samples.append(sample)
        return samples

    cache = torch.load(args.descriptor_cache, map_location="cpu", weights_only=False)
    estimator = DensityUQEstimator.from_checkpoint(args.density_checkpoint).eval()

    descriptors = cache["descriptors"]
    score_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, descriptors.shape[0], 1024):
            end = min(start + 1024, descriptors.shape[0])
            _, scores, _, _ = estimator.encode_descriptor(descriptors[start:end])
            score_chunks.append(scores.cpu())
    scores = torch.cat(score_chunks, dim=0).flatten()

    samples: list[Sample] = []
    for idx, filename in enumerate(cache["filenames"]):
        route, frame_idx = parse_filename(str(filename))
        sample = Sample(
            filename=str(filename),
            route=route,
            frame_idx=frame_idx,
            weather_id=int(cache["weather_ids"][idx].item()),
            scene_type=str(cache["scene_types"][idx]),
            score=float(scores[idx].item()),
        )
        if (args.feature_dir / sample.feature_name).exists() and image_path(
            args.bench_root, sample, args.camera
        ).exists():
            samples.append(sample)
    if not samples:
        raise RuntimeError("No samples have both cached features and camera images.")
    return samples


def select_examples(samples: list[Sample], n: int) -> list[tuple[str, Sample]]:
    if any(np.isnan(sample.score) for sample in samples):
        return [("selected", sample) for sample in samples]

    ranked = sorted(samples, key=lambda item: item.score)
    normal = [sample for sample in ranked if sample.scene_type == "normal"]
    adverse = [sample for sample in ranked if sample.scene_type != "normal"]

    def diverse(candidates: list[Sample], count: int) -> list[Sample]:
        selected: list[Sample] = []
        routes: set[str] = set()
        for sample in candidates:
            if sample.route in routes:
                continue
            selected.append(sample)
            routes.add(sample.route)
            if len(selected) == count:
                break
        for sample in candidates:
            if len(selected) == count:
                break
            if sample not in selected:
                selected.append(sample)
        return selected

    low = diverse(normal or ranked, n)
    high = diverse((adverse or ranked)[::-1], n)
    mid_start = max(len(ranked) // 2 - n, 0)
    mid = diverse(ranked[mid_start : mid_start + 4 * n], n)

    tagged: list[tuple[str, Sample]] = []
    for tag, group in (("low", low), ("mid", mid), ("high", high)):
        tagged.extend((tag, sample) for sample in group)
    return tagged


def resize_keep(img: Image.Image, width: int) -> Image.Image:
    scale = width / float(img.width)
    height = int(round(img.height * scale))
    return img.resize((width, height), Image.Resampling.LANCZOS)


def heatmap_rgba(values: np.ndarray, size: tuple[int, int], cmap_name: str = "magma") -> Image.Image:
    vals = values.astype(np.float32)
    vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 0.0
    if vmax > 0:
        vals = vals / vmax
    else:
        vals = np.zeros_like(vals)
    cmap = plt.get_cmap(cmap_name)
    rgba = (cmap(vals) * 255).astype(np.uint8)
    img = Image.fromarray(rgba, mode="RGBA")
    return img.resize(size, Image.Resampling.BILINEAR)


def overlay_heatmap(base: Image.Image, values: np.ndarray, alpha: int = 145) -> Image.Image:
    heat = heatmap_rgba(values, base.size)
    a = heat.getchannel("A").point(lambda _: alpha)
    heat.putalpha(a)
    out = base.convert("RGBA")
    out.alpha_composite(heat)
    return out.convert("RGB")


def draw_black_block(img: Image.Image, grid: int, block_rc: tuple[int, int]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    row, col = block_rc
    x0 = int(col * img.width / grid)
    x1 = int((col + 1) * img.width / grid)
    y0 = int(row * img.height / grid)
    y1 = int((row + 1) * img.height / grid)
    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
    draw.rectangle([x0, y0, x1, y1], outline=(255, 230, 64), width=4)
    return out


def add_header(img: Image.Image, text: str, font: ImageFont.ImageFont) -> Image.Image:
    pad = 10
    header_h = 48
    out = Image.new("RGB", (img.width, img.height + header_h), (250, 250, 250))
    out.paste(img, (0, header_h))
    draw = ImageDraw.Draw(out)
    draw.text((pad, 12), text, fill=(20, 20, 20), font=font)
    return out


def make_panel(
    sample: Sample,
    camera_img: Image.Image,
    l2_grid: np.ndarray,
    delta_grid: np.ndarray,
    strongest: tuple[int, int],
    stats: dict[str, float],
    out_path: Path,
    width: int,
) -> None:
    font = load_font(20)
    small = load_font(16)
    base = resize_keep(camera_img, width)
    masked = draw_black_block(base, l2_grid.shape[0], strongest)
    l2_overlay = overlay_heatmap(base, l2_grid)
    delta_overlay = overlay_heatmap(base, np.abs(delta_grid), alpha=135)

    tiles = [
        add_header(base, "original front camera", font),
        add_header(masked, "token block masked at max embedding response", font),
        add_header(l2_overlay, "active embedding L2 sensitivity", font),
        add_header(delta_overlay, "absolute Density-UQ score change", font),
    ]

    gap = 14
    footer_h = 92
    canvas_w = width * 2 + gap
    canvas_h = tiles[0].height * 2 + gap + footer_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    positions = [(0, 0), (width + gap, 0), (0, tiles[0].height + gap), (width + gap, tiles[0].height + gap)]
    for tile, pos in zip(tiles, positions):
        canvas.paste(tile, pos)

    draw = ImageDraw.Draw(canvas)
    footer_y = tiles[0].height * 2 + gap + 12
    lines = [
        f"{sample.short_name} | weather {sample.weather_id} | {sample.scene_type}",
        (
            f"score {stats['orig_score']:.3f} -> {stats['masked_score']:.3f} "
            f"(delta {stats['masked_score_delta']:+.3f}); "
            f"active embedding L2 {stats['masked_embedding_l2']:.3f}"
        ),
    ]
    for i, line in enumerate(lines):
        draw.text((12, footer_y + i * 28), line, fill=(30, 30, 30), font=small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_contact_sheet(panel_paths: list[Path], out_path: Path, thumb_width: int = 520) -> None:
    thumbs = []
    for path in panel_paths:
        img = Image.open(path).convert("RGB")
        thumbs.append(resize_keep(img, thumb_width))
    if not thumbs:
        return
    gap = 18
    cols = 2
    rows = int(np.ceil(len(thumbs) / cols))
    cell_h = max(img.height for img in thumbs)
    canvas = Image.new("RGB", (cols * thumb_width + (cols - 1) * gap, rows * cell_h + (rows - 1) * gap), (255, 255, 255))
    for idx, img in enumerate(thumbs):
        row, col = divmod(idx, cols)
        canvas.paste(img, (col * (thumb_width + gap), row * (cell_h + gap)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def replacement_tokens(tokens: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(tokens[:, :1, :])
    if mode == "global_mean":
        return tokens.reshape(-1, tokens.shape[-1]).mean(dim=0, keepdim=True).reshape(1, 1, -1)
    if mode == "view_mean":
        return tokens.mean(dim=1, keepdim=True)
    raise ValueError(mode)


def encode(estimator: DensityUQEstimator, tokens: torch.Tensor) -> tuple[float, torch.Tensor]:
    descriptor = compute_view_moments(tokens.unsqueeze(0))
    _, score, _, active = estimator.encode_descriptor(descriptor)
    return float(score.item()), active.squeeze(0).cpu()


def compute_sensitivity(
    estimator: DensityUQEstimator,
    tokens: torch.Tensor,
    camera_index: int,
    grid: int,
    mask_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    patch_side = int(round(tokens.shape[1] ** 0.5))
    if patch_side * patch_side != tokens.shape[1]:
        raise ValueError(f"Expected square patch grid, got {tokens.shape[1]} patches")
    if patch_side % grid != 0:
        raise ValueError(f"Patch side {patch_side} must be divisible by grid {grid}")

    block = patch_side // grid
    p_count = tokens.shape[1]
    orig_descriptor = compute_view_moments(tokens.unsqueeze(0))
    _, orig_score_t, _, orig_active_t = estimator.encode_descriptor(orig_descriptor)
    orig_score = float(orig_score_t.item())
    orig_active = orig_active_t.squeeze(0).cpu()

    view_tokens = tokens[camera_index].float()
    if mask_mode == "zero":
        repl = torch.zeros(tokens.shape[-1], dtype=torch.float32)
    elif mask_mode == "global_mean":
        repl = tokens.float().reshape(-1, tokens.shape[-1]).mean(dim=0)
    elif mask_mode == "view_mean":
        repl = view_tokens.mean(dim=0)
    else:
        raise ValueError(mask_mode)

    sum_orig = view_tokens.sum(dim=0)
    sumsq_orig = (view_tokens * view_tokens).sum(dim=0)
    descriptors = orig_descriptor.repeat(grid * grid, 1)
    block_positions: list[tuple[int, int]] = []

    # Descriptor layout is view-major: [view0_mean, view0_std, view1_mean, ...].
    d_model = tokens.shape[-1]
    mean_start = camera_index * 2 * d_model
    std_start = mean_start + d_model
    for row in range(grid):
        for col in range(grid):
            patch_indices = []
            for rr in range(row * block, (row + 1) * block):
                start = rr * patch_side + col * block
                patch_indices.extend(range(start, start + block))
            idx = torch.tensor(patch_indices, dtype=torch.long)
            sum_s = view_tokens.index_select(0, idx).sum(dim=0)
            tok_s = view_tokens.index_select(0, idx)
            sumsq_s = (tok_s * tok_s).sum(dim=0)
            n_mask = float(len(patch_indices))
            sum_new = sum_orig - sum_s + n_mask * repl
            sumsq_new = sumsq_orig - sumsq_s + n_mask * repl * repl
            mean_new = sum_new / float(p_count)
            var_new = (sumsq_new / float(p_count) - mean_new * mean_new).clamp_min(0.0)
            std_new = var_new.sqrt()
            desc_idx = len(block_positions)
            descriptors[desc_idx, mean_start : mean_start + d_model] = mean_new
            descriptors[desc_idx, std_start : std_start + d_model] = std_new
            block_positions.append((row, col))

    with torch.no_grad():
        _, scores, _, active = estimator.encode_descriptor(descriptors)
    l2_values = torch.linalg.vector_norm(active.cpu() - orig_active.unsqueeze(0), dim=-1)
    delta_values = scores.flatten().cpu() - orig_score
    l2_grid = l2_values.reshape(grid, grid).numpy().astype(np.float32)
    delta_grid = delta_values.reshape(grid, grid).numpy().astype(np.float32)

    max_pos = np.unravel_index(np.argmax(l2_grid), l2_grid.shape)
    row, col = int(max_pos[0]), int(max_pos[1])
    flat_idx = row * grid + col
    masked_score = float(scores.flatten()[flat_idx].item())
    masked_l2 = float(l2_values[flat_idx].item())
    stats = {
        "orig_score": orig_score,
        "masked_score": masked_score,
        "masked_score_delta": masked_score - orig_score,
        "masked_embedding_l2": masked_l2,
        "max_l2": float(l2_grid.max()),
        "max_positive_score_delta": float(np.maximum(delta_grid, 0.0).max()),
        "mean_l2": float(l2_grid.mean()),
        "mean_abs_score_delta": float(np.abs(delta_grid).mean()),
        "max_row": row,
        "max_col": col,
    }
    return l2_grid, delta_grid, stats


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    estimator = DensityUQEstimator.from_checkpoint(args.density_checkpoint).eval()
    samples = load_samples(args)
    selected = select_examples(samples, args.examples_per_band)
    camera_index = CAMERAS.index(args.camera)

    rows: list[dict[str, object]] = []
    panel_paths: list[Path] = []
    for idx, (band, sample) in enumerate(selected, start=1):
        feature = torch.load(args.feature_dir / sample.feature_name, map_location="cpu", weights_only=False)
        tokens = feature["tokens"].float()
        l2_grid, delta_grid, stats = compute_sensitivity(
            estimator, tokens, camera_index, args.grid, args.mask_mode
        )
        stem = f"{idx:02d}_{band}_{sample.short_name}"
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
        np.save(args.out_dir / f"{safe_stem}_embedding_l2.npy", l2_grid)
        np.save(args.out_dir / f"{safe_stem}_score_delta.npy", delta_grid)
        img = Image.open(image_path(args.bench_root, sample, args.camera)).convert("RGB")
        panel_path = args.out_dir / f"{safe_stem}_panel.png"
        make_panel(
            sample,
            img,
            l2_grid,
            delta_grid,
            (int(stats["max_row"]), int(stats["max_col"])),
            stats,
            panel_path,
            args.image_width,
        )
        panel_paths.append(panel_path)
        row = {
            "band": band,
            "filename": sample.filename,
            "route": sample.route,
            "frame_idx": sample.frame_idx,
            "weather_id": sample.weather_id,
            "scene_type": sample.scene_type,
            "camera": args.camera,
            **stats,
        }
        rows.append(row)
        print(
            f"[{idx}/{len(selected)}] {band} {sample.short_name}: "
            f"score {stats['orig_score']:.3f}->{stats['masked_score']:.3f}, "
            f"emb_l2={stats['masked_embedding_l2']:.3f}"
        )

    csv_path = args.out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    make_contact_sheet(panel_paths, args.out_dir / "contact_sheet.png")
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
