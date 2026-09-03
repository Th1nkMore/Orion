"""Render Bench2Drive camera GIFs and Density-UQ visual examples.

This script is CPU-only. It reads cached Density-UQ descriptors and the
Bench2Drive validation image folders, then renders report-ready camera mosaics,
front-camera GIFs, and low/high UQ comparison sheets.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import torch
from PIL import Image, ImageDraw, ImageFont

from uq_estimator.density import DensityUQEstimator


CAMERAS = (
    ("rgb_front_left", "front left"),
    ("rgb_front", "front"),
    ("rgb_front_right", "front right"),
    ("rgb_back_left", "back left"),
    ("rgb_back", "back"),
    ("rgb_back_right", "back right"),
)


@dataclass(frozen=True)
class Sample:
    route: str
    frame_idx: int
    weather_id: int
    scene_type: str
    score: float
    distance: float
    filename: str

    @property
    def image_stem(self) -> str:
        return f"{self.frame_idx:05d}.jpg"

    @property
    def short_name(self) -> str:
        return f"{self.route}__{self.frame_idx:05d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", required=True, type=Path)
    parser.add_argument("--descriptor-cache", required=True, type=Path)
    parser.add_argument("--density-checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--gif-frames", type=int, default=24)
    parser.add_argument("--gif-stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--front-width", type=int, default=760)
    parser.add_argument("--tile-width", type=int, default=360)
    parser.add_argument("--examples-per-band", type=int, default=4)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def parse_filename(name: str) -> tuple[str, int]:
    match = re.match(r"(.+)__(\d+)\.pt$", name)
    if not match:
        raise ValueError(f"Unexpected descriptor filename: {name}")
    return match.group(1), int(match.group(2))


def load_samples(descriptor_cache: Path, density_checkpoint: Path) -> list[Sample]:
    cache = torch.load(descriptor_cache, map_location="cpu", weights_only=False)
    estimator = DensityUQEstimator.from_checkpoint(density_checkpoint, map_location="cpu")
    descriptors = cache["descriptors"]
    score_chunks: list[torch.Tensor] = []
    distance_chunks: list[torch.Tensor] = []
    # Keep peak memory low enough for CPU-only AutoDL instances.
    with torch.no_grad():
        for start in range(0, descriptors.shape[0], 1024):
            end = min(start + 1024, descriptors.shape[0])
            _, chunk_scores, chunk_distances, _ = estimator.encode_descriptor(
                descriptors[start:end]
            )
            score_chunks.append(chunk_scores.cpu())
            distance_chunks.append(chunk_distances.cpu())
    scores = torch.cat(score_chunks, dim=0).flatten()
    distances = torch.cat(distance_chunks, dim=0).flatten()
    filenames = cache["filenames"]
    weather_ids = cache["weather_ids"].cpu().tolist()
    scene_types = cache["scene_types"]
    samples: list[Sample] = []
    for idx, filename in enumerate(filenames):
        route, frame_idx = parse_filename(filename)
        samples.append(
            Sample(
                route=route,
                frame_idx=frame_idx,
                weather_id=int(weather_ids[idx]),
                scene_type=str(scene_types[idx]),
                score=float(scores[idx].item()),
                distance=float(distances[idx].item()),
                filename=str(filename),
            )
        )
    return samples


def image_path(bench_root: Path, sample: Sample, camera: str, frame_idx: int | None = None) -> Path:
    idx = sample.frame_idx if frame_idx is None else frame_idx
    return bench_root / sample.route / "camera" / camera / f"{idx:05d}.jpg"


def compact_route(route: str) -> str:
    scenario = route.split("_Town", 1)[0]
    town_match = re.search(r"Town[^_]+", route)
    route_match = re.search(r"Route\d+", route)
    weather_match = re.search(r"Weather\d+", route)
    parts = [scenario]
    suffix = " ".join(
        item.group(0)
        for item in (town_match, route_match, weather_match)
        if item is not None
    )
    if suffix:
        parts.append(suffix)
    return " | ".join(parts)


def sample_has_all_cameras(bench_root: Path, sample: Sample) -> bool:
    return all(image_path(bench_root, sample, camera).exists() for camera, _ in CAMERAS)


def route_has_frames(bench_root: Path, sample: Sample, frames: list[int]) -> bool:
    return all(image_path(bench_root, sample, "rgb_front", idx).exists() for idx in frames)


def select_examples(samples: list[Sample], bench_root: Path, n: int) -> dict[str, list[Sample]]:
    valid = [sample for sample in samples if sample_has_all_cameras(bench_root, sample)]
    if not valid:
        raise RuntimeError(f"No valid camera samples under {bench_root}")
    ranked = sorted(valid, key=lambda item: item.score)
    normal_ranked = [sample for sample in ranked if sample.scene_type == "normal"]
    adverse_ranked = [sample for sample in ranked if sample.scene_type != "normal"]
    low = (normal_ranked or ranked)[: max(n * 4, n)]
    high_source = adverse_ranked or ranked
    high = high_source[-max(n * 4, n) :]
    mid_start = max(len(ranked) // 2 - n * 2, 0)
    mid = ranked[mid_start : mid_start + max(n * 4, n)]

    def diverse(candidates: list[Sample]) -> list[Sample]:
        selected: list[Sample] = []
        routes: set[str] = set()
        for sample in candidates:
            if sample.route in routes:
                continue
            selected.append(sample)
            routes.add(sample.route)
            if len(selected) == n:
                break
        if len(selected) < n:
            for sample in candidates:
                if sample not in selected:
                    selected.append(sample)
                if len(selected) == n:
                    break
        return selected

    return {
        "low": diverse(low),
        "mid": diverse(mid),
        "high": diverse(list(reversed(high))),
    }


def choose_gif_sample(candidates: list[Sample], bench_root: Path, frames: int, stride: int) -> Sample:
    for sample in candidates:
        start = max(sample.frame_idx - frames * stride // 3, 0)
        frame_ids = [start + i * stride for i in range(frames)]
        if route_has_frames(bench_root, sample, frame_ids):
            return Sample(
                route=sample.route,
                frame_idx=start,
                weather_id=sample.weather_id,
                scene_type=sample.scene_type,
                score=sample.score,
                distance=sample.distance,
                filename=sample.filename,
            )
    raise RuntimeError("No candidate had enough consecutive rgb_front frames")


def add_label(image: Image.Image, text: str, font: ImageFont.ImageFont, fill=(255, 255, 255)) -> Image.Image:
    draw = ImageDraw.Draw(image)
    pad = 8
    bbox = draw.textbbox((0, 0), text, font=font)
    rect = (0, 0, bbox[2] + pad * 2, bbox[3] + pad * 2)
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=fill, font=font)
    return image


def resize_keep(image: Image.Image, width: int) -> Image.Image:
    ratio = width / image.width
    return image.resize((width, int(image.height * ratio)), Image.Resampling.LANCZOS)


def render_mosaic(bench_root: Path, sample: Sample, frame_idx: int, tile_width: int) -> Image.Image:
    font = load_font(18)
    small_font = load_font(15)
    tiles: list[Image.Image] = []
    for camera, label in CAMERAS:
        path = image_path(bench_root, sample, camera, frame_idx)
        tile = resize_keep(Image.open(path).convert("RGB"), tile_width)
        tile = add_label(tile, label, small_font)
        tiles.append(tile)
    w, h = tiles[0].size
    header_h = 76
    canvas = Image.new("RGB", (w * 3, h * 2 + header_h), (245, 245, 242))
    draw = ImageDraw.Draw(canvas)
    title = compact_route(sample.route)
    subtitle = (
        f"frame {frame_idx:05d} | Density-UQ {sample.score:.3f} | "
        f"weather {sample.weather_id} | {sample.scene_type} | six-camera RGB input"
    )
    draw.text((12, 8), title, fill=(20, 35, 38), font=font)
    draw.text((12, 42), subtitle, fill=(75, 75, 75), font=small_font)
    for i, tile in enumerate(tiles):
        x = (i % 3) * w
        y = header_h + (i // 3) * h
        canvas.paste(tile, (x, y))
    return canvas


def render_front(bench_root: Path, sample: Sample, frame_idx: int, width: int) -> Image.Image:
    font = load_font(20)
    img = resize_keep(Image.open(image_path(bench_root, sample, "rgb_front", frame_idx)).convert("RGB"), width)
    label = (
        f"{compact_route(sample.route)} | frame {frame_idx:05d} | "
        f"UQ {sample.score:.3f} | W{sample.weather_id} | {sample.scene_type}"
    )
    return add_label(img, label, font)


def write_gif(frames: list[Image.Image], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, [frame.convert("RGB") for frame in frames], fps=fps)


def render_gifs(args: argparse.Namespace, band_samples: dict[str, list[Sample]]) -> list[dict]:
    rows: list[dict] = []
    for band in ("low", "mid", "high"):
        sample = choose_gif_sample(
            band_samples[band], args.bench_root, args.gif_frames, args.gif_stride
        )
        frame_ids = [sample.frame_idx + i * args.gif_stride for i in range(args.gif_frames)]
        front_frames = [
            render_front(args.bench_root, sample, idx, args.front_width)
            for idx in frame_ids
        ]
        mosaic_frames = [
            render_mosaic(args.bench_root, sample, idx, args.tile_width)
            for idx in frame_ids
        ]
        write_gif(front_frames, args.out_dir / "gifs" / f"{band}_uq_front.gif", args.fps)
        write_gif(mosaic_frames, args.out_dir / "gifs" / f"{band}_uq_six_camera.gif", args.fps)
        rows.append(
            {
                "kind": "gif",
                "band": band,
                "route": sample.route,
                "start_frame": sample.frame_idx,
                "end_frame": frame_ids[-1],
                "score": sample.score,
                "weather_id": sample.weather_id,
                "scene_type": sample.scene_type,
                "front_gif": str(args.out_dir / "gifs" / f"{band}_uq_front.gif"),
                "six_camera_gif": str(args.out_dir / "gifs" / f"{band}_uq_six_camera.gif"),
            }
        )
    return rows


def render_static_examples(args: argparse.Namespace, band_samples: dict[str, list[Sample]]) -> list[dict]:
    rows: list[dict] = []
    sheet_pairs: list[tuple[Sample, Sample]] = []
    for low_sample, high_sample in zip(
        band_samples.get("low", []), band_samples.get("high", [])
    ):
        sheet_pairs.append((low_sample, high_sample))
    for band, samples in band_samples.items():
        for rank, sample in enumerate(samples, start=1):
            out = args.out_dir / "static_examples" / f"{band}_uq_{rank}_{sample.short_name}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            image = render_mosaic(args.bench_root, sample, sample.frame_idx, args.tile_width)
            image.save(out)
            rows.append(
                {
                    "kind": "static",
                    "band": band,
                    "rank": rank,
                    "route": sample.route,
                    "frame_idx": sample.frame_idx,
                    "score": sample.score,
                    "distance": sample.distance,
                    "weather_id": sample.weather_id,
                    "scene_type": sample.scene_type,
                    "path": str(out),
                }
            )
    if sheet_pairs:
        font = load_font(22)
        small_font = load_font(17)
        tile_w = 520
        header_h = 78
        images: list[tuple[str, Image.Image]] = []
        for low_sample, high_sample in sheet_pairs:
            images.append(
                (
                    f"LOW UQ {low_sample.score:.3f} | W{low_sample.weather_id} | {low_sample.scene_type}",
                    render_front(args.bench_root, low_sample, low_sample.frame_idx, tile_w),
                )
            )
            images.append(
                (
                    f"HIGH UQ {high_sample.score:.3f} | W{high_sample.weather_id} | {high_sample.scene_type}",
                    render_front(args.bench_root, high_sample, high_sample.frame_idx, tile_w),
                )
            )
        tile_h = images[0][1].size[1]
        cols = 2
        rows_n = len(sheet_pairs)
        canvas = Image.new("RGB", (tile_w * cols, tile_h * rows_n + header_h), (245, 245, 242))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), "Density-UQ case comparison", fill=(20, 35, 38), font=font)
        draw.text((12, 43), "Left: low score examples. Right: high score examples.", fill=(75, 75, 75), font=small_font)
        for i, (label, tile) in enumerate(images):
            x = (i % cols) * tile_w
            y = header_h + (i // cols) * tile_h
            canvas.paste(tile, (x, y))
            draw.rectangle((x, y, x + tile_w, y + 30), fill=(0, 0, 0))
            draw.text((x + 8, y + 6), label, fill=(255, 255, 255), font=small_font)
        sheet = args.out_dir / "density_uq_low_high_contact_sheet.png"
        canvas.save(sheet)
        rows.append({"kind": "contact_sheet", "path": str(sheet)})
    return rows


def write_metadata(args: argparse.Namespace, rows: list[dict], samples: list[Sample]) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "manifest.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    scores = torch.tensor([sample.score for sample in samples])
    summary = {
        "num_samples": len(samples),
        "score_min": float(scores.min()),
        "score_p25": float(scores.quantile(0.25)),
        "score_p50": float(scores.quantile(0.50)),
        "score_p75": float(scores.quantile(0.75)),
        "score_max": float(scores.max()),
        "manifest": str(csv_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    samples = load_samples(args.descriptor_cache, args.density_checkpoint)
    band_samples = select_examples(samples, args.bench_root, args.examples_per_band)
    rows = []
    rows.extend(render_gifs(args, band_samples))
    rows.extend(render_static_examples(args, band_samples))
    write_metadata(args, rows, samples)
    print(f"Wrote {len(rows)} assets under {args.out_dir}")


if __name__ == "__main__":
    main()
