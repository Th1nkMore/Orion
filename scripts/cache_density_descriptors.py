"""Build compact per-frame descriptors from pre-extracted EVAViT tokens."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from tqdm import tqdm

from uq_estimator.density import compute_view_moments


WEATHER_RE = re.compile(r"_Weather(\d+)__")


def parse_feature_name(filename: str) -> tuple[str, int]:
    stem = Path(filename).stem
    if "__" not in stem:
        raise ValueError(f"Cannot parse route from feature filename: {filename}")
    route = stem.rsplit("__", 1)[0]
    match = WEATHER_RE.search(filename)
    if match is None:
        raise ValueError(f"Cannot parse weather id from feature filename: {filename}")
    return route, int(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_dir = Path(args.feature_dir)
    files = sorted(feature_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {feature_dir}")

    output_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    device = torch.device(args.device)
    first = torch.load(files[0], map_location="cpu", weights_only=True)["tokens"]
    descriptor_dim = compute_view_moments(first.to(device)).numel()
    descriptors = torch.empty((len(files), descriptor_dim), dtype=output_dtype)
    filenames: list[str] = []
    routes: list[str] = []
    weather_ids = torch.empty(len(files), dtype=torch.int16)
    scene_types: list[str] = []

    for index, path in enumerate(tqdm(files, desc="Caching descriptors")):
        data = torch.load(path, map_location="cpu", weights_only=True)
        descriptor = compute_view_moments(
            data["tokens"].to(device, non_blocking=True)
        ).cpu().to(output_dtype)
        if descriptor.numel() != descriptor_dim:
            raise ValueError(
                f"Descriptor dimension mismatch for {path.name}: "
                f"{descriptor.numel()} != {descriptor_dim}"
            )
        route, weather_id = parse_feature_name(path.name)
        expected_type = "normal" if weather_id in {0, 1, 2, 3} else "adverse"
        actual_type = data.get("scene_type", expected_type)
        if actual_type != expected_type:
            raise ValueError(
                f"scene_type mismatch for {path.name}: "
                f"{actual_type!r} != {expected_type!r}"
            )
        descriptors[index].copy_(descriptor)
        filenames.append(path.name)
        routes.append(route)
        weather_ids[index] = weather_id
        scene_types.append(expected_type)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "descriptors": descriptors,
            "filenames": filenames,
            "routes": routes,
            "weather_ids": weather_ids,
            "scene_types": scene_types,
            "descriptor": "per_view_patch_mean_std",
        },
        output,
    )
    size_mb = output.stat().st_size / 1e6
    print(
        f"Saved {len(files)} descriptors with shape "
        f"{tuple(descriptors.shape)} to {output} ({size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    main()
