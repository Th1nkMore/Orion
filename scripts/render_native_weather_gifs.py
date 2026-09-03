#!/usr/bin/env python3
"""Render side-by-side front/BEV GIFs from native CARLA weather captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CONDITIONS = ("clear", "fog_light", "fog_heavy")


def _font(size, bold=False):
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _load_manifest(capture_root):
    path = capture_root / "capture_manifest.json"
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "orion.native-carla-weather-capture/v1":
        raise RuntimeError("unexpected capture manifest schema: %s" % path)
    routes = {str(item["route_id"]) for item in payload["items"]}
    if len(routes) != 1:
        raise RuntimeError("each visual capture root must contain exactly one route")
    return payload, next(iter(routes))


def _render_comparison(capture_root, route_id, directory, output, fps):
    route_directory = route_id.replace("/", "_")
    paths_by_condition = {}
    for condition in CONDITIONS:
        paths = sorted((capture_root / condition / route_directory / directory).glob("*.png"))
        if not paths:
            raise RuntimeError("missing %s/%s frames" % (condition, directory))
        paths_by_condition[condition] = paths
    frame_count = len(paths_by_condition["clear"])
    if any(len(paths) != frame_count for paths in paths_by_condition.values()):
        raise RuntimeError("condition frame counts do not match")
    frames = []
    for index in range(frame_count):
        images = []
        for condition in CONDITIONS:
            image = Image.open(paths_by_condition[condition][index]).convert("RGB")
            target_width = 640 if directory == "rgb_front" else 420
            target_height = int(image.height * target_width / image.width)
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            card = Image.new("RGB", (target_width, target_height + 54), "white")
            card.paste(image, (0, 54))
            draw = ImageDraw.Draw(card)
            draw.text((18, 13), condition.replace("_", " ").title(), font=_font(24, True), fill="#172033")
            images.append(card)
        canvas = Image.new("RGB", (sum(image.width for image in images), images[0].height + 52), "#F3F6FA")
        x = 0
        for image in images:
            canvas.paste(image, (x, 52))
            x += image.width
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 12), "%s — %s — pose %02d" % (route_id, directory, index), font=_font(26, True), fill="#172033")
        frames.append(canvas)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=max(40, int(1000 / fps)),
        loop=0,
        optimize=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=4.0)
    args = parser.parse_args(argv)
    if args.fps <= 0:
        raise SystemExit("fps must be positive")
    outputs = []
    for capture_root in args.capture_root:
        _, route_id = _load_manifest(capture_root)
        route_tag = route_id.replace("/", "_")
        for directory, label in (("rgb_front", "front"), ("bev", "bev")):
            output = args.output_dir / ("%s_%s_weather_comparison.gif" % (route_tag, label))
            _render_comparison(capture_root, route_id, directory, output, args.fps)
            outputs.append(str(output.resolve()))
    print(json.dumps({"outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
