#!/usr/bin/env python3
"""Render a human-review bundle for the Route151 clean-camera A/B/C test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


PROFILES = ("none", "intensity_zero_only", "clean")


def _read_trace(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError("empty capture trace: %s" % path)
    return rows


def _nearest(rows, progress):
    return min(rows, key=lambda row: abs(float(row["route_progress"]) - progress))


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _panel(images, labels, size=(480, 270)):
    header = 34
    canvas = Image.new("RGB", (size[0] * len(images), size[1] + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        fitted = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        x = index * size[0]
        canvas.paste(fitted, (x, header))
        draw.text((x + 8, 9), label, fill="black")
    return canvas


def _save_gif(path, frames, fps):
    duration = max(20, int(round(1000.0 / fps)))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )


def render(root: Path, protocol_path: Path, output: Path):
    protocol = json.loads(protocol_path.read_text())
    if protocol["schema"] != "orion.clean_camera_render_diagnostic.v1":
        raise ValueError("unexpected diagnostic protocol schema")
    traces = {
        profile: _read_trace(
            next((root / "captures" / profile).glob("**/capture_trace.jsonl"))
        )
        for profile in PROFILES
    }
    output.mkdir(parents=True, exist_ok=False)
    reference = traces["none"]
    maximum_frames = int(protocol["render"]["maximum_frames"])
    stride = max(1, len(reference) // maximum_frames)
    reference_rows = reference[::stride][:maximum_frames]
    panels = []
    realized = []
    for row in reference_rows:
        progress = float(row["route_progress"])
        matched = {profile: _nearest(traces[profile], progress) for profile in PROFILES}
        images = [Image.open(matched[profile]["front"]) for profile in PROFILES]
        panels.append(_panel(images, PROFILES))
        realized.append(
            {
                profile: {
                    "route_progress": float(matched[profile]["route_progress"]),
                    "front": matched[profile]["front"],
                }
                for profile in PROFILES
            }
        )
    contact_progress = float(protocol["route"]["contact_progress"])
    contact_rows = {
        profile: _nearest(traces[profile], contact_progress) for profile in PROFILES
    }
    contact = _panel(
        [Image.open(contact_rows[profile]["front"]) for profile in PROFILES],
        ["%s @ %.5f" % (profile, contact_rows[profile]["route_progress"]) for profile in PROFILES],
    )
    contact_path = output / "front_contact.png"
    gif_path = output / "front_diagnostic.gif"
    contact.save(contact_path)
    _save_gif(gif_path, panels, float(protocol["render"]["fps"]))
    raw_evidence = {}
    for profile, row in contact_rows.items():
        source = Path(row["front"])
        target = output / ("raw_%s.png" % profile)
        target.write_bytes(source.read_bytes())
        with Image.open(target) as image:
            raw_evidence[profile] = {
                "path": str(target.resolve()),
                "sha256": _sha256(target),
                "size": list(image.size),
                "route_progress": float(row["route_progress"]),
                "camera_postprocess_readback": row["camera_postprocess_readback"],
                "weather": row["weather"],
            }
    result = {
        "schema": "orion.clean_camera_render_diagnostic_result.v1",
        "status": "pending_human_visual_review",
        "orion_loaded": False,
        "safety_outcome_used": False,
        "profiles": list(PROFILES),
        "frame_count": len(panels),
        "contact": {"path": str(contact_path.resolve()), "sha256": _sha256(contact_path)},
        "gif": {"path": str(gif_path.resolve()), "sha256": _sha256(gif_path)},
        "raw_evidence": raw_evidence,
        "matched_frames": realized,
        "decision_rule": protocol["decision_rule"],
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.root.resolve(), args.protocol.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
