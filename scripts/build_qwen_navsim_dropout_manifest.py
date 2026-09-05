#!/usr/bin/env python3
"""Build an exact frame manifest from a Qwen benchmark scene JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


NUM_CAMERAS = 3


def selected_paths(scene_path: Path, camera_index: int, scope: str) -> list[str]:
    selected = []
    with scene_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            content = record["messages"][0]["content"]
            images = [str(item["image"]).replace("\\", "/") for item in content if "image" in item]
            per_camera, remainder = divmod(len(images), NUM_CAMERAS)
            if remainder or per_camera == 0:
                raise ValueError(
                    f"{scene_path}:{line_number} has {len(images)} images; expected "
                    "equal non-empty groups for three cameras"
                )
            group = images[camera_index * per_camera : (camera_index + 1) * per_camera]
            selected.extend(group[-1:] if scope == "current" else group)
    if not selected:
        raise ValueError(f"no images found in {scene_path}")
    return list(dict.fromkeys(selected))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--camera-index", type=int, default=0, choices=range(NUM_CAMERAS))
    parser.add_argument("--scope", choices=("current", "all-history"), default="current")
    args = parser.parse_args()
    paths = selected_paths(args.scenes, args.camera_index, args.scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(path + "\n" for path in paths), encoding="utf-8")
    print(f"wrote {len(paths)} unique frame paths to {args.output}")


if __name__ == "__main__":
    main()
