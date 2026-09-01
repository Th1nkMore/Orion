#!/usr/bin/env python3
"""Generate CPU Route214 six-view projection overlays; never load ORION."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uq_estimator.projection_overlay_preflight import (  # noqa: E402
    DEFAULT_CANDIDATE_FRAME,
    DEFAULT_FRAMES,
    ProjectionOverlayPreflightError,
    build_mock_route214_projection_frames,
    generate_route214_projection_overlays,
    load_route214_frames_from_dedicated_pipeline,
)


DEFAULT_CONFIG = REPO_ROOT / "adzoo" / "orion" / "configs" / "orion_stage3_agent.py"


def _parse_frames(value: str) -> tuple[int, ...]:
    try:
        frames = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("frames must be comma-separated integers") from exc
    if not frames or any(frame < 0 for frame in frames) or len(set(frames)) != len(frames):
        raise argparse.ArgumentTypeError("frames must be unique non-negative integers")
    return frames


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic CPU fixtures; artifacts are explicitly marked mock.",
    )
    mode.add_argument(
        "--dataset",
        action="store_true",
        help="Run the real dedicated target geometry pipeline on Route214.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=_parse_frames, default=DEFAULT_FRAMES)
    parser.add_argument("--candidate-frame", type=int, default=DEFAULT_CANDIDATE_FRAME)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--infos", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--map-file", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.candidate_frame < 0:
        raise ProjectionOverlayPreflightError("candidate-frame must be non-negative")
    if 0 not in args.frames or args.candidate_frame not in args.frames:
        raise ProjectionOverlayPreflightError(
            "--frames must include frame 0 and --candidate-frame"
        )
    if args.mock:
        if tuple(args.frames) != DEFAULT_FRAMES or args.candidate_frame != DEFAULT_CANDIDATE_FRAME:
            raise ProjectionOverlayPreflightError(
                "mock fixture is frozen to Route214 frames 0 and 39"
            )
        frames = build_mock_route214_projection_frames()
    else:
        missing = [
            name
            for name, value in (
                ("--infos", args.infos),
                ("--dataset-root", args.dataset_root),
                ("--map-file", args.map_file),
            )
            if value is None
        ]
        if missing:
            raise ProjectionOverlayPreflightError(
                "dataset mode requires " + ", ".join(missing)
            )
        frames, _ = load_route214_frames_from_dedicated_pipeline(
            repo_root=REPO_ROOT,
            config_path=args.config,
            infos_path=args.infos,
            dataset_root=args.dataset_root,
            map_file=args.map_file,
            frames=args.frames,
        )
    manifest = generate_route214_projection_overlays(
        frames, args.output_dir, candidate_frame=args.candidate_frame
    )
    summary = {
        "schema_version": manifest["schema_version"],
        "route_key": manifest["route_key"],
        "input_mode": manifest["input_mode"],
        "selected_frames": manifest["selected_frames"],
        "candidate_frame": manifest["candidate_frame"],
        "automated_preflight": manifest["automated_preflight"],
        "claim_boundary": manifest["claim_boundary"],
        "manifest_path": manifest["manifest_path"],
        "manifest_sha256": manifest["manifest_sha256"],
    }
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
