#!/usr/bin/env python3
"""Fetch a tiny auditable ICRA 2023 paired/real waterdrop reference bank."""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from PIL import Image


URL = "https://huggingface.co/datasets/WayneWenOfficial/ICRA2023/resolve/main/dataset/test.zip"
DATASET_REPOSITORY = "https://huggingface.co/datasets/WayneWenOfficial/ICRA2023"
PAPER_REPOSITORY = "https://github.com/csqiangwen/Video_Waterdrop_Removal_in_Driving_Scenes"
DATASET_COMMIT = "70c3f8f1c3ae8e2e712c1f64322b22ca7ef3b60e"
ARCHIVE_SIZE_BYTES = 2840485103
ARCHIVE_LINKED_SHA256 = "51c1455f4cc31ac8cd0d7ea1d41160044f7d64b1e4253832c30a4fee29723b1a"
MEMBERS = (
    "test/syn/clean_vid/0003/000075.png",
    "test/syn/rainy_vid/0003/000075.png",
    "test/real/6/40.png",
    "test/real/6/120.png",
    "test/real/6/200.png",
)


def _flatten(member: str) -> str:
    return member.replace("/", "__")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reuse-members-from",
        type=Path,
        help="Reuse previously range-extracted members named with '/' replaced by '__'.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite ICRA 2023 waterdrop reference bank")
    output.mkdir(parents=True)
    archive = None
    if args.reuse_members_from is None:
        try:
            from remotezip import RemoteZip
        except ImportError as error:  # pragma: no cover - acquisition-only dependency
            raise SystemExit("remotezip is required for direct HTTP ZIP range extraction") from error
        archive = RemoteZip(URL)
        names = set(archive.namelist())
        missing = sorted(set(MEMBERS) - names)
        if missing:
            raise RuntimeError("source archive members missing: %s" % missing)
    assets = []
    try:
        for member in MEMBERS:
            filename = _flatten(member)
            if args.reuse_members_from is not None:
                source = args.reuse_members_from.resolve() / filename
                if not source.is_file():
                    raise FileNotFoundError("reused range-extracted member missing: %s" % source)
                payload = source.read_bytes()
            else:
                payload = archive.read(member)
            with Image.open(BytesIO(payload)) as image:
                width, height = image.size
                if member.startswith("test/syn/") and (width, height) != (960, 540):
                    raise ValueError("paired synthetic source resolution differs")
                if member.startswith("test/real/") and (width, height) != (1280, 704):
                    raise ValueError("real reference source resolution differs")
                target = output / filename
                target.write_bytes(payload)
                assets.append(
                    {
                        "member": member,
                        "file": filename,
                        "sha256": sha256(payload).hexdigest(),
                        "width": width,
                        "height": height,
                        "role": (
                            "paired_template_source"
                            if member.startswith("test/syn/")
                            else "real_driving_visual_reference_only"
                        ),
                    }
                )
    finally:
        if archive is not None:
            archive.close()
    metadata = {
        "schema": "orion.icra2023_waterdrop_reference_bank.v1",
        "source": {
            "paper": "Video Waterdrop Removal via Spatio-Temporal Fusion in Driving Scenes",
            "venue": "ICRA 2023",
            "dataset_repository": DATASET_REPOSITORY,
            "paper_repository": PAPER_REPOSITORY,
            "dataset_commit": DATASET_COMMIT,
            "license_as_declared_by_dataset_card": "MIT",
            "archive_url": URL,
            "archive_size_bytes": ARCHIVE_SIZE_BYTES,
            "archive_linked_sha256": ARCHIVE_LINKED_SHA256,
        },
        "selection": {
            "paired_source": "One prospectively selected clean/rainy synthetic pair (sequence 0003, frame 000075) with moderate, distributed small droplets.",
            "real_reference": "Three fixed frames from real driving sequence 6, used only to compare visual morphology and temporal persistence.",
            "selection_inputs_excluded": ["ORION output", "TTC", "collision", "route outcome", "actor bounding boxes"],
        },
        "acquisition": {
            "full_archive_downloaded": False,
            "method": (
                "reused individually HTTP-range-extracted members"
                if args.reuse_members_from is not None
                else "direct HTTP ZIP range extraction"
            ),
        },
        "assets": assets,
        "claim_boundary": "The paired template source is published synthetic data, not a photographed clean/rainy pair. The real frames are unpaired visual references and are not composited into CARLA.",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "asset_count": len(assets)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
