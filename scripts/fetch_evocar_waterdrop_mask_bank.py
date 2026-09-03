#!/usr/bin/env python3
"""Fetch a small CC-BY-4.0 real windshield-drop mask bank by ZIP ranges."""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from PIL import Image


URL = "https://zenodo.org/api/records/4680442/files/RaindropsOnWindshield.zip/content"
RECORD = "https://doi.org/10.5281/zenodo.4680442"
ARCHIVE_MD5 = "0ea1b373c981f8ed3ecd311d6596ec0f"
UPSTREAM_COMMIT = "01ba7ff8a7c7d7e7865059de69d80447e3c9fb1c"
MEMBERS = (
    "masks/D1/D1_0095.png",
    "masks/D2/D2_0050.png",
    "masks/D3/D3_0/D3_0950.png",
    "masks/D4/D4_0/D4_0950.png",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reuse-members-from",
        type=Path,
        help="Reuse previously range-verified flattened members after a network interruption.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite real-waterdrop mask bank")
    output.mkdir(parents=True)
    assets = []
    archive = None
    if args.reuse_members_from is None:
        try:
            from remotezip import RemoteZip
        except ImportError as error:  # pragma: no cover - developer acquisition path
            raise SystemExit(
                "remotezip is required only for asset acquisition; install it "
                "into a temporary target and add that target to PYTHONPATH"
            ) from error
        archive = RemoteZip(URL)
        names = set(archive.namelist())
        missing = sorted(set(MEMBERS) - names)
        if missing:
            raise RuntimeError("upstream mask members missing: %s" % missing)
    try:
        for member in MEMBERS:
            filename = member.replace("masks/", "").replace("/", "__")
            if args.reuse_members_from is not None:
                reused = args.reuse_members_from.resolve() / filename
                if not reused.is_file():
                    raise FileNotFoundError("verified reused member missing: %s" % reused)
                payload = reused.read_bytes()
            else:
                payload = archive.read(member)
            with Image.open(BytesIO(payload)) as image:
                mask = image.convert("L")
                if min(mask.size) < 640:
                    raise ValueError("source mask is unexpectedly small: %s" % member)
                value_range = mask.getextrema()
                if value_range[1] <= value_range[0]:
                    raise ValueError("source mask is constant: %s" % member)
                histogram = mask.histogram()
                unique_value_count = sum(value > 0 for value in histogram)
                target = output / filename
                target.write_bytes(payload)
                assets.append(
                    {
                        "member": member,
                        "file": filename,
                        "sha256": sha256(payload).hexdigest(),
                        "width": mask.width,
                        "height": mask.height,
                        "source_value_range": list(value_range),
                        "source_unique_value_count": unique_value_count,
                    }
                )
    finally:
        if archive is not None:
            archive.close()
    metadata = {
        "schema": "orion.real_waterdrop_mask_bank.v1",
        "source": {
            "title": "Raindrops on Windshield Dataset",
            "authors": ["Vera Soboleva", "Oleg Shipitko"],
            "record": RECORD,
            "archive_url": URL,
            "archive_size_bytes": 7403293582,
            "archive_md5": ARCHIVE_MD5,
            "license": "CC-BY-4.0",
            "upstream_repository": "https://github.com/Evocargo/RaindropsOnWindshield",
            "upstream_commit": UPSTREAM_COMMIT,
        },
        "selection": "One fixed mid-sequence non-empty annotated real mask from each of the four D-series waterdrop sequences; no scene RGB is redistributed. Empty mid-sequence masks from the other released sequences were excluded prospectively.",
        "acquisition": {
            "method": (
                "reused previously range-verified members after an interrupted retry"
                if args.reuse_members_from is not None
                else "direct HTTP ZIP range extraction"
            ),
            "full_archive_downloaded": False,
        },
        "assets": assets,
        "claim_boundary": "Real binary silhouette masks only. Alpha, displacement, refraction, edge, and highlight fields remain renderer-derived and must be reviewed separately.",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(output), "asset_count": len(assets)}, indent=2))


if __name__ == "__main__":
    main()
