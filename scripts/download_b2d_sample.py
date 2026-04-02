"""Download a small Bench2Drive sample: one normal + one adverse-weather scenario.

Downloads the two smallest tar.gz archives (~330 MB total) and extracts only
the first MAX_FRAMES camera frames to keep disk usage low.

Usage:
    source .venv/bin/activate
    python scripts/download_b2d_sample.py
"""

import io
import os
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

# ── Config ────────────────────────────────────────────────────────────
REPO_ID = "rethinklab/Bench2Drive"
REPO_TYPE = "dataset"
OUT_DIR = Path("data/b2d_sample")
MAX_FRAMES = 15  # number of timesteps per scenario to keep

# Weather codes (B2D convention):
#   0  ClearNoon      1  ClearSunset   2  CloudyNoon   3  CloudySunset
#   13 HardRainNight  14 MidRainyNight 15 WetNight
SCENARIOS = {
    "normal_w3": "AccidentTwoWays_Town12_Route1121_Weather3.tar.gz",
    "adverse_w13": "AccidentTwoWays_Town12_Route1105_Weather13.tar.gz",
}

CAM_DIRS = [
    "rgb_front", "rgb_front_left", "rgb_front_right",
    "rgb_back", "rgb_back_left", "rgb_back_right",
]


def download_and_extract(scenario_key: str, filename: str):
    out_path = OUT_DIR / scenario_key
    if out_path.exists() and any(out_path.rglob("*.jpg")):
        print(f"[skip] {scenario_key} already extracted.")
        return out_path

    size_mb = {"normal_w3": 182, "adverse_w13": 146}.get(scenario_key, "?")
    print(f"\n[download] {filename}  (~{size_mb} MB)")
    local_tar = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type=REPO_TYPE,
        local_dir=str(OUT_DIR / "_cache"),
    )

    print(f"[extract]  → {out_path}  (first {MAX_FRAMES} frames only)")
    out_path.mkdir(parents=True, exist_ok=True)

    # Extract selectively: keep annotations + first MAX_FRAMES of each camera dir
    frame_counts: dict[str, int] = {}  # cam_dir → count
    with tarfile.open(local_tar, "r:gz") as tf:
        for member in tf.getmembers():
            name = member.name
            # Always keep: non-image files (annotations, metadata)
            if not any(f"/{cd}/" in name for cd in CAM_DIRS):
                tf.extract(member, path=str(out_path))
                continue
            # For camera dirs, keep only first MAX_FRAMES images
            cam_dir = next(
                (cd for cd in CAM_DIRS if f"/{cd}/" in name), None
            )
            if cam_dir is None:
                continue
            count = frame_counts.get(cam_dir, 0)
            if count < MAX_FRAMES:
                tf.extract(member, path=str(out_path))
                frame_counts[cam_dir] = count + 1

    # Cleanup cache tar
    os.remove(local_tar)
    print(f"[done]     {sum(frame_counts.values())} images extracted: {frame_counts}")
    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, fname in SCENARIOS.items():
        paths[key] = download_and_extract(key, fname)

    print("\n=== Download complete ===")
    for key, p in paths.items():
        imgs = list(p.rglob("*.jpg"))
        print(f"  {key}: {len(imgs)} images in {p}")


if __name__ == "__main__":
    main()
