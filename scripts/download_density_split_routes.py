"""Download only the Bench2Drive routes referenced by a Density UQ split."""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--density-checkpoint", default="checkpoints/density_uq/best.pt"
    )
    parser.add_argument(
        "--splits",
        default="train,calibration,test",
        help="comma-separated Density UQ splits to download",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--routes",
        default="",
        help="optional comma-separated route names; bypasses checkpoint loading",
    )
    parser.add_argument(
        "--endpoint",
        default="https://hf-mirror.com/datasets/rethinklab/Bench2Drive/resolve/main",
    )
    return parser.parse_args()


def download_route(route: str, out: Path, endpoint: str) -> tuple[str, int]:
    final_path = out / f"{route}.tar.gz"
    partial_path = out / f".{route}.tar.gz.partial"
    if final_path.is_file() and final_path.stat().st_size > 0:
        return route, final_path.stat().st_size
    url = f"{endpoint.rstrip('/')}/{quote(route)}.tar.gz"
    subprocess.run(
        [
            "curl",
            "-4",
            "--fail",
            "--location",
            "--retry",
            "20",
            "--retry-delay",
            "5",
            "--continue-at",
            "-",
            "--output",
            str(partial_path),
            url,
        ],
        check=True,
    )
    os.replace(partial_path, final_path)
    return route, final_path.stat().st_size


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    requested_splits = {
        item.strip() for item in args.splits.split(",") if item.strip()
    }
    explicit_routes = [
        item.strip() for item in args.routes.split(",") if item.strip()
    ]
    if explicit_routes:
        routes = sorted(set(explicit_routes))
        selection_label = "explicit route list"
    else:
        import torch

        payload = torch.load(
            args.density_checkpoint, map_location="cpu", weights_only=True
        )
        assignment = payload["split_assignment"]
        unknown = requested_splits.difference(assignment.values())
        if unknown:
            raise ValueError(f"Unknown Density UQ splits: {sorted(unknown)}")
        routes = sorted(
            route
            for route, split in assignment.items()
            if split in requested_splits
        )
        selection_label = str(sorted(requested_splits))
    if not routes:
        raise RuntimeError("No routes matched the requested splits")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading {len(routes)} routes for {selection_label} "
        f"to {out}",
        flush=True,
    )
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_route, route, out, args.endpoint): route
            for route in routes
        }
        for future in as_completed(futures):
            route = futures[future]
            try:
                _, size = future.result()
                print(f"[OK] {route}: {size} bytes", flush=True)
            except Exception as error:
                failures.append((route, str(error)))
                print(f"[FAIL] {route}: {error}", flush=True)
    if failures:
        raise RuntimeError(f"Failed downloads: {failures}")


if __name__ == "__main__":
    main()
