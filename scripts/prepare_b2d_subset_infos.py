"""Prepare Bench2Drive info files from only the routes present on disk.

The upstream converter assumes the complete Base dataset and its fixed
train/validation split. This wrapper reuses the upstream conversion functions
while limiting work to a downloaded route subset, such as the 50-route Density
UQ set used by this project.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zoo-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split-name", default="val")
    parser.add_argument("--tmp-dir", default="tmp_density_subset")
    parser.add_argument("--skip-routes", action="store_true")
    parser.add_argument("--skip-maps", action="store_true")
    return parser.parse_args()


def load_upstream_converter(zoo_root: Path):
    dataset_dir = zoo_root / "mmcv" / "datasets"
    converter_path = dataset_dir / "prepare_B2D.py"
    if not converter_path.is_file():
        raise FileNotFoundError(f"Missing upstream converter: {converter_path}")
    # Upstream vis_utils imports open3d for interactive visualization, but the
    # data conversion functions used here never reference it. Avoid pulling a
    # large GUI/visualization wheel into the headless preprocessing runtime.
    try:
        import open3d  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["open3d"] = types.ModuleType("open3d")
    sys.path.insert(0, str(dataset_dir))
    spec = importlib.util.spec_from_file_location(
        "bench2drive_subset_converter", converter_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load converter: {converter_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.skip_routes and args.skip_maps:
        raise ValueError("--skip-routes and --skip-maps cannot both be set")

    route_root = args.data_root / "v1"
    map_root = args.data_root / "maps"
    routes = sorted(
        f"v1/{path.name}"
        for path in route_root.iterdir()
        if path.is_dir()
        and "Town" in path.name
        and "Route" in path.name
        and "Weather" in path.name
    )
    if not routes:
        raise RuntimeError(f"No Bench2Drive routes found under {route_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    converter = load_upstream_converter(args.zoo_root)
    converter.DATAROOT = str(args.data_root)
    converter.MAP_ROOT = str(map_root)
    converter.OUT_DIR = str(args.output_dir)
    converter.process_list = []

    if not args.skip_routes:
        print(
            f"Preparing {len(routes)} routes as split {args.split_name!r} "
            f"with {args.workers} workers",
            flush=True,
        )
        converter.generate_infos(
            routes, args.workers, args.split_name, args.tmp_dir
        )
    if not args.skip_maps:
        print(f"Preparing map infos from {map_root}", flush=True)
        converter.gengrate_map(str(map_root))
    print("B2D_SUBSET_INFO_PREP_OK", flush=True)


if __name__ == "__main__":
    main()
