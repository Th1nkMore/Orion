#!/usr/bin/env python3
"""Build hazard/no-hazard route pairs from official Bench2Drive splits."""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench2drive-root", type=Path, required=True)
    parser.add_argument(
        "--route-indices", type=int, nargs="+", default=(203, 195, 146)
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def scenario_metadata(route: ET.Element) -> list[dict[str, str]]:
    return [dict(scenario.attrib) for scenario in route.findall(".//scenario")]


def remove_scenarios(root: ET.Element) -> int:
    removed = 0
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "scenario":
                parent.remove(child)
                removed += 1
    return removed


def main() -> None:
    args = parse_args()
    data_dir = args.bench2drive_root / "leaderboard" / "data"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(data_dir), "routes": []}

    for index in args.route_indices:
        source = data_dir / f"bench2drive220_{index}_orion_traj.xml"
        if not source.is_file():
            raise FileNotFoundError(source)
        tree = ET.parse(source)
        root = tree.getroot()
        route = root.find("route")
        if route is None:
            raise ValueError(f"No <route> in {source}")
        scenarios = scenario_metadata(route)
        if not scenarios:
            raise ValueError(f"Route {index} has no scenario to remove")

        hazard_path = args.out_dir / f"route_{index}_hazard.xml"
        nohazard_path = args.out_dir / f"route_{index}_nohazard.xml"
        shutil.copy2(source, hazard_path)
        nohazard_tree = ET.parse(source)
        removed = remove_scenarios(nohazard_tree.getroot())
        if removed != len(scenarios):
            raise RuntimeError(
                f"Expected to remove {len(scenarios)} scenarios, removed {removed}"
            )
        nohazard_tree.write(nohazard_path, encoding="utf-8", xml_declaration=True)
        manifest["routes"].append(
            {
                "route_index": index,
                "route_id": route.get("id"),
                "town": route.get("town"),
                "scenarios": scenarios,
                "hazard": str(hazard_path),
                "nohazard": str(nohazard_path),
            }
        )

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
