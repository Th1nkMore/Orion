#!/usr/bin/env python3
"""Create an audited low-sun Route151 derivative for native-glare capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


SCHEMA = "orion.native_glare_route_derivation.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_route(source: Path, protocol: Path, output: Path, manifest: Path) -> dict:
    for candidate in (source, protocol):
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
    for target in (output, manifest):
        if target.exists():
            raise FileExistsError("refusing to overwrite %s" % target)

    spec = json.loads(protocol.read_text(encoding="utf-8"))
    weather = spec["methods"]["carla_native_low_sun"][
        "weather_shared_by_all_profiles"
    ]
    tree = ET.parse(source)
    root = tree.getroot()
    routes = root.findall("route")
    if len(routes) != 1:
        raise RuntimeError("native glare derivative requires exactly one route")
    route = routes[0]
    scenarios = route.findall("./scenarios/scenario")
    if route.attrib.get("town") != "Town02":
        raise RuntimeError("expected Route151 in Town02")
    if len(scenarios) != 1 or scenarios[0].attrib.get("type") != "ParkingCrossingPedestrian":
        raise RuntimeError("Route151 scenario contract changed")
    waypoints = route.findall("./waypoints/position")
    if len(waypoints) < 50:
        raise RuntimeError("Route151 waypoint contract changed")
    weather_nodes = route.findall("./weathers/weather")
    if [node.attrib.get("route_percentage") for node in weather_nodes] != ["0", "100"]:
        raise RuntimeError("Route151 weather endpoints changed")

    for node in weather_nodes:
        percentage = node.attrib["route_percentage"]
        node.attrib.clear()
        for key, value in weather.items():
            node.set(key, str(float(value)))
        node.set("route_percentage", percentage)

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    # ``ElementTree.indent`` was added in Python 3.9, while the frozen CARLA
    # environment is Python 3.7.  Formatting is optional; route semantics and
    # the derived artifact hash are not.
    if hasattr(ET, "indent"):
        ET.indent(tree, space="   ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    payload = {
        "schema": SCHEMA,
        "source_route": str(source.resolve()),
        "source_sha256": _sha256(source),
        "protocol": str(protocol.resolve()),
        "protocol_sha256": _sha256(protocol),
        "derived_route": str(output.resolve()),
        "derived_sha256": _sha256(output),
        "town": route.attrib["town"],
        "route_id": route.attrib.get("id"),
        "scenario_name": scenarios[0].attrib.get("name"),
        "scenario_type": scenarios[0].attrib.get("type"),
        "waypoint_count": len(waypoints),
        "weather": weather,
        "claim_boundary": "visual capture input only; no safety or model claim",
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-route", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-route", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = prepare_route(
        args.source_route.resolve(),
        args.protocol.resolve(),
        args.output_route.resolve(),
        args.manifest.resolve(),
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
