#!/usr/bin/env python3
"""Derive an audited low-sun route while changing weather nodes only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


SCHEMA = "orion.native_glare_confirmation_route_derivation.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _structure_without_weather(root) -> bytes:
    clone = copy.deepcopy(root)
    for route in clone.findall("route"):
        for weathers in list(route.findall("weathers")):
            route.remove(weathers)
    return ET.tostring(clone, encoding="utf-8")


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def prepare_route(source: Path, protocol: Path, output: Path, manifest: Path) -> dict:
    for candidate in (source, protocol):
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
    for target in (output, manifest):
        if target.exists():
            raise FileExistsError("refusing to overwrite %s" % target)
    spec = json.loads(protocol.read_text(encoding="utf-8"))
    route_spec = spec["route"]
    if _sha256(source) != route_spec["source_sha256"]:
        raise RuntimeError("source route hash differs from frozen protocol")
    tree = ET.parse(source)
    root = tree.getroot()
    routes = root.findall("route")
    if len(routes) != 1:
        raise RuntimeError("confirmation derivative requires exactly one route")
    route = routes[0]
    scenarios = route.findall("./scenarios/scenario")
    if route.attrib.get("town") != route_spec["town"]:
        raise RuntimeError("route town differs from frozen protocol")
    if len(scenarios) != 1 or scenarios[0].attrib.get("type") != route_spec["scenario_type"]:
        raise RuntimeError("route scenario differs from frozen protocol")
    weather_nodes = route.findall("./weathers/weather")
    percentages = [node.attrib.get("route_percentage") for node in weather_nodes]
    if percentages != ["0", "100"]:
        raise RuntimeError("expected weather endpoints at 0 and 100 percent")
    structure_before = _digest_bytes(_structure_without_weather(root))
    weather = spec["weather"]
    for node in weather_nodes:
        percentage = node.attrib["route_percentage"]
        node.attrib.clear()
        for key, value in weather.items():
            node.set(key, str(float(value)))
        node.set("route_percentage", percentage)
    structure_after = _digest_bytes(_structure_without_weather(root))
    if structure_after != structure_before:
        raise RuntimeError("route structure changed outside weather nodes")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
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
        "route_index": int(route_spec["route_index"]),
        "town": route.attrib["town"],
        "scenario_type": scenarios[0].attrib["type"],
        "waypoint_count": len(route.findall("./waypoints/position")),
        "weather": weather,
        "structure_without_weather_sha256_before": structure_before,
        "structure_without_weather_sha256_after": structure_after,
        "only_weather_changed": structure_before == structure_after,
        "claim_boundary": "renderer confirmation input only; no model or safety claim",
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
    print(json.dumps(prepare_route(
        args.source_route.resolve(),
        args.protocol.resolve(),
        args.output_route.resolve(),
        args.manifest.resolve(),
    ), sort_keys=True))


if __name__ == "__main__":
    main()
