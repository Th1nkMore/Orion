import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_native_glare_route import prepare_route


def _source_xml(path):
    points = "".join(
        '<position x="%s" y="0" z="0" />' % index for index in range(50)
    )
    path.write_text(
        '<routes><route id="24294" town="Town02"><waypoints>%s</waypoints>'
        '<scenarios><scenario name="ParkingCrossingPedestrian_1" '
        'type="ParkingCrossingPedestrian" /></scenarios><weathers>'
        '<weather route_percentage="0" sun_altitude_angle="15" />'
        '<weather route_percentage="100" sun_altitude_angle="15" />'
        '</weathers></route></routes>' % points
    )


def test_prepare_route_changes_only_weather_contract(tmp_path):
    source = tmp_path / "source.xml"
    protocol = tmp_path / "protocol.json"
    output = tmp_path / "derived.xml"
    manifest = tmp_path / "manifest.json"
    _source_xml(source)
    protocol.write_text(json.dumps({
        "methods": {"carla_native_low_sun": {"weather_shared_by_all_profiles": {
            "sun_altitude_angle": 8.0,
            "sun_azimuth_angle": 180.0,
            "fog_density": 10.0,
        }}}
    }))

    payload = prepare_route(source, protocol, output, manifest)

    assert payload["waypoint_count"] == 50
    assert payload["scenario_type"] == "ParkingCrossingPedestrian"
    tree = ET.parse(output)
    nodes = tree.findall("./route/weathers/weather")
    assert [node.attrib["route_percentage"] for node in nodes] == ["0", "100"]
    assert all(node.attrib["sun_altitude_angle"] == "8.0" for node in nodes)
    assert all(node.attrib["sun_azimuth_angle"] == "180.0" for node in nodes)
    assert json.loads(manifest.read_text())["derived_sha256"] == payload["derived_sha256"]


def test_prepare_route_refuses_overwrite(tmp_path):
    source = tmp_path / "source.xml"
    protocol = tmp_path / "protocol.json"
    output = tmp_path / "derived.xml"
    manifest = tmp_path / "manifest.json"
    _source_xml(source)
    protocol.write_text(json.dumps({
        "methods": {"carla_native_low_sun": {"weather_shared_by_all_profiles": {}}}
    }))
    output.write_text("occupied")
    try:
        prepare_route(source, protocol, output, manifest)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected fail-closed output policy")
