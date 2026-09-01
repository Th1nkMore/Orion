import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "team_code/glare_capture_agent.py"


def _literal(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError("missing literal: %s" % name)


def test_native_profiles_are_monotonic_and_clean_is_disabled():
    profiles = _literal("PROFILES")
    assert tuple(profiles) == ("clean", "light", "medium", "heavy")
    assert profiles["clean"] == {
        "lens_flare_intensity": 0.0,
        "bloom_intensity": 0.0,
    }
    for field in ("lens_flare_intensity", "bloom_intensity"):
        values = [profiles[name][field] for name in profiles]
        assert values == sorted(values)
        assert len(set(values)) == len(values)


def test_native_weather_is_low_sun_and_profile_invariant():
    weather = _literal("WEATHER_DEFAULTS")
    assert weather["sun_altitude_angle"] == 8.0
    assert weather["mie_scattering_scale"] >= 0.0
    source = SOURCE.read_text(encoding="utf-8")
    assert "self.weather_values = native_glare_weather()" in source
    assert "CarlaDataProvider.get_world().set_weather(weather)" in source


def test_capture_is_orion_free_and_pose_auditable():
    source = SOURCE.read_text(encoding="utf-8")
    assert "import orion" not in source.lower()
    assert "_load_orion" not in source
    assert '"orion_loaded": False' in source
    for field in (
        '"ego_location"',
        '"ego_rotation"',
        '"route_progress"',
        '"nearby_actors"',
    ):
        assert field in source


def test_controller_plan_extends_past_official_finish():
    source = SOURCE.read_text(encoding="utf-8")
    assert "GLARE_CAPTURE_FINISH_EXTENSION_M" in source
    assert "_extended_finish_xy(route_xy, self.finish_extension_m)" in source
    assert "plan.extend(self._agent.trace_route(previous, extended_waypoint))" in source


def test_front_receives_glare_and_bev_disables_postprocess():
    source = SOURCE.read_text(encoding="utf-8")
    assert 'sensor_id == "CAM_FRONT"' in source
    assert 'sensor_id == "bev"' in source
    assert '"enable_postprocess_effects": "false"' in source
    assert '"lens_flare_intensity"' in source
    assert '"bloom_intensity"' in source


def test_generic_capture_supports_native_motion_blur_without_orion():
    source = SOURCE.read_text(encoding="utf-8")
    assert 'self.capture_family == "native_motion_blur"' in source
    assert "install_orion_native_motion_blur_sensor_patch(" in source
    assert "readback_native_motion_blur_condition(" in source
    assert '"orion.native_motion_blur_capture_trace.v1"' in source
    assert '"HARDCAPTURE_START_PROGRESS"' in source
    assert '"HARDCAPTURE_END_PROGRESS"' in source
    assert "self.step % self.capture_stride == 0" in source
