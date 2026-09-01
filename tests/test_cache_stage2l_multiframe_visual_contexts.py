import json

import team_code.orion_b2d_agent as orion_agent_module
from scripts.cache_closedloop_orion_visual_context import (
    DET_VISUAL_TOKENS,
    MAP_VISUAL_TOKENS,
    TOTAL_VISUAL_TOKENS,
    _make_offline_orion_agent,
    _setup_offline_orion_agent,
)
from scripts.cache_stage2l_multiframe_visual_contexts import _factory_inputs
from scripts.scenario_factory_lib import sha256_file
from team_code.orion_b2d_agent import OrionAgent


def test_offline_agent_factory_bypasses_online_constructor(monkeypatch):
    def fail_if_online_constructor_runs(*_args, **_kwargs):
        raise AssertionError("offline cache must not resolve a CARLA hero actor")

    monkeypatch.setattr(OrionAgent, "__init__", fail_if_online_constructor_runs)
    agent = _make_offline_orion_agent()
    assert isinstance(agent, OrionAgent)


def test_offline_setup_temporarily_supplies_only_vehicle_control(monkeypatch):
    agent = _make_offline_orion_agent()
    carla_module = orion_agent_module.carla
    monkeypatch.delattr(carla_module, "VehicleControl", raising=False)

    def fake_setup(config):
        control = carla_module.VehicleControl()
        assert (control.steer, control.throttle, control.brake) == (0.0, 0.0, 0.0)
        assert config == "offline-config"

    monkeypatch.setattr(agent, "setup", fake_setup)
    _setup_offline_orion_agent(agent, "offline-config")
    assert not hasattr(carla_module, "VehicleControl")


def test_native_orion_visual_token_contract_includes_temporal_memory():
    assert DET_VISUAL_TOKENS == 256 + 16 + 1
    assert MAP_VISUAL_TOKENS == 256
    assert TOTAL_VISUAL_TOKENS == 529


def test_factory_inputs_resolve_three_fixed_observed_keyframes(tmp_path):
    frame_reports = []
    for frame in (10, 12, 14):
        scenario = tmp_path / "scenario"
        image = scenario / "rgb_front_model_input" / ("%04d.png" % frame)
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"png")
        meta = scenario / "meta" / ("%04d.json" % frame)
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text("{}")
        bundle = tmp_path / ("bundle_%04d.json" % frame)
        bundle.write_text(json.dumps({
            "schema": "orion.uq_relevance_frame_bundle.v1",
            "counterfactual": {
                "variant": "observed",
                "group_id": "route1_step20_saved_%04d" % frame,
            },
            "model_input": {"observation": {"camera_files": [
                {"view": "CAM_FRONT", "path": str(image)}
            ]}},
            "provenance": {"selected_saved_frame_index": frame},
        }))
        batch = tmp_path / ("batch_%04d.json" % frame)
        batch.write_text(json.dumps({
            "schema": "orion.uq_relevance_frame_bundle_batch.v1",
            "bundles": [{
                "variant": "observed",
                "path": str(bundle),
                "sha256": sha256_file(bundle),
            }],
        }))
        frame_reports.append({
            "selected_saved_frame_index": frame,
            "frame_bundle_batch": {
                "path": str(batch),
                "sha256": sha256_file(batch),
            },
        })
    report = tmp_path / "factory.json"
    report.write_text(json.dumps({
        "schema": "orion.uq_relevance_multiframe_event_factory.v1",
        "status": "pending_multiframe_human_geometry_review",
        "keyframe_count": 3,
        "frame_reports": frame_reports,
    }))
    rows = _factory_inputs(report)
    assert len(rows) == 3
    assert sorted(row["frame"] for row in rows.values()) == [10, 12, 14]
