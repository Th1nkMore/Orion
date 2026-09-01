import json

from uq_estimator.counterfactual_evidence_extraction import (
    deterministic_condition_view,
    deterministic_window_balanced_view,
    load_counterfactual_protocol,
    projected_feature_counts,
    split_interventions,
)


def test_frozen_protocol_and_split_family_isolation(tmp_path):
    source = "configs/observation_uq_counterfactual_evidence_v1.json"
    payload = json.load(open(source))
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload))
    protocol = load_counterfactual_protocol(path)
    assert split_interventions("train", protocol) == (
        ("local_blur", 1),
        ("local_blur", 3),
        ("local_dark", 1),
        ("local_dark", 3),
    )
    assert ("local_glare", 1) not in split_interventions("train", protocol)
    assert ("local_glare", 1) in split_interventions("validation", protocol)
    assert split_interventions("held_out", protocol) == (
        ("local_glare", 1),
        ("local_glare", 3),
    )


def test_v2_protocol_freezes_window_schedule_without_changing_family_split(tmp_path):
    source = "configs/observation_uq_counterfactual_evidence_v2.json"
    payload = json.load(open(source))
    path = tmp_path / "protocol_v2.json"
    path.write_text(json.dumps(payload))
    protocol = load_counterfactual_protocol(path)
    assert protocol["intervention_split"]["view_schedule"] == {
        "version": "route_condition_window_cycle/v2",
        "window_frames": 4,
        "offset": "sha256(seed, route_id, family, severity) modulo six",
        "view": "(offset + floor(frame_idx / 4)) modulo six",
        "rationale": "preserve short temporal sensor continuity while preventing one route/family/severity key from labeling one camera for all 16 frames",
    }
    assert split_interventions("train", protocol) == (
        ("local_blur", 1),
        ("local_blur", 3),
        ("local_dark", 1),
        ("local_dark", 3),
    )


def test_v3_protocol_retains_v2_window_schedule_contract(tmp_path):
    payload = json.load(open("configs/observation_uq_counterfactual_evidence_v2.json"))
    payload["schema_version"] = "orion.observation-uq-counterfactual-evidence/v3"
    payload["intervention_split"]["heldout_family_development"] = (
        "local_glare on validation and held-out B2D routes; read only after train diagnostics"
    )
    path = tmp_path / "protocol_v3.json"
    path.write_text(json.dumps(payload))
    protocol = load_counterfactual_protocol(path)
    assert protocol["intervention_split"]["view_schedule"]["window_frames"] == 4
    assert split_interventions("held_out", protocol) == (
        ("local_glare", 1),
        ("local_glare", 3),
    )


def test_projected_counts_use_all_560_train_frames_without_family_leakage():
    counts = projected_feature_counts(
        {"train": 35, "validation": 5, "held_out": 5}, 16
    )
    assert counts == {
        "reference": 720,
        "observed": 2880,
        "total": 3600,
        "observed_train": 2240,
        "observed_validation": 480,
        "observed_held_out": 160,
    }


def test_projected_counts_double_for_frozen_100_route_expansion():
    counts = projected_feature_counts(
        {"train": 70, "validation": 10, "held_out": 10}, 16
    )
    assert counts == {
        "reference": 1440,
        "observed": 5760,
        "total": 7200,
        "observed_train": 4480,
        "observed_validation": 960,
        "observed_held_out": 320,
    }


def test_condition_view_is_replayable_and_covers_all_cameras():
    routes = ["route_%02d" % index for index in range(35)]
    views = {
        deterministic_condition_view(route, family, severity, 6, 20260826)
        for route in routes
        for family in ("local_blur", "local_dark")
        for severity in (1, 3)
    }
    assert views == set(range(6))
    value = deterministic_condition_view(
        routes[0], "local_blur", 1, 6, 20260826
    )
    assert value == deterministic_condition_view(
        routes[0], "local_blur", 1, 6, 20260826
    )


def test_window_balanced_view_breaks_fixed_route_condition_label():
    route = "route_00"
    values = [
        deterministic_window_balanced_view(
            route, frame, "local_blur", 1, 6, 20260827, window_frames=4
        )
        for frame in range(16)
    ]
    assert len(set(values)) == 4
    assert all(len(set(values[start : start + 4])) == 1 for start in range(0, 16, 4))
    assert values == [
        deterministic_window_balanced_view(
            route, frame, "local_blur", 1, 6, 20260827, window_frames=4
        )
        for frame in range(16)
    ]
