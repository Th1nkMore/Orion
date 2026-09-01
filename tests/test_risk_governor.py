import json

import pytest

from uq_estimator.risk_governor import UQRiskGovernor, load_score_trace


def test_off_mode_preserves_controls():
    governor = UQRiskGovernor(mode="off")
    throttle, brake, decision = governor.apply(
        throttle=0.7,
        brake=0.0,
        speed=4.0,
        raw_score=0.9,
        step=0,
    )
    assert throttle == pytest.approx(0.7)
    assert brake == pytest.approx(0.0)
    assert decision.intensity == 0.0


def test_aligned_learned_mode_caps_speed_without_commanding_a_stop():
    governor = UQRiskGovernor(mode="aligned_learned")
    throttle, brake, decision = governor.apply(
        throttle=0.7,
        brake=0.0,
        speed=3.0,
        raw_score=1.0,
        step=0,
    )
    assert decision.speed_cap == pytest.approx(1.5)
    assert throttle == 0.0
    assert 0.0 < brake <= 0.5


def test_trace_mode_cycles_a_matched_score_sequence():
    governor = UQRiskGovernor(mode="trace", trace_scores=[0.2, 0.8])
    scores = [governor.resolve_score(None, step) for step in range(4)]
    assert scores == [0.2, 0.8, 0.2, 0.8]


def test_aligned_learned_mode_requires_a_raw_score():
    governor = UQRiskGovernor(mode="aligned_learned")
    with pytest.raises(RuntimeError):
        governor.apply(
            throttle=0.1,
            brake=0.0,
            speed=1.0,
            raw_score=None,
            step=0,
        )


@pytest.mark.parametrize("score", [0.0, 0.2, 0.4])
def test_subthreshold_learned_score_is_strict_noop_even_above_max_speed(score):
    governor = UQRiskGovernor(
        mode="aligned_learned",
        threshold=0.4,
        saturation=0.8,
        max_speed=5.0,
    )
    throttle, brake, decision = governor.apply(
        throttle=0.73,
        brake=0.12,
        speed=8.0,
        raw_score=score,
        step=0,
    )
    assert decision.intensity == 0.0
    assert throttle == pytest.approx(0.73)
    assert brake == pytest.approx(0.12)


def test_oracle_uses_known_corruption_state_with_the_same_bounds():
    governor = UQRiskGovernor(mode="oracle")
    clear_throttle, clear_brake, clear = governor.apply(
        throttle=0.7,
        brake=0.0,
        speed=3.0,
        raw_score=0.95,
        step=0,
        oracle_active=False,
    )
    corrupt_throttle, corrupt_brake, corrupt = governor.apply(
        throttle=0.7,
        brake=0.0,
        speed=3.0,
        raw_score=0.05,
        step=1,
        oracle_active=True,
    )
    assert clear.applied_score == 0.0
    assert clear_throttle == pytest.approx(0.7)
    assert clear_brake == 0.0
    assert corrupt.applied_score == 1.0
    assert corrupt.speed_cap == pytest.approx(1.5)
    assert corrupt_throttle == 0.0
    assert 0.0 < corrupt_brake <= 0.5


def test_inactive_oracle_is_strict_noop_even_above_max_speed():
    governor = UQRiskGovernor(mode="oracle", max_speed=5.0)
    throttle, brake, decision = governor.apply(
        throttle=0.81,
        brake=0.07,
        speed=8.0,
        raw_score=1.0,
        step=0,
        oracle_active=False,
    )
    assert decision.applied_score == 0.0
    assert decision.intensity == 0.0
    assert throttle == pytest.approx(0.81)
    assert brake == pytest.approx(0.07)


def test_oracle_requires_known_corruption_state():
    governor = UQRiskGovernor(mode="oracle")
    with pytest.raises(RuntimeError):
        governor.apply(
            throttle=0.1,
            brake=0.0,
            speed=1.0,
            raw_score=0.5,
            step=0,
        )


def test_oracle_can_command_controlled_stop_for_complete_dropout():
    governor = UQRiskGovernor(mode="oracle", min_speed=0.0)
    throttle, brake, decision = governor.apply(
        throttle=0.7,
        brake=0.0,
        speed=3.0,
        raw_score=0.1,
        step=0,
        oracle_active=True,
    )
    assert decision.speed_cap == 0.0
    assert throttle == 0.0
    assert brake == pytest.approx(0.5)


def test_load_score_trace_accepts_wrapped_payload(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"scores": [0.1, 0.9]}))
    assert load_score_trace(path) == [0.1, 0.9]
