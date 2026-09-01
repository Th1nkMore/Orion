"""Transparent UQ-conditioned speed governor for closed-loop causal pilots."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def load_score_trace(path: str | Path) -> list[float]:
    """Load a score trace stored as a JSON list or ``{"scores": [...]}``."""
    payload = json.loads(Path(path).read_text())
    scores = payload.get("scores") if isinstance(payload, dict) else payload
    if not isinstance(scores, list) or not scores:
        raise ValueError("score trace must be a non-empty JSON list")
    values = [float(score) for score in scores]
    if any(score < 0.0 or score > 1.0 for score in values):
        raise ValueError("score trace values must lie in [0, 1]")
    return values


@dataclass(frozen=True)
class RiskDecision:
    mode: str
    raw_score: float | None
    applied_score: float | None
    intensity: float
    speed_cap: float
    base_throttle: float
    base_brake: float
    throttle: float
    brake: float

    def to_dict(self) -> dict:
        return asdict(self)


class UQRiskGovernor:
    """Convert UQ into a bounded speed cap without changing steering.

    The governor is intentionally small and inspectable. ``min_speed`` may be
    zero when a complete sensor outage warrants a controlled stop. Constant
    and trace modes provide intervention-budget and temporal-alignment controls.
    """

    MODES = {"off", "aligned_learned", "oracle", "constant", "trace"}

    def __init__(
        self,
        mode: str = "off",
        *,
        threshold: float = 0.4,
        saturation: float = 0.8,
        min_speed: float = 1.5,
        max_speed: float = 5.0,
        slowdown_margin: float = 1.0,
        brake_gain: float = 0.5,
        max_brake: float = 0.5,
        constant_score: float = 0.6209,
        trace_scores: Iterable[float] | None = None,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        if not 0.0 <= threshold < saturation <= 1.0:
            raise ValueError("require 0 <= threshold < saturation <= 1")
        if not 0.0 <= min_speed < max_speed:
            raise ValueError("require 0 <= min_speed < max_speed")
        if slowdown_margin <= 0.0:
            raise ValueError("slowdown_margin must be positive")
        if brake_gain < 0.0 or not 0.0 <= max_brake <= 1.0:
            raise ValueError("invalid brake parameters")
        if not 0.0 <= constant_score <= 1.0:
            raise ValueError("constant_score must lie in [0, 1]")

        self.mode = mode
        self.threshold = float(threshold)
        self.saturation = float(saturation)
        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        self.slowdown_margin = float(slowdown_margin)
        self.brake_gain = float(brake_gain)
        self.max_brake = float(max_brake)
        self.constant_score = float(constant_score)
        self.trace_scores = (
            [float(score) for score in trace_scores]
            if trace_scores is not None else []
        )
        if self.mode == "trace" and not self.trace_scores:
            raise ValueError("trace mode requires trace_scores")
        if any(score < 0.0 or score > 1.0 for score in self.trace_scores):
            raise ValueError("trace scores must lie in [0, 1]")

    def resolve_score(
        self,
        raw_score: float | None,
        step: int,
        *,
        oracle_active: bool | None = None,
    ) -> float | None:
        if self.mode == "off":
            return None
        if self.mode == "aligned_learned":
            if raw_score is None:
                raise RuntimeError(
                    "aligned_learned risk mode requires an observation-UQ score"
                )
            return _clip(raw_score, 0.0, 1.0)
        if self.mode == "oracle":
            if oracle_active is None:
                raise RuntimeError(
                    "oracle risk mode requires the known corruption state"
                )
            return 1.0 if oracle_active else 0.0
        if self.mode == "constant":
            return self.constant_score
        return self.trace_scores[int(step) % len(self.trace_scores)]

    def apply(
        self,
        *,
        throttle: float,
        brake: float,
        speed: float,
        raw_score: float | None,
        step: int,
        oracle_active: bool | None = None,
    ) -> tuple[float, float, RiskDecision]:
        base_throttle = _clip(throttle, 0.0, 1.0)
        base_brake = _clip(brake, 0.0, 1.0)
        applied_score = self.resolve_score(
            raw_score, step, oracle_active=oracle_active
        )
        if applied_score is None:
            decision = RiskDecision(
                mode=self.mode,
                raw_score=raw_score,
                applied_score=None,
                intensity=0.0,
                speed_cap=self.max_speed,
                base_throttle=base_throttle,
                base_brake=base_brake,
                throttle=base_throttle,
                brake=base_brake,
            )
            return base_throttle, base_brake, decision

        intensity = _clip(
            (applied_score - self.threshold)
            / (self.saturation - self.threshold),
            0.0,
            1.0,
        )
        speed_cap = self.max_speed - intensity * (
            self.max_speed - self.min_speed
        )
        # A sub-threshold score is an explicit no-op, even when the current
        # vehicle speed is above ``max_speed``.  Without this branch the
        # nominal maximum cap changes the base ORION command before any UQ
        # trigger and confounds every off-versus-governed comparison.
        if intensity == 0.0:
            decision = RiskDecision(
                mode=self.mode,
                raw_score=raw_score,
                applied_score=applied_score,
                intensity=0.0,
                speed_cap=speed_cap,
                base_throttle=base_throttle,
                base_brake=base_brake,
                throttle=base_throttle,
                brake=base_brake,
            )
            return base_throttle, base_brake, decision

        governed_throttle = base_throttle
        governed_brake = base_brake
        headroom = speed_cap - float(speed)
        if headroom < self.slowdown_margin:
            throttle_scale = _clip(
                headroom / self.slowdown_margin, 0.0, 1.0
            )
            governed_throttle *= throttle_scale
        if headroom < 0.0:
            governed_throttle = 0.0
            governed_brake = max(
                governed_brake,
                min(
                    self.max_brake,
                    self.brake_gain * (-headroom) / self.slowdown_margin,
                ),
            )
        if governed_brake > governed_throttle:
            governed_throttle = 0.0

        decision = RiskDecision(
            mode=self.mode,
            raw_score=raw_score,
            applied_score=applied_score,
            intensity=intensity,
            speed_cap=speed_cap,
            base_throttle=base_throttle,
            base_brake=base_brake,
            throttle=_clip(governed_throttle, 0.0, 1.0),
            brake=_clip(governed_brake, 0.0, 1.0),
        )
        return decision.throttle, decision.brake, decision
