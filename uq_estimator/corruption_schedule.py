"""Helpers for route-aligned closed-loop corruption schedules."""

from __future__ import annotations

import math

import numpy as np


class RouteTriggeredTimedWindow:
    """One-shot window triggered by route progress and ended by simulation time.

    Unlike a progress-to-progress window, this schedule always releases after a
    fixed exposure duration even when conservative control stops the vehicle.
    Once released it cannot retrigger if route-progress estimates later jitter.
    """

    def __init__(self, start_progress, duration_seconds):
        self.start_progress = float(start_progress)
        self.duration_seconds = float(duration_seconds)
        if not 0.0 <= self.start_progress <= 1.0:
            raise ValueError("start_progress must be in [0, 1]")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0.0:
            raise ValueError("duration_seconds must be finite and positive")
        self.trigger_time_seconds = None

    def is_active(self, route_progress, sim_time_seconds):
        route_progress = float(route_progress)
        sim_time_seconds = float(sim_time_seconds)
        if self.trigger_time_seconds is None and route_progress >= self.start_progress:
            self.trigger_time_seconds = sim_time_seconds
        if self.trigger_time_seconds is None:
            return False
        elapsed = sim_time_seconds - self.trigger_time_seconds
        return 0.0 <= elapsed < self.duration_seconds

    def elapsed_seconds(self, sim_time_seconds):
        if self.trigger_time_seconds is None:
            return None
        return float(sim_time_seconds) - self.trigger_time_seconds


def project_route_progress(position, route_points):
    """Project a 2-D position onto a route polyline and return [0, 1] progress.

    The closest projected point is used, so small lateral deviations do not
    change the event window.  This keeps corruption exposure aligned by route
    position even when a risk governor changes the vehicle's speed.
    """
    point = np.asarray(position, dtype=np.float64).reshape(-1)[:2]
    route = np.asarray(route_points, dtype=np.float64)
    if route.ndim != 2 or route.shape[1] < 2 or len(route) < 2:
        raise ValueError("route_points must contain at least two 2-D points")
    route = route[:, :2]
    segments = route[1:] - route[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    total_length = float(lengths.sum())
    if total_length <= 0.0:
        raise ValueError("route polyline must have positive length")

    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    best_distance_sq = float("inf")
    best_distance_along = 0.0
    for index, (start, segment, length) in enumerate(
        zip(route[:-1], segments, lengths)
    ):
        if length <= 0.0:
            continue
        fraction = float(np.dot(point - start, segment) / (length * length))
        fraction = float(np.clip(fraction, 0.0, 1.0))
        projected = start + fraction * segment
        distance_sq = float(np.dot(point - projected, point - projected))
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_distance_along = cumulative[index] + fraction * length

    return float(np.clip(best_distance_along / total_length, 0.0, 1.0))
