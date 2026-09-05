"""Run Bench2Drive against an already-started CARLA server.

Bench2Drive 0.0.4 normally starts its own CARLA process inside
``LeaderboardEvaluator._setup_simulation``.  The cluster launcher already
starts CARLA and verifies RPC readiness, so this entry point replaces only
that setup method and leaves the official evaluator and route logic intact.
"""

from __future__ import annotations

import carla
import faulthandler
import os

from leaderboard import leaderboard_evaluator
from leaderboard.autoagents import agent_wrapper
from leaderboard.envs import sensor_interface
from leaderboard.scenarios import scenario_manager
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from uq_estimator.closedloop_sensor_diagnostics import (
    install_exact_frame_speedometer,
    install_oracle_depth_camera_support,
    install_sensor_queue_diagnostics,
)


def setup_external_simulation(self, args):
    client_timeout = float(args.timeout or self.client_timeout)
    client = carla.Client(args.host, args.port)
    client.set_timeout(client_timeout)
    print(
        f"external CARLA ready on {args.host}:{args.port} "
        f"rpc_timeout={client_timeout:.1f}s",
        flush=True,
    )

    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
    # Keep the bootstrap world and Traffic Manager asynchronous until the
    # requested map has loaded. Loading a new map while both are synchronous
    # can leave CARLA permanently spinning at frame zero with no actors.
    traffic_manager.set_synchronous_mode(False)
    traffic_manager.set_hybrid_physics_mode(True)
    print(
        f"external traffic manager ready on {args.traffic_manager_port}",
        flush=True,
    )
    return client, client_timeout, traffic_manager


def load_external_world_then_enable_sync(self, args, town):
    """Load the requested map asynchronously, then establish sync mode."""

    self.world = self.client.load_world(town, reset_settings=True)
    settings = self.world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / self.frame_rate
    settings.deterministic_ragdolls = True
    settings.spectator_as_ego = False
    settings.tile_stream_distance = 650
    settings.actor_active_distance = 650
    self.world.apply_settings(settings)

    self.world.reset_all_traffic_lights()
    CarlaDataProvider.set_client(self.client)
    CarlaDataProvider.set_traffic_manager_port(args.traffic_manager_port)
    CarlaDataProvider.set_world(self.world)
    self.traffic_manager.set_random_device_seed(args.traffic_manager_seed)
    self.traffic_manager.set_synchronous_mode(True)
    self.traffic_manager.set_hybrid_physics_mode(True)
    self.world.tick()

    map_name = CarlaDataProvider.get_map().name.split("/")[-1]
    if map_name != town:
        raise RuntimeError(
            "CARLA external server loaded {}, expected {}".format(map_name, town)
        )
    print(
        "external CARLA map ready after async load: {}".format(map_name),
        flush=True,
    )


leaderboard_evaluator.LeaderboardEvaluator._setup_simulation = (
    setup_external_simulation
)
leaderboard_evaluator.LeaderboardEvaluator._load_and_wait_for_world = (
    load_external_world_then_enable_sync
)
install_exact_frame_speedometer(sensor_interface)
install_sensor_queue_diagnostics(sensor_interface)
install_oracle_depth_camera_support(
    agent_wrapper,
    carla,
    sensor_icons=leaderboard_evaluator.sensors_to_icons,
)


def install_scenario_traceback_diagnostics():
    """Dump Python stacks only when one route tick exceeds the threshold."""

    interval = float(
        os.environ.get("ORION_SCENARIO_TRACEBACK_INTERVAL_SECONDS", "75")
    )
    if interval <= 0:
        return
    original_tick_scenario = scenario_manager.ScenarioManager._tick_scenario

    def tick_scenario_with_traceback(self):
        faulthandler.enable()
        faulthandler.dump_traceback_later(interval, repeat=False)
        try:
            return original_tick_scenario(self)
        finally:
            faulthandler.cancel_dump_traceback_later()

    scenario_manager.ScenarioManager._tick_scenario = tick_scenario_with_traceback
    print(
        "[ScenarioTracebackDiagnostic] per_tick_threshold={}s".format(interval),
        flush=True,
    )


install_scenario_traceback_diagnostics()


if __name__ == "__main__":
    leaderboard_evaluator.main()
