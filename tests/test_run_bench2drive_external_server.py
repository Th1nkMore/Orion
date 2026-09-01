from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_external_runner_loads_map_before_enabling_sync_mode():
    source = (ROOT / "scripts/run_bench2drive_external_server.py").read_text(
        encoding="utf-8"
    )
    setup = source[
        source.index("def setup_external_simulation") : source.index(
            "\n\ndef load_external_world_then_enable_sync"
        )
    ]
    loader = source[
        source.index("def load_external_world_then_enable_sync") : source.index(
            "\n\nleaderboard_evaluator.LeaderboardEvaluator._setup_simulation"
        )
    ]

    assert "get_world().apply_settings" not in setup
    assert "set_synchronous_mode(False)" in setup
    load_position = loader.index("load_world(town, reset_settings=True)")
    settings_position = loader.index("settings.synchronous_mode = True")
    tm_position = loader.index("self.traffic_manager.set_synchronous_mode(True)")
    tick_position = loader.index("self.world.tick()")
    assert load_position < settings_position < tm_position < tick_position
