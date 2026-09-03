import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_scenario_review_queue.py"
SPEC = importlib.util.spec_from_file_location("scenario_review_queue", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_package(root: Path, route_index: int, valid: bool, town: str, family: str):
    root.mkdir(parents=True)
    package_path = root / "event_package.json"
    package = {
        "schema": "orion.scenario_event_package.v1",
        "split": "development_screen",
        "route": {"route_index": route_index},
        "runtime": {"valid": valid},
        "outcome_class": "VALID_NEAR_MISS_OR_CONFLICT" if valid else "INVALID_RUNTIME",
        "qa_input_ready": valid,
        "critical_event": (
            {"step": 20 + route_index, "actor": {"actor_id": route_index}}
            if valid
            else None
        ),
        "official_endpoint": {},
        "continuous_safety": {"safety": {"min_obb_ttc_seconds": 1.5}},
        "visualization": None,
    }
    package_path.write_text(json.dumps(package))
    return package_path, {
        "route_index": route_index,
        "town": town,
        "scenario_type": family,
        "screen_role": "dynamic_path_conflict",
    }


def test_queue_excludes_invalid_runtime_and_hash_binds_decisions(tmp_path):
    package1, row1 = _write_package(tmp_path / "route1", 1, True, "Town01", "A")
    _, row2 = _write_package(tmp_path / "route2", 2, False, "Town02", "B")
    row2.update(
        {
            "outcome_class": "INVALID_RUNTIME_UNPACKAGEABLE",
            "runtime_valid": False,
            "attempts": [{"job_id": 9, "package_buildable": False}],
        }
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "orion.scenario_factory.batch_screen_report.v1",
                "routes": [row1, row2],
                "event_packages": [
                    {
                        "route_index": 1,
                        "path": str(package1),
                        "sha256": MODULE.sha256_file(package1),
                    },
                ],
            }
        )
    )

    queue = MODULE.build_review_queue([report])

    assert queue["human_review_count"] == 1
    assert queue["automatically_excluded_count"] == 1
    assert queue["review_order"][0]["event_id"] == "route1_step21"
    assert queue["automatically_excluded"][0]["automatic_exclusion_reasons"] == [
        "runtime_invalid",
        "event_package_unbuildable",
    ]
    assert queue["automatically_excluded"][0]["runtime_attempts"][0]["job_id"] == 9


def test_diversity_order_prefers_new_family_before_outcome_priority():
    rows = [
        {
            "event_id": "e1",
            "route_index": 1,
            "town": "Town01",
            "scenario_family": "A",
            "screen_role": "role",
            "outcome_class": "VALID_COLLISION",
        },
        {
            "event_id": "e2",
            "route_index": 2,
            "town": "Town01",
            "scenario_family": "A",
            "screen_role": "role",
            "outcome_class": "VALID_COLLISION",
        },
        {
            "event_id": "e3",
            "route_index": 3,
            "town": "Town02",
            "scenario_family": "B",
            "screen_role": "other",
            "outcome_class": "VALID_NEAR_MISS_OR_CONFLICT",
        },
    ]

    ordered = MODULE._diversity_order(rows)

    assert [row["event_id"] for row in ordered] == ["e1", "e3", "e2"]


def test_review_warnings_surface_boundary_and_liveness_without_exclusion():
    package = {
        "critical_event": {"route_progress": 0.01},
        "continuous_safety": {
            "duration_seconds": 20.0,
            "efficiency": {"stopped_below_0_25_mps_seconds": 12.0},
        },
    }

    assert MODULE._review_warnings(package) == [
        "critical_event_near_route_start",
        "majority_of_route_nearly_stopped",
    ]
