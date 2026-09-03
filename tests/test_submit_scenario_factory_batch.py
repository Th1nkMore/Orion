import json
import os
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "submit_scenario_factory_batch.sh"


def test_dry_run_exports_absolute_route_directory(tmp_path):
    batch_dir = tmp_path / "relative" / "batch"
    batch_dir.mkdir(parents=True)
    batch = batch_dir / "batch_manifest.json"
    batch.write_text(
        json.dumps(
            {
                "schema": "orion.scenario_factory.batch.v1",
                "status": "prepared_no_jobs_submitted",
                "run_id": "coverage_absolute_path_test",
                "split": "train_coverage_repair",
                "runtime_contract": {
                    "condition": "clean_off",
                    "variant": "hazard",
                    "stage2_spatial_uq_source": "disabled",
                    "stage1_adapter_control_influence": False,
                    "legacy_density_uq": False,
                    "risk_mode": "off",
                    "planning_response": "off",
                    "carla_quality": "Epic",
                },
                "routes": [
                    {
                        "route_index": 167,
                        "town": "Town03",
                        "scenario_type": "YieldToEmergencyVehicle",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    asset_root = tmp_path / "assets"
    environment = dict(os.environ)
    environment.update(
        {
            "PROJECT_ROOT": str(tmp_path / "project"),
            "ASSET_ROOT": str(asset_root),
            "SCENARIO_FACTORY_PYTHON": "python3",
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), str(batch.relative_to(tmp_path)), "--dry-run"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = "PILOT_ROUTE_DIR=%s" % batch_dir.resolve()
    assert expected in result.stdout
    assert "SCENARIO_FACTORY_SPLIT=train_coverage_repair" in result.stdout
