import json
import os
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_official_closedloop_smoke.sh"


def test_progress_watchdog_marks_stalled_trace_as_runtime_invalid(tmp_path):
    bench2drive_root = tmp_path / "Bench2Drive"
    bench2drive_root.mkdir()
    route = tmp_path / "route.xml"
    route.write_text("<routes />\n")
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p \"${SAVE_PATH}/fake-route\"\n"
        "printf '{}\\n' > \"${SAVE_PATH}/fake-route/control_trace.jsonl\"\n"
        "exec sleep 30\n"
    )
    fake_python.chmod(0o755)
    output_root = tmp_path / "output"
    env = os.environ.copy()
    env.update(
        {
            "BENCH2DRIVE_ROOT": str(bench2drive_root),
            "BENCH2DRIVE_ZOO_ROOT": str(tmp_path / "Bench2DriveZoo"),
            "CARLA_ROOT": str(tmp_path / "CARLA"),
            "PYTHON_BIN": str(fake_python),
            "ROUTE_SPLIT_OVERRIDE": str(route),
            "OUTPUT_ROOT": str(output_root),
            "RUN_VULKAN_PRECHECK": "0",
            "CLOSEDLOOP_PROGRESS_WATCHDOG_SECONDS": "1",
            "CLOSEDLOOP_PROGRESS_STARTUP_GRACE_SECONDS": "10",
            "CLOSEDLOOP_PROGRESS_WATCHDOG_POLL_SECONDS": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(RUNNER)],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 124
    marker = json.loads(
        (output_root / "runtime_progress_watchdog.json").read_text()
    )
    assert marker["reason"] == "control_trace_stalled"
    assert marker["scientific_classification"] == "runtime_environment_invalid"
    assert marker["stall_threshold_seconds"] == 1
