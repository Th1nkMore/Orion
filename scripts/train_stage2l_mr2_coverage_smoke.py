#!/usr/bin/env python3
"""Run one bounded 17-event Stage2-L coverage diagnostic.

This entry point intentionally reuses the frozen MR1 optimizer and evaluation
implementation while changing only the hash-bound dataset scope and run
identity.  Every optimizer step still consumes one complete matched group
from every train event, so the 40-step comparison preserves per-event exposure
while increasing route, town, scenario and task-field coverage.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import scripts.train_stage2l_mr1_smoke as base
from scripts.scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2l_mr2_expanded_coverage_smoke.v1"
PROTOCOL_SCHEMA = "orion.stage2l_mr2_training_protocol.v1"
DATASET_SCHEMA = "orion.stage2l_expanded_coverage_dataset.v1"
PREFLIGHT_SCHEMA = "orion.stage2l_mr2_trainer_preflight.v1"
EXPECTED_EVENT_COUNT = 17
EXPECTED_TRAIN_EVENT_COUNT = 13
EXPECTED_DEV_EVENT_COUNT = 4
EXPECTED_GROUP_COUNT = 80
EXPECTED_RECORD_COUNT = 1600
ALLOWED_BOUNDED_OPTIMIZER_STEPS = (40,)


def _configure_base() -> None:
    base.SCHEMA = SCHEMA
    base.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    base.DATASET_SCHEMA = DATASET_SCHEMA
    base.PREFLIGHT_SCHEMA = PREFLIGHT_SCHEMA
    base.EXPECTED_EVENT_COUNT = EXPECTED_EVENT_COUNT
    base.EXPECTED_TRAIN_EVENT_COUNT = EXPECTED_TRAIN_EVENT_COUNT
    base.EXPECTED_DEV_EVENT_COUNT = EXPECTED_DEV_EVENT_COUNT
    base.EXPECTED_GROUP_COUNT = EXPECTED_GROUP_COUNT
    base.EXPECTED_RECORD_COUNT = EXPECTED_RECORD_COUNT
    base.ALLOWED_BOUNDED_OPTIMIZER_STEPS = ALLOWED_BOUNDED_OPTIMIZER_STEPS


def _patch_validated_input_lineage() -> None:
    original = base._validated_input_hashes

    def expanded(**kwargs) -> Dict[str, Any]:
        value = original(**kwargs)
        value["base_mr1_trainer_sha256"] = value["trainer_sha256"]
        value["trainer_sha256"] = sha256_file(Path(__file__).resolve())
        return value

    base._validated_input_hashes = expanded


def _argument_path(name: str) -> Optional[Path]:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return None
    if index + 1 >= len(sys.argv):
        return None
    return Path(sys.argv[index + 1]).resolve()


def _annotate_report(output_dir: Path) -> None:
    report_path = output_dir / "report.json"
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks = report.get("checks", {})
    old_key = "every_step_is_six_event_balanced"
    if old_key in checks:
        checks["every_step_covers_all_13_train_events"] = checks.pop(old_key)
    report["diagnostic_identity"] = {
        "name": "MR2 expanded event/class coverage smoke",
        "intended_change_from_mr1_40": (
            "8 events (6/2) and 37 groups -> 17 events (13/4) and 80 groups; "
            "optimizer steps, per-event primary exposure, architecture, losses, "
            "seed and release thresholds remain fixed"
        ),
        "not_a_duration_extension": True,
        "not_formal_training": True,
    }
    report.setdefault("provenance", {})["runtime_entrypoint"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__).resolve()),
    }
    report["claim_boundary"] = (
        "One bounded expanded-coverage engineering diagnostic. It does not "
        "authorize formal Stage2-L, Stage2-P, locked-test reading, planning, "
        "closed-loop, generalization, or safety claims."
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    _configure_base()
    _patch_validated_input_lineage()
    output_dir = _argument_path("--output-dir")
    result = base.main()
    if output_dir is not None and "--preflight-only" not in sys.argv:
        _annotate_report(output_dir)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
