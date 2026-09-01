#!/usr/bin/env python3
"""Preflight the Route214 frozen-ORION actual-target runner; never run GPU work."""

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uq_estimator.orion_actual_target_runner import (  # noqa: E402
    OrionActualTargetRunnerError,
    assert_real_execution_ready,
    build_runner_preflight,
    load_stage3_agent_config,
    mutate_stage3_agent_config_for_actual_targets,
    verify_box_z_origin_lineage,
    verify_local_traffic_formatter_fix,
)
from uq_estimator.orion_replay_smoke import (  # noqa: E402
    build_replay_smoke_plan,
    load_pilot_manifest,
    verify_source_infos,
)


DEFAULT_CONFIG = REPO_ROOT / "adzoo" / "orion" / "configs" / "orion_stage3_agent.py"
DEFAULT_PILOT = (
    REPO_ROOT
    / "configs"
    / "spatial_uq_route_manifests"
    / "b2d_val_exploratory_pilot10_seed20260826.json"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pilot-manifest", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--infos", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Fail closed: this preflight CLI has no real ORION/raster/support "
            "integration factory and therefore cannot execute."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.dataset_root is not None and args.infos is None:
        raise OrionActualTargetRunnerError("--dataset-root requires --infos")
    source_config, config_lineage = load_stage3_agent_config(args.config)
    _, pipeline_audit = mutate_stage3_agent_config_for_actual_targets(source_config)
    formatter_audit = verify_local_traffic_formatter_fix(REPO_ROOT)
    box_z_origin_audit = verify_box_z_origin_lineage(REPO_ROOT)

    pilot, pilot_lineage = load_pilot_manifest(args.pilot_manifest)
    source_verification = None
    if args.infos is not None:
        provisional = build_replay_smoke_plan(pilot, pilot_lineage)
        source_verification = verify_source_infos(
            args.infos,
            pilot,
            provisional["route"]["folder"],
            63,
            dataset_root=args.dataset_root,
        )
    plan = build_replay_smoke_plan(
        pilot,
        pilot_lineage,
        source_verification=source_verification,
    )
    preflight = build_runner_preflight(
        plan,
        config_lineage=config_lineage,
        pipeline_audit=pipeline_audit,
        formatter_audit=formatter_audit,
        box_z_origin_audit=box_z_origin_audit,
        hooks=None,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(preflight, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    json.dump(preflight, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    if args.execute:
        assert_real_execution_ready(preflight)
        raise OrionActualTargetRunnerError(
            "preflight CLI intentionally has no model/dataloader/runtime-hook factory"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
