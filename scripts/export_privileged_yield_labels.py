#!/usr/bin/env python3
"""Export planning-layer privileged yield labels from a closed-loop run."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uq_estimator.privileged_yield_labels import (
    TrajectoryConflictConfig,
    export_run_labels,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-time-seconds", type=float)
    parser.add_argument("--safety-margin-m", type=float, default=0.75)
    parser.add_argument("--imminent-horizon-seconds", type=float, default=1.5)
    parser.add_argument("--clearance-seconds", type=float, default=1.0)
    parser.add_argument("--release-seconds", type=float, default=0.5)
    parser.add_argument("--stop-buffer-m", type=float, default=2.0)
    parser.add_argument("--release-creep-distance-m", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrajectoryConflictConfig(
        safety_margin_m=args.safety_margin_m,
        imminent_horizon_seconds=args.imminent_horizon_seconds,
        clearance_seconds=args.clearance_seconds,
        release_seconds=args.release_seconds,
        stop_buffer_m=args.stop_buffer_m,
        release_creep_distance_m=args.release_creep_distance_m,
    )
    report = export_run_labels(
        args.run_dir,
        args.output_dir,
        config=config,
        audit_time_seconds=args.audit_time_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
