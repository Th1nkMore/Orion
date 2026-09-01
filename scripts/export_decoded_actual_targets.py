#!/usr/bin/env python3
"""CPU smoke/export boundary for decoded actual perception-error targets.

No real ORION inference hook is implemented here.  Production callers must
adapt frozen-model decoded outputs to ``DecodedORIONFrameV1`` and call the
library exporter.  ``--mock`` exercises the full bundle and v2-record bridge
with deterministic CPU tensors; ``--dry-run`` performs no writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uq_estimator.decoded_actual_target_export import (  # noqa: E402
    ActualTargetExportError,
    bridge_actual_target_bundle_to_v2_record,
    build_cpu_mock_actual_target_bundle,
    save_paired_actual_target_bundle,
)
from uq_estimator.spatial_training import save_paired_feature_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="use deterministic decoded/GT/support CPU fixtures",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and summarize without writing files",
    )
    parser.add_argument("--output", type=Path, help="paired target bundle .pt path")
    parser.add_argument(
        "--record-output",
        type=Path,
        help="optional one-record Stage-1 v2 paired-feature dataset path",
    )
    parser.add_argument("--feature-dim", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.mock:
        raise SystemExit(
            "A real frozen-ORION decode hook is intentionally not implemented. "
            "Use the Python adapter API, or pass --mock for the CPU smoke."
        )
    bundle, observed_features, clean_features, corruption_mask = (
        build_cpu_mock_actual_target_bundle(feature_dim=args.feature_dim)
    )
    record = bridge_actual_target_bundle_to_v2_record(
        bundle,
        observed_features,
        clean_features,
        record_id=f"{bundle.bundle_id}/severity_2",
        pair_id=bundle.bundle_id,
        corruption_mask=corruption_mask,
    )

    writes = []
    if not args.dry_run:
        if args.output is not None:
            save_paired_actual_target_bundle(args.output, bundle)
            writes.append(str(args.output))
        if args.record_output is not None:
            if args.record_output.exists():
                raise ActualTargetExportError(
                    f"refusing to overwrite existing record output: {args.record_output}"
                )
            save_paired_feature_records(args.record_output, [record])
            writes.append(str(args.record_output))

    paired_valid = bundle.paired_valid_mask
    summary = {
        "schema_version": bundle.schema_version,
        "bundle_id": bundle.bundle_id,
        "mock": True,
        "dry_run": bool(args.dry_run),
        "real_orion_hook_executed": bundle.real_orion_hook_executed,
        "real_orion_hook_implemented_by_this_cli": False,
        "patch_attribution_is_causal": bundle.patch_attribution_is_causal,
        "target_shape": list(bundle.observed.error_severity_target.shape),
        "component_error_names": list(bundle.observed.component_error_names),
        "paired_valid_cells": int(paired_valid.sum()),
        "observed_failure_event_cells": int(
            bundle.observed.failure_event_target.sum()
        ),
        "clean_failure_event_cells": int(bundle.clean.failure_event_target.sum()),
        "mean_delta_error_on_paired_valid": (
            float(bundle.delta_error[paired_valid].mean())
            if bool(paired_valid.any())
            else None
        ),
        "duplicate_source_queries_preserved": (
            bundle.observed.duplicate_source_queries_present
        ),
        "with_light_state_required": True,
        "v2_record_bridge_ok": record.target_provenance
        == "actual_perception_failure",
        "writes_performed": bool(writes),
        "written_paths": writes,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
