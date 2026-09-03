#!/usr/bin/env python3
"""Run frame-bundle, QA-record and U/R/K visualization postprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

try:
    from scripts.build_uq_relevance_frame_bundles import build_frame_bundles
    from scripts.render_uq_relevance_bundle import render_bundle
    from scripts.scenario_factory_lib import sha256_file
except ModuleNotFoundError:
    from build_uq_relevance_frame_bundles import build_frame_bundles
    from render_uq_relevance_bundle import render_bundle
    from scenario_factory_lib import sha256_file


SCHEMA = "orion.uq_relevance_event_factory_report.v1"


def finalize_event(
    *,
    project_root: Path,
    event_package: Path,
    stage1_manifest: Path,
    split: str,
    output_root: Path,
    town: str = None,
    scenario_family: str = None,
) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty event-factory output")
    bundle_root = output_root / "frame_bundles"
    route_override = None
    if town or scenario_family:
        if not town or not scenario_family:
            raise ValueError("town and scenario-family overrides must be supplied together")
        route_override = {"town": town, "scenario_type": scenario_family}
    bundle_report = build_frame_bundles(
        event_package_path=event_package,
        stage1_manifest_path=stage1_manifest,
        split=split,
        output_dir=bundle_root,
        variants=(
            "observed",
            "zero_uq",
            "on_path_uq",
            "off_path_uq",
            "view_shuffled_uq",
        ),
        counterfactual_peak=0.9,
        route_override=route_override,
    )
    bundle_list_path = output_root / "bundle_list.json"
    bundle_list_path.write_text(
        json.dumps(
            {"bundles": [item["path"] for item in bundle_report["bundles"]]},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    qa_root = output_root / "qa_dataset"
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "build_uq_relevance_qa_dataset.py"),
            "--bundle-list",
            str(bundle_list_path),
            "--config",
            str(project_root / "configs" / "scenario_factory" / "qa_factory_v1.json"),
            "--output-dir",
            str(qa_root),
        ],
        check=True,
    )
    visual_rows = []
    for item in bundle_report["bundles"]:
        bundle_path = Path(item["path"])
        visual_root = output_root / "visuals" / item["variant"]
        visual = render_bundle(bundle_path, visual_root)
        visual_rows.append({
            "variant": item["variant"],
            "manifest": str((visual_root / "visualization_manifest.json").resolve()),
            "contact_sheet": visual["output"],
        })
    audit_path = qa_root / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    report = {
        "schema": SCHEMA,
        "status": "pending_human_review",
        "event_package": {"path": str(event_package), "sha256": sha256_file(event_package)},
        "stage1_manifest": {"path": str(stage1_manifest), "sha256": sha256_file(stage1_manifest)},
        "frame_bundle_batch": {
            "path": str((bundle_root / "frame_bundle_batch.json").resolve()),
            "sha256": sha256_file(bundle_root / "frame_bundle_batch.json"),
        },
        "qa_dataset": {
            "path": str((qa_root / "dataset.json").resolve()),
            "sha256": sha256_file(qa_root / "dataset.json"),
            "audit": str(audit_path.resolve()),
            "formal_training_ready": audit["formal_training_ready"],
        },
        "visualizations": visual_rows,
        "claim_boundary": (
            "Single development-event factory smoke awaiting human map review; "
            "formal Stage2-L gates intentionally remain closed."
        ),
    }
    report_path = output_root / "event_factory_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--event-package", type=Path, required=True)
    parser.add_argument("--stage1-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--town")
    parser.add_argument("--scenario-family")
    args = parser.parse_args()
    report = finalize_event(
        project_root=args.project_root.resolve(),
        event_package=args.event_package.resolve(),
        stage1_manifest=args.stage1_manifest.resolve(),
        split=args.split,
        output_root=args.output_root.resolve(),
        town=args.town,
        scenario_family=args.scenario_family,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
