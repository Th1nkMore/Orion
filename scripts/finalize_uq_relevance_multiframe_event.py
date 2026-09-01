#!/usr/bin/env python3
"""Build one event's 3-5 keyframe Stage2-L QA package in one pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Mapping

try:
    from scripts.build_uq_relevance_frame_bundles import build_frame_bundles
    from scripts.render_uq_relevance_bundle import render_bundle
    from scripts.scenario_factory_lib import sha256_file
    from scripts.upgrade_stage2l_v9_qa_records import (
        QA_CONTRACT_SCHEMA as QA_CONTRACT_V5_SCHEMA,
        upgrade_records as upgrade_records_v5,
    )
    from uq_estimator.task_relevance_geometry import TaskRelevanceGeometryError
except ModuleNotFoundError:
    from build_uq_relevance_frame_bundles import build_frame_bundles
    from render_uq_relevance_bundle import render_bundle
    from scenario_factory_lib import sha256_file
    from upgrade_stage2l_v9_qa_records import (
        QA_CONTRACT_SCHEMA as QA_CONTRACT_V5_SCHEMA,
        upgrade_records as upgrade_records_v5,
    )
    from uq_estimator.task_relevance_geometry import TaskRelevanceGeometryError


SCHEMA = "orion.uq_relevance_multiframe_event_factory.v1"
STAGE1_SCHEMA = "orion.stage1_observation_uq_multiframe.v1"
QA_FACTORY_V2_SCHEMA = "orion.uq_relevance_qa_factory_config.v2"
QA_FACTORY_V5_SCHEMA = "orion.uq_relevance_qa_factory_config.v5"


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _resolve_reference(reference: Mapping[str, Any], base: Path, name: str) -> Path:
    path = Path(str(reference.get("path", "")))
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.is_file():
        raise FileNotFoundError("%s is missing: %s" % (name, path))
    if sha256_file(path) != reference.get("sha256"):
        raise ValueError("%s SHA-256 mismatch" % name)
    return path


def validate_multiframe_manifest(
    path: Path, event_package: Path
) -> List[Dict[str, Any]]:
    root = _load_json(path)
    if root.get("schema") != STAGE1_SCHEMA or root.get("control_influence") is not False:
        raise ValueError("unsupported or control-active Stage1 multi-frame manifest")
    event_ref = root.get("event_package", {})
    if event_ref.get("sha256") != sha256_file(event_package):
        raise ValueError("Stage1 multi-frame event-package hash mismatch")
    rows = root.get("sequences")
    if not isinstance(rows, list) or not 3 <= len(rows) <= 5:
        raise ValueError("Stage1 multi-frame manifest must contain three to five sequences")
    result = []
    seen = set()
    for row in rows:
        frame = int(row["selected_saved_frame_index"])
        if frame in seen:
            raise ValueError("Stage1 multi-frame manifest duplicates a saved frame")
        seen.add(frame)
        manifest_path = _resolve_reference(row["manifest"], path.parent, "Stage1 keyframe sequence")
        manifest = _load_json(manifest_path)
        if manifest.get("schema") != "orion.stage1_observation_uq_sequence.v1":
            raise ValueError("unsupported Stage1 keyframe sequence schema")
        if int(manifest.get("latest_frame_index", -1)) != frame:
            raise ValueError("Stage1 keyframe sequence frame mismatch")
        result.append({"frame": frame, "manifest": manifest_path})
    return sorted(result, key=lambda row: row["frame"])


def _write_v5_qa_dataset(
    *, base_qa_root: Path, output_root: Path, qa_factory_config: Path
) -> Dict[str, Any]:
    """Upgrade geometric v2 records to the frozen v5 task-field contract."""

    if output_root.exists():
        raise FileExistsError("refusing to overwrite v5 QA dataset output")
    base_records_path = base_qa_root / "records.jsonl"
    base_sidecars = base_qa_root / "map_sidecars"
    if not base_records_path.is_file() or not base_sidecars.is_dir():
        raise FileNotFoundError("base v2 QA records or map sidecars are missing")
    base_records = [
        json.loads(line)
        for line in base_records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records, audit = upgrade_records_v5(base_records)
    if not audit.get("passed"):
        raise RuntimeError("v5 QA upgrade failed its frozen contract audit")

    output_root.mkdir(parents=True)
    shutil.copytree(base_sidecars, output_root / "map_sidecars")
    shutil.copyfile(qa_factory_config, output_root / "qa_factory_config.json")
    records_path = output_root / "records.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n"
            for row in records
        ),
        encoding="utf-8",
    )
    audit_path = output_root / "audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    dataset = {
        "schema": "orion.uq_relevance_qa_dataset.v5",
        "status": "prepared_training_locked",
        "config": {
            "path": "qa_factory_config.json",
            "sha256": sha256_file(output_root / "qa_factory_config.json"),
        },
        "source_v2_records": {
            "path": str(base_records_path.resolve()),
            "sha256": sha256_file(base_records_path),
        },
        "records": records,
        "audit_summary": {
            "passed": True,
            "checks": audit["checks"],
        },
        "qa_contract": QA_CONTRACT_V5_SCHEMA,
        "formal_training_ready": False,
        "stage2p_allowed": False,
        "claim_boundary": (
            "CPU-only semantic target construction under the frozen v5 task-field "
            "contract; no model learning, planning, control or safety evidence."
        ),
    }
    dataset_path = output_root / "dataset.json"
    dataset_path.write_text(
        json.dumps(dataset, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "orion.uq_relevance_qa_dataset_manifest.v5",
        "status": "prepared_training_locked",
        "source_v2_records": dataset["source_v2_records"],
        "records": {"path": str(records_path.resolve()), "sha256": sha256_file(records_path)},
        "audit": {"path": str(audit_path.resolve()), "sha256": sha256_file(audit_path)},
        "dataset": {"path": str(dataset_path.resolve()), "sha256": sha256_file(dataset_path)},
        "qa_contract": QA_CONTRACT_V5_SCHEMA,
        "training_started": False,
        "stage2p_allowed": False,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "dataset_path": dataset_path,
        "records_path": records_path,
        "audit_path": audit_path,
        "record_count": len(records),
        "formal_training_ready": False,
        "manifest_path": manifest_path,
    }


def finalize_multiframe_event(
    *,
    project_root: Path,
    event_package: Path,
    stage1_multiframe_manifest: Path,
    split: str,
    output_root: Path,
    town: str = None,
    scenario_family: str = None,
    qa_factory_config: Path = None,
    base_qa_factory_config: Path = None,
) -> Dict[str, Any]:
    if split not in ("train", "dev", "test"):
        raise ValueError("split must be train, dev or test")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite multi-frame event output")
    if qa_factory_config is None:
        qa_factory_config = (
            project_root
            / "configs"
            / "scenario_factory"
            / "qa_factory_v2_matched_supervision.json"
        )
    qa_factory_config = qa_factory_config.resolve()
    if not qa_factory_config.is_file():
        raise FileNotFoundError("QA factory config is missing: %s" % qa_factory_config)
    qa_factory_payload = _load_json(qa_factory_config)
    qa_factory_schema = str(qa_factory_payload.get("schema", ""))
    if qa_factory_schema == QA_FACTORY_V5_SCHEMA:
        if base_qa_factory_config is None:
            base_qa_factory_config = (
                project_root
                / "configs"
                / "scenario_factory"
                / "qa_factory_v2_matched_supervision.json"
            )
        base_qa_factory_config = base_qa_factory_config.resolve()
        if not base_qa_factory_config.is_file():
            raise FileNotFoundError(
                "base QA factory config is missing: %s" % base_qa_factory_config
            )
        if _load_json(base_qa_factory_config).get("schema") != QA_FACTORY_V2_SCHEMA:
            raise ValueError("v5 construction requires the frozen v2 geometric base config")
        build_qa_factory_config = base_qa_factory_config
    else:
        if qa_factory_schema not in {
            "orion.uq_relevance_qa_factory_config.v1",
            QA_FACTORY_V2_SCHEMA,
        }:
            raise ValueError("unsupported QA factory config schema")
        if base_qa_factory_config is not None:
            raise ValueError("base QA factory config is only valid for a v5 upgrade")
        build_qa_factory_config = qa_factory_config
    event_payload = _load_json(event_package)
    if event_payload.get("schema") != "orion.scenario_event_package.v1":
        raise ValueError("unsupported event-package schema")
    critical = event_payload.get("critical_event") or {}
    event_id = "route%s_step%s" % (
        event_payload["route"]["route_index"], critical["step"]
    )
    route_override = None
    if town or scenario_family:
        if not town or not scenario_family:
            raise ValueError("town and scenario-family overrides must be supplied together")
        route_override = {"town": town, "scenario_type": scenario_family}
    sequences = validate_multiframe_manifest(
        stage1_multiframe_manifest, event_package
    )
    bundle_rows = []
    frame_reports = []
    visual_rows = []
    excluded_keyframes = []
    for sequence in sequences:
        frame = sequence["frame"]
        frame_root = output_root / ("frame_%04d" % frame)
        try:
            report = build_frame_bundles(
                event_package_path=event_package,
                stage1_manifest_path=sequence["manifest"],
                split=split,
                output_dir=frame_root / "frame_bundles",
                variants=(
                    "observed",
                    "zero_uq",
                    "on_path_uq",
                    "off_path_uq",
                    "view_shuffled_uq",
                ),
                counterfactual_peak=0.9,
                route_override=route_override,
                selected_frame_index=frame,
            )
        except TaskRelevanceGeometryError as error:
            if str(error) not in {
                "the ORION route has no visible camera support",
                "task relevance has no visible route or conflict-actor support",
            }:
                raise
            excluded_keyframes.append({
                "selected_saved_frame_index": frame,
                "reason": (
                    "orion_route_and_conflict_actor_support_not_visible"
                    if "task relevance" in str(error)
                    else "orion_route_has_no_visible_camera_support"
                ),
                "uses_observation_uq": False,
                "uses_actor_or_outcome_selection": False,
                "uses_stage2_outputs": False,
            })
            continue
        frame_reports.append({
            "selected_saved_frame_index": frame,
            "frame_bundle_batch": {
                "path": str((frame_root / "frame_bundles" / "frame_bundle_batch.json").resolve()),
                "sha256": sha256_file(frame_root / "frame_bundles" / "frame_bundle_batch.json"),
            },
        })
        for item in report["bundles"]:
            bundle_rows.append(item["path"])
            bundle_path = Path(item["path"])
            visual_root = frame_root / "visuals" / item["variant"]
            visual = render_bundle(bundle_path, visual_root)
            visual_manifest_path = visual_root / "visualization_manifest.json"
            output_reference = visual.get("output")
            if not isinstance(output_reference, Mapping):
                raise RuntimeError("U/R/K visualization output reference is malformed")
            contact_sheet_path = Path(str(output_reference.get("path", "")))
            if (
                not contact_sheet_path.is_file()
                or sha256_file(contact_sheet_path) != output_reference.get("sha256")
            ):
                raise RuntimeError("U/R/K contact-sheet provenance mismatch")
            visual_rows.append({
                "selected_saved_frame_index": frame,
                "variant": item["variant"],
                "manifest": {
                    "path": str(visual_manifest_path.resolve()),
                    "sha256": sha256_file(visual_manifest_path),
                },
                "contact_sheet": {
                    "path": str(contact_sheet_path.resolve()),
                    "sha256": sha256_file(contact_sheet_path),
                },
            })
    if len(frame_reports) < 3:
        raise RuntimeError(
            "fewer than three fixed-offset keyframes retain visible ORION route support"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    bundle_list_path = output_root / "bundle_list.json"
    bundle_list_path.write_text(
        json.dumps({"bundles": bundle_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    qa_root = output_root / "qa_dataset"
    base_qa_root = (
        output_root / "qa_dataset_v2_source"
        if qa_factory_schema == QA_FACTORY_V5_SCHEMA
        else qa_root
    )
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "build_uq_relevance_qa_dataset.py"),
            "--bundle-list",
            str(bundle_list_path),
            "--config",
            str(build_qa_factory_config),
            "--output-dir",
            str(base_qa_root),
        ],
        check=True,
    )
    if qa_factory_schema == QA_FACTORY_V5_SCHEMA:
        qa_outputs = _write_v5_qa_dataset(
            base_qa_root=base_qa_root,
            output_root=qa_root,
            qa_factory_config=qa_factory_config,
        )
        audit_path = qa_outputs["audit_path"]
        audit = _load_json(audit_path)
        dataset_path = qa_outputs["dataset_path"]
        records_path = qa_outputs["records_path"]
        formal_training_ready = False
    else:
        audit_path = qa_root / "audit.json"
        audit = _load_json(audit_path)
        dataset_path = qa_root / "dataset.json"
        records_path = qa_root / "records.jsonl"
        formal_training_ready = bool(audit["formal_training_ready"])
    dataset = _load_json(dataset_path)
    expected_records = len(frame_reports) * 5 * 4
    if len(dataset.get("records", [])) != expected_records:
        raise RuntimeError("multi-frame QA record count differs from 20 per keyframe")
    report = {
        "schema": SCHEMA,
        "status": "pending_multiframe_human_geometry_review",
        "event_id": event_id,
        "event_package": {
            "path": str(event_package.resolve()),
            "sha256": sha256_file(event_package),
        },
        "stage1_multiframe_manifest": {
            "path": str(stage1_multiframe_manifest.resolve()),
            "sha256": sha256_file(stage1_multiframe_manifest),
        },
        "requested_keyframe_count": len(sequences),
        "keyframe_count": len(frame_reports),
        "selected_saved_frames": [
            row["selected_saved_frame_index"] for row in frame_reports
        ],
        "excluded_keyframes": excluded_keyframes,
        "geometry_eligibility_policy": {
            "policy_id": "fixed_offsets_visible_route_or_conflict_actor_support_v2",
            "candidate_offsets_unchanged": True,
            "minimum_retained_keyframes": 3,
            "support": "visible ORION route corridor union visible conflict-actor boxes",
            "uses_observation_uq": False,
            "uses_conflict_actor_geometry": True,
            "uses_recorded_ttc": False,
            "uses_collision_outcome": False,
            "uses_actor_or_outcome_selection": False,
            "uses_stage2_outputs": False,
        },
        "qa_records_per_keyframe": 20,
        "qa_record_count": expected_records,
        "frame_reports": frame_reports,
        "qa_dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": sha256_file(dataset_path),
            "records": {
                "path": str(records_path.resolve()),
                "sha256": sha256_file(records_path),
            },
            "audit": str(audit_path.resolve()),
            "formal_training_ready": formal_training_ready,
        },
        "qa_factory_config": {
            "path": str(qa_factory_config),
            "sha256": sha256_file(qa_factory_config),
        },
        "base_qa_factory_config": (
            {
                "path": str(base_qa_factory_config),
                "sha256": sha256_file(base_qa_factory_config),
            }
            if base_qa_factory_config is not None
            else None
        ),
        "qa_contract": (
            QA_CONTRACT_V5_SCHEMA
            if qa_factory_schema == QA_FACTORY_V5_SCHEMA
            else qa_factory_schema
        ),
        "visualizations": visual_rows,
        "claim_boundary": (
            "One reviewed event expanded from fixed temporal offsets for Stage2-L "
            "pilot construction. Geometry-invalid frames may only be dropped when "
            "neither the unmodified ORION route nor a visible conflict actor has "
            "camera support; repeated frames do not count as independent events."
        ),
    }
    report_path = output_root / "multiframe_event_factory_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--event-package", type=Path, required=True)
    parser.add_argument("--stage1-multiframe-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--town")
    parser.add_argument("--scenario-family")
    parser.add_argument("--qa-factory-config", type=Path)
    parser.add_argument("--base-qa-factory-config", type=Path)
    args = parser.parse_args()
    report = finalize_multiframe_event(
        project_root=args.project_root.resolve(),
        event_package=args.event_package.resolve(),
        stage1_multiframe_manifest=args.stage1_multiframe_manifest.resolve(),
        split=args.split,
        output_root=args.output_root.resolve(),
        town=args.town,
        scenario_family=args.scenario_family,
        qa_factory_config=args.qa_factory_config,
        base_qa_factory_config=args.base_qa_factory_config,
    )
    print(json.dumps({
        "report": str((args.output_root / "multiframe_event_factory_report.json").resolve()),
        "keyframe_count": report["keyframe_count"],
        "qa_record_count": report["qa_record_count"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
