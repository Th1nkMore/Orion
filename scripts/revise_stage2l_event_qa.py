#!/usr/bin/env python3
"""Rebuild only Stage2-L QA text/sidecars from an immutable event factory report.

Frame bundles, Stage-1 UQ, R maps, camera observations, and visualizations are
hash-verified and reused.  The output is a new report that explicitly
supersedes the source report; the source directory is never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.scenario_factory_lib import sha256_file


FACTORY_SCHEMA = "orion.uq_relevance_multiframe_event_factory.v1"
REVISION_SCHEMA = "orion.stage2l_qa_only_revision.v1"
ALLOWED_SPLITS = {"train", "dev", "test"}


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _verified(reference: Mapping[str, Any], name: str) -> Path:
    path = Path(str(reference.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError("%s is missing: %s" % (name, path))
    if sha256_file(path) != reference.get("sha256"):
        raise ValueError("%s SHA-256 mismatch" % name)
    return path


def collect_bundle_paths(report: Mapping[str, Any]) -> List[Path]:
    if report.get("schema") != FACTORY_SCHEMA:
        raise ValueError("unsupported source factory report schema")
    frame_reports = report.get("frame_reports")
    if not isinstance(frame_reports, list) or not 3 <= len(frame_reports) <= 5:
        raise ValueError("source report must contain three to five frame reports")
    paths: List[Path] = []
    seen_frames = set()
    expected_variants = {
        "observed", "zero_uq", "on_path_uq", "off_path_uq",
        "view_shuffled_uq",
    }
    for frame_row in frame_reports:
        frame = int(frame_row["selected_saved_frame_index"])
        if frame in seen_frames:
            raise ValueError("source report duplicates a selected frame")
        seen_frames.add(frame)
        batch_path = _verified(
            frame_row["frame_bundle_batch"], "frame bundle batch"
        )
        batch = _load_json(batch_path)
        if int(batch.get("frame_id", "saved_-1").split("_")[-1]) != frame:
            raise ValueError("frame bundle batch is not aligned to its report row")
        bundles = batch.get("bundles")
        if not isinstance(bundles, list) or len(bundles) != 5:
            raise ValueError("each retained frame must contain five bundles")
        variants = {str(item.get("variant")) for item in bundles}
        if variants != expected_variants:
            raise ValueError("frame bundle variants do not match the frozen contract")
        for item in bundles:
            paths.append(_verified(item, "frame bundle"))
    if len(paths) != len(frame_reports) * 5:
        raise RuntimeError("bundle inventory count mismatch")
    return paths


def materialize_split_overridden_bundles(
    bundle_paths: List[Path],
    *,
    output_dir: Path,
    split_override: str,
) -> Dict[str, Any]:
    """Create immutable split-only bundle revisions without touching payload assets."""
    if split_override not in ALLOWED_SPLITS:
        raise ValueError("unsupported split override: %s" % split_override)
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite split-overridden bundles")
    output_dir.mkdir(parents=True)
    revised_paths: List[Path] = []
    source_splits = set()
    inventory = []
    for source_path in bundle_paths:
        source = _load_json(source_path)
        source_split = str(source.get("split", ""))
        if source_split not in ALLOWED_SPLITS:
            raise ValueError("source bundle has unsupported split: %s" % source_path)
        source_splits.add(source_split)
        frame_id = str(source.get("frame_id", "unknown_frame"))
        variant = str(source.get("counterfactual", {}).get("variant", "unknown"))
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", "%s_%s" % (frame_id, variant))
        revised_path = output_dir / (safe_stem + ".json")
        if revised_path.exists():
            raise ValueError("duplicate revised bundle filename: %s" % revised_path.name)
        source_sha256 = sha256_file(source_path)
        revised = dict(source)
        revised["split"] = split_override
        revised["split_only_revision"] = {
            "schema": "orion.stage2l_bundle_split_only_revision.v1",
            "source_bundle": {
                "path": str(source_path.resolve()),
                "sha256": source_sha256,
            },
            "source_split": source_split,
            "revised_split": split_override,
            "unchanged_fields": [
                "model_input",
                "route",
                "supervision",
                "counterfactual",
                "provenance",
            ],
            "claim_boundary": (
                "Split assignment metadata only; no image, U/R/K, geometry, "
                "supervision, or counterfactual content was changed."
            ),
        }
        revised_path.write_text(
            json.dumps(revised, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        revised_paths.append(revised_path)
        inventory.append({
            "source": {"path": str(source_path.resolve()), "sha256": source_sha256},
            "revised": {
                "path": str(revised_path.resolve()),
                "sha256": sha256_file(revised_path),
            },
            "frame_id": frame_id,
            "variant": variant,
            "source_split": source_split,
            "revised_split": split_override,
        })
    return {
        "paths": revised_paths,
        "source_splits": sorted(source_splits),
        "revised_split": split_override,
        "inventory": inventory,
    }


def revise_event_qa(
    *,
    source_report_path: Path,
    config_path: Path,
    amendment_path: Path,
    output_root: Path,
    split_override: Optional[str] = None,
) -> Dict[str, Any]:
    if output_root.exists():
        raise FileExistsError("refusing to overwrite QA-only revision output")
    source_report = _load_json(source_report_path)
    bundle_paths = collect_bundle_paths(source_report)
    if not config_path.is_file() or not amendment_path.is_file():
        raise FileNotFoundError("QA config and amendment must both exist")
    output_root.mkdir(parents=True)
    split_revision = None
    if split_override is not None:
        split_revision = materialize_split_overridden_bundles(
            bundle_paths,
            output_dir=output_root / "split_overridden_bundles",
            split_override=split_override,
        )
        bundle_paths = split_revision["paths"]
    bundle_list_path = output_root / "bundle_list.json"
    bundle_list_path.write_text(
        json.dumps(
            {"bundles": [str(path.resolve()) for path in bundle_paths]},
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    qa_root = output_root / "qa_dataset"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_uq_relevance_qa_dataset.py"),
            "--bundle-list",
            str(bundle_list_path),
            "--config",
            str(config_path.resolve()),
            "--output-dir",
            str(qa_root),
        ],
        check=True,
    )
    dataset_path = qa_root / "dataset.json"
    records_path = qa_root / "records.jsonl"
    audit_path = qa_root / "audit.json"
    dataset = _load_json(dataset_path)
    audit = _load_json(audit_path)
    expected_records = len(source_report["frame_reports"]) * 5 * 4
    if len(dataset.get("records", [])) != expected_records:
        raise RuntimeError("revised QA record count differs from 20 per keyframe")

    revised_report = dict(source_report)
    revised_report["status"] = "pending_multiframe_human_geometry_review"
    revised_report["qa_dataset"] = {
        "path": str(dataset_path.resolve()),
        "sha256": sha256_file(dataset_path),
        "records": {
            "path": str(records_path.resolve()),
            "sha256": sha256_file(records_path),
        },
        "audit": str(audit_path.resolve()),
        "formal_training_ready": audit["formal_training_ready"],
    }
    revised_report["qa_only_revision"] = {
        "schema": REVISION_SCHEMA,
        "source_factory_report": {
            "path": str(source_report_path.resolve()),
            "sha256": sha256_file(source_report_path),
        },
        "qa_factory_config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "protocol_amendment": {
            "path": str(amendment_path.resolve()),
            "sha256": sha256_file(amendment_path),
        },
        "reused_unchanged": [
            "event package",
            "Stage1 multiframe UQ manifests",
            "frame bundles and U/R/K arrays",
            "camera observations and route context",
            "visualization manifests and contact sheets",
        ],
        "rebuilt": ["QA records", "QA map sidecars", "QA dataset audit"],
    }
    if split_revision is not None:
        revised_report["qa_only_revision"].update({
            "split_override": split_revision["revised_split"],
            "source_splits": split_revision["source_splits"],
            "split_overridden_bundle_inventory": split_revision["inventory"],
        })
        revised_report["qa_only_revision"]["rebuilt"].insert(
            0, "bundle split metadata with immutable source provenance"
        )
    report_path = output_root / "multiframe_event_factory_report.json"
    report_path.write_text(
        json.dumps(revised_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "event_id": revised_report["event_id"],
        "qa_record_count": expected_records,
        "report": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-override", choices=sorted(ALLOWED_SPLITS))
    args = parser.parse_args()
    result = revise_event_qa(
        source_report_path=args.source_report.resolve(),
        config_path=args.config.resolve(),
        amendment_path=args.amendment.resolve(),
        output_root=args.output_root.resolve(),
        split_override=args.split_override,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
