#!/usr/bin/env python3
"""Run a hash-bound batch of Stage2-L visual-cache re-attestations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    from scripts.reattest_stage2l_visual_cache import reattest_cache
    from scripts.scenario_factory_lib import sha256_file
except ModuleNotFoundError:
    from reattest_stage2l_visual_cache import reattest_cache
    from scenario_factory_lib import sha256_file


SCHEMA = "orion.stage2l_visual_cache_reuse_batch.v1"
REPORT_SCHEMA = "orion.stage2l_visual_cache_reuse_batch_report.v1"


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _resolve(reference: Mapping[str, Any], base: Path, name: str) -> Path:
    path = Path(str(reference.get("path", "")))
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.is_file() or sha256_file(path) != reference.get("sha256"):
        raise ValueError("%s is absent or has a SHA-256 mismatch" % name)
    return path


def run_batch(*, manifest_path: Path, output_root: Path) -> Dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = _load(manifest_path)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported visual-cache reuse batch schema")
    checkpoint = str(manifest.get("expected_orion_checkpoint_sha256", ""))
    if len(checkpoint) != 64:
        raise ValueError("batch lacks the frozen ORION checkpoint hash")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("visual-cache reuse batch has no entries")
    event_ids = [str(row.get("event_id", "")) for row in entries]
    if any(not event_id for event_id in event_ids) or len(set(event_ids)) != len(event_ids):
        raise ValueError("visual-cache reuse batch event ids are absent or duplicated")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite visual-cache reuse batch output")
    output_root.mkdir(parents=True, exist_ok=True)
    completed = []
    for row in entries:
        event_id = str(row["event_id"])
        source = _resolve(
            row.get("source_cache_manifest", {}),
            manifest_path.parent,
            "%s source cache manifest" % event_id,
        )
        target = _resolve(
            row.get("target_factory_report", {}),
            manifest_path.parent,
            "%s target factory report" % event_id,
        )
        event_root = output_root / event_id
        output_manifest = event_root / "visual_cache_manifest.json"
        output_attestation = event_root / "reuse_attestation.json"
        result = reattest_cache(
            source_manifest_path=source,
            target_factory_report_path=target,
            expected_orion_checkpoint_sha256=checkpoint,
            output_manifest_path=output_manifest,
            output_attestation_path=output_attestation,
        )
        completed.append(
            {
                "event_id": event_id,
                "group_count": len(result["group_ids"]),
                "visual_cache_manifest": {
                    "path": str(output_manifest.resolve()),
                    "sha256": sha256_file(output_manifest.resolve()),
                },
                "reuse_attestation": {
                    "path": str(output_attestation.resolve()),
                    "sha256": sha256_file(output_attestation.resolve()),
                },
            }
        )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "completed_all_cache_reuse_attestations",
        "source_batch_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "expected_orion_checkpoint_sha256": checkpoint,
        "event_count": len(completed),
        "events": completed,
        "formal_training_ready": False,
        "stage2p_allowed": False,
        "claim_boundary": "Visual-cache observation-equivalence and lineage only; no QA, model, planning or safety result.",
    }
    report_path = output_root / "batch_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = run_batch(manifest_path=args.manifest, output_root=args.output_root)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
