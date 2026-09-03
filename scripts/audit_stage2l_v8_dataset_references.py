#!/usr/bin/env python3
"""Verify that every external artifact referenced by v8 QA records exists."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


SCHEMA = "orion.stage2l_v8_reference_audit.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _references(row: Mapping[str, Any]) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    observation = row["model_input"]["observation"]
    for camera in observation["camera_files"]:
        yield "camera", camera
    yield "stage1_observation_uq", row["model_input"]["stage1_observation_uq"]
    relevance = row["provenance"]["relevance_supervision"]
    yield "relevance_supervision", relevance
    yield "relevance_geometry", relevance["geometry_manifest"]
    yield "map_sidecar", row["target"]["map_sidecar"]


def audit_references(
    records: Sequence[Mapping[str, Any]], *, records_parent: Path
) -> Dict[str, Any]:
    unique: Dict[Tuple[str, str], Tuple[str, Path, str]] = {}
    for row in records:
        for kind, reference in _references(row):
            raw_path = str(reference.get("path", ""))
            if not raw_path:
                raise ValueError("artifact reference has no path")
            path = Path(raw_path)
            if not path.is_absolute():
                path = (records_parent / path).resolve()
            expected = str(reference.get("sha256", ""))
            unique[(str(path), expected)] = (kind, path, expected)
    missing = []
    mismatched = []
    verified_hashes = 0
    kind_counts: Dict[str, int] = {}
    for kind, path, expected in unique.values():
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        if not path.is_file():
            missing.append(str(path))
            continue
        if expected:
            verified_hashes += 1
            actual = _sha256(path)
            if actual != expected:
                mismatched.append(
                    {"path": str(path), "expected": expected, "actual": actual}
                )
    checks = {
        "all_referenced_files_exist": not missing,
        "all_declared_hashes_match": not mismatched,
        "map_sidecars_present": kind_counts.get("map_sidecar", 0) > 0,
        "stage1_uq_maps_present": kind_counts.get("stage1_observation_uq", 0) > 0,
        "camera_evidence_present": kind_counts.get("camera", 0) > 0,
    }
    return {
        "schema": SCHEMA,
        "passed": all(checks.values()),
        "checks": checks,
        "record_count": len(records),
        "unique_reference_count": len(unique),
        "verified_sha256_count": verified_hashes,
        "reference_kind_counts": dict(sorted(kind_counts.items())),
        "missing": missing,
        "hash_mismatches": mismatched,
        "claim_boundary": "Artifact availability and declared-hash integrity only; no model or label-validity claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = audit_references(records, records_parent=args.records.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
