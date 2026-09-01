#!/usr/bin/env python3
"""Materialize hash-bound v3 QA records from an existing matched v2 file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.scenario_factory_lib import sha256_file
from scripts.uq_relevance_qa_factory_v3_lib import upgrade_records


SCHEMA = "orion.stage2l_v7_qa_upgrade_manifest.v1"


def _load_json(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _rebase_relative_sidecar_references(
    records, *, source_dir: Path, output_dir: Path
) -> int:
    """Keep v2 sidecar references valid after records move to a sibling dir."""

    count = 0
    for row in records:
        reference = row["target"]["map_sidecar"]
        raw = Path(str(reference["path"]))
        if raw.is_absolute():
            continue
        source = (source_dir / raw).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        reference["path"] = os.path.relpath(source, output_dir.resolve())
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-records", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite v3 QA output")
    input_path = args.input_records.resolve()
    config_path = args.config.resolve()
    source = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    upgraded, audit = upgrade_records(source, config=_load_json(config_path))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rebased_sidecar_count = _rebase_relative_sidecar_references(
        upgraded,
        source_dir=input_path.parent,
        output_dir=args.output_dir,
    )
    records_path = args.output_dir / "records.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in upgraded
        ),
        encoding="utf-8",
    )
    audit_path = args.output_dir / "audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": SCHEMA,
        "status": "v3_qa_upgrade_pass",
        "training_authorized": False,
        "input_records": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "output_records": {
            "path": str(records_path.resolve()),
            "sha256": sha256_file(records_path),
        },
        "relative_sidecar_references_rebased": rebased_sidecar_count,
        "audit": {
            "path": str(audit_path.resolve()),
            "sha256": sha256_file(audit_path),
            "passed": audit["passed"],
        },
        "claim_boundary": audit["claim_boundary"],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
