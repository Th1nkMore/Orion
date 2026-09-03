#!/usr/bin/env python3
"""Fail-closed offline audit of the Stage2-L v6 matched dataset contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from uq_estimator.stage2l_matched_objective import audit_matched_training_records


PROTOCOL_SCHEMA = "orion.stage2l_uq_language_grounding_protocol.v2"
REPORT_SCHEMA = "orion.stage2l_v6_dataset_contract_audit.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(records_path: Path, protocol_path: Path) -> Dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported Stage2-L v6 protocol")
    locks = protocol.get("launch_locks", {})
    if locks.get("stage2l_pilot_training_allowed") is not False:
        raise ValueError("offline audit requires pilot training to remain locked")
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts = audit_matched_training_records(records)
    if counts["record_count"] != counts["matched_group_count"] * 20:
        raise RuntimeError("matched group arithmetic is inconsistent")
    if counts["optimizer_step_count_per_epoch"] != counts["matched_group_count"]:
        raise RuntimeError("optimizer boundary arithmetic is inconsistent")
    return {
        "schema": REPORT_SCHEMA,
        "status": "passed_offline_dataset_contract_training_still_locked",
        "records": {"path": str(records_path.resolve()), "sha256": _sha256(records_path)},
        "protocol": {"path": str(protocol_path.resolve()), "sha256": _sha256(protocol_path)},
        "counts": counts,
        "pilot_training_allowed": False,
        "claim_boundary": "Offline dataset/optimizer-boundary audit only; no model training or evaluation evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Stage2-L v6 audit report")
    report = audit_dataset(args.records.resolve(), args.protocol.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
