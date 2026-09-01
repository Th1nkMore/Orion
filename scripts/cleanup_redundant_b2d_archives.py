#!/usr/bin/env python3
"""Delete redundant official B2D archives after exact extraction checks.

Only direct ``*.tar.gz`` children of the declared archive directory are in
scope.  Every archive must have an exactly named extracted route directory,
and the expected counts must match, before ``--execute`` can remove anything.
An audit/tombstone JSON is written after the checks and updated after deletion.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument(
        "--official-source",
        default="https://huggingface.co/datasets/rethinklab/Bench2Drive",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_count <= 0:
        raise ValueError("--expected-count must be positive")
    if not args.archive_root.is_dir() or not args.extracted_root.is_dir():
        raise FileNotFoundError("archive and extracted roots must both exist")

    archives = sorted(args.archive_root.glob("*.tar.gz"))
    extracted = {path.name for path in args.extracted_root.iterdir() if path.is_dir()}
    expected = {path.name[:-7] for path in archives}
    unmatched_archives = sorted(expected - extracted)
    unmatched_extracted = sorted(extracted - expected)
    if len(archives) != args.expected_count:
        raise RuntimeError(
            f"refusing cleanup: found {len(archives)} archives, "
            f"expected {args.expected_count}"
        )
    if unmatched_archives:
        raise RuntimeError(
            f"refusing cleanup: archives lack extracted directories: {unmatched_archives}"
        )

    records = [
        {
            "archive_name": path.name,
            "archive_path": str(path),
            "extracted_directory": str(args.extracted_root / path.name[:-7]),
            "size_bytes": path.stat().st_size,
        }
        for path in archives
    ]
    payload = {
        "schema_version": "redundant-b2d-archive-cleanup/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry_run",
        "official_source": args.official_source,
        "archive_root": str(args.archive_root),
        "extracted_root": str(args.extracted_root),
        "archive_count": len(records),
        "archive_bytes": sum(record["size_bytes"] for record in records),
        "all_archives_have_matching_extracted_directories": True,
        "extracted_directories_without_archives": unmatched_extracted,
        "records": records,
        "deletion": {
            "performed": False,
            "deleted_count": 0,
            "deleted_bytes": 0,
            "remaining_archive_count": len(records),
        },
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.execute:
        for path in archives:
            path.unlink()
        remaining = sorted(args.archive_root.glob("*.tar.gz"))
        if remaining:
            raise RuntimeError(f"cleanup incomplete; archives remain: {remaining}")
        payload["deletion"] = {
            "performed": True,
            "deleted_count": len(records),
            "deleted_bytes": sum(record["size_bytes"] for record in records),
            "remaining_archive_count": 0,
        }
        payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.audit_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(json.dumps({
        "audit_output": str(args.audit_output),
        "mode": payload["mode"],
        "archive_count": payload["archive_count"],
        "archive_bytes": payload["archive_bytes"],
        "deletion": payload["deletion"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
