#!/usr/bin/env python3
"""Download exactly the archives in a frozen B2D expansion plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Mapping, Sequence
from urllib.parse import quote


DEFAULT_ENDPOINTS = (
    "https://hf-mirror.com/datasets/rethinklab/Bench2Drive/resolve/main",
    "https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/main",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--endpoint", action="append", default=[])
    return parser.parse_args()


def validate_plan(path: Path) -> tuple[Mapping[str, object], Sequence[Mapping[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "b2d-expansion-plan/v1" or payload.get(
        "status"
    ) != "pre_download_plan_not_a_training_manifest":
        raise ValueError("input is not a frozen pre-download B2D expansion plan")
    additions = payload.get("additions")
    if not isinstance(additions, list) or len(additions) != 50:
        raise ValueError("expansion plan must contain exactly 50 additions")
    paths = [str(row.get("archive_path", "")) for row in additions]
    if len(paths) != len(set(paths)) or any(Path(value).name != value for value in paths):
        raise ValueError("expansion archive paths must be unique basenames")
    if any(int(row.get("size_bytes", 0)) <= 0 for row in additions):
        raise ValueError("expansion archive sizes must be positive")
    planned_bytes = int(payload["budget"]["new_archive_bytes"])
    if sum(int(row["size_bytes"]) for row in additions) != planned_bytes:
        raise ValueError("expansion archive byte budget differs from additions")
    return payload, additions


def download_one(
    row: Mapping[str, object], archive_dir: Path, endpoints: Sequence[str]
) -> Dict[str, object]:
    archive_name = str(row["archive_path"])
    expected_size = int(row["size_bytes"])
    final_path = archive_dir / archive_name
    partial_path = archive_dir / ("." + archive_name + ".partial")
    if final_path.exists():
        if not final_path.is_file() or final_path.stat().st_size != expected_size:
            raise RuntimeError("existing archive has wrong size: %s" % final_path)
        status = "existing_exact_size"
    else:
        errors = []
        for endpoint in endpoints:
            url = "%s/%s" % (endpoint.rstrip("/"), quote(archive_name))
            try:
                subprocess.run(
                    [
                        "curl",
                        "-4",
                        "--silent",
                        "--show-error",
                        "--fail",
                        "--location",
                        "--retry",
                        "4",
                        "--retry-delay",
                        "5",
                        "--connect-timeout",
                        "20",
                        "--speed-limit",
                        "1024",
                        "--speed-time",
                        "30",
                        "--continue-at",
                        "-",
                        "--output",
                        str(partial_path),
                        url,
                    ],
                    check=True,
                )
                if partial_path.stat().st_size != expected_size:
                    raise RuntimeError(
                        "downloaded size %d, expected %d"
                        % (partial_path.stat().st_size, expected_size)
                    )
                os.replace(partial_path, final_path)
                status = "downloaded"
                break
            except Exception as error:
                errors.append("%s: %s" % (endpoint, error))
        else:
            raise RuntimeError("all endpoints failed for %s: %s" % (archive_name, errors))
    return {
        "archive_path": archive_name,
        "local_path": str(final_path.resolve()),
        "size_bytes": final_path.stat().st_size,
        "sha256": sha256_file(final_path),
        "split": row["split"],
        "status": status,
    }


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.receipt.exists():
        raise FileExistsError("refusing to overwrite receipt %s" % args.receipt)
    plan, additions = validate_plan(args.plan)
    endpoints = tuple(args.endpoint) if args.endpoint else DEFAULT_ENDPOINTS
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    print(
        "Downloading %d frozen routes (%.3f GiB) with %d workers"
        % (
            len(additions),
            int(plan["budget"]["new_archive_bytes"]) / 2**30,
            args.workers,
        ),
        flush=True,
    )
    records = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, row, args.archive_dir, endpoints): row
            for row in additions
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                record = future.result()
                records.append(record)
                print(
                    "[OK] %s %d" % (record["archive_path"], record["size_bytes"]),
                    flush=True,
                )
            except Exception as error:
                failures.append(
                    {"archive_path": row["archive_path"], "error": str(error)}
                )
                print("[FAIL] %s %s" % (row["archive_path"], error), flush=True)
    if failures:
        raise RuntimeError("download failures: %s" % failures)
    records.sort(key=lambda row: str(row["archive_path"]))
    receipt = {
        "schema_version": "b2d-expansion-download-receipt/v1",
        "plan_path": str(args.plan.resolve()),
        "plan_sha256": sha256_file(args.plan),
        "archive_count": len(records),
        "archive_bytes": sum(int(row["size_bytes"]) for row in records),
        "all_sizes_match_plan": True,
        "endpoints": list(endpoints),
        "records": records,
        "extraction_performed": False,
        "training_performed": False,
    }
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "receipt": str(args.receipt.resolve()),
        "archive_count": receipt["archive_count"],
        "archive_bytes": receipt["archive_bytes"],
    }, indent=2, sort_keys=True))
    print("B2D_EXPANSION_DOWNLOAD_OK=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
