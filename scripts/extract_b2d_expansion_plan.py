#!/usr/bin/env python3
"""Safely extract a frozen B2D expansion and optionally remove its archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Dict, Mapping, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--download-receipt", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--delete-archives", action="store_true")
    return parser.parse_args()


def load_inputs(
    plan_path: Path, download_receipt_path: Path
) -> tuple[Mapping[str, object], Sequence[Mapping[str, object]], Dict[str, Mapping[str, object]]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "b2d-expansion-plan/v1":
        raise ValueError("invalid B2D expansion plan")
    additions = plan.get("additions")
    if not isinstance(additions, list) or len(additions) != 50:
        raise ValueError("B2D expansion plan must contain 50 additions")
    download = json.loads(download_receipt_path.read_text(encoding="utf-8"))
    if download.get("schema_version") != "b2d-expansion-download-receipt/v1":
        raise ValueError("invalid B2D download receipt")
    if download.get("plan_sha256") != sha256_file(plan_path):
        raise ValueError("download receipt and expansion plan differ")
    records = download.get("records")
    if not isinstance(records, list) or len(records) != len(additions):
        raise ValueError("download receipt count differs")
    by_name = {str(row["archive_path"]): row for row in records}
    if len(by_name) != len(records):
        raise ValueError("download receipt contains duplicate archives")
    return plan, additions, by_name


def safe_members(
    archive: tarfile.TarFile, expected_root: str
) -> tuple[Sequence[tarfile.TarInfo], int, int]:
    members = archive.getmembers()
    if not members:
        raise RuntimeError("archive is empty")
    file_count = 0
    file_bytes = 0
    for member in members:
        normalized = member.name
        while normalized.startswith("./"):
            normalized = normalized[2:]
        path = PurePosixPath(normalized)
        if (
            not normalized
            or path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[0] != expected_root
        ):
            raise RuntimeError("unsafe or unexpected archive member: %r" % member.name)
        if member.issym() or member.islnk() or member.isdev():
            raise RuntimeError("links/devices are forbidden in B2D archives")
        if member.isfile():
            file_count += 1
            file_bytes += int(member.size)
    if file_count == 0:
        raise RuntimeError("archive contains no regular files")
    return members, file_count, file_bytes


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.parent / (path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.receipt.exists():
        raise FileExistsError("refusing to overwrite extraction receipt %s" % args.receipt)
    plan, additions, downloaded = load_inputs(args.plan, args.download_receipt)
    args.route_root.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.receipt.parent / (args.receipt.name + ".partial")
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = {
            str(row["archive_path"]): row for row in progress.get("records", [])
        }
    else:
        completed = {}

    for index, row in enumerate(additions, 1):
        archive_name = str(row["archive_path"])
        expected_root = archive_name[:-7]
        archive_path = args.archive_dir / archive_name
        final_path = args.route_root / expected_root
        download_row = downloaded.get(archive_name)
        if download_row is None:
            raise RuntimeError("archive absent from download receipt: %s" % archive_name)
        if archive_name in completed:
            if not final_path.is_dir():
                raise RuntimeError("progress says extracted but route is absent: %s" % final_path)
            print("[SKIP %d/50] %s" % (index, archive_name), flush=True)
            continue
        if final_path.exists():
            raise FileExistsError("untracked extracted route already exists: %s" % final_path)
        if (
            not archive_path.is_file()
            or archive_path.stat().st_size != int(download_row["size_bytes"])
            or sha256_file(archive_path) != download_row["sha256"]
        ):
            raise RuntimeError("archive differs from download receipt: %s" % archive_name)

        temporary = args.route_root / (
            ".extract-" + expected_root + ".tmp-" + uuid.uuid4().hex
        )
        temporary.mkdir(parents=False, exist_ok=False)
        try:
            with tarfile.open(archive_path, "r:gz") as archive:
                members, member_file_count, member_file_bytes = safe_members(
                    archive, expected_root
                )
                archive.extractall(path=temporary, members=members)
            extracted_root = temporary / expected_root
            if not extracted_root.is_dir():
                raise RuntimeError("archive did not create expected route root")
            other_roots = [path.name for path in temporary.iterdir() if path != extracted_root]
            if other_roots:
                raise RuntimeError("archive created unexpected top-level paths: %s" % other_roots)
            actual_files = [path for path in extracted_root.rglob("*") if path.is_file()]
            actual_bytes = sum(path.stat().st_size for path in actual_files)
            if len(actual_files) != member_file_count or actual_bytes != member_file_bytes:
                raise RuntimeError("extracted file count/bytes differ from tar metadata")
            extracted_root.rename(final_path)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

        completed[archive_name] = {
            "archive_path": archive_name,
            "archive_sha256": download_row["sha256"],
            "archive_size_bytes": int(download_row["size_bytes"]),
            "route_path": str(final_path.resolve()),
            "route_file_count": member_file_count,
            "route_file_bytes": actual_bytes,
            "split": row["split"],
        }
        progress = {
            "schema_version": "b2d-expansion-extraction-progress/v1",
            "plan_sha256": sha256_file(args.plan),
            "records": [completed[name] for name in sorted(completed)],
        }
        write_json_atomic(progress_path, progress)
        print("[OK %d/50] %s" % (index, archive_name), flush=True)

    records = [completed[str(row["archive_path"])] for row in additions]
    receipt = {
        "schema_version": "b2d-expansion-extraction-receipt/v1",
        "plan_path": str(args.plan.resolve()),
        "plan_sha256": sha256_file(args.plan),
        "download_receipt_path": str(args.download_receipt.resolve()),
        "download_receipt_sha256": sha256_file(args.download_receipt),
        "route_count": len(records),
        "route_file_count": sum(int(row["route_file_count"]) for row in records),
        "route_file_bytes": sum(int(row["route_file_bytes"]) for row in records),
        "records": records,
        "archive_deletion": {
            "requested": args.delete_archives,
            "performed": False,
            "deleted_count": 0,
            "deleted_bytes": 0,
        },
        "infos_rebuilt": False,
        "training_performed": False,
    }
    write_json_atomic(args.receipt, receipt)

    if args.delete_archives:
        deleted_bytes = 0
        for row in additions:
            archive_name = str(row["archive_path"])
            archive_path = args.archive_dir / archive_name
            if not archive_path.is_file():
                raise RuntimeError("archive vanished before audited deletion: %s" % archive_path)
            deleted_bytes += archive_path.stat().st_size
            archive_path.unlink()
        receipt["archive_deletion"] = {
            "requested": True,
            "performed": True,
            "deleted_count": len(additions),
            "deleted_bytes": deleted_bytes,
        }
        write_json_atomic(args.receipt, receipt)
    progress_path.unlink(missing_ok=True)
    print(json.dumps({
        "receipt": str(args.receipt.resolve()),
        "route_count": receipt["route_count"],
        "route_file_bytes": receipt["route_file_bytes"],
        "archive_deletion": receipt["archive_deletion"],
    }, indent=2, sort_keys=True))
    print("B2D_EXPANSION_EXTRACTION_OK=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
