#!/usr/bin/env python3
"""Create versioned records with the hash-bound ORION speedometer reading."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA = "orion.stage2l_v11_route_context_upgrade.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _load_reference(reference: Mapping[str, Any], name: str) -> Tuple[Dict[str, Any], Path]:
    path = Path(str(reference.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError("%s is missing: %s" % (name, path))
    if _sha256(path) != reference.get("sha256"):
        raise ValueError("%s SHA-256 differs: %s" % (name, path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("%s must contain a JSON object" % name)
    return payload, path


def _bound_speed(row: Mapping[str, Any]) -> Tuple[float, Dict[str, str]]:
    relevance = row.get("provenance", {}).get("relevance_supervision") or {}
    geometry_reference = relevance.get("geometry_manifest") or {}
    geometry, geometry_path = _load_reference(
        geometry_reference, "task-relevance geometry manifest"
    )
    meta_reference = geometry.get("source_meta") or {}
    meta, meta_path = _load_reference(meta_reference, "source frame metadata")
    speed = float(meta["speed"])
    if not math.isfinite(speed):
        raise ValueError("source frame speedometer reading is invalid")
    old_payload = row["model_input"]["route_context"]["payload"]
    if int(meta["command"]) != int(old_payload["command"]):
        raise ValueError("source frame command differs from route context")
    if _canonical(meta["plan"]) != _canonical(
        old_payload["orion_unmodified_plan_right_forward_m"]
    ):
        raise ValueError("source frame plan differs from route context")
    return speed, {
        "geometry_manifest_path": str(geometry_path.resolve()),
        "geometry_manifest_sha256": _sha256(geometry_path),
        "source_meta_path": str(meta_path.resolve()),
        "source_meta_sha256": _sha256(meta_path),
    }


def upgrade_records(source_records: Path, output_dir: Path) -> Dict[str, Any]:
    rows = [
        json.loads(line)
        for line in source_records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("source records are empty")
    speed_by_group: Dict[str, float] = {}
    binding_by_group: Dict[str, Dict[str, str]] = {}
    for row in rows:
        group_id = str(row["counterfactual"]["group_id"])
        speed, binding = _bound_speed(row)
        if group_id in speed_by_group and not math.isclose(
            speed_by_group[group_id], speed, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "matched group resolves to multiple speedometer readings"
            )
        if group_id in binding_by_group and binding_by_group[group_id] != binding:
            raise ValueError("matched group resolves to multiple source metadata files")
        speed_by_group[group_id] = speed
        binding_by_group[group_id] = binding

    upgraded: List[Dict[str, Any]] = []
    for source_row in rows:
        row = copy.deepcopy(source_row)
        group_id = str(row["counterfactual"]["group_id"])
        source_row_sha = hashlib.sha256(_canonical(source_row)).hexdigest()
        route_context = row["model_input"]["route_context"]
        payload = copy.deepcopy(route_context["payload"])
        if "ego_state" in payload:
            raise ValueError("source record already contains ego_state")
        payload["ego_state"] = {
            "speedometer_mps": speed_by_group[group_id]
        }
        route_context.clear()
        route_context.update({
            "schema": "orion.route_context.v2",
            "payload": payload,
            "sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        })
        provenance = row.setdefault("provenance", {})
        provenance["route_context_v11_upgrade"] = {
            "schema": SCHEMA,
            "source_record_sha256": source_row_sha,
            "route_context_only_change": True,
            **binding_by_group[group_id],
        }
        upgraded.append(row)

    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "records.jsonl"
    temporary = output_dir / ".records.jsonl.tmp"
    temporary.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in upgraded
        ),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    speeds = list(speed_by_group.values())
    splits = {}
    events = set()
    for row in upgraded:
        split = str(row["split"])
        group_id = str(row["counterfactual"]["group_id"])
        splits.setdefault(split, set()).add(group_id)
        events.add(str(row["event_id"]))
    report = {
        "schema": SCHEMA,
        "status": "complete_route_context_v2_records_not_yet_tensor_audited",
        "source": {
            "path": str(source_records.resolve()),
            "sha256": _sha256(source_records),
            "record_count": len(rows),
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": _sha256(output_path),
            "record_count": len(upgraded),
        },
        "event_count": len(events),
        "group_count": len(speed_by_group),
        "groups_by_split": {
            split: len(group_ids) for split, group_ids in sorted(splits.items())
        },
        "ego_speedometer_mps": {
            "minimum": min(speeds),
            "maximum": max(speeds),
            "mean": sum(speeds) / len(speeds),
        },
        "checks": {
            "source_record_count_preserved": len(rows) == len(upgraded),
            "one_speed_per_matched_group": True,
            "speed_bound_through_hashed_geometry_and_meta": True,
            "visual_u_and_supervision_references_unchanged": True,
            "desired_speed_route_progress_ttc_and_outcome_not_added": True,
        },
        "claim_boundary": (
            "Route-context version upgrade only. It does not validate U, R, "
            "language understanding, planning or closed-loop safety."
        ),
    }
    report_path = output_dir / "upgrade_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = upgrade_records(
        args.source_records.resolve(), args.output_dir.resolve()
    )
    print(json.dumps({
        "event_count": report["event_count"],
        "group_count": report["group_count"],
        "record_count": report["output"]["record_count"],
        "output": report["output"]["path"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
