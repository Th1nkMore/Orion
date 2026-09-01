#!/usr/bin/env python3
"""Freeze a development-screen batch and its hazard/no-hazard route pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


SCHEMA = "orion.scenario_factory.batch.v1"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_scenarios(root: ET.Element) -> int:
    removed = 0
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "scenario":
                parent.remove(child)
                removed += 1
    return removed


def nohazard_bytes(source: Path) -> bytes:
    tree = ET.parse(source)
    scenarios = tree.getroot().findall(".//scenario")
    removed = remove_scenarios(tree.getroot())
    if not scenarios or removed != len(scenarios):
        raise RuntimeError(
            "route %s has no removable scenario or scenario count changed" % source
        )
    return ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)


def _candidate_by_index(payload: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("candidate manifest has no candidates list")
    result: Dict[int, Mapping[str, Any]] = {}
    for candidate in candidates:
        index = int(candidate["route_index"])
        if index in result:
            raise ValueError("candidate manifest duplicates route %d" % index)
        result[index] = candidate
    return result


def build_batch(
    *,
    candidate_manifest: Path,
    source_routes_dir: Path,
    baseline_source: Optional[Path],
    protocol: Path,
    out_dir: Path,
    run_id: str,
    route_indices: Sequence[int],
    limit: Optional[int],
    writes_performed: bool,
    split: str = "development_screen",
) -> Dict[str, Any]:
    if not run_id or any(character.isspace() for character in run_id):
        raise ValueError("run-id must be non-empty and contain no whitespace")
    candidate_payload = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    if split not in ("development_screen", "locked_test"):
        raise ValueError("split must be development_screen or locked_test")
    selection_inputs = candidate_payload.get("selection_inputs", {})
    published_outcomes_used = bool(
        selection_inputs.get(
            "published_orion_outcomes_used",
            candidate_payload.get("clean_baseline_required", False),
        )
    )
    learned_uq_outcomes_used = bool(
        selection_inputs.get("learned_uq_outcomes_used", False)
    )
    stage2_outcomes_used = bool(selection_inputs.get("stage2_outcomes_used", False))
    if split == "locked_test" and (
        published_outcomes_used or learned_uq_outcomes_used or stage2_outcomes_used
    ):
        raise ValueError(
            "locked_test selection must not use published ORION, learned-UQ, or Stage2 outcomes"
        )
    by_index = _candidate_by_index(candidate_payload)
    development_failure_candidates_allowed = bool(
        candidate_payload.get("development_failure_candidates_allowed", False)
    )
    if route_indices:
        selected = []
        for index in route_indices:
            if int(index) not in by_index:
                raise ValueError("route %d is absent from candidate manifest" % index)
            selected.append(by_index[int(index)])
    else:
        candidates = list(candidate_payload.get("candidates", []))
        selected = candidates[:limit] if limit is not None else candidates
    if not selected:
        raise ValueError("batch selection is empty")
    if len({int(row["route_index"]) for row in selected}) != len(selected):
        raise ValueError("batch selection contains duplicate routes")

    route_rows: List[Dict[str, Any]] = []
    generated: List[tuple] = []
    for candidate in selected:
        clean = candidate.get("clean_baseline", {})
        if split == "development_screen" and clean.get("valid") is not True:
            if not (
                development_failure_candidates_allowed
                and candidate.get("development_selection_role")
                == "published_failure_hard_case"
            ):
                raise ValueError(
                    "route %s lacks the required published clean-valid prior"
                    % candidate["route_index"]
                )
        index = int(candidate["route_index"])
        source = source_routes_dir / str(candidate["source_xml"])
        if not source.is_file():
            raise FileNotFoundError(source)
        hazard_payload = source.read_bytes()
        nohazard_payload = nohazard_bytes(source)
        hazard_path = out_dir / ("route_%d_hazard.xml" % index)
        nohazard_path = out_dir / ("route_%d_nohazard.xml" % index)
        generated.append((hazard_path, hazard_payload, nohazard_path, nohazard_payload))
        route_rows.append(
            {
                "route_index": index,
                "xml_route_id": str(candidate["xml_route_id"]),
                "town": candidate["town"],
                "scenario_type": candidate["scenario_type"],
                "scenario_name": candidate["scenario_name"],
                "screen_role": candidate["screen_role"],
                "priority": int(candidate["priority"]),
                "trigger_progress": float(candidate["trigger_progress"]),
                "route_length_m": float(candidate["route_length_m"]),
                "published_clean_prior": clean,
                "development_selection_role": candidate.get(
                    "development_selection_role", "published_clean_valid"
                ),
                "formal_split": candidate.get("formal_split"),
                "source_xml": {
                    "path": str(source.resolve()),
                    "sha256": sha256_bytes(hazard_payload),
                },
                "hazard_xml": {
                    "path": str(hazard_path.resolve()),
                    "sha256": sha256_bytes(hazard_payload),
                },
                "nohazard_xml": {
                    "path": str(nohazard_path.resolve()),
                    "sha256": sha256_bytes(nohazard_payload),
                },
                "initial_condition": "clean_off",
                "review_status": "awaiting_current_environment_replay",
            }
        )

    payload = {
        "schema": SCHEMA,
        "status": (
            "prepared_no_jobs_submitted"
            if writes_performed
            else "dry_run_no_files_written_no_jobs_submitted"
        ),
        "run_id": run_id,
        "split": split,
        "route_count": len(route_rows),
        "routes": route_rows,
        "runtime_contract": {
            "condition": "clean_off",
            "variant": "hazard",
            "agent_config": "adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py",
            "stage2_spatial_uq_source": "disabled",
            "stage1_adapter_control_influence": False,
            "legacy_density_uq": False,
            "risk_mode": "off",
            "planning_response": "off",
            "carla_quality": "Epic",
            "gpus_per_route": 1,
            "cpus_per_route": 2,
            "memory_per_route": "192G",
        },
        "lineage": {
            "candidate_manifest": {
                "path": str(candidate_manifest.resolve()),
                "sha256": sha256_file(candidate_manifest),
            },
            "scenario_factory_protocol": {
                "path": str(protocol.resolve()),
                "sha256": sha256_file(protocol),
            },
        },
        "postprocessing": {
            "event_package_schema": "orion.scenario_event_package.v1",
            "initial_review_status": "pending_human_review",
            "stage1_uq": "offline precomputation only; never affects clean replay control",
        },
        "audit": {
            "writes_performed": writes_performed,
            "jobs_submitted": False,
            "selection_uses_published_orion_outcomes": published_outcomes_used,
            "selection_uses_learned_uq_outcomes": learned_uq_outcomes_used,
            "selection_uses_stage2_outcomes": stage2_outcomes_used,
            "development_failure_candidates_allowed": (
                split == "development_screen"
                and development_failure_candidates_allowed
            ),
            "eligible_for_locked_test_claim": split == "locked_test",
        },
        "claim_boundary": (
            (
                "Locked-test routes were frozen without published ORION, learned-UQ, "
                "or Stage2 outcomes. Post-freeze removal is permitted only for attested "
                "runtime/environment invalidity."
            )
            if split == "locked_test"
            else (
                "Development screening batch may use published ORION priors; not an "
                "untouched test set and not learned-UQ evidence."
            )
        ),
    }
    if baseline_source is not None:
        payload["lineage"]["published_orion_baseline"] = {
            "path": str(baseline_source.resolve()),
            "sha256": sha256_file(baseline_source),
            "used_for_selection": published_outcomes_used,
        }

    if writes_performed:
        if out_dir.exists() and any(out_dir.iterdir()):
            raise FileExistsError("refusing to overwrite non-empty batch directory")
        out_dir.mkdir(parents=True, exist_ok=True)
        for hazard_path, hazard_payload, nohazard_path, nohazard_payload in generated:
            hazard_path.write_bytes(hazard_payload)
            nohazard_path.write_bytes(nohazard_payload)
        manifest_path = out_dir / "batch_manifest.json"
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--source-routes-dir", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--route-index", type=int, action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--split",
        choices=("development_screen", "locked_test"),
        default="development_screen",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    payload = build_batch(
        candidate_manifest=args.candidate_manifest,
        source_routes_dir=args.source_routes_dir,
        baseline_source=args.baseline_source,
        protocol=args.protocol,
        out_dir=args.out_dir,
        run_id=args.run_id,
        route_indices=args.route_index,
        limit=args.limit,
        writes_performed=not args.dry_run,
        split=args.split,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
