#!/usr/bin/env python3
"""Plan a deterministic Bench2Drive expansion without downloading any data.

The output is deliberately an expansion *plan*, not a training manifest.  A
real route manifest must be rebuilt from the expanded infos file after every
archive has been downloaded, extracted, and validated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


SCHEMA_VERSION = "b2d-expansion-plan/v1"
DEFAULT_SOURCE_URL = "https://huggingface.co/datasets/rethinklab/Bench2Drive"
CATALOG_RE = re.compile(
    r"^(?P<scenario>.+)_(?P<town>Town\d+(?:HD)?)_"
    r"(?P<route>Route\d+)_Weather(?P<weather>\d+)\.tar\.gz$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seeded_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def parse_route_name(path: str) -> Dict[str, object]:
    name = Path(path).name
    match = CATALOG_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid Bench2Drive archive name: {path!r}")
    parsed: Dict[str, object] = dict(match.groupdict())
    parsed["weather"] = int(str(parsed["weather"]))
    parsed["path"] = path
    parsed["folder"] = f"v1/{name[:-7]}"
    parsed["canonical_route_key"] = f"{parsed['town']}/{parsed['route']}"
    return parsed


def load_catalog(path: Path) -> Tuple[List[Dict[str, object]], List[str]]:
    rows: List[Dict[str, object]] = []
    invalid: List[str] = []
    seen_paths = set()
    seen_routes = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, values in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if len(values) != 2:
                raise ValueError(f"catalog line {line_number} must have two columns")
            archive_path, size_text = values
            if archive_path in seen_paths:
                raise ValueError(f"duplicate catalog path: {archive_path}")
            seen_paths.add(archive_path)
            try:
                parsed = parse_route_name(archive_path)
            except ValueError:
                invalid.append(archive_path)
                continue
            size = int(size_text)
            if size <= 0:
                raise ValueError(f"non-positive catalog size for {archive_path}")
            canonical = str(parsed["canonical_route_key"])
            if canonical in seen_routes:
                raise ValueError(f"duplicate canonical route in catalog: {canonical}")
            seen_routes.add(canonical)
            parsed["size_bytes"] = size
            rows.append(parsed)
    if not rows:
        raise ValueError("catalog contains no valid route archives")
    return rows, invalid


def load_existing_manifest(path: Path) -> Tuple[Dict[str, List[str]], List[Dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    split_payload = payload.get("splits")
    if not isinstance(split_payload, Mapping):
        raise ValueError("existing manifest has no splits mapping")
    splits: Dict[str, List[str]] = {}
    parsed: List[Dict[str, object]] = []
    for split in ("train", "validation", "calibration", "held_out"):
        route_ids = split_payload.get(split, {}).get("route_ids")
        if not isinstance(route_ids, list) or not all(isinstance(x, str) for x in route_ids):
            raise ValueError(f"existing manifest split {split!r} has invalid route_ids")
        splits[split] = list(route_ids)
        for route_id in route_ids:
            archive = Path(route_id).name + ".tar.gz"
            row = parse_route_name(archive)
            row["split"] = split
            parsed.append(row)
    canonical = [str(row["canonical_route_key"]) for row in parsed]
    if len(canonical) != len(set(canonical)):
        raise ValueError("existing manifest is not canonical-route disjoint")
    return splits, parsed


def counter_dict(values: Iterable[str]) -> Dict[str, int]:
    return dict(sorted(Counter(values).items()))


def route_brief(row: Mapping[str, object], split: str) -> Dict[str, object]:
    return {
        "archive_path": row["path"],
        "canonical_route_key": row["canonical_route_key"],
        "folder": row["folder"],
        "scenario": row["scenario"],
        "size_bytes": row["size_bytes"],
        "split": split,
        "town": row["town"],
        "weather": row["weather"],
    }


def select_balanced_routes(
    candidates: Sequence[Mapping[str, object]],
    *,
    count: int,
    scenario_counts: MutableMapping[str, int],
    town_counts: MutableMapping[str, int],
    weather_counts: MutableMapping[int, int],
    seed: int,
) -> List[Mapping[str, object]]:
    remaining = list(candidates)
    selected: List[Mapping[str, object]] = []
    if count > len(remaining):
        raise ValueError(f"requested {count} routes from only {len(remaining)} candidates")
    for _ in range(count):
        # Scenario coverage is primary, town coverage secondary, and weather
        # coverage tertiary.  Do not optimize archive size: it is reported as
        # a budget only, because selecting short clips would bias the data.
        chosen = min(
            remaining,
            key=lambda row: (
                scenario_counts.get(str(row["scenario"]), 0),
                town_counts.get(str(row["town"]), 0),
                weather_counts.get(int(row["weather"]), 0),
                seeded_key(seed, str(row["path"])),
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
        scenario = str(chosen["scenario"])
        town = str(chosen["town"])
        weather = int(chosen["weather"])
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        town_counts[town] = town_counts.get(town, 0) + 1
        weather_counts[weather] = weather_counts.get(weather, 0) + 1
    return selected


def assign_eval_splits(
    selected: Sequence[Mapping[str, object]],
    existing_rows: Sequence[Mapping[str, object]],
    *,
    validation_count: int,
    calibration_count: int,
    seed: int,
) -> Dict[str, List[Mapping[str, object]]]:
    remaining = list(selected)
    assigned: Dict[str, List[Mapping[str, object]]] = {
        "train": [],
        "validation": [],
        "calibration": [],
    }
    quotas = {"validation": validation_count, "calibration": calibration_count}
    per_split_scenario = {
        split: Counter(
            str(row["scenario"])
            for row in existing_rows
            if row.get("split") == split
        )
        for split in quotas
    }
    per_split_town = {
        split: Counter(
            str(row["town"])
            for row in existing_rows
            if row.get("split") == split
        )
        for split in quotas
    }
    # Alternate the two evaluation splits so neither gets first choice of all
    # strata.  The seeded order is frozen and recorded in the plan.
    order = sorted(quotas, key=lambda split: seeded_key(seed, split))
    while any(quotas.values()):
        for split in order:
            if quotas[split] == 0:
                continue
            chosen = min(
                remaining,
                key=lambda row: (
                    per_split_scenario[split][str(row["scenario"])],
                    per_split_town[split][str(row["town"])],
                    seeded_key(seed, f"{split}:{row['path']}"),
                ),
            )
            assigned[split].append(chosen)
            remaining.remove(chosen)
            per_split_scenario[split][str(chosen["scenario"])] += 1
            per_split_town[split][str(chosen["town"])] += 1
            quotas[split] -= 1
    assigned["train"] = remaining
    return assigned


def build_plan(args: argparse.Namespace) -> Dict[str, object]:
    catalog, invalid_catalog_paths = load_catalog(args.catalog)
    existing_splits, existing_rows = load_existing_manifest(args.existing_manifest)
    catalog_by_canonical = {
        str(row["canonical_route_key"]): row for row in catalog
    }
    existing_canonical = {str(row["canonical_route_key"]) for row in existing_rows}
    missing_existing = sorted(existing_canonical - set(catalog_by_canonical))
    if missing_existing:
        raise ValueError(f"existing routes absent from official catalog: {missing_existing}")

    held_out_towns = {
        str(row["town"]) for row in existing_rows if row["split"] == "held_out"
    }
    nonheld_towns = {
        str(row["town"]) for row in existing_rows if row["split"] != "held_out"
    }
    if held_out_towns & nonheld_towns:
        raise ValueError("existing held-out towns already overlap non-held-out splits")
    if args.new_heldout_town in held_out_towns | nonheld_towns:
        raise ValueError("new held-out town must not occur in the existing manifest")

    available = [
        row for row in catalog
        if str(row["canonical_route_key"]) not in existing_canonical
    ]
    heldout_additions = [
        row for row in available if row["town"] == args.new_heldout_town
    ]
    heldout_additions.sort(key=lambda row: str(row["path"]))
    if len(heldout_additions) != args.heldout_additions:
        raise ValueError(
            f"expected exactly {args.heldout_additions} routes in new held-out town "
            f"{args.new_heldout_town}, found {len(heldout_additions)}"
        )

    forbidden_nonheld_towns = held_out_towns | {args.new_heldout_town}
    nonheld_candidates = [
        row for row in available if str(row["town"]) not in forbidden_nonheld_towns
    ]
    scenario_counts: MutableMapping[str, int] = Counter(
        str(row["scenario"]) for row in existing_rows + heldout_additions
    )
    town_counts: MutableMapping[str, int] = Counter(
        str(row["town"]) for row in existing_rows if row["split"] != "held_out"
    )
    weather_counts: MutableMapping[int, int] = Counter(
        int(row["weather"]) for row in existing_rows + heldout_additions
    )
    nonheld_count = args.total_additions - args.heldout_additions
    nonheld_selected = select_balanced_routes(
        nonheld_candidates,
        count=nonheld_count,
        scenario_counts=scenario_counts,
        town_counts=town_counts,
        weather_counts=weather_counts,
        seed=args.seed,
    )
    assigned = assign_eval_splits(
        nonheld_selected,
        existing_rows,
        validation_count=args.validation_additions,
        calibration_count=args.calibration_additions,
        seed=args.seed,
    )
    expected_train = (
        args.total_additions
        - args.heldout_additions
        - args.validation_additions
        - args.calibration_additions
    )
    if len(assigned["train"]) != expected_train:
        raise AssertionError("train assignment quota mismatch")
    assigned["held_out"] = heldout_additions

    additions = [
        route_brief(row, split)
        for split in ("train", "validation", "calibration", "held_out")
        for row in sorted(assigned[split], key=lambda item: str(item["path"]))
    ]
    addition_canonical = [str(row["canonical_route_key"]) for row in additions]
    if len(addition_canonical) != len(set(addition_canonical)):
        raise AssertionError("selected additions are not canonical-route disjoint")
    if set(addition_canonical) & existing_canonical:
        raise AssertionError("selected additions overlap existing routes")

    resulting_rows = existing_rows + additions
    archive_bytes = sum(int(row["size_bytes"]) for row in additions)
    projected_int8_feature_bytes = round(
        args.current_feature_bytes
        * args.planned_reference_frames
        / args.current_reference_frames
        * args.int8_payload_ratio
    )
    resulting_split_counts = {
        split: len(existing_splits[split]) + sum(row["split"] == split for row in additions)
        for split in existing_splits
    }
    nonheld_addition_towns = {
        str(row["town"]) for row in additions if row["split"] != "held_out"
    }
    resulting_heldout_towns = held_out_towns | {args.new_heldout_town}

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pre_download_plan_not_a_training_manifest",
        "seed": args.seed,
        "source": {
            "catalog_path": str(args.catalog),
            "catalog_sha256": sha256_file(args.catalog),
            "official_dataset_url": args.source_url,
            "valid_archive_count": len(catalog),
            "invalid_catalog_paths_excluded": invalid_catalog_paths,
        },
        "existing": {
            "manifest_path": str(args.existing_manifest),
            "manifest_sha256": sha256_file(args.existing_manifest),
            "route_count": len(existing_rows),
            "split_counts": {split: len(rows) for split, rows in existing_splits.items()},
        },
        "selection_contract": {
            "canonical_identity": "Town/Route; weather and scenario are not split keys",
            "held_out_rule": (
                f"all official {args.new_heldout_town} routes go to held_out; "
                "existing and new held-out towns are forbidden in all other additions"
            ),
            "nonheld_selection_order": [
                "lowest resulting scenario count",
                "lowest resulting non-held-out town count",
                "lowest resulting weather count",
                "seeded SHA-256 tie break",
            ],
            "eval_assignment": (
                "alternate validation/calibration and minimize existing split-level "
                "scenario count, then town count, then seeded SHA-256"
            ),
            "download_or_extraction_performed": False,
            "must_rebuild_manifest_from_expanded_infos_before_training": True,
        },
        "budget": {
            "new_archive_bytes": archive_bytes,
            "new_archive_gib": archive_bytes / 2**30,
            "current_fp16_feature_bytes": args.current_feature_bytes,
            "current_reference_frames": args.current_reference_frames,
            "planned_reference_frames": args.planned_reference_frames,
            "int8_payload_ratio": args.int8_payload_ratio,
            "projected_int8_feature_payload_bytes": projected_int8_feature_bytes,
            "projected_int8_feature_payload_gib": projected_int8_feature_bytes / 2**30,
            "note": "feature estimate excludes shard metadata, targets, scales, and filesystem overhead",
        },
        "audit": {
            "addition_count": len(additions),
            "addition_split_counts": counter_dict(str(row["split"]) for row in additions),
            "resulting_split_counts": resulting_split_counts,
            "addition_scenario_counts": counter_dict(str(row["scenario"]) for row in additions),
            "resulting_scenario_counts": counter_dict(str(row["scenario"]) for row in resulting_rows),
            "addition_town_counts": counter_dict(str(row["town"]) for row in additions),
            "resulting_town_counts": counter_dict(str(row["town"]) for row in resulting_rows),
            "addition_weather_counts": counter_dict(str(row["weather"]) for row in additions),
            "resulting_weather_counts": counter_dict(str(row["weather"]) for row in resulting_rows),
            "held_out_towns": sorted(resulting_heldout_towns),
            "nonheld_addition_towns": sorted(nonheld_addition_towns),
            "heldout_town_overlap": sorted(resulting_heldout_towns & nonheld_addition_towns),
            "canonical_overlap_with_existing": [],
        },
        "additions": additions,
    }


def write_download_tsv(path: Path, plan: Mapping[str, object]) -> None:
    rows = plan["additions"]
    assert isinstance(rows, list)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("split", "archive_path", "size_bytes"))
        for row in rows:
            assert isinstance(row, Mapping)
            writer.writerow((row["split"], row["archive_path"], row["size_bytes"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--existing-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-tsv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--total-additions", type=int, default=50)
    parser.add_argument("--heldout-additions", type=int, default=5)
    parser.add_argument("--validation-additions", type=int, default=5)
    parser.add_argument("--calibration-additions", type=int, default=5)
    parser.add_argument("--new-heldout-town", default="Town11")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--current-feature-bytes", type=int, default=70_837_587_338)
    parser.add_argument("--current-reference-frames", type=int, default=720)
    parser.add_argument("--planned-reference-frames", type=int, default=1600)
    parser.add_argument("--int8-payload-ratio", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(
        args.total_additions,
        args.heldout_additions,
        args.validation_additions,
        args.calibration_additions,
        args.current_feature_bytes,
        args.current_reference_frames,
        args.planned_reference_frames,
    ) <= 0:
        raise ValueError("counts, byte sizes, and frame counts must be positive")
    if not 0 < args.int8_payload_ratio <= 1:
        raise ValueError("--int8-payload-ratio must be in (0, 1]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.download_tsv.parent.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_download_tsv(args.download_tsv, plan)
    print(json.dumps({
        "output": str(args.output),
        "download_tsv": str(args.download_tsv),
        "addition_count": plan["audit"]["addition_count"],
        "new_archive_gib": plan["budget"]["new_archive_gib"],
        "projected_int8_feature_payload_gib": plan["budget"]["projected_int8_feature_payload_gib"],
        "heldout_town_overlap": plan["audit"]["heldout_town_overlap"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
