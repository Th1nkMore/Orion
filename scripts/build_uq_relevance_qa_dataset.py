#!/usr/bin/env python3
"""Build an immutable Stage2-L UQ/relevance QA dataset from frame bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from uq_relevance_qa_factory_lib import (
    QA_DATASET_SCHEMA,
    audit_dataset,
    build_records_for_bundle,
    sha256_file,
)


def _bundle_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(item).resolve() for item in args.bundle]
    if args.bundle_list:
        list_path = Path(args.bundle_list).resolve()
        payload = json.loads(list_path.read_text(encoding="utf-8"))
        values = payload.get("bundles") if isinstance(payload, dict) else payload
        if not isinstance(values, list):
            raise ValueError("bundle list must be a JSON list or contain a bundles list")
        for value in values:
            path = Path(value)
            if not path.is_absolute():
                path = (list_path.parent / path).resolve()
            paths.append(path)
    unique = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    if not unique:
        raise ValueError("at least one --bundle or --bundle-list entry is required")
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="append", default=[])
    parser.add_argument("--bundle-list")
    parser.add_argument(
        "--config",
        default="configs/scenario_factory/qa_factory_v1.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--require-formal-gates", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite QA dataset output directory")
    output_dir.mkdir(parents=True)
    sidecar_dir = output_dir / "map_sidecars"
    sidecar_dir.mkdir()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    shutil.copyfile(config_path, output_dir / "qa_factory_config.json")
    records = []
    bundle_inventory = []
    for index, bundle_path in enumerate(_bundle_paths(args)):
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        variant = bundle.get("counterfactual", {}).get("variant", "unknown")
        stem = "%04d_%s_%s" % (
            index,
            str(bundle.get("event_id", "event")).replace("/", "_"),
            str(variant).replace("/", "_"),
        )
        relative_sidecar = "map_sidecars/%s.npz" % stem
        built, sidecar_metadata, arrays = build_records_for_bundle(
            bundle,
            bundle_path=bundle_path,
            config=config,
            sidecar_relative_path=relative_sidecar,
        )
        sidecar_path = output_dir / relative_sidecar
        np.savez_compressed(
            sidecar_path,
            task_relevance=arrays[0],
            task_risk=arrays[1],
        )
        sidecar_sha = sha256_file(sidecar_path)
        for record in built:
            record["target"]["map_sidecar"].update({
                "sha256": sidecar_sha,
                "metadata": sidecar_metadata,
            })
        records.extend(built)
        bundle_inventory.append({
            "path": str(bundle_path),
            "sha256": sha256_file(bundle_path),
            "event_id": bundle.get("event_id"),
            "variant": variant,
            "map_sidecar": relative_sidecar,
            "map_sidecar_sha256": sidecar_sha,
        })

    dataset = {
        "schema": QA_DATASET_SCHEMA,
        "status": "engineering_interface_only",
        "config": {
            "path": "qa_factory_config.json",
            "sha256": sha256_file(output_dir / "qa_factory_config.json"),
        },
        "bundle_inventory": bundle_inventory,
        "records": records,
        "claim_boundary": config["claim_boundary"],
    }
    audit = audit_dataset(dataset, config=config, dataset_dir=output_dir)
    dataset["audit_summary"] = {
        "formal_training_ready": audit["formal_training_ready"],
        "checks": audit["checks"],
    }
    dataset_path = output_dir / "dataset.json"
    dataset_path.write_text(
        json.dumps(dataset, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.require_formal_gates and not audit["formal_training_ready"]:
        raise RuntimeError("formal Stage2-L gates did not pass; see audit.json")
    print(json.dumps({
        "dataset": str(dataset_path),
        "record_count": len(records),
        "formal_training_ready": audit["formal_training_ready"],
        "audit": str(output_dir / "audit.json"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
