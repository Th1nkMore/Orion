#!/usr/bin/env python3
"""Run one bounded synthetic/native pairwise Stage-1 repair."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uq_estimator.counterfactual_evidence import (  # noqa: E402
    CLAIM_BOUNDARY,
    EVIDENCE_COMPONENTS,
    ObservationEvidenceHurdleAdapter,
)
from uq_estimator.counterfactual_evidence_pairwise import (  # noqa: E402
    records_from_native_weather_payload,
    run_pairwise_hurdle_epoch,
)
from uq_estimator.counterfactual_evidence_training import (  # noqa: E402
    CounterfactualEvidenceRecord,
)
from uq_estimator.counterfactual_sharded_dataset import (  # noqa: E402
    FP16_DIRECT_DATASET_SCHEMA_VERSION,
    load_fp16_dataset_manifest,
    load_fp16_route_shard_records_selective,
)


CONFIG_SCHEMA = "orion.counterfactual-evidence-pairwise-native-training/v4"
PROTOCOL_SCHEMA = "orion.counterfactual-evidence-pairwise-native-protocol/v4"
INITIAL_CHECKPOINT_SCHEMA = (
    "orion.counterfactual-evidence-no-view-repair-checkpoint/v1"
)
CHECKPOINT_SCHEMA = "orion.counterfactual-evidence-pairwise-native-checkpoint/v1"
REPORT_SCHEMA = "orion.counterfactual-evidence-pairwise-native-report/v1"
OPTIMIZER_FAMILIES = ("local_blur", "local_dark")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _route_id(row: Mapping[str, object]) -> str:
    route_ids = row.get("route_ids")
    if not isinstance(route_ids, list) or len(route_ids) != 1:
        raise RuntimeError("route shard must own exactly one route")
    return str(route_ids[0])


def _selected_rows(manifest: Mapping[str, object], split: str) -> List[dict]:
    return sorted(
        [dict(row) for row in manifest["shards"] if row.get("split") == split],
        key=_route_id,
    )


def _check_row_file(dataset_root: Path, row: Mapping[str, object]) -> None:
    path = dataset_root / str(row["file"])
    if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
        raise RuntimeError("route shard file/size differs: %s" % path)
    if _sha256(path) != str(row["sha256"]):
        raise RuntimeError("route shard SHA256 differs: %s" % path)


def _load_synthetic_records(
    dataset_root: Path, row: Mapping[str, object], expected_split: str
) -> List[CounterfactualEvidenceRecord]:
    records = load_fp16_route_shard_records_selective(
        dataset_root / str(row["file"]), families=OPTIMIZER_FAMILIES
    )
    route_id = _route_id(row)
    if (
        len(records) != 64
        or {record.route_id for record in records} != {route_id}
        or {record.split for record in records} != {expected_split}
        or {record.family for record in records} != set(OPTIMIZER_FAMILIES)
        or {record.severity for record in records} != {1.0, 3.0}
    ):
        raise RuntimeError("synthetic route population differs: %s" % route_id)
    return records


def _weighted_metrics(
    rows: Iterable[Tuple[Mapping[str, float], int]]
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    count = 0
    for metrics, weight in rows:
        count += int(weight)
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value) * int(weight)
    if count <= 0:
        raise RuntimeError("weighted metric population is empty")
    return {name: value / count for name, value in totals.items()}


def _clone_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _evaluate_domains(
    model: ObservationEvidenceHurdleAdapter,
    synthetic_validation: Sequence[CounterfactualEvidenceRecord],
    native_development: Sequence[CounterfactualEvidenceRecord],
    scales: torch.Tensor,
    device: torch.device,
    common: Mapping[str, object],
    native_selection_weight: float,
) -> Dict[str, object]:
    synthetic = run_pairwise_hurdle_epoch(
        model,
        synthetic_validation,
        scales,
        device,
        optimizer=None,
        **common,
    )
    native = run_pairwise_hurdle_epoch(
        model,
        native_development,
        scales,
        device,
        optimizer=None,
        **common,
    )
    composite = (
        (1.0 - native_selection_weight) * float(synthetic["total"])
        + native_selection_weight * float(native["total"])
    )
    return {
        "synthetic_route_validation": synthetic,
        "native_development": native,
        "selection_composite": composite,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--native-features", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    report_path = args.output.with_suffix(".report.json")
    training_state_path = args.output.with_suffix(".training_state.pt")
    if args.output.exists() or report_path.exists() or training_state_path.exists():
        raise SystemExit("refusing to overwrite pairwise native repair")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA pairwise native repair requested but unavailable")

    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise RuntimeError("pairwise native training config schema differs")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise RuntimeError("pairwise native protocol schema differs")
    hashes = {
        "dataset_manifest_sha256": _sha256(args.dataset_manifest),
        "native_feature_sha256": _sha256(args.native_features),
        "initial_checkpoint_sha256": _sha256(args.initial_checkpoint),
        "protocol_sha256": _sha256(args.protocol),
    }
    for name, value in hashes.items():
        if value != str(config["inputs"][name]):
            raise RuntimeError("pairwise native input hash changed: %s" % name)

    manifest = load_fp16_dataset_manifest(args.dataset_manifest, verify_shards=False)
    if (
        manifest.get("schema_version") != FP16_DIRECT_DATASET_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("written_route_count") != 90
    ):
        raise RuntimeError("pairwise synthetic manifest contract differs")
    train_rows = _selected_rows(manifest, "train")
    validation_rows = _selected_rows(manifest, "validation")
    if len(train_rows) != 70 or len(validation_rows) != 10:
        raise RuntimeError("pairwise synthetic route split differs")
    dataset_root = args.dataset_manifest.parent
    for index, row in enumerate(train_rows + validation_rows, start=1):
        _check_row_file(dataset_root, row)
        if index % 10 == 0 or index == len(train_rows) + len(validation_rows):
            print(
                "[PairwiseRepair] integrity=%d/%d"
                % (index, len(train_rows) + len(validation_rows)),
                flush=True,
            )

    native_payload = torch.load(
        args.native_features, map_location="cpu", weights_only=False
    )
    native_records = records_from_native_weather_payload(
        native_payload, split="native_development_pool"
    )
    split = config["native_development_split"]
    train_route = str(split["optimizer_route"])
    development_route = str(split["checkpoint_selection_route"])
    observed_routes = {record.route_id for record in native_records}
    if (
        observed_routes != {train_route, development_route}
        or train_route == development_route
    ):
        raise RuntimeError("native development route population differs")
    native_train = [
        record for record in native_records if record.route_id == train_route
    ]
    native_development = [
        record for record in native_records if record.route_id == development_route
    ]
    if len(native_train) != 32 or len(native_development) != 32:
        raise RuntimeError("native development split record count differs")

    synthetic_validation: List[CounterfactualEvidenceRecord] = []
    for index, row in enumerate(validation_rows, start=1):
        synthetic_validation.extend(
            _load_synthetic_records(dataset_root, row, "validation")
        )
        print(
            "[PairwiseRepair] validation_mmap=%d/%d route=%s"
            % (index, len(validation_rows), _route_id(row)),
            flush=True,
        )
    if len(synthetic_validation) != 640:
        raise RuntimeError("synthetic validation record count differs")

    initial = torch.load(
        args.initial_checkpoint, map_location="cpu", weights_only=False
    )
    if initial.get("schema_version") != INITIAL_CHECKPOINT_SCHEMA:
        raise RuntimeError("pairwise initialization checkpoint differs")
    hparams = config["optimization"]
    expected_model_config = {
        "feature_dim": int(hparams["feature_dim"]),
        "hidden_dim": int(hparams["hidden_dim"]),
        "max_views": int(hparams["max_views"]),
        "presence_bias": float(hparams["presence_bias"]),
        "magnitude_bias": float(hparams["magnitude_bias"]),
        "use_view_embedding": bool(hparams["use_view_embedding"]),
    }
    if initial.get("model_config") != expected_model_config:
        raise RuntimeError("pairwise initialization architecture differs")
    scales = initial["component_scales"].detach().cpu().float()
    expected_scales = torch.tensor(
        [config["component_scales"][name] for name in EVIDENCE_COMPONENTS]
    )
    if not torch.allclose(scales, expected_scales, rtol=1e-6, atol=1e-7):
        raise RuntimeError("pairwise component scales changed")

    seed = int(hparams["seed"])
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = ObservationEvidenceHurdleAdapter(**expected_model_config).to(device)
    model.load_state_dict(initial["student_state"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hparams["learning_rate"]),
        weight_decay=float(hparams["weight_decay"]),
    )
    common = {
        "pair_batch_size": int(hparams["pair_batch_size"]),
        "responsive_weight": float(hparams["responsive_weight"]),
        "ranking_weight": float(hparams["ranking_weight"]),
        "response_floor": float(hparams["response_floor"]),
    }
    native_selection_weight = float(hparams["native_selection_weight"])
    if not 0.0 < native_selection_weight < 1.0:
        raise RuntimeError("native selection weight must lie in (0,1)")
    native_repeats = int(hparams["native_optimizer_passes_per_epoch"])
    if native_repeats <= 0 or native_repeats >= len(train_rows):
        raise RuntimeError("native optimizer pass count is not bounded")

    initial_validation = _evaluate_domains(
        model,
        synthetic_validation,
        native_development,
        scales.to(device),
        device,
        common,
        native_selection_weight,
    )
    print(
        "[PairwiseRepair] initial synthetic_val=%.6f native_dev=%.6f composite=%.6f"
        % (
            initial_validation["synthetic_route_validation"]["total"],
            initial_validation["native_development"]["total"],
            initial_validation["selection_composite"],
        ),
        flush=True,
    )

    history = []
    best_state = None
    best_epoch = 0
    best_value = float("inf")
    early_stopped = False
    for epoch_index in range(int(hparams["epochs"])):
        epoch_rows = list(train_rows)
        random.Random(seed + epoch_index).shuffle(epoch_rows)
        insertion_after = {
            max(1, math.ceil((repeat + 1) * len(epoch_rows) / native_repeats))
            for repeat in range(native_repeats)
        }
        synthetic_metric_rows = []
        native_metric_rows = []
        native_pass = 0
        for route_index, row in enumerate(epoch_rows, start=1):
            records = _load_synthetic_records(dataset_root, row, "train")
            metrics = run_pairwise_hurdle_epoch(
                model,
                records,
                scales.to(device),
                device,
                optimizer=optimizer,
                seed=seed + epoch_index * 10000 + route_index,
                **common,
            )
            synthetic_metric_rows.append((metrics, len(records)))
            del records
            gc.collect()
            if route_index in insertion_after:
                native_pass += 1
                metrics = run_pairwise_hurdle_epoch(
                    model,
                    native_train,
                    scales.to(device),
                    device,
                    optimizer=optimizer,
                    seed=seed + epoch_index * 10000 + 1000 + native_pass,
                    **common,
                )
                native_metric_rows.append((metrics, len(native_train)))
            if route_index % 10 == 0 or route_index == len(epoch_rows):
                print(
                    "[PairwiseRepair] epoch=%d synthetic_routes=%d/%d native_passes=%d/%d"
                    % (
                        epoch_index + 1,
                        route_index,
                        len(epoch_rows),
                        native_pass,
                        native_repeats,
                    ),
                    flush=True,
                )
        if native_pass != native_repeats:
            raise RuntimeError("native optimizer schedule did not complete")
        validation = _evaluate_domains(
            model,
            synthetic_validation,
            native_development,
            scales.to(device),
            device,
            common,
            native_selection_weight,
        )
        row = {
            "epoch": epoch_index + 1,
            "synthetic_train": _weighted_metrics(synthetic_metric_rows),
            "native_train": _weighted_metrics(native_metric_rows),
            **validation,
        }
        history.append(row)
        value = float(validation["selection_composite"])
        if value < best_value:
            best_value = value
            best_epoch = epoch_index + 1
            best_state = _clone_state(model)
        print(
            "[PairwiseRepair] epoch=%d/%d synthetic_val=%.6f native_dev=%.6f composite=%.6f"
            % (
                epoch_index + 1,
                int(hparams["epochs"]),
                validation["synthetic_route_validation"]["total"],
                validation["native_development"]["total"],
                value,
            ),
            flush=True,
        )
        if len(history) >= int(hparams["early_stop_minimum_epochs"]):
            recent = [float(item["selection_composite"]) for item in history[-3:]]
            if len(recent) == 3 and recent[0] < recent[1] < recent[2]:
                early_stopped = True
                print(
                    "[PairwiseRepair] EARLY_STOP_DEVELOPMENT_OVERFIT=1 completed_epochs=%d"
                    % len(history),
                    flush=True,
                )
                break

    if best_state is None:
        raise RuntimeError("pairwise native checkpoint selection failed")
    model.load_state_dict(best_state)
    best_validation = _evaluate_domains(
        model,
        synthetic_validation,
        native_development,
        scales.to(device),
        device,
        common,
        native_selection_weight,
    )
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "student_state": best_state,
        "model_config": expected_model_config,
        "component_scales": scales,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "history": history,
        "requires_inference_baseline_calibration": True,
        "identified_quantity": "within-pair score increment",
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    torch.save(
        {
            "schema_version": "orion.counterfactual-evidence-pairwise-native-training-state/v1",
            "student_state": best_state,
            "best_epoch": best_epoch,
            "best_selection_composite": best_value,
            "initial_validation": initial_validation,
            "best_validation": best_validation,
            "history": history,
            "inputs": {
                **hashes,
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            },
        },
        training_state_path,
    )
    torch.save(checkpoint, args.output)
    report = {
        "schema_version": REPORT_SCHEMA,
        "inputs": {
            **hashes,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "native_development_split": split,
        "best_epoch": best_epoch,
        "completed_epochs": len(history),
        "early_stopped": early_stopped,
        "best_selection_composite": best_value,
        "initial_validation": initial_validation,
        "best_validation": best_validation,
        "history": history,
        "scope_attestation": {
            "synthetic_optimizer_routes": 70,
            "synthetic_checkpoint_selection_routes": 10,
            "native_optimizer_route": train_route,
            "native_checkpoint_selection_route": development_route,
            "final_native_held_out_routes_read": False,
            "local_glare_tensor_values_read": False,
            "blanket_reference_presence_zero_loss": False,
            "blanket_reference_score_zero_loss": False,
            "corruption_mask_optimizer_weight": 0.0,
            "actual_target_optimizer_weight": 0.0,
            "orion_finetuning": False,
            "stage_b": False,
        },
        "decision": "freeze_checkpoint_then_run_unchanged_glare_and_native_heldout_gates",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "best_epoch": best_epoch,
                "completed_epochs": len(history),
                "initial_composite": initial_validation["selection_composite"],
                "best_composite": best_value,
                "decision": report["decision"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print("COUNTERFACTUAL_PAIRWISE_NATIVE_REPAIR_OK=1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
