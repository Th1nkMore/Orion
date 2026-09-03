#!/usr/bin/env python3
"""Evaluate a frozen pairwise adapter on final Town-disjoint native fog."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uq_estimator.counterfactual_evidence import (  # noqa: E402
    ObservationEvidenceHurdleAdapter,
)
from uq_estimator.native_appearance_audit import (  # noqa: E402
    audit_native_appearance_score_maps,
)
from uq_estimator.native_weather_audit import (  # noqa: E402
    EXPECTED_CONDITIONS,
    validate_native_weather_payload,
)


SCHEMA_VERSION = "orion.counterfactual-evidence-pairwise-native-fog-report/v1"
CONFIG_SCHEMA = "orion.counterfactual-evidence-pairwise-native-fog-evaluation/v4"
CHECKPOINT_SCHEMA = "orion.counterfactual-evidence-pairwise-native-checkpoint/v1"
GLARE_REPORT_SCHEMA = "orion.counterfactual-evidence-pairwise-glare-report/v1"
CANDIDATE = "counterfactual_pairwise_component_mean"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _previous_for_condition(features, items):
    key_to_index = {
        (str(item["route_id"]), int(item["sequence_index"])): index
        for index, item in enumerate(items)
    }
    previous = []
    valid = []
    for item in items:
        index = key_to_index.get(
            (str(item["route_id"]), int(item["sequence_index"]) - 1)
        )
        previous.append(
            features[index] if index is not None else torch.zeros_like(features[0])
        )
        valid.append(index is not None)
    return torch.stack(previous), torch.tensor(valid, dtype=torch.bool)


@torch.no_grad()
def _score_native(model, payload, device, batch_size):
    scores = {}
    for condition in EXPECTED_CONDITIONS:
        features = payload["features_by_condition"][condition]
        previous, valid = _previous_for_condition(features, payload["items"])
        chunks = []
        for start in range(0, int(features.shape[0]), batch_size):
            stop = min(start + batch_size, int(features.shape[0]))
            component_score = model(
                features[start:stop].to(device=device, dtype=torch.float32),
                previous[start:stop].to(device=device, dtype=torch.float32),
                valid[start:stop].to(device),
            )
            chunks.append(component_score.mean(dim=-1).cpu().float())
        scores[condition] = torch.cat(chunks)
    return scores


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-features", type=Path, required=True)
    parser.add_argument("--upstream-glare-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--score-calibration",
        choices=("absolute", "exact-pair-diagnostic"),
        default="absolute",
        help=(
            "Use the deployable absolute score for the frozen gate, or subtract "
            "the matched clear score as a privileged diagnostic only."
        ),
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("refusing to overwrite pairwise native-fog report")
    if args.batch_size <= 0:
        raise SystemExit("batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA pairwise native-fog evaluation requested but unavailable")

    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise RuntimeError("pairwise native-fog config differs")
    if _sha256(args.checkpoint) != config["inputs"]["checkpoint_sha256"]:
        raise RuntimeError("pairwise native-fog checkpoint hash differs")
    if _sha256(args.native_features) != config["inputs"]["native_feature_sha256"]:
        raise RuntimeError("pairwise native-fog feature hash differs")
    upstream = json.loads(args.upstream_glare_report.read_text(encoding="utf-8"))
    if (
        upstream.get("schema_version") != GLARE_REPORT_SCHEMA
        or upstream.get("gates", {}).get("both_splits_passed") is not True
        or upstream.get("decision")
        != "proceed_to_frozen_pairwise_native_heldout_evaluation"
        or upstream.get("inputs", {}).get("checkpoint_sha256")
        != config["inputs"]["checkpoint_sha256"]
    ):
        raise RuntimeError("pairwise upstream glare gate did not pass")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.get("requires_inference_baseline_calibration") is not True
    ):
        raise RuntimeError("pairwise native-fog checkpoint contract differs")
    device = torch.device(args.device)
    model = ObservationEvidenceHurdleAdapter(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["student_state"])
    model.eval()

    payload = torch.load(args.native_features, map_location="cpu", weights_only=False)
    validate_native_weather_payload(payload)
    if sorted({str(item["route_id"]) for item in payload["items"]}) != sorted(
        config["routes"]
    ):
        raise RuntimeError("pairwise native-fog held-out routes differ")
    if set(config["routes"]) & set(config["native_development_routes"]):
        raise RuntimeError("pairwise native development/held-out routes overlap")
    scores = _score_native(model, payload, device, args.batch_size)
    if args.score_calibration == "exact-pair-diagnostic":
        clear_score = scores["clear"]
        scores = {
            "clear": torch.zeros_like(clear_score),
            "fog_light": scores["fog_light"] - clear_score,
            "fog_heavy": scores["fog_heavy"] - clear_score,
        }
    audit = audit_native_appearance_score_maps(
        payload,
        {CANDIDATE: scores},
        candidate_tails={CANDIDATE: "positive"},
    )
    passed = bool(audit["candidate_passes"][CANDIDATE])
    report = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "checkpoint_sha256": _sha256(args.checkpoint),
            "native_features_sha256": _sha256(args.native_features),
            "upstream_glare_report_sha256": _sha256(args.upstream_glare_report),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "candidate": audit["candidates"][CANDIDATE],
        "gate_passed": passed,
        "scope_attestation": {
            "checkpoint_updated": False,
            "adapter_training": False,
            "paired_clear_feature_used_for_model_score": (
                args.score_calibration == "exact-pair-diagnostic"
            ),
            "paired_clear_feature_used_for_evaluation_only": True,
            "score_calibration": args.score_calibration,
            "deployable_gate": args.score_calibration == "absolute",
            "native_held_out_used_for_optimizer_or_checkpoint_selection": False,
            "requires_inference_baseline_calibration": True,
            "orion_finetuning": False,
            "stage_b": False,
        },
        "decision": (
            "exact_pair_calibration_diagnostic_only_no_execution_authority"
            if args.score_calibration == "exact-pair-diagnostic"
            else (
                "proceed_to_minimum_learned_uq_closed_loop"
                if passed
                else "stop_before_learned_closed_loop_and_expand_independent_native_supervision"
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "gate_passed": passed,
                "decision": report["decision"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
