#!/usr/bin/env python3
"""Post-failure component diagnosis for the frozen native-fog evaluation."""

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
    EVIDENCE_COMPONENTS,
    ObservationEvidenceHurdleAdapter,
)
from uq_estimator.native_appearance_audit import (  # noqa: E402
    audit_native_appearance_score_maps,
)
from uq_estimator.native_weather_audit import (  # noqa: E402
    EXPECTED_CONDITIONS,
    validate_native_weather_payload,
)


SCHEMA_VERSION = "orion.counterfactual-evidence-native-fog-component-diagnostic/v1"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _previous(features, items):
    indices = {
        (str(item["route_id"]), int(item["sequence_index"])): index
        for index, item in enumerate(items)
    }
    values = []
    valid = []
    for item in items:
        index = indices.get(
            (str(item["route_id"]), int(item["sequence_index"]) - 1)
        )
        values.append(
            features[index] if index is not None else torch.zeros_like(features[0])
        )
        valid.append(index is not None)
    return torch.stack(values), torch.tensor(valid, dtype=torch.bool)


@torch.no_grad()
def _predict(model, payload, device, batch_size):
    names = []
    for component in EVIDENCE_COMPONENTS:
        names.extend(
            (
                "score/%s" % component,
                "presence/%s" % component,
                "conditional_magnitude/%s" % component,
            )
        )
    outputs = {
        name: {condition: [] for condition in EXPECTED_CONDITIONS}
        for name in names
    }
    for condition in EXPECTED_CONDITIONS:
        features = payload["features_by_condition"][condition]
        previous, valid = _previous(features, payload["items"])
        for start in range(0, int(features.shape[0]), batch_size):
            stop = min(start + batch_size, int(features.shape[0]))
            parts = model.predict_parts(
                features[start:stop].to(device=device, dtype=torch.float32),
                previous[start:stop].to(device=device, dtype=torch.float32),
                valid[start:stop].to(device),
            )
            tensors = {
                "score": parts.score,
                "presence": parts.presence_probability,
                "conditional_magnitude": parts.conditional_magnitude,
            }
            for component_index, component in enumerate(EVIDENCE_COMPONENTS):
                for head, tensor in tensors.items():
                    outputs["%s/%s" % (head, component)][condition].append(
                        tensor[..., component_index].cpu().float()
                    )
    return {
        name: {condition: torch.cat(chunks) for condition, chunks in rows.items()}
        for name, rows in outputs.items()
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-features", type=Path, required=True)
    parser.add_argument("--failed-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("refusing to overwrite native-fog component diagnosis")
    if not torch.cuda.is_available():
        raise SystemExit("component diagnosis requires CUDA")
    failed = json.loads(args.failed_report.read_text(encoding="utf-8"))
    if (
        failed.get("schema_version")
        != "orion.counterfactual-evidence-native-fog-report/v1"
        or failed.get("gate_passed") is not False
        or failed.get("decision")
        != "stop_before_closed_loop_and_revise_stage1_supervision"
    ):
        raise RuntimeError("native-fog failure prerequisite differs")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ObservationEvidenceHurdleAdapter(**checkpoint["model_config"]).cuda().eval()
    model.load_state_dict(checkpoint["student_state"])
    payload = torch.load(args.native_features, map_location="cpu", weights_only=False)
    validate_native_weather_payload(payload)
    scores = _predict(model, payload, torch.device("cuda"), args.batch_size)
    audit = audit_native_appearance_score_maps(
        payload,
        scores,
        candidate_tails={name: "positive" for name in scores},
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "checkpoint_sha256": _sha256(args.checkpoint),
            "native_features_sha256": _sha256(args.native_features),
            "failed_report_sha256": _sha256(args.failed_report),
        },
        "candidates": audit["candidates"],
        "candidate_passes_diagnostic_only": audit["candidate_passes"],
        "attestation": {
            "performed_after_native_gate_failure": True,
            "cannot_retroactively_rescue_failed_gate": True,
            "cannot_select_a_deployable_component_on_this_data": True,
            "purpose": "distinguish aggregation failure from supervision-domain failure",
            "adapter_training": False,
            "checkpoint_updated": False,
            "closed_loop_authorized": False,
        },
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
                "candidate_passes_diagnostic_only": audit["candidate_passes"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
