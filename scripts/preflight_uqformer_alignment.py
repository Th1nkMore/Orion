#!/usr/bin/env python3
"""CPU-only architecture/gradient preflight for the task-free UQFormer.

This script deliberately uses synthetic normalized U tensors.  It validates
interfaces and numerical behavior only; it cannot authorize alignment
training, an ORION load, a GPU job, or any downstream claim.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uq_estimator.uq_modality_bridge import UQFormerBridge  # noqa: E402
from uq_estimator.uqformer_alignment import (  # noqa: E402
    UQFormerReconstructionHead,
    symmetric_u_text_alignment_loss,
    uqformer_equivariance_terms,
    uqformer_reconstruction_terms,
)


PROTOCOL_SCHEMA = "orion.uqformer-alignment-protocol/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != PROTOCOL_SCHEMA
        or value.get("status")
        != "cpu_only_architecture_preflight_not_training_authority"
    ):
        raise ValueError("UQFormer CPU preflight protocol is absent or stale")
    architecture = value.get("architecture", {})
    authority = value.get("resource_authority", {})
    if (
        architecture.get("source_summary_width") != 9
        or architecture.get("language_width_projection_position")
        != "final_orion_boundary_only"
        or architecture.get("task_risk_language_bridge_present") is not False
        or authority
        != {
            "cpu_preflight_allowed": True,
            "gpu_job_allowed": False,
            "slurm_submission_allowed": False,
            "orion_load_allowed": False,
            "automatic_retry_allowed": False,
        }
    ):
        raise ValueError("UQFormer architecture/resource locks differ")
    if len(value.get("forbidden_inputs", [])) != 9:
        raise ValueError("UQFormer forbidden-input boundary is incomplete")
    if any(value.get("downstream_locks", {}).values()):
        raise ValueError("a downstream stage was unexpectedly authorized")
    return value


def _bridge(config: Dict[str, Any], language_width: int) -> UQFormerBridge:
    return UQFormerBridge(
        component_dim=3,
        model_dim=language_width,
        bridge_dim=int(config["latent_width"]),
        grid_hw=tuple(config["grid_hw"]),
        max_views=int(config["views"]),
        view_query_hw=tuple(config["view_query_hw"]),
        temporal_queries=3,
        include_component_queries=True,
        global_queries=4,
        num_heads=int(config["heads"]),
        num_layers=int(config["layers"]),
        feedforward_dim=int(config["feedforward_width"]),
        dropout=0.0,
    ).to(device=torch.device("cpu"))


def run_preflight(protocol_path: Path) -> Dict[str, Any]:
    protocol = _load_protocol(protocol_path)
    config = protocol["cpu_preflight"]
    seed = int(config["seed"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = torch.float32
    shape = (
        int(config["batch"]),
        int(config["time"]),
        int(config["views"]),
        int(config["input_hw"][0]),
        int(config["input_hw"][1]),
        3,
    )
    components = torch.rand(
        shape, dtype=dtype, device="cpu", generator=generator
    )
    components[0].zero_()
    torch.manual_seed(seed)
    bridge = _bridge(config, int(config["language_width"]))
    output = bridge(components)
    zero_output = bridge(torch.zeros_like(components[:1]))

    decoder = UQFormerReconstructionHead(
        bridge_dim=int(config["latent_width"]),
        component_dim=3,
        max_views=int(config["views"]),
        num_heads=int(config["heads"]),
        hidden_dim=int(config["feedforward_width"]),
    ).to(device=torch.device("cpu"))
    reconstruction = uqformer_reconstruction_terms(
        output=output,
        reconstruction_head=decoder,
        zero_output=zero_output,
    )
    identity = uqformer_equivariance_terms(
        reference=output,
        transformed=output,
        global_invariant=True,
    )
    text_embedding = output.alignment_embedding().detach().roll(1, dims=0)
    alignment = symmetric_u_text_alignment_loss(
        output.alignment_embedding(), text_embedding
    )
    total = reconstruction.loss + identity.loss + 0.01 * alignment
    if not bool(torch.isfinite(total)):
        raise RuntimeError("UQFormer CPU objective is non-finite")
    total.backward()
    parameters = list(bridge.parameters()) + list(decoder.parameters())
    gradients = [parameter.grad for parameter in parameters if parameter.requires_grad]
    if not gradients or any(
        gradient is None or not bool(torch.isfinite(gradient).all())
        for gradient in gradients
    ):
        raise RuntimeError("UQFormer CPU gradients are absent or non-finite")

    # The source and latent topology must not depend on ORION's language width.
    torch.manual_seed(seed)
    alternate = _bridge(config, int(config["alternate_language_width"]))
    alternate_output = alternate(components)
    if (
        output.source_summary.shape != alternate_output.source_summary.shape
        or output.source_features.shape != alternate_output.source_features.shape
        or output.compact_tokens.shape != alternate_output.compact_tokens.shape
        or output.attention_maps.shape != alternate_output.attention_maps.shape
    ):
        raise RuntimeError("language width changed U source/latent topology")
    if output.source_summary.shape[-1] != 9:
        raise RuntimeError("U source is not the native 9-d summary")
    if (
        not torch.equal(output.source_summary, alternate_output.source_summary)
        or not torch.equal(output.source_features, alternate_output.source_features)
        or not torch.equal(output.compact_tokens, alternate_output.compact_tokens)
    ):
        raise RuntimeError("language width changed U source/latent values")
    if output.language_tokens.shape[-1] == alternate_output.language_tokens.shape[-1]:
        raise RuntimeError("alternate language boundary width was not exercised")
    forward_parameters = list(inspect.signature(UQFormerBridge.forward).parameters)
    if forward_parameters != ["self", "components"]:
        raise RuntimeError("UQFormer accepted an input other than Stage-1 components")

    attention_sums = output.attention_maps.flatten(2).sum(dim=-1)
    report = {
        "schema": "orion.uqformer-cpu-preflight-report/v1",
        "status": "passed_architecture_and_gradient_preflight_only",
        "device": str(output.language_tokens.device),
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": _sha256(protocol_path),
        },
        "shapes": {
            "input_components": list(components.shape),
            "source_summary_9d": list(output.source_summary.shape),
            "source_latent": list(output.source_features.shape),
            "compact_latent": list(output.compact_tokens.shape),
            "language_tokens": list(output.language_tokens.shape),
            "attention_maps": list(output.attention_maps.shape),
            "alternate_language_tokens": list(
                alternate_output.language_tokens.shape
            ),
        },
        "checks": {
            "forward_accepts_components_only": True,
            "native_source_width_is_9": True,
            "language_width_independent_source_topology": True,
            "zero_input_is_audited_without_presence_embedding": bool(
                output.zero_input_mask[0]
            ),
            "zero_output_finite": bool(
                torch.isfinite(zero_output.language_tokens).all()
            ),
            "attention_normalized": bool(
                torch.allclose(attention_sums, torch.ones_like(attention_sums))
            ),
            "all_gradients_finite": True,
            "no_task_labels_consumed": not reconstruction.task_labels_consumed,
            "no_route_context_consumed": not reconstruction.route_context_consumed,
            "no_corruption_metadata_consumed": (
                not reconstruction.corruption_metadata_consumed
            ),
        },
        "objectives": {
            "reconstruction": float(reconstruction.reconstruction_loss.detach()),
            "zero_anchor": float(reconstruction.zero_anchor_loss.detach()),
            "identity_equivariance": float(identity.loss.detach()),
            "u_text_alignment_smoke": float(alignment.detach()),
        },
        "parameter_counts": {
            "bridge": sum(parameter.numel() for parameter in bridge.parameters()),
            "disposable_reconstruction_head": sum(
                parameter.numel() for parameter in decoder.parameters()
            ),
        },
        "gpu_job_submitted": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    if not all(report["checks"].values()):
        raise RuntimeError("one or more UQFormer CPU checks failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_preflight(args.protocol.resolve())
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
