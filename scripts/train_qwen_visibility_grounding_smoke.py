#!/usr/bin/env python3
"""Run the one-step V1b full-model visibility-grounding gradient smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
import types

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "orion.qwen-visibility-grounding-smoke-config/v1"
REPORT_SCHEMA = "orion.qwen-visibility-grounding-smoke-report/v1"


def _load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load local module from %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package = types.ModuleType("_orion_qwen_visibility_training_package")
package.__path__ = [str(PROJECT_ROOT / "uq_estimator")]
sys.modules[package.__name__] = package
_belief = _load_local_module(
    package.__name__ + ".qwen_visibility_belief",
    PROJECT_ROOT / "uq_estimator" / "qwen_visibility_belief.py",
)
_grounding = _load_local_module(
    package.__name__ + ".qwen_visibility_grounding",
    PROJECT_ROOT / "uq_estimator" / "qwen_visibility_grounding.py",
)
_vlm = _load_local_module(
    package.__name__ + ".qwen_visibility_vlm",
    PROJECT_ROOT / "uq_estimator" / "qwen_visibility_vlm.py",
)
_training = _load_local_module(
    package.__name__ + ".qwen_visibility_training",
    PROJECT_ROOT / "uq_estimator" / "qwen_visibility_training.py",
)
_bridge = _load_local_module(
    "_orion_qwen_visibility_training_bridge",
    PROJECT_ROOT / "uq_estimator" / "qwen_drive_bridge.py",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected grounding smoke config schema")
    if protocol.get("stage") != "V1b_gradient_smoke":
        raise ValueError("this entry point is limited to V1b_gradient_smoke")
    training = protocol["training"]
    if int(training["optimizer_steps"]) != 1:
        raise ValueError("V1b gradient smoke must take exactly one optimizer step")
    if protocol["claim_boundary"] != {
        "plumbing_overfit_only": True,
        "reportable_generalization": False,
        "safety_claim_allowed": False,
    }:
        raise ValueError("V1b claim boundary changed")
    if protocol["evaluation"]["controls"] != [
        "true_u",
        "zero_u",
        "spatial_shuffle",
    ]:
        raise ValueError("V1b must evaluate all paired U controls")
    return protocol


def _verify_manifest(path, sample_ids):
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != _grounding.VISIBILITY_GROUNDING_MANIFEST_SCHEMA:
        raise ValueError("unexpected grounding manifest schema")
    if (
        manifest.get("reportable_generalization") is not False
        or manifest.get("controls_used_for_optimizer") is not False
        or manifest.get("hidden_actor_labels_used") is not False
        or manifest.get("planning_expert_used_for_optimizer") is not False
    ):
        raise ValueError("grounding manifest violates the V1b boundary")
    by_id = {record["sample_id"]: record for record in manifest["records"]}
    if len(by_id) != len(manifest["records"]):
        raise ValueError("grounding manifest contains duplicate sample ids")
    selected = []
    for sample_id in sample_ids:
        if sample_id not in by_id:
            raise ValueError("missing configured sample id: %s" % sample_id)
        record = by_id[sample_id]
        if _sha256(record["token_artifact"]) != record["token_sha256"]:
            raise ValueError("token hash changed for %s" % sample_id)
        for image, digest in zip(record["camera_images"], record["camera_sha256"]):
            if _sha256(image) != digest:
                raise ValueError("image hash changed for %s" % sample_id)
        selected.append(record)
    return manifest, selected


def _load_control_tokens(record, control):
    prefixes = {
        "true_u": "visibility_tokens",
        "zero_u": "visibility_tokens_zero_u",
        "spatial_shuffle": "visibility_tokens_spatial_shuffle",
    }
    prefix = prefixes[control]
    with np.load(record["token_artifact"], allow_pickle=False) as artifact:
        global_tokens = np.asarray(artifact[prefix + "_global"], dtype=np.float32)
        frontier_tokens = np.asarray(
            artifact[prefix + "_frontier"], dtype=np.float32
        )
        global_mask = np.asarray(artifact[prefix + "_global_mask"], dtype=bool)
        frontier_mask = np.asarray(
            artifact[prefix + "_frontier_mask"], dtype=bool
        )
        names = tuple(str(value) for value in artifact[prefix + "_feature_names"].tolist())
    if names != tuple(_belief.VISIBILITY_TOKEN_FEATURE_NAMES):
        raise ValueError("control feature order changed")
    permutation = record["frontier_permutation_new_to_old"]
    frontier_tokens, frontier_mask = _grounding.permute_frontier_rows(
        frontier_tokens, frontier_mask, permutation
    )
    return (
        np.concatenate([global_tokens, frontier_tokens], axis=0),
        np.concatenate([global_mask, frontier_mask], axis=0),
    )


def _gradient_report(named_parameters):
    rows = []
    for name, parameter in named_parameters:
        gradient = parameter.grad
        rows.append(
            {
                "name": name,
                "has_gradient": gradient is not None,
                "finite": bool(
                    gradient is not None and torch.isfinite(gradient).all().item()
                ),
                "norm": float(gradient.float().norm().item())
                if gradient is not None
                else 0.0,
            }
        )
    return rows


def _parse_answer(answer):
    try:
        value = json.loads(answer)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or set(value) != {
        "frontier",
        "route",
        "margin",
        "action",
    }:
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to reuse output directory: %s" % args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    protocol = _load_protocol(args.protocol)
    manifest, records = _verify_manifest(
        protocol["manifest"], protocol["sample_ids"]
    )
    seed = int(protocol["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    bridge_path = Path(protocol["base_bridge_config"])
    if not bridge_path.is_absolute():
        bridge_path = PROJECT_ROOT / bridge_path
    bridge = _bridge.load_bridge_config(bridge_path)
    runtime = bridge["runtime"]
    from qwen_drive import QwenDriveForPlanning

    load_started = time.monotonic()
    model = QwenDriveForPlanning.from_pretrained(
        runtime["model"],
        planner=runtime["planner"],
        dtype=getattr(torch, str(runtime["dtype"])),
        attn_implementation=runtime["attention_implementation"],
    ).to(runtime["device"])
    load_seconds = time.monotonic() - load_started
    _training.freeze_qwen_for_visibility_grounding(model)
    lora_config = _training.VisibilityLoRAConfig(**protocol["lora"])
    installed = _training.install_upper_full_attention_lora(model, lora_config)
    projector_config = protocol["projector"]
    projector = _vlm.VisibilityTokenProjector(**projector_config).to(model.device)
    scope = _training.visibility_grounding_trainable_scope(model, projector)
    if protocol["training"]["gradient_checkpointing"]:
        model.vlm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.vlm.model.visual.eval()
    model.vlm.model.language_model.train()
    projector.train()

    prepared = []
    for record in records:
        inputs = model.processor.encode_vqa(
            record["camera_images"],
            manifest["question"],
            system=manifest["system_prompt"],
            device="cpu",
        )
        inputs = {
            key: value.to(model.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        with torch.no_grad():
            base_embeddings = _vlm._official_multimodal_embeddings(
                model, inputs
            ).detach()
        true_tokens, true_mask = _load_control_tokens(record, "true_u")
        answer_ids = _training.encode_grounding_answer(
            model.processor, record["canonical_answer"], model.device
        )
        prepared.append(
            {
                "record": record,
                "inputs": inputs,
                "base_embeddings": base_embeddings,
                "true_tokens": torch.from_numpy(true_tokens).to(model.device),
                "true_mask": torch.from_numpy(true_mask).to(model.device),
                "answer_ids": answer_ids,
            }
        )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(projector.parameters()),
                "lr": float(protocol["training"]["projector_learning_rate"]),
            },
            {
                "params": [
                    parameter
                    for _, parameter in model.named_parameters()
                    if parameter.requires_grad
                ],
                "lr": float(protocol["training"]["lora_learning_rate"]),
            },
        ],
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    history = []
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for optimizer_step in range(1, int(protocol["training"]["optimizer_steps"]) + 1):
        example = prepared[(optimizer_step - 1) % len(prepared)]
        result = _training.visibility_grounding_answer_loss(
            model,
            example["inputs"],
            example["true_tokens"],
            example["true_mask"],
            projector,
            example["answer_ids"],
            base_embeddings=example["base_embeddings"],
        )
        if not torch.isfinite(result.loss):
            raise RuntimeError("V1b grounding loss is non-finite")
        result.loss.backward()
        projector_gradients = _gradient_report(projector.named_parameters())
        lora_gradients = _gradient_report(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        projector_connected = any(
            row["finite"] and row["norm"] > 0.0
            and ("output_projection" in row["name"] or "boundary_embeddings" in row["name"])
            for row in projector_gradients
        )
        lora_connected = any(
            row["finite"] and row["norm"] > 0.0 for row in lora_gradients
        )
        if not projector_connected or not lora_connected:
            raise RuntimeError(
                "V1b gradient path failed: projector=%s lora=%s"
                % (projector_connected, lora_connected)
            )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(projector.parameters())
            + [
                parameter
                for _, parameter in model.named_parameters()
                if parameter.requires_grad
            ],
            float(protocol["training"]["maximum_gradient_norm"]),
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        history.append(
            {
                "optimizer_step": optimizer_step,
                "sample_id": example["record"]["sample_id"],
                "loss": float(result.loss.detach().item()),
                "gradient_norm_before_clip": float(gradient_norm.item()),
                "answer_token_count": result.answer_token_count,
                "base_prompt_length": result.base_prompt_length,
                "augmented_prompt_length": result.augmented_prompt_length,
                "full_sequence_length": result.full_sequence_length,
                "insertion_index": result.insertion_index,
                "visibility_token_count": result.visibility_token_count,
            }
        )

    model.vlm.model.language_model.eval()
    projector.eval()
    evaluations = []
    for example in prepared:
        target = example["record"]["target"]
        for control in protocol["evaluation"]["controls"]:
            tokens, mask = _load_control_tokens(example["record"], control)
            answer = _training.generate_visibility_grounding_answer(
                model,
                example["inputs"],
                torch.from_numpy(tokens).to(model.device),
                torch.from_numpy(mask).to(model.device),
                projector,
                max_new_tokens=int(protocol["evaluation"]["max_new_tokens"]),
            )
            parsed = _parse_answer(answer)
            evaluations.append(
                {
                    "sample_id": example["record"]["sample_id"],
                    "control": control,
                    "answer": answer,
                    "parsed": parsed,
                    "canonical_exact": answer == example["record"]["canonical_answer"],
                    "field_correct": {
                        field: bool(parsed is not None and parsed.get(field) == value)
                        for field, value in target.items()
                    },
                }
            )

    adaptation = _training.adaptation_state_dict(model, projector)
    checkpoint = {
        "schema": _training.VISIBILITY_GROUNDING_TRAINING_SCHEMA,
        "status": "v1b_gradient_smoke_complete",
        "base_model": runtime["model"],
        "base_planner": runtime["planner"],
        "protocol": protocol,
        "installed_lora_modules": list(installed),
        "adaptation": adaptation,
    }
    checkpoint_path = args.output_dir / "adaptation.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = _sha256(checkpoint_path)
    (args.output_dir / "adaptation.sha256").write_text(
        checkpoint_sha256 + "  " + checkpoint_path.name + "\n", encoding="utf-8"
    )

    gpu_memory = {}
    if torch.cuda.is_available():
        gpu_memory = {
            "allocated_mb": float(torch.cuda.memory_allocated() / 1024**2),
            "peak_allocated_mb": float(
                torch.cuda.max_memory_allocated() / 1024**2
            ),
            "reserved_mb": float(torch.cuda.memory_reserved() / 1024**2),
            "peak_reserved_mb": float(
                torch.cuda.max_memory_reserved() / 1024**2
            ),
        }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "claim_boundary": protocol["claim_boundary"],
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": _sha256(args.protocol),
        "manifest_path": str(Path(protocol["manifest"]).resolve()),
        "manifest_sha256": _sha256(protocol["manifest"]),
        "sample_ids": protocol["sample_ids"],
        "optimizer_controls": [],
        "hidden_actor_labels_used": False,
        "planning_expert_in_optimizer": False,
        "load_seconds": float(load_seconds),
        "scope": scope,
        "lora": lora_config.as_dict(),
        "installed_lora_modules": list(installed),
        "projector_gradients_before_first_optimizer_step": projector_gradients,
        "lora_gradients_before_first_optimizer_step": lora_gradients,
        "history": history,
        "evaluations": evaluations,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "bytes": checkpoint_path.stat().st_size,
            "contains_optimizer_state": False,
            "contains_base_model_weights": False,
            "projector_tensor_count": len(adaptation["projector"]),
            "lora_tensor_count": len(adaptation["lora"]),
        },
        "gpu_memory": gpu_memory,
        "torch_version": torch.__version__,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

