#!/usr/bin/env python3
"""Run the one-step V1b full-model visibility-grounding gradient smoke."""

from __future__ import annotations

import argparse
from collections import Counter
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
SMOKE_CONFIG_SCHEMA = "orion.qwen-visibility-grounding-smoke-config/v1"
OVERFIT_CONFIG_SCHEMA = "orion.qwen-visibility-grounding-overfit-config/v1"
FACTORIZED_CONFIG_SCHEMA = "orion.qwen-visibility-grounding-factorized-config/v1"
ROW_CONFIG_SCHEMA = "orion.qwen-visibility-row-grounding-config/v1"
ROUTE_READOUT_CONFIG_SCHEMA = "orion.qwen-visibility-route-readout-config/v1"
TYPED_ROUTE_READOUT_CONFIG_SCHEMA = (
    "orion.qwen-visibility-typed-route-readout-config/v1"
)
SLOT_TYPED_ROUTE_READOUT_CONFIG_SCHEMA = (
    "orion.qwen-visibility-slot-typed-route-readout-config/v1"
)
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
_curriculum = _load_local_module(
    package.__name__ + ".qwen_visibility_grounding_curriculum",
    PROJECT_ROOT / "uq_estimator" / "qwen_visibility_grounding_curriculum.py",
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
    schema = protocol.get("schema")
    stage = protocol.get("stage")
    if (schema, stage) not in {
        (SMOKE_CONFIG_SCHEMA, "V1b_gradient_smoke"),
        (OVERFIT_CONFIG_SCHEMA, "V1c_route151_plumbing_overfit"),
        (FACTORIZED_CONFIG_SCHEMA, "V1d_route151_factorized_overfit"),
        (ROW_CONFIG_SCHEMA, "V1e_route151_row_addressed_overfit"),
        (ROUTE_READOUT_CONFIG_SCHEMA, "V1f_route151_route_readout_overfit"),
        (
            TYPED_ROUTE_READOUT_CONFIG_SCHEMA,
            "V1g_route151_typed_route_readout_overfit",
        ),
        (
            SLOT_TYPED_ROUTE_READOUT_CONFIG_SCHEMA,
            "V1h_route151_slot_typed_route_readout_overfit",
        ),
    }:
        raise ValueError("unexpected grounding training config schema/stage")
    training = protocol["training"]
    if stage == "V1b_gradient_smoke" and int(training["optimizer_steps"]) != 1:
        raise ValueError("V1b gradient smoke must take exactly one optimizer step")
    if stage == "V1c_route151_plumbing_overfit":
        if int(training["optimizer_steps"]) != 15:
            raise ValueError("V1c bounded overfit must take exactly 15 optimizer steps")
        if training.get("separate_gradient_clipping") is not True:
            raise ValueError("V1c requires separate projector/LoRA gradient clipping")
        expected_samples = {
            "route151-step-000000",
            "route151-step-000200",
            "route151-step-000260",
            "route151-step-000280",
            "route151-step-000300",
        }
        if set(protocol.get("sample_ids", ())) != expected_samples:
            raise ValueError("V1c must use all five immutable plumbing records")
    if stage == "V1d_route151_factorized_overfit":
        if int(training["optimizer_steps"]) != 60:
            raise ValueError("V1d factorized overfit must take exactly 60 steps")
        if training.get("separate_gradient_clipping") is not True:
            raise ValueError("V1d requires separate projector/LoRA clipping")
        if set(protocol.get("sample_ids", ())) != {
            "route151-step-000000",
            "route151-step-000200",
            "route151-step-000260",
            "route151-step-000280",
            "route151-step-000300",
        }:
            raise ValueError("V1d must use all five immutable plumbing records")
        if protocol.get("objective") != {
            "type": "factorized_fields",
            "fields": ["frontier", "route", "margin", "action"],
        }:
            raise ValueError("V1d requires the balanced four-field objective")
    if stage == "V1e_route151_row_addressed_overfit":
        if int(training["optimizer_steps"]) != 360:
            raise ValueError("V1e row-addressed overfit must take exactly 360 steps")
        if training.get("separate_gradient_clipping") is not True:
            raise ValueError("V1e requires separate projector/LoRA clipping")
        if set(protocol.get("sample_ids", ())) != {
            "route151-step-000000",
            "route151-step-000200",
            "route151-step-000260",
            "route151-step-000280",
            "route151-step-000300",
        }:
            raise ValueError("V1e must use all five immutable plumbing records")
        if protocol.get("objective") != {
            "type": "row_addressed_fields",
            "fields": ["frontier", "route", "margin", "action"],
        }:
            raise ValueError("V1e requires the balanced row-addressed objective")
        if not protocol.get("curriculum"):
            raise ValueError("V1e requires an immutable curriculum manifest")
    if stage in {
        "V1f_route151_route_readout_overfit",
        "V1g_route151_typed_route_readout_overfit",
        "V1h_route151_slot_typed_route_readout_overfit",
    }:
        if int(training["optimizer_steps"]) != 240:
            raise ValueError("V1f route readout must take exactly 240 steps")
        if training.get("separate_gradient_clipping") is not True:
            raise ValueError("V1f requires separate projector/LoRA clipping")
        if set(protocol.get("sample_ids", ())) != {
            "route151-step-000000",
            "route151-step-000200",
            "route151-step-000260",
            "route151-step-000280",
            "route151-step-000300",
        }:
            raise ValueError("V1f must use all five immutable plumbing records")
        if protocol.get("objective") != {
            "type": "route_readout_pairs",
            "fields": ["route"],
        }:
            raise ValueError("V1f requires the matched-pair route objective")
        if protocol.get("evaluation", {}).get("split") != "held_out_order":
            raise ValueError("V1f must evaluate unseen decoy-row orders")
        if not protocol.get("curriculum"):
            raise ValueError("route-readout stage requires an immutable curriculum")
        projector_type = protocol.get("projector", {}).get("type", "generic_mlp")
        if stage == "V1f_route151_route_readout_overfit" and projector_type != "generic_mlp":
            raise ValueError("V1f requires the generic MLP projector")
        if (
            stage == "V1g_route151_typed_route_readout_overfit"
            and projector_type != "typed_scalar_basis"
        ):
            raise ValueError("V1g requires the typed scalar-basis projector")
        if (
            stage == "V1h_route151_slot_typed_route_readout_overfit"
            and projector_type != "slot_typed_scalar_basis"
        ):
            raise ValueError("V1h requires explicit slot-typed scalar basis")
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


def _verify_row_curriculum(path, manifest_path, records):
    curriculum_path = Path(path)
    curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
    if curriculum.get("schema") != _curriculum.ROW_ADDRESSED_CURRICULUM_SCHEMA:
        raise ValueError("unexpected row-addressed curriculum schema")
    if Path(curriculum.get("base_manifest_path", "")).resolve() != Path(
        manifest_path
    ).resolve():
        raise ValueError("row curriculum base manifest path changed")
    if curriculum.get("base_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("row curriculum base manifest hash changed")
    if (
        curriculum.get("reportable_generalization") is not False
        or curriculum.get("controls_used_for_optimizer") is not False
        or curriculum.get("hidden_actor_labels_used") is not False
        or curriculum.get("planning_expert_used_for_optimizer") is not False
        or curriculum.get("complete_row_permutations_only") is not True
        or curriculum.get("spatial_shuffle_target_changed_for_every_example")
        is not True
    ):
        raise ValueError("row curriculum violates the V1e boundary")
    if curriculum.get("field_counts") != {
        "action": 6,
        "frontier": 15,
        "margin": 12,
        "route": 10,
    }:
        raise ValueError("row curriculum field counts changed")
    if curriculum.get("label_counts") != {
        "action": {"KEEP": 2, "SLOW": 2, "STOP": 2},
        "frontier": {"F03": 5, "F13": 5, "F23": 5},
        "margin": {"CLEAR": 4, "INSIDE": 4, "NEAR": 4},
        "route": {"OFF_ROUTE": 5, "ON_ROUTE": 5},
    }:
        raise ValueError("row curriculum label counts changed")
    if int(curriculum.get("example_count", -1)) != 43:
        raise ValueError("V1e requires exactly 43 row-addressed examples")
    if int(curriculum.get("steps_per_field", -1)) != 90:
        raise ValueError("V1e requires exactly 90 optimizer steps per field")
    schedule = curriculum.get("training_schedule", [])
    if int(curriculum.get("optimizer_steps", -1)) != 360 or len(schedule) != 360:
        raise ValueError("V1e curriculum schedule must contain 360 steps")
    selected_ids = {record["sample_id"] for record in records}
    examples = curriculum.get("examples", [])
    by_id = {example.get("example_id"): example for example in examples}
    if len(by_id) != 43 or None in by_id:
        raise ValueError("V1e curriculum example ids are not unique")
    if any(example_id not in by_id for example_id in schedule):
        raise ValueError("V1e schedule names an unknown example")
    for example in examples:
        field = example.get("task_field")
        if example.get("sample_id") not in selected_ids:
            raise ValueError("V1e curriculum names an unselected sample")
        if field not in _curriculum.ROW_ADDRESSED_FIELDS:
            raise ValueError("V1e curriculum names an invalid field")
        control_answers = example.get("control_expected_answers", {})
        if control_answers.get("true_u") != example.get("expected_answer"):
            raise ValueError("V1e true-U target changed")
        if control_answers.get("spatial_shuffle") == example.get(
            "expected_answer"
        ):
            raise ValueError("V1e spatial shuffle does not change a target")
        permutation = example.get("sequence_permutation_new_to_manifest", [])
        if sorted(permutation) != list(range(32)):
            raise ValueError("V1e example lacks a complete row permutation")
    return curriculum


def _verify_route_readout_curriculum(path, manifest_path, records):
    curriculum_path = Path(path)
    curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
    if curriculum.get("schema") != _curriculum.ROUTE_READOUT_CURRICULUM_SCHEMA:
        raise ValueError("unexpected route-readout curriculum schema")
    if Path(curriculum.get("base_manifest_path", "")).resolve() != Path(
        manifest_path
    ).resolve():
        raise ValueError("route-readout curriculum base manifest path changed")
    if curriculum.get("base_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("route-readout curriculum base manifest hash changed")
    required_true_flags = (
        "complete_row_permutations_only",
        "matched_pairs_differ_only_by_query_row_swap",
        "non_query_order_randomized",
        "held_out_order_evaluation",
        "held_out_order_disjoint_verified",
        "spatial_shuffle_target_changed_for_every_example",
    )
    if (
        curriculum.get("reportable_generalization") is not False
        or curriculum.get("controls_used_for_optimizer") is not False
        or curriculum.get("hidden_actor_labels_used") is not False
        or curriculum.get("planning_expert_used_for_optimizer") is not False
        or any(curriculum.get(name) is not True for name in required_true_flags)
    ):
        raise ValueError("route-readout curriculum violates the V1f boundary")
    if curriculum.get("query_frontier") != "F00":
        raise ValueError("V1f query frontier changed")
    if curriculum.get("example_count") != 50:
        raise ValueError("V1f requires exactly 50 route-readout examples")
    if curriculum.get("train_pair_variants_per_sample") != 3:
        raise ValueError("V1f train pair variants changed")
    if curriculum.get("evaluation_pair_variants_per_sample") != 2:
        raise ValueError("V1f evaluation pair variants changed")
    if curriculum.get("training_label_counts") != {
        "OFF_ROUTE": 15,
        "ON_ROUTE": 15,
    }:
        raise ValueError("V1f training labels changed")
    if curriculum.get("evaluation_label_counts") != {
        "OFF_ROUTE": 10,
        "ON_ROUTE": 10,
    }:
        raise ValueError("V1f evaluation labels changed")
    if curriculum.get("optimizer_label_counts") != {
        "OFF_ROUTE": 120,
        "ON_ROUTE": 120,
    }:
        raise ValueError("V1f optimizer labels changed")
    schedule = curriculum.get("training_schedule", [])
    if curriculum.get("optimizer_steps") != 240 or len(schedule) != 240:
        raise ValueError("V1f curriculum schedule must contain 240 steps")
    examples = curriculum.get("examples", [])
    by_id = {example.get("example_id"): example for example in examples}
    if len(by_id) != 50 or None in by_id:
        raise ValueError("V1f curriculum example ids are not unique")
    training_ids = set(curriculum.get("training_example_ids", []))
    evaluation_ids = set(curriculum.get("evaluation_example_ids", []))
    if (
        len(training_ids) != 30
        or len(evaluation_ids) != 20
        or training_ids & evaluation_ids
        or training_ids | evaluation_ids != set(by_id)
    ):
        raise ValueError("V1f train/evaluation split changed")
    if any(example_id not in training_ids for example_id in schedule):
        raise ValueError("V1f optimizer schedule contains a held-out example")
    if Counter(schedule) != Counter({example_id: 8 for example_id in training_ids}):
        raise ValueError("V1f optimizer example balance changed")
    selected_ids = {record["sample_id"] for record in records}
    pairs = {}
    for example in examples:
        example_id = example["example_id"]
        if example.get("sample_id") not in selected_ids:
            raise ValueError("V1f curriculum names an unselected sample")
        if example.get("task_field") != "route":
            raise ValueError("V1f curriculum contains a non-route task")
        expected_split = (
            "train" if example_id in training_ids else "held_out_order"
        )
        if example.get("split") != expected_split:
            raise ValueError("V1f example split changed")
        if example.get("query_frontier") != "F00":
            raise ValueError("V1f example query slot changed")
        expected_answer = example.get("expected_answer")
        if expected_answer not in {"ON_ROUTE", "OFF_ROUTE"}:
            raise ValueError("V1f example answer changed")
        controls = example.get("control_expected_answers", {})
        if controls.get("true_u") != expected_answer:
            raise ValueError("V1f true-U target changed")
        if controls.get("spatial_shuffle") == expected_answer:
            raise ValueError("V1f spatial shuffle target did not change")
        permutation = example.get("sequence_permutation_new_to_manifest", [])
        if sorted(permutation) != list(range(32)):
            raise ValueError("V1f example lacks a complete row permutation")
        pair_id = example.get("pair_id")
        pairs.setdefault(pair_id, []).append(example)
    for sample_id in selected_ids:
        for label in ("ON_ROUTE", "OFF_ROUTE"):
            train_orders = {
                tuple(by_id[value]["sequence_permutation_new_to_manifest"])
                for value in training_ids
                if by_id[value]["sample_id"] == sample_id
                and by_id[value]["expected_answer"] == label
            }
            evaluation_orders = {
                tuple(by_id[value]["sequence_permutation_new_to_manifest"])
                for value in evaluation_ids
                if by_id[value]["sample_id"] == sample_id
                and by_id[value]["expected_answer"] == label
            }
            if len(train_orders) != 3 or len(evaluation_orders) != 2:
                raise ValueError("V1f row-order variant count changed")
            if train_orders & evaluation_orders:
                raise ValueError("V1f held-out row order leaked into training")
    if len(pairs) != 25 or any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError("V1f matched-pair count changed")
    for pair in pairs.values():
        if {example["expected_answer"] for example in pair} != {
            "ON_ROUTE",
            "OFF_ROUTE",
        }:
            raise ValueError("V1f matched pair lacks both route labels")
        first, second = pair
        if first["sample_id"] != second["sample_id"] or first["split"] != second["split"]:
            raise ValueError("V1f matched pair crosses sample or split")
        first_order = first["sequence_permutation_new_to_manifest"]
        second_order = second["sequence_permutation_new_to_manifest"]
        changed = [
            index
            for index, values in enumerate(zip(first_order, second_order))
            if values[0] != values[1]
        ]
        if len(changed) != 2 or 0 not in changed:
            raise ValueError("V1f matched pair changes more than one row swap")
        other = changed[1] if changed[0] == 0 else changed[0]
        if not (
            first_order[0] == second_order[other]
            and first_order[other] == second_order[0]
        ):
            raise ValueError("V1f matched-pair rows are not swapped")
    return curriculum


def _load_control_tokens(record, control, sequence_permutation=None):
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
    if sequence_permutation is not None:
        frontier_tokens, frontier_mask = _grounding.permute_frontier_rows(
            frontier_tokens, frontier_mask, sequence_permutation
        )
    return (
        np.concatenate([global_tokens, frontier_tokens], axis=0),
        np.concatenate([global_mask, frontier_mask], axis=0),
    )


def _build_projector(config):
    values = dict(config)
    projector_type = values.pop("type", "generic_mlp")
    if projector_type == "generic_mlp":
        return _vlm.VisibilityTokenProjector(**values)
    if projector_type == "typed_scalar_basis":
        return _vlm.TypedScalarVisibilityTokenProjector(**values)
    if projector_type == "slot_typed_scalar_basis":
        return _vlm.SlotTypedScalarVisibilityTokenProjector(**values)
    raise ValueError("unsupported visibility projector type: %s" % projector_type)


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


def _run_evaluations(model, projector, prepared, controls, max_new_tokens):
    rows = []
    model.vlm.model.language_model.eval()
    projector.eval()
    for example in prepared:
        target = example["record"]["target"]
        for control in controls:
            tokens, mask = _load_control_tokens(
                example["record"],
                control,
                example.get("sequence_permutation"),
            )
            answer = _training.generate_visibility_grounding_answer(
                model,
                example["inputs"],
                torch.from_numpy(tokens).to(model.device),
                torch.from_numpy(mask).to(model.device),
                projector,
                max_new_tokens=int(max_new_tokens),
            )
            task_field = example["task_field"]
            if task_field is None:
                parsed = _parse_answer(answer)
                canonical_exact = answer == example["expected_answer"]
                field_correct = {
                    field: bool(parsed is not None and parsed.get(field) == value)
                    for field, value in target.items()
                }
            else:
                parsed = answer.strip()
                canonical_exact = parsed == example["expected_answer"]
                field_correct = {task_field: canonical_exact}
            control_expected_answer = example.get(
                "control_expected_answers", {}
            ).get(control, example["expected_answer"])
            rows.append(
                {
                    "example_id": example.get("example_id"),
                    "sample_id": example["record"]["sample_id"],
                    "task_field": task_field,
                    "split": example.get("split"),
                    "pair_id": example.get("pair_id"),
                    "control": control,
                    "answer": answer,
                    "parsed": parsed,
                    "expected_answer": example["expected_answer"],
                    "control_expected_answer": control_expected_answer,
                    "control_semantic_exact": (
                        answer.strip() == control_expected_answer
                    ),
                    "canonical_exact": canonical_exact,
                    "field_correct": field_correct,
                }
            )
    return rows


def _parameter_delta_norm(parameters, before):
    squared = 0.0
    for parameter, reference in zip(parameters, before):
        difference = parameter.detach().float() - reference
        squared += float(torch.sum(difference * difference).item())
    return float(squared**0.5)


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
    curriculum = None
    if protocol["stage"] == "V1e_route151_row_addressed_overfit":
        curriculum = _verify_row_curriculum(
            protocol["curriculum"], protocol["manifest"], records
        )
    if protocol["stage"] in {
        "V1f_route151_route_readout_overfit",
        "V1g_route151_typed_route_readout_overfit",
        "V1h_route151_slot_typed_route_readout_overfit",
    }:
        curriculum = _verify_route_readout_curriculum(
            protocol["curriculum"], protocol["manifest"], records
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
    projector = _build_projector(projector_config).to(model.device)
    scope = _training.visibility_grounding_trainable_scope(model, projector)
    if protocol["training"]["gradient_checkpointing"]:
        model.vlm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.vlm.model.visual.eval()
    model.vlm.model.language_model.train()
    projector.train()

    prepare_started = time.monotonic()
    objective = protocol.get("objective", {"type": "composite_json"})
    if objective.get("type") == "factorized_fields":
        task_fields = list(objective["fields"])
    elif objective.get("type") == "composite_json":
        task_fields = [None]
    elif objective.get("type") in {"row_addressed_fields", "route_readout_pairs"}:
        task_fields = []
    else:
        raise ValueError("unsupported grounding objective")
    record_by_id = {record["sample_id"]: record for record in records}
    if curriculum is None:
        preparation_specs = [
            {
                "record": record,
                "task_field": task_field,
                "question": (
                    manifest["question"]
                    if task_field is None
                    else _grounding.FACTORIZED_GROUNDING_QUESTIONS[task_field]
                ),
                "expected_answer": (
                    record["canonical_answer"]
                    if task_field is None
                    else record["target"][task_field]
                ),
                "example_id": None,
                "sequence_permutation": None,
                "control_expected_answers": {},
            }
            for record in records
            for task_field in task_fields
        ]
    else:
        preparation_specs = [
            {
                "record": record_by_id[example["sample_id"]],
                "task_field": example["task_field"],
                "question": example["question"],
                "expected_answer": example["expected_answer"],
                "example_id": example["example_id"],
                "sequence_permutation": example[
                    "sequence_permutation_new_to_manifest"
                ],
                "control_expected_answers": example[
                    "control_expected_answers"
                ],
                "split": example.get("split"),
                "pair_id": example.get("pair_id"),
            }
            for example in curriculum["examples"]
        ]
    prepared = []
    for spec in preparation_specs:
        record = spec["record"]
        true_tokens, true_mask = _load_control_tokens(
            record, "true_u", spec["sequence_permutation"]
        )
        inputs = model.processor.encode_vqa(
            record["camera_images"],
            spec["question"],
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
        answer_ids = _training.encode_grounding_answer(
            model.processor, spec["expected_answer"], model.device
        )
        prepared.append(
            {
                "record": record,
                "task_field": spec["task_field"],
                "question": spec["question"],
                "expected_answer": spec["expected_answer"],
                "example_id": spec["example_id"],
                "sequence_permutation": spec["sequence_permutation"],
                "control_expected_answers": spec[
                    "control_expected_answers"
                ],
                "split": spec.get("split"),
                "pair_id": spec.get("pair_id"),
                "inputs": inputs,
                "base_embeddings": base_embeddings,
                "true_tokens": torch.from_numpy(true_tokens).to(model.device),
                "true_mask": torch.from_numpy(true_mask).to(model.device),
                "answer_ids": answer_ids,
            }
        )
    prepare_seconds = time.monotonic() - prepare_started

    evaluation_example_ids = (
        set(curriculum.get("evaluation_example_ids", []))
        if curriculum is not None
        else set()
    )
    evaluation_prepared = (
        [
            example
            for example in prepared
            if example["example_id"] in evaluation_example_ids
        ]
        if evaluation_example_ids
        else prepared
    )
    pre_training_started = time.monotonic()
    pre_training_evaluations = _run_evaluations(
        model,
        projector,
        evaluation_prepared,
        protocol["evaluation"].get("pre_training_controls", []),
        protocol["evaluation"]["max_new_tokens"],
    )
    pre_training_evaluation_seconds = time.monotonic() - pre_training_started
    model.vlm.model.language_model.train()
    projector.train()

    projector_parameters = list(projector.parameters())
    lora_named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    lora_parameters = [parameter for _, parameter in lora_named_parameters]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": projector_parameters,
                "lr": float(protocol["training"]["projector_learning_rate"]),
            },
            {
                "params": lora_parameters,
                "lr": float(protocol["training"]["lora_learning_rate"]),
            },
        ],
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    history = []
    first_projector_gradients = None
    first_lora_gradients = None
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if curriculum is None:
        optimizer_examples = [
            prepared[index % len(prepared)]
            for index in range(int(protocol["training"]["optimizer_steps"]))
        ]
    else:
        prepared_by_id = {example["example_id"]: example for example in prepared}
        optimizer_examples = [
            prepared_by_id[example_id]
            for example_id in curriculum["training_schedule"]
        ]
    if len(optimizer_examples) != int(protocol["training"]["optimizer_steps"]):
        raise RuntimeError("optimizer schedule length changed")
    for optimizer_step, example in enumerate(optimizer_examples, start=1):
        step_started = time.monotonic()
        projector_before = [
            parameter.detach().float().clone()
            for parameter in projector_parameters
        ]
        lora_before = [parameter.detach().float().clone() for parameter in lora_parameters]
        forward_started = time.monotonic()
        result = _training.visibility_grounding_answer_loss(
            model,
            example["inputs"],
            example["true_tokens"],
            example["true_mask"],
            projector,
            example["answer_ids"],
            base_embeddings=example["base_embeddings"],
        )
        forward_seconds = time.monotonic() - forward_started
        if not torch.isfinite(result.loss):
            raise RuntimeError("V1b grounding loss is non-finite")
        backward_started = time.monotonic()
        result.loss.backward()
        backward_seconds = time.monotonic() - backward_started
        projector_gradients = _gradient_report(projector.named_parameters())
        lora_gradients = _gradient_report(
            lora_named_parameters
        )
        if optimizer_step == 1:
            first_projector_gradients = projector_gradients
            first_lora_gradients = lora_gradients
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
        projector_maximum_gradient_norm = float(
            protocol["training"].get(
                "projector_maximum_gradient_norm",
                protocol["training"]["maximum_gradient_norm"],
            )
        )
        lora_maximum_gradient_norm = float(
            protocol["training"].get(
                "lora_maximum_gradient_norm",
                protocol["training"]["maximum_gradient_norm"],
            )
        )
        projector_gradient_norm = torch.nn.utils.clip_grad_norm_(
            projector_parameters, projector_maximum_gradient_norm
        )
        lora_gradient_norm = torch.nn.utils.clip_grad_norm_(
            lora_parameters, lora_maximum_gradient_norm
        )
        optimizer.step()
        projector_update_norm = _parameter_delta_norm(
            projector_parameters, projector_before
        )
        lora_update_norm = _parameter_delta_norm(lora_parameters, lora_before)
        optimizer.zero_grad(set_to_none=True)
        history.append(
            {
                "optimizer_step": optimizer_step,
                "sample_id": example["record"]["sample_id"],
                "example_id": example["example_id"],
                "task_field": example["task_field"],
                "loss": float(result.loss.detach().item()),
                "projector_gradient_norm_before_clip": float(
                    projector_gradient_norm.item()
                ),
                "lora_gradient_norm_before_clip": float(lora_gradient_norm.item()),
                "projector_update_norm": projector_update_norm,
                "lora_update_norm": lora_update_norm,
                "projector_nonzero_gradient_tensors": sum(
                    row["finite"] and row["norm"] > 0.0
                    for row in projector_gradients
                ),
                "lora_nonzero_gradient_tensors": sum(
                    row["finite"] and row["norm"] > 0.0
                    for row in lora_gradients
                ),
                "forward_seconds": float(forward_seconds),
                "backward_seconds": float(backward_seconds),
                "optimizer_step_seconds": float(time.monotonic() - step_started),
                "answer_token_count": result.answer_token_count,
                "base_prompt_length": result.base_prompt_length,
                "augmented_prompt_length": result.augmented_prompt_length,
                "full_sequence_length": result.full_sequence_length,
                "insertion_index": result.insertion_index,
                "visibility_token_count": result.visibility_token_count,
            }
        )

    evaluation_started = time.monotonic()
    evaluations = _run_evaluations(
        model,
        projector,
        evaluation_prepared,
        protocol["evaluation"]["controls"],
        protocol["evaluation"]["max_new_tokens"],
    )
    evaluation_seconds = time.monotonic() - evaluation_started

    adaptation = _training.adaptation_state_dict(model, projector)
    checkpoint = {
        "schema": _training.VISIBILITY_GROUNDING_TRAINING_SCHEMA,
        "status": protocol["stage"].lower() + "_complete",
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
        "stage": protocol["stage"],
        "objective": objective,
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
        "prepare_seconds": float(prepare_seconds),
        "pre_training_evaluation_seconds": float(
            pre_training_evaluation_seconds
        ),
        "training_seconds": float(
            sum(row["optimizer_step_seconds"] for row in history)
        ),
        "evaluation_seconds": float(evaluation_seconds),
        "scope": scope,
        "lora": lora_config.as_dict(),
        "installed_lora_modules": list(installed),
        "projector_gradients_before_first_optimizer_step": first_projector_gradients,
        "lora_gradients_before_first_optimizer_step": first_lora_gradients,
        "history": history,
        "pre_training_evaluations": pre_training_evaluations,
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
    if curriculum is not None:
        report.update(
            {
                "curriculum_path": str(Path(protocol["curriculum"]).resolve()),
                "curriculum_sha256": _sha256(protocol["curriculum"]),
                "curriculum_example_count": len(curriculum["examples"]),
            }
        )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
