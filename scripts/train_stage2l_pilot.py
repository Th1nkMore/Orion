#!/usr/bin/env python3
"""Train and evaluate the frozen 6/2 multi-event Stage2-L pilot.

Only the ORION LLM LoRA, spatial UQ tokenizer, and explicit task-relevance
head are trainable.  The Stage-1 observation-UQ adapter, ORION visual stack,
and trajectory decoder remain frozen and are not loaded into the optimizer.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mmcv.utils import set_random_seed

from scripts.train_stage2l_route196_overfit_smoke import (
    ORION_VISUAL_TOKENS,
    SPATIAL_UQ_TOKENS,
    _forward_record,
    _load_orion_lm,
    _prompt_tokens,
    _route_text,
)
from uq_estimator.stage2l_pilot import (
    PilotInputs,
    balanced_driving_epoch,
    binary_auroc,
    driving_stance_counts,
    load_pilot_inputs,
    matched_answer_preference_loss,
    parse_planning_stance,
    planning_stance,
    resolve_reference,
    sha256_file,
)
from uq_estimator.uq_relevance_tokenizer import (
    TaskRelevanceMapHead,
    UQComponentTokenizer,
)


SCHEMA = "orion.stage2l_pilot_training.v1"
CACHE_SCHEMA = "orion.stage2l_multiframe_visual_context_cache.v1"
PROTOCOL_SCHEMA = "orion.stage2l_uq_language_grounding_protocol.v1"


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


class PilotAssets:
    """Eagerly validated CPU assets for bounded one-record GPU training."""

    def __init__(self, inputs: PilotInputs) -> None:
        self.inputs = inputs
        self.records = inputs.records
        self.by_group_variant_family: Dict[
            Tuple[str, str, str], Dict[str, Any]
        ] = {}
        self.event_by_group: Dict[str, str] = {}
        self.split_by_group: Dict[str, str] = {}
        for row in self.records:
            group = str(row["counterfactual"]["group_id"])
            variant = str(row["counterfactual"]["variant"])
            family = str(row["question_family"])
            key = (group, variant, family)
            if key in self.by_group_variant_family:
                raise ValueError("duplicate pilot group/variant/family record")
            self.by_group_variant_family[key] = row
            event_id = str(row["event_id"])
            split = str(row["split"])
            if group in self.event_by_group and self.event_by_group[group] != event_id:
                raise ValueError("counterfactual group crosses events")
            self.event_by_group[group] = event_id
            self.split_by_group[group] = split

        self.visual_contexts: Dict[str, torch.Tensor] = {}
        for event_id, cache_path in inputs.event_cache_paths.items():
            cache = torch.load(cache_path, map_location="cpu")
            if cache.get("schema") != CACHE_SCHEMA:
                raise ValueError("unsupported visual cache payload for %s" % event_id)
            contexts = cache.get("contexts", {})
            if not isinstance(contexts, dict):
                raise ValueError("visual cache contexts are malformed")
            for group, value in contexts.items():
                group = str(group)
                if group in self.visual_contexts:
                    raise ValueError("visual context group is duplicated across events")
                if tuple(value.shape) != (1, ORION_VISUAL_TOKENS, 4096):
                    raise ValueError("visual context shape mismatch for %s" % group)
                self.visual_contexts[group] = value.detach().to(
                    dtype=torch.float16, device="cpu"
                )
        if set(self.visual_contexts) != set(self.event_by_group):
            raise ValueError("loaded visual contexts do not cover every pilot group")

        self.components: Dict[Tuple[str, str], torch.Tensor] = {}
        self.relevance: Dict[str, torch.Tensor] = {}
        self.route_text: Dict[str, str] = {}
        for group in sorted(self.event_by_group):
            relevance_arrays = []
            route_context = None
            for variant in (
                "observed", "zero_uq", "on_path_uq", "off_path_uq",
                "view_shuffled_uq",
            ):
                row = self.by_group_variant_family[(
                    group, variant, "task_relevance"
                )]
                uq_ref = row["model_input"]["stage1_observation_uq"]
                uq_path = resolve_reference(
                    uq_ref, inputs.records_path.parent,
                    "Stage1 component map for %s/%s" % (group, variant),
                )
                with np.load(uq_path) as archive:
                    value = archive[uq_ref["component_key"]].astype(np.float32)
                if value.shape != (4, 6, 40, 40, 3):
                    raise ValueError("unexpected Stage1 component shape")
                self.components[(group, variant)] = torch.from_numpy(value).unsqueeze(0)

                sidecar_ref = row["target"]["map_sidecar"]
                sidecar = resolve_reference(
                    sidecar_ref, inputs.records_path.parent,
                    "task-relevance sidecar for %s/%s" % (group, variant),
                )
                with np.load(sidecar) as archive:
                    current = archive[sidecar_ref["relevance_key"]].astype(np.float32)
                if current.shape != (6, 40, 40):
                    raise ValueError("unexpected task-relevance target shape")
                relevance_arrays.append(current)

                current_route = row["model_input"]["route_context"]["payload"]
                if route_context is None:
                    route_context = current_route
                elif current_route != route_context:
                    raise ValueError("matched counterfactuals have different route context")
            if any(
                not np.array_equal(relevance_arrays[0], value)
                for value in relevance_arrays[1:]
            ):
                raise ValueError("matched counterfactuals have different R targets")
            target = torch.from_numpy(relevance_arrays[0]).unsqueeze(0)
            self.relevance[group] = F.adaptive_avg_pool2d(target, (10, 10))
            self.route_text[group] = _route_text(route_context)

    def row(self, group: str, variant: str, family: str) -> Dict[str, Any]:
        return self.by_group_variant_family[(group, variant, family)]

    def groups(self, split: str) -> List[str]:
        return sorted(group for group, value in self.split_by_group.items() if value == split)

    def opposite_stance_answer(self, row: Mapping[str, Any]) -> str:
        """Return a matched answer with the opposite driving stance.

        The visual/UQ/route input stays identical in the second forward pass;
        only the supervised answer changes.  Answers are always drawn from the
        same counterfactual group, never from another event or frame.
        """
        if row["question_family"] != "driving_implication":
            raise ValueError("opposite stance is defined only for driving QA")
        group = str(row["counterfactual"]["group_id"])
        target_stance = str(
            row["target"]["structured_summary"]["planning_implication"]["stance"]
        )
        alternative_variant = (
            "on_path_uq" if target_stance == "maintain" else "zero_uq"
        )
        alternative = self.row(group, alternative_variant, "driving_implication")
        alternative_stance = str(
            alternative["target"]["structured_summary"]
            ["planning_implication"]["stance"]
        )
        if target_stance == "maintain" and alternative_stance == "maintain":
            raise ValueError("matched on-path answer is not conservative")
        if target_stance != "maintain" and alternative_stance != "maintain":
            raise ValueError("matched zero-UQ answer is not maintain")
        return str(alternative["conversation"][1]["value"])


def _call_forward(
    *, lm, tokenizer, uq_tokenizer, relevance_head, assets: PilotAssets,
    row: Mapping[str, Any], answer: str = None,
) -> Dict[str, torch.Tensor]:
    group = str(row["counterfactual"]["group_id"])
    variant = str(row["counterfactual"]["variant"])
    return _forward_record(
        lm=lm,
        tokenizer=tokenizer,
        uq_tokenizer=uq_tokenizer,
        relevance_head=relevance_head,
        baseline_vision=assets.visual_contexts[group].float().cuda(non_blocking=True),
        components=assets.components[(group, variant)],
        relevance_target=assets.relevance[group].cuda(non_blocking=True),
        on_components=assets.components[(group, "on_path_uq")],
        off_components=assets.components[(group, "off_path_uq")],
        row=row,
        route_text=assets.route_text[group],
        answer=answer,
    )


@torch.no_grad()
def _generate_stance(
    *, lm, tokenizer, uq_tokenizer, assets: PilotAssets,
    row: Mapping[str, Any],
) -> Tuple[str, Any]:
    group = str(row["counterfactual"]["group_id"])
    variant = str(row["counterfactual"]["variant"])
    prompt = _prompt_tokens(tokenizer, row, assets.route_text[group]).cuda()
    uq = uq_tokenizer(assets.components[(group, variant)].cuda(non_blocking=True))
    vision = torch.cat((assets.visual_contexts[group].float().cuda(), uq.tokens), dim=1)
    output_ids = lm.generate(
        inputs=prompt,
        images=vision,
        do_sample=False,
        num_beams=1,
        max_new_tokens=72,
        use_cache=True,
    )
    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return text, parse_planning_stance(text)


@torch.no_grad()
def evaluate(
    *, lm, tokenizer, uq_tokenizer, relevance_head, assets: PilotAssets,
    generate_text: bool,
) -> Dict[str, Any]:
    lm.eval()
    uq_tokenizer.eval()
    relevance_head.eval()
    language_nll = []
    map_bce = []
    risk_ordering = []
    zero_off_preferences = []
    on_preferences = []
    map_scores = []
    map_labels = []
    generated_rows = []
    map_text_consistency = []
    target_text_accuracy = []

    for row in assets.records:
        if row["split"] != "dev":
            continue
        result = _call_forward(
            lm=lm, tokenizer=tokenizer, uq_tokenizer=uq_tokenizer,
            relevance_head=relevance_head, assets=assets, row=row,
        )
        language_nll.append(float(result["language_loss"].item()))

    for group in assets.groups("dev"):
        canonical = assets.row(group, "observed", "task_relevance")
        canonical_result = _call_forward(
            lm=lm, tokenizer=tokenizer, uq_tokenizer=uq_tokenizer,
            relevance_head=relevance_head, assets=assets, row=canonical,
        )
        map_bce.append(float(canonical_result["map_loss"].item()))
        risk_ordering.append(
            float((canonical_result["on_peak"] - canonical_result["off_peak"]).item())
            >= 0.2
        )
        scores = canonical_result["relevance_logits"].sigmoid().detach().cpu().numpy().ravel()
        labels = assets.relevance[group].numpy().ravel() >= 0.5
        map_scores.extend(scores.tolist())
        map_labels.extend(labels.tolist())

        implication = {
            variant: assets.row(group, variant, "driving_implication")
            for variant in ("zero_uq", "off_path_uq", "on_path_uq")
        }
        maintain_answer = implication["zero_uq"]["conversation"][1]["value"]
        conservative_answer = implication["on_path_uq"]["conversation"][1]["value"]
        for variant in ("zero_uq", "off_path_uq", "on_path_uq"):
            row = implication[variant]
            expected = _call_forward(
                lm=lm, tokenizer=tokenizer, uq_tokenizer=uq_tokenizer,
                relevance_head=relevance_head, assets=assets, row=row,
            )
            alternative_answer = (
                conservative_answer if variant != "on_path_uq" else maintain_answer
            )
            alternative = _call_forward(
                lm=lm, tokenizer=tokenizer, uq_tokenizer=uq_tokenizer,
                relevance_head=relevance_head, assets=assets, row=row,
                answer=alternative_answer,
            )
            preferred = float(expected["language_loss"].item()) < float(
                alternative["language_loss"].item()
            )
            if variant == "on_path_uq":
                on_preferences.append(preferred)
            else:
                zero_off_preferences.append(preferred)
            if generate_text:
                text, generated_stance = _generate_stance(
                    lm=lm, tokenizer=tokenizer, uq_tokenizer=uq_tokenizer,
                    assets=assets, row=row,
                )
                risk = expected["current_risk"]
                flat_peak_index = int(risk.reshape(-1).argmax().item())
                patches_per_view = int(risk.shape[-2] * risk.shape[-1])
                predicted_peak_view = (
                    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
                )[flat_peak_index // patches_per_view]
                predicted_stance = planning_stance(
                    float(expected["current_peak"].item()), predicted_peak_view
                )
                target_stance = str(
                    row["target"]["structured_summary"]["planning_implication"]["stance"]
                )
                map_text_consistency.append(generated_stance == predicted_stance)
                target_text_accuracy.append(generated_stance == target_stance)
                generated_rows.append({
                    "group_id": group,
                    "variant": variant,
                    "text": text,
                    "parsed_stance": generated_stance,
                    "predicted_map_stance": predicted_stance,
                    "predicted_peak_view": predicted_peak_view,
                    "target_stance": target_stance,
                    "predicted_peak_task_risk": float(expected["current_peak"].item()),
                })

    metrics = {
        "dev_mean_language_nll": float(np.mean(language_nll)),
        "dev_mean_relevance_bce": float(np.mean(map_bce)),
        "heldout_on_path_risk_above_matched_off_path_fraction": float(np.mean(risk_ordering)),
        "heldout_zero_and_off_path_nonconservative_answer_fraction": float(np.mean(zero_off_preferences)),
        "heldout_on_path_conservative_answer_fraction": float(np.mean(on_preferences)),
        "heldout_task_relevance_auroc": binary_auroc(map_scores, map_labels),
        "generated_driving_implications": generated_rows,
    }
    if generate_text:
        metrics["map_text_consistency_fraction"] = float(np.mean(map_text_consistency))
        metrics["generated_target_stance_accuracy"] = float(np.mean(target_text_accuracy))
    lm.train()
    uq_tokenizer.train()
    relevance_head.train()
    return metrics


def _gate_checks(metrics: Mapping[str, Any]) -> Dict[str, bool]:
    return {
        "heldout_on_path_risk_above_matched_off_path_fraction_gte_0_8": (
            metrics["heldout_on_path_risk_above_matched_off_path_fraction"] >= 0.8
        ),
        "heldout_zero_and_off_path_nonconservative_answer_fraction_gte_0_8": (
            metrics["heldout_zero_and_off_path_nonconservative_answer_fraction"] >= 0.8
        ),
        "heldout_on_path_conservative_answer_fraction_gte_0_8": (
            metrics["heldout_on_path_conservative_answer_fraction"] >= 0.8
        ),
        "heldout_task_relevance_auroc_gte_0_75": (
            metrics["heldout_task_relevance_auroc"] >= 0.75
        ),
        "map_text_consistency_fraction_gte_0_9": (
            metrics.get("map_text_consistency_fraction", 0.0) >= 0.9
        ),
        "generated_target_stance_accuracy_gte_0_8": (
            metrics.get("generated_target_stance_accuracy", 0.0) >= 0.8
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate-lora", type=float, default=2e-5)
    parser.add_argument("--learning-rate-head", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--log-interval", type=int, default=20)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite Stage2-L pilot output")
    if args.epochs < 1 or (args.max_steps is not None and args.max_steps < 1):
        raise ValueError("epochs and max-steps must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("Stage2-L pilot training requires CUDA")
    protocol = _load_json(args.training_protocol.resolve())
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported Stage2-L training protocol")
    if protocol.get("launch_locks", {}).get("stage2l_pilot_training_allowed") is not True:
        raise ValueError(
            "Stage2-L pilot training remains locked by the supplied protocol"
        )
    losses = protocol["losses"]
    if float(losses.get("trajectory", -1)) != 0.0:
        raise ValueError("Stage2-L pilot protocol may not train trajectory loss")
    lambda_map = float(losses["dense_soft_relevance_bce"])
    lambda_ranking = float(losses["matched_onpath_offpath_risk_ranking"])
    preference = losses.get("matched_answer_preference")
    if not isinstance(preference, dict):
        raise ValueError("pilot protocol must define matched answer preference")
    lambda_answer_preference = float(preference["weight"])
    answer_preference_margin = float(preference["margin"])
    if lambda_answer_preference <= 0.0 or answer_preference_margin < 0.0:
        raise ValueError("pilot matched answer preference must be enabled")
    balance_driving_stances = bool(
        protocol.get("sampling", {}).get("balance_driving_stances", False)
    )

    set_random_seed(args.seed, deterministic=True)
    random.seed(args.seed)
    inputs = load_pilot_inputs(args.pilot_manifest.resolve())
    assets = PilotAssets(inputs)
    train_records = [row for row in assets.records if row["split"] == "train"]
    if not train_records or not assets.groups("dev"):
        raise ValueError("pilot train/dev partitions must both be non-empty")

    lm, tokenizer = _load_orion_lm(args.config.resolve(), args.checkpoint.resolve())
    uq_tokenizer = UQComponentTokenizer(
        model_dim=4096, hidden_dim=256, grid_hw=(10, 10)
    ).cuda()
    relevance_head = TaskRelevanceMapHead(model_dim=4096, hidden_dim=256).cuda()
    lora_parameters = [parameter for parameter in lm.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": args.learning_rate_lora},
            {
                "params": list(uq_tokenizer.parameters()) + list(relevance_head.parameters()),
                "lr": args.learning_rate_head,
            },
        ],
        weight_decay=1e-4,
    )

    before = evaluate(
        lm=lm, tokenizer=tokenizer, uq_tokenizer=uq_tokenizer,
        relevance_head=relevance_head, assets=assets, generate_text=False,
    )
    history = []
    step = 0
    stop = False
    for epoch in range(1, args.epochs + 1):
        order = (
            balanced_driving_epoch(train_records, random)
            if balance_driving_stances else list(train_records)
        )
        if not balance_driving_stances:
            random.shuffle(order)
        for row in order:
            if args.max_steps is not None and step >= args.max_steps:
                stop = True
                break
            step += 1
            optimizer.zero_grad(set_to_none=True)
            result = _call_forward(
                lm=lm, tokenizer=tokenizer, uq_tokenizer=uq_tokenizer,
                relevance_head=relevance_head, assets=assets, row=row,
            )
            preference_loss = result["language_loss"].new_zeros(())
            if row["question_family"] == "driving_implication":
                alternative = _call_forward(
                    lm=lm, tokenizer=tokenizer, uq_tokenizer=uq_tokenizer,
                    relevance_head=relevance_head, assets=assets, row=row,
                    answer=assets.opposite_stance_answer(row),
                )
                preference_loss = matched_answer_preference_loss(
                    result["language_loss"], alternative["language_loss"],
                    margin=answer_preference_margin,
                )
            loss = (
                result["language_loss"]
                + lambda_map * result["map_loss"]
                + lambda_ranking * result["ranking_loss"]
                + lambda_answer_preference * preference_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                lora_parameters
                + list(uq_tokenizer.parameters())
                + list(relevance_head.parameters()),
                1.0,
            )
            optimizer.step()
            if step == 1 or step % args.log_interval == 0:
                row_log = {
                    "epoch": epoch,
                    "step": step,
                    "event_id": row["event_id"],
                    "variant": row["counterfactual"]["variant"],
                    "question_family": row["question_family"],
                    "loss": float(loss.item()),
                    "language_nll": float(result["language_loss"].item()),
                    "relevance_bce": float(result["map_loss"].item()),
                    "ranking_loss": float(result["ranking_loss"].item()),
                    "answer_preference_loss": float(preference_loss.item()),
                }
                history.append(row_log)
                print("[Stage2LPilot] " + json.dumps(row_log, sort_keys=True), flush=True)
        epoch_metrics = evaluate(
            lm=lm, tokenizer=tokenizer, uq_tokenizer=uq_tokenizer,
            relevance_head=relevance_head, assets=assets, generate_text=False,
        )
        history.append({"epoch": epoch, "step": step, "dev": epoch_metrics})
        print("[Stage2LPilotDev] " + json.dumps(epoch_metrics, sort_keys=True), flush=True)
        if stop:
            break

    after = evaluate(
        lm=lm, tokenizer=tokenizer, uq_tokenizer=uq_tokenizer,
        relevance_head=relevance_head, assets=assets, generate_text=True,
    )
    checks = _gate_checks(after)
    passed = all(checks.values())
    status = "stage2l_pilot_pass" if passed else "stage2l_pilot_failed_gate"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "stage2l_pilot.pt"
    torch.save({
        "schema": SCHEMA,
        "status": status,
        "formal_training_ready": False,
        "stage2p_engineering_pilot_ready": passed,
        "uq_tokenizer": uq_tokenizer.state_dict(),
        "relevance_head": relevance_head.state_dict(),
        "lora": {
            name: value.detach().cpu()
            for name, value in lm.state_dict().items() if "lora_" in name
        },
        "steps": step,
        "epochs_requested": args.epochs,
    }, checkpoint_path)
    report = {
        "schema": SCHEMA,
        "status": status,
        "formal_training_ready": False,
        "stage2p_engineering_pilot_ready": passed,
        "steps": step,
        "epochs_requested": args.epochs,
        "event_count": 8,
        "train_event_count": 6,
        "dev_event_count": 2,
        "qa_record_count": len(assets.records),
        "sampling": {
            "balance_driving_stances": balance_driving_stances,
            "source_train_driving_stance_counts": driving_stance_counts(train_records),
            "non_driving_records_repeated": False,
        },
        "before": before,
        "after": after,
        "checks": checks,
        "history": history,
        "loss_weights": {
            "qa_causal_language_modeling": 1.0,
            "dense_soft_relevance_bce": lambda_map,
            "matched_onpath_offpath_risk_ranking": lambda_ranking,
            "matched_answer_preference": {
                "weight": lambda_answer_preference,
                "margin": answer_preference_margin,
                "scope": "driving_implication only",
                "matched_same_input_opposite_stance": True,
            },
            "trajectory": 0.0,
        },
        "architecture": {
            "trainable": [
                "ORION LLM LoRA", "spatial UQ tokenizer",
                "explicit task-relevance map head",
            ],
            "frozen": [
                "Stage1 observation-UQ adapter", "ORION visual encoder",
                "ORION detection/map heads", "ORION trajectory decoder",
            ],
            "legacy_density_uq_used": False,
            "hard_governor_used": False,
            "task_risk": "K = U * R",
        },
        "provenance": {
            "pilot_manifest": {
                "path": str(args.pilot_manifest.resolve()),
                "sha256": sha256_file(args.pilot_manifest.resolve()),
            },
            "training_protocol": {
                "path": str(args.training_protocol.resolve()),
                "sha256": sha256_file(args.training_protocol.resolve()),
            },
            "base_orion_checkpoint": {
                "path": str(args.checkpoint.resolve()),
                "sha256": sha256_file(args.checkpoint.resolve()),
            },
            "orion_training_config": {
                "path": str(args.config.resolve()),
                "sha256": sha256_file(args.config.resolve()),
            },
            "output_checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": sha256_file(checkpoint_path.resolve()),
            },
            "source_sha256": {
                str(path.relative_to(Path(__file__).resolve().parents[1])): sha256_file(path)
                for path in (
                    Path(__file__).resolve(),
                    Path(__file__).resolve().with_name(
                        "train_stage2l_route196_overfit_smoke.py"
                    ),
                    Path(__file__).resolve().parents[1]
                    / "uq_estimator" / "stage2l_pilot.py",
                    Path(__file__).resolve().parents[1]
                    / "uq_estimator" / "uq_relevance_tokenizer.py",
                )
            },
        },
        "claim_boundary": (
            "Held-out 6/2 Stage2-L engineering pilot only. Passing may unlock "
            "Stage2-P engineering work, but it is not formal generalization or "
            "closed-loop safety evidence."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    del assets
    gc.collect()
    print(json.dumps({
        "status": status,
        "checks": checks,
        "report": str((args.output_dir / "report.json").resolve()),
    }, indent=2, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
