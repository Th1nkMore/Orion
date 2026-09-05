"""Training primitives for structured Qwen visibility-token grounding.

This module belongs only in the Qwen/Torch environment.  It keeps every base
parameter frozen, adds explicit LoRA residuals to declared upper full-attention
layers, and computes answer-only language loss from the same continuous-token
insertion contract accepted in V0.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .qwen_visibility_vlm import (
    VisibilityTokenProjector,
    _last_vision_end_insertion_index,
    _official_multimodal_embeddings,
    _validate_single_scene_tokens,
    prefill_with_visibility_tokens,
)


VISIBILITY_GROUNDING_TRAINING_SCHEMA = (
    "orion.qwen-visibility-grounding-training/v1"
)


@dataclass(frozen=True)
class VisibilityLoRAConfig:
    """Declared low-rank capacity for the first VLM grounding pilot."""

    layer_indices: Tuple[int, ...] = (27, 31)
    module_names: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        layers = tuple(int(value) for value in self.layer_indices)
        modules = tuple(str(value) for value in self.module_names)
        if not layers or len(set(layers)) != len(layers) or min(layers) < 0:
            raise ValueError("LoRA layer indices must be unique non-negative integers")
        if not modules or len(set(modules)) != len(modules):
            raise ValueError("LoRA module names must be unique and non-empty")
        if any(name not in {"q_proj", "k_proj", "v_proj", "o_proj"} for name in modules):
            raise ValueError("V1 LoRA supports q/k/v/o projections only")
        if isinstance(self.rank, bool) or int(self.rank) != self.rank or int(self.rank) <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        if not math.isfinite(float(self.alpha)) or float(self.alpha) <= 0.0:
            raise ValueError("LoRA alpha must be finite and positive")
        if not math.isfinite(float(self.dropout)) or not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("LoRA dropout must lie in [0,1)")
        object.__setattr__(self, "layer_indices", layers)
        object.__setattr__(self, "module_names", modules)
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "dropout", float(self.dropout))

    def as_dict(self) -> Dict[str, object]:
        return {
            "layer_indices": list(self.layer_indices),
            "module_names": list(self.module_names),
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
        }


class VisibilityLoRALinear(nn.Module):
    """A frozen linear layer plus a small float32 low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("VisibilityLoRALinear can wrap nn.Linear only")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        device = base.weight.device
        self.lora_a = nn.Parameter(
            torch.empty(self.rank, base.in_features, device=device, dtype=torch.float32)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, self.rank, device=device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        adapter_inputs = self.dropout(inputs).to(self.lora_a.dtype)
        update = F.linear(F.linear(adapter_inputs, self.lora_a), self.lora_b)
        return base_output + update.to(base_output.dtype) * self.scaling


def freeze_qwen_for_visibility_grounding(model) -> None:
    """Freeze the released Qwen-Drive model before adapters are installed."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()


def install_upper_full_attention_lora(
    model, config: VisibilityLoRAConfig
) -> Tuple[str, ...]:
    """Install fail-closed LoRA modules in declared full-attention layers."""

    language_model = model.vlm.model.language_model
    layers = language_model.layers
    installed = []
    for layer_index in config.layer_indices:
        if layer_index >= len(layers):
            raise ValueError("LoRA layer index %d is out of range" % layer_index)
        layer = layers[layer_index]
        if getattr(layer, "block_type", None) != "full_attention" or not hasattr(
            layer, "self_attn"
        ):
            raise ValueError("LoRA layer %d is not full attention" % layer_index)
        for module_name in config.module_names:
            base = getattr(layer.self_attn, module_name, None)
            if isinstance(base, VisibilityLoRALinear):
                raise ValueError("LoRA already installed at layer %d %s" % (layer_index, module_name))
            if not isinstance(base, nn.Linear):
                raise TypeError("expected nn.Linear at layer %d %s" % (layer_index, module_name))
            adapter = VisibilityLoRALinear(
                base,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            )
            setattr(layer.self_attn, module_name, adapter)
            installed.append(
                "vlm.model.language_model.layers.%d.self_attn.%s"
                % (layer_index, module_name)
            )
    return tuple(installed)


def visibility_grounding_trainable_scope(model, projector: nn.Module) -> dict:
    """Audit that no released base or Planning Expert weight can be optimized."""

    model_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    projector_trainable = [
        (name, parameter)
        for name, parameter in projector.named_parameters()
        if parameter.requires_grad
    ]
    unexpected = [
        name
        for name, _ in model_trainable
        if not (name.endswith(".lora_a") or name.endswith(".lora_b"))
    ]
    if unexpected:
        raise RuntimeError("unexpected trainable Qwen parameters: %s" % unexpected)
    if any(name.startswith("planning_expert.") for name, _ in model_trainable):
        raise RuntimeError("Planning Expert must remain frozen in V1")
    if not model_trainable or not projector_trainable:
        raise RuntimeError("V1 requires both LoRA and projector trainable parameters")
    if any(not parameter.requires_grad for parameter in projector.parameters()):
        raise RuntimeError("the complete visibility projector must be trainable")
    return {
        "schema": VISIBILITY_GROUNDING_TRAINING_SCHEMA,
        "model_trainable_names": [name for name, _ in model_trainable],
        "model_trainable_parameter_count": int(
            sum(parameter.numel() for _, parameter in model_trainable)
        ),
        "projector_trainable_names": [name for name, _ in projector_trainable],
        "projector_trainable_parameter_count": int(
            sum(parameter.numel() for _, parameter in projector_trainable)
        ),
        "planning_expert_trainable_parameter_count": int(
            sum(
                parameter.numel()
                for parameter in model.planning_expert.parameters()
                if parameter.requires_grad
            )
        ),
        "vision_trainable_parameter_count": int(
            sum(
                parameter.numel()
                for parameter in model.vlm.model.visual.parameters()
                if parameter.requires_grad
            )
        ),
        "embedding_trainable": bool(
            model.vlm.get_input_embeddings().weight.requires_grad
        ),
        "lm_head_trainable": bool(model.vlm.lm_head.weight.requires_grad),
    }


def encode_grounding_answer(processor, canonical_answer: str, device) -> torch.Tensor:
    """Encode the exact JSON plus the released ChatML assistant turn ending."""

    if not canonical_answer or "\n" in canonical_answer:
        raise ValueError("canonical grounding answer must be one non-empty line")
    content = processor.tokenizer.encode(canonical_answer, add_special_tokens=False)
    if not content:
        raise ValueError("canonical grounding answer tokenized to an empty sequence")
    token_ids = content + [int(processor.im_end_id)] + [
        int(value) for value in processor.newline_ids
    ]
    return torch.tensor(token_ids, dtype=torch.long, device=device)


@dataclass
class VisibilityGroundingLossResult:
    loss: torch.Tensor
    answer_token_count: int
    base_prompt_length: int
    augmented_prompt_length: int
    full_sequence_length: int
    insertion_index: int
    visibility_token_count: int


def _augmented_prompt_components(
    model,
    inputs: dict,
    token_features: torch.Tensor,
    token_mask: torch.Tensor,
    projector: VisibilityTokenProjector,
    base_embeddings: Optional[torch.Tensor] = None,
):
    input_ids = inputs["input_ids"]
    features, mask = _validate_single_scene_tokens(
        token_features, token_mask, projector.feature_dim
    )
    features = features.to(input_ids.device)
    mask = mask.to(input_ids.device)
    projected = projector(features[mask]).to(
        dtype=model.vlm.get_input_embeddings().weight.dtype
    )
    boundaries = projector.boundary_embeddings.to(projected.dtype)
    visibility_block = torch.cat(
        [boundaries[0:1], projected, boundaries[1:2]], dim=0
    ).unsqueeze(0)
    if base_embeddings is None:
        base_embeddings = _official_multimodal_embeddings(model, inputs)
    if base_embeddings.shape[:2] != input_ids.shape:
        raise ValueError("precomputed base embeddings do not match prompt ids")
    insertion = _last_vision_end_insertion_index(model, input_ids)
    prompt_embeddings = torch.cat(
        [
            base_embeddings[:, :insertion],
            visibility_block,
            base_embeddings[:, insertion:],
        ],
        dim=1,
    )
    block_length = int(visibility_block.shape[1])
    dummy_ids = torch.zeros(
        (1, block_length), dtype=input_ids.dtype, device=input_ids.device
    )
    if int(model.config.vlm_config.image_token_id) == 0:
        raise ValueError("visibility dummy id collides with image token")
    shadow_ids = torch.cat(
        [input_ids[:, :insertion], dummy_ids, input_ids[:, insertion:]], dim=1
    )
    return prompt_embeddings, shadow_ids, insertion, int(mask.sum().item())


def visibility_grounding_answer_loss(
    model,
    inputs: dict,
    token_features: torch.Tensor,
    token_mask: torch.Tensor,
    projector: VisibilityTokenProjector,
    answer_token_ids: torch.Tensor,
    base_embeddings: Optional[torch.Tensor] = None,
) -> VisibilityGroundingLossResult:
    """Compute causal CE only for the canonical assistant answer and turn end."""

    if answer_token_ids.ndim != 1 or not int(answer_token_ids.numel()):
        raise ValueError("answer_token_ids must be a non-empty 1D tensor")
    prompt_embeddings, shadow_ids, insertion, valid_count = (
        _augmented_prompt_components(
            model,
            inputs,
            token_features,
            token_mask,
            projector,
            base_embeddings=base_embeddings,
        )
    )
    answer_token_ids = answer_token_ids.to(shadow_ids.device)
    answer_embeddings = model.vlm.get_input_embeddings()(
        answer_token_ids.unsqueeze(0)
    )
    full_embeddings = torch.cat([prompt_embeddings, answer_embeddings], dim=1)
    full_shadow_ids = torch.cat(
        [shadow_ids, answer_token_ids.unsqueeze(0)], dim=1
    )
    positions = model._rope_positions(
        full_shadow_ids, inputs["image_grid_thw"]
    )
    outputs = model.vlm.model.language_model(
        input_ids=None,
        inputs_embeds=full_embeddings,
        position_ids=positions,
        use_cache=False,
    )
    answer_start = int(prompt_embeddings.shape[1])
    answer_count = int(answer_token_ids.numel())
    prediction_hidden = outputs.last_hidden_state[
        :, answer_start - 1 : answer_start + answer_count - 1
    ]
    if prediction_hidden.shape[1] != answer_count:
        raise RuntimeError("answer-only hidden-state slice has the wrong length")
    logits = model.vlm.lm_head(prediction_hidden).float()
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), answer_token_ids.reshape(-1)
    )
    return VisibilityGroundingLossResult(
        loss=loss,
        answer_token_count=answer_count,
        base_prompt_length=int(inputs["input_ids"].shape[1]),
        augmented_prompt_length=answer_start,
        full_sequence_length=int(full_embeddings.shape[1]),
        insertion_index=insertion,
        visibility_token_count=valid_count,
    )


@torch.no_grad()
def generate_visibility_grounding_answer(
    model,
    inputs: dict,
    token_features: torch.Tensor,
    token_mask: torch.Tensor,
    projector: VisibilityTokenProjector,
    max_new_tokens: int = 64,
) -> str:
    """Greedily decode a structured answer from the augmented VLM prompt."""

    if int(max_new_tokens) <= 0:
        raise ValueError("max_new_tokens must be positive")
    prompt = prefill_with_visibility_tokens(
        model,
        inputs,
        token_features=token_features,
        token_mask=token_mask,
        projector=projector,
        enabled=True,
    )
    cache = prompt.past_key_values
    logits = model.vlm.lm_head(prompt.last_hidden_state[:, -1]).float()
    content = []
    terminators = {
        int(model.processor.im_end_id),
        int(model.config.vlm_config.text_config.eos_token_id),
    }
    for offset in range(1, int(max_new_tokens) + 1):
        token_id = int(torch.argmax(logits, dim=-1).item())
        if token_id in terminators:
            break
        content.append(token_id)
        token = torch.tensor(
            [[token_id]], dtype=torch.long, device=prompt.anchor.device
        )
        outputs = model.vlm.model.language_model(
            input_ids=None,
            inputs_embeds=model.vlm.get_input_embeddings()(token),
            position_ids=prompt.anchor.unsqueeze(-1) + offset,
            past_key_values=cache,
            cache_position=torch.tensor(
                [prompt.augmented_sequence_length + offset - 1],
                dtype=torch.long,
                device=prompt.anchor.device,
            ),
            use_cache=True,
        )
        cache = outputs.past_key_values
        logits = model.vlm.lm_head(outputs.last_hidden_state[:, -1]).float()
    return model.processor.tokenizer.decode(
        content, skip_special_tokens=True
    ).strip()


def adaptation_state_dict(model, projector: nn.Module) -> dict:
    """Return adaptation tensors only; released base weights are never copied."""

    projector_state = {
        name: tensor.detach().cpu()
        for name, tensor in projector.state_dict().items()
    }
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, VisibilityLoRALinear):
            lora_state[name + ".lora_a"] = module.lora_a.detach().cpu()
            lora_state[name + ".lora_b"] = module.lora_b.detach().cpu()
    if not projector_state or not lora_state:
        raise RuntimeError("adaptation checkpoint would be empty")
    if any(".base." in name or name.endswith(".weight") for name in lora_state):
        raise RuntimeError("base model tensor leaked into adaptation checkpoint")
    return {"projector": projector_state, "lora": lora_state}

