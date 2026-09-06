import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package = types.ModuleType("_qwen_visibility_training_test_package")
package.__path__ = [str(PROJECT_ROOT / "uq_estimator")]
sys.modules[package.__name__] = package
_load_local_module(
    package.__name__ + ".qwen_visibility_vlm",
    PROJECT_ROOT / "uq_estimator" / "qwen_visibility_vlm.py",
)
training = _load_local_module(
    package.__name__ + ".qwen_visibility_training",
    PROJECT_ROOT / "uq_estimator" / "qwen_visibility_training.py",
)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 8, bias=False)
        self.k_proj = nn.Linear(4, 2, bias=False)
        self.v_proj = nn.Linear(4, 2, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class _Layer(nn.Module):
    def __init__(self, full):
        super().__init__()
        self.block_type = "full_attention" if full else "linear_attention"
        if full:
            self.self_attn = _Attention()


class _Language(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(False) for _ in range(32)])
        self.layers[27] = _Layer(True)
        self.layers[31] = _Layer(True)


class _Visual(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class _VLMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _Language()
        self.visual = _Visual()


class _VLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _VLMModel()
        self.embedding = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)

    def get_input_embeddings(self):
        return self.embedding


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = _VLM()
        self.planning_expert = nn.Linear(4, 4)
        self.config = SimpleNamespace(
            vlm_config=SimpleNamespace(
                image_token_id=98,
                vision_end_token_id=99,
            )
        )


def test_lora_installation_is_identity_then_receives_gradient():
    model = _Model()
    original = model.vlm.model.language_model.layers[27].self_attn.q_proj
    inputs = torch.randn(3, 4)
    expected = original(inputs).detach()
    training.freeze_qwen_for_visibility_grounding(model)
    installed = training.install_upper_full_attention_lora(
        model, training.VisibilityLoRAConfig()
    )
    assert len(installed) == 8
    adapted = model.vlm.model.language_model.layers[27].self_attn.q_proj
    torch.testing.assert_close(adapted(inputs), expected)
    adapted(inputs).sum().backward()
    assert adapted.lora_b.grad is not None
    assert adapted.lora_b.grad.norm() > 0
    assert adapted.base.weight.grad is None


def test_scope_and_checkpoint_exclude_every_base_parameter():
    model = _Model()
    training.freeze_qwen_for_visibility_grounding(model)
    training.install_upper_full_attention_lora(
        model, training.VisibilityLoRAConfig()
    )
    vlm_module = sys.modules[package.__name__ + ".qwen_visibility_vlm"]
    projector = vlm_module.VisibilityTokenProjector(23, 8, 4)
    scope = training.visibility_grounding_trainable_scope(model, projector)
    assert scope["planning_expert_trainable_parameter_count"] == 0
    assert scope["vision_trainable_parameter_count"] == 0
    assert scope["embedding_trainable"] is False
    assert scope["lm_head_trainable"] is False
    assert all(
        name.endswith((".lora_a", ".lora_b"))
        for name in scope["model_trainable_names"]
    )
    state = training.adaptation_state_dict(model, projector)
    assert state["projector"]
    assert len(state["lora"]) == 16
    assert not any("base" in name for name in state["lora"])


def test_typed_scalar_projector_preserves_field_identity_and_absolute_value():
    vlm_module = sys.modules[package.__name__ + ".qwen_visibility_vlm"]
    projector = vlm_module.TypedScalarVisibilityTokenProjector(23, 8, 4)
    features = torch.zeros((2, 23), dtype=torch.float32)
    features[0, 16] = 0.2
    features[1, 17] = 0.2
    basis = projector.scalar_basis(features)
    assert basis.shape == (2, 23, 4)
    torch.testing.assert_close(
        basis[0, 16],
        torch.tensor([0.2, 0.04, torch.sin(torch.tensor(torch.pi * 0.2)), torch.cos(torch.tensor(torch.pi * 0.2))]),
    )
    assert basis[0, 17, 0] == 0.0
    assert basis[1, 16, 0] == 0.0
    assert basis[1, 17, 0] == pytest.approx(0.2)
    output = projector(features)
    torch.testing.assert_close(output, torch.zeros_like(output))
    full = vlm_module.TypedScalarVisibilityTokenProjector(23, 512, 2560)
    assert sum(parameter.numel() for parameter in full.parameters()) == 1_367_040
    assert len(full.state_dict()) == 7


def test_slot_typed_projector_adds_deterministic_sequence_identity():
    vlm_module = sys.modules[package.__name__ + ".qwen_visibility_vlm"]
    projector = vlm_module.SlotTypedScalarVisibilityTokenProjector(
        23, 4, 2, maximum_token_slots=3
    )
    assert projector.field_basis_projection.in_features == 23 * 4 + 3
    with torch.no_grad():
        projector.field_basis_projection.weight.zero_()
        projector.field_basis_projection.bias.zero_()
        projector.field_basis_projection.weight[0, 23 * 4] = 1.0
        projector.field_basis_projection.weight[1, 23 * 4 + 1] = 1.0
        projector.output_projection.weight.zero_()
        projector.output_projection.weight[0, 0] = 1.0
        projector.output_projection.weight[1, 1] = 1.0
        projector.output_projection.bias.zero_()
    output = projector(torch.zeros((2, 23)))
    assert output.shape == (2, 2)
    assert not torch.equal(output[0], output[1])
    with pytest.raises(ValueError, match="slot budget"):
        projector(torch.zeros((4, 23)))
    full = vlm_module.SlotTypedScalarVisibilityTokenProjector(23, 512, 2560)
    assert sum(parameter.numel() for parameter in full.parameters()) == 1_391_616
    assert len(full.state_dict()) == 7


def test_non_full_attention_target_fails_closed():
    model = _Model()
    training.freeze_qwen_for_visibility_grounding(model)
    with pytest.raises(ValueError, match="not full attention"):
        training.install_upper_full_attention_lora(
            model, training.VisibilityLoRAConfig(layer_indices=(26,))
        )


def test_training_prompt_inserts_once_and_preserves_official_suffix():
    model = _Model()
    vlm_module = sys.modules[package.__name__ + ".qwen_visibility_vlm"]
    projector = vlm_module.VisibilityTokenProjector(23, 8, 4)
    input_ids = torch.tensor([[1, 99, 2, 99, 3, 4]], dtype=torch.long)
    base = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    tokens = torch.zeros((3, 23), dtype=torch.float32)
    mask = torch.tensor([True, False, True])
    prompt, shadow, insertion, valid_count = training._augmented_prompt_components(
        model,
        {"input_ids": input_ids},
        tokens,
        mask,
        projector,
        base_embeddings=base,
    )
    assert insertion == 4
    assert valid_count == 2
    assert prompt.shape[1] == base.shape[1] + valid_count + 2
    assert shadow.shape[1] == prompt.shape[1]
    torch.testing.assert_close(prompt[:, :insertion], base[:, :insertion])
    torch.testing.assert_close(prompt[:, insertion + valid_count + 2 :], base[:, insertion:])
    torch.testing.assert_close(shadow[:, :insertion], input_ids[:, :insertion])
    torch.testing.assert_close(
        shadow[:, insertion + valid_count + 2 :], input_ids[:, insertion:]
    )
