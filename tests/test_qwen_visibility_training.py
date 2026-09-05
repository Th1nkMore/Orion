import importlib.util
from pathlib import Path
import sys
import types

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


def test_non_full_attention_target_fails_closed():
    model = _Model()
    training.freeze_qwen_for_visibility_grounding(model)
    with pytest.raises(ValueError, match="not full attention"):
        training.install_upper_full_attention_lora(
            model, training.VisibilityLoRAConfig(layer_indices=(26,))
        )

