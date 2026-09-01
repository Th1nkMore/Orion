import torch

from scripts.train_stage2l_v121_factorized_r_smoke import (
    copy_single_head_to_factorized,
    factorized_gate,
)
from uq_estimator.stage2l_factorized_relevance_v121 import (
    FactorizedTaskRelevanceMapHead,
)
from uq_estimator.uq_relevance_tokenizer import TaskRelevanceMapHead


def test_single_head_warm_start_preserves_both_components_and_union():
    torch.manual_seed(3)
    single = TaskRelevanceMapHead(model_dim=8, hidden_dim=4)
    factorized = FactorizedTaskRelevanceMapHead(model_dim=8, hidden_dim=4)
    result = copy_single_head_to_factorized(factorized, single.state_dict())
    grid = torch.randn(2, 3, 2, 2, 8)
    with torch.no_grad():
        expected = single(grid).sigmoid()
        output = factorized(grid)
    assert result["route_actor_outputs_initially_identical"] is True
    assert torch.equal(expected, output.route_probability)
    assert torch.equal(expected, output.actor_probability)
    assert torch.equal(expected, output.derived_union_probability)


def _metrics(nonfront=0.5, train_macro=0.9, route=0.7, actor_front=0.5, fpr=0.01):
    def split(macro):
        cells = {
            component: {
                view: {
                    "mean_group_foreground_recall": (
                        route if component == "route" and view == "CAM_FRONT"
                        else actor_front if component == "actor" and view == "CAM_FRONT"
                        else nonfront if component == "actor" else None
                    ),
                    "mean_group_background_false_positive_rate": fpr,
                }
                for view in (
                    "CAM_FRONT",
                    "CAM_FRONT_LEFT",
                    "CAM_FRONT_RIGHT",
                    "CAM_BACK",
                    "CAM_BACK_LEFT",
                    "CAM_BACK_RIGHT",
                )
            }
            for component in ("route", "actor")
        }
        return {
            "supported_component_view_macro_recall": macro,
            "actor_nonfront_macro_recall": nonfront,
            "per_component_view": cells,
        }
    return {"train": split(train_macro), "dev": split(0.5)}


def _gates():
    return {
        "train_min_supported_macro_recall": 0.8,
        "dev_min_route_front_recall": 0.62,
        "dev_min_actor_front_recall": 0.4,
        "dev_min_actor_nonfront_macro_recall": 0.35,
        "dev_min_each_actor_nonfront_recall": 0.05,
        "dev_max_mean_background_fpr": 0.1,
        "baseline_dev_actor_nonfront_macro_recall": 0.09806547619047619,
        "minimum_actor_nonfront_absolute_improvement": 0.25,
        "background_fpr_cells": [
            "route/CAM_FRONT",
            "actor/CAM_FRONT",
            "actor/CAM_FRONT_LEFT",
            "actor/CAM_FRONT_RIGHT",
            "actor/CAM_BACK",
            "actor/CAM_BACK_LEFT",
        ],
    }


def test_factorized_gate_passes_meaningful_actor_repair():
    checks = factorized_gate(_metrics(), _gates())
    assert all(checks.values())


def test_factorized_gate_rejects_zero_side_actor_recall():
    checks = factorized_gate(_metrics(nonfront=0.0), _gates())
    assert checks["dev_actor_nonfront_macro"] is False
    assert checks["dev_actor_nonfront_each_positive"] is False
    assert checks["dev_actor_nonfront_absolute_improvement"] is False
