import pytest
import torch

from uq_estimator.uq_relevance_tokenizer import (
    ViewAlignedTaskRelevanceQueryTokenizer,
)


def test_view_aligned_queries_change_only_the_bound_view_before_vlm_fusion():
    torch.manual_seed(4)
    module = ViewAlignedTaskRelevanceQueryTokenizer(
        model_dim=16,
        image_feature_dim=8,
        hidden_dim=12,
        grid_hw=(2, 2),
        max_views=3,
    )
    baseline = torch.zeros(1, 3, 2, 2, 8)
    changed = baseline.clone()
    changed[:, 1] = torch.arange(8, dtype=torch.float32)
    first = module(baseline).reshape(1, 3, 2, 2, 16)
    second = module(changed).reshape(1, 3, 2, 2, 16)
    assert torch.equal(first[:, 0], second[:, 0])
    assert not torch.equal(first[:, 1], second[:, 1])
    assert torch.equal(first[:, 2], second[:, 2])


def test_view_aligned_queries_reject_wrong_grid_or_feature_dim():
    module = ViewAlignedTaskRelevanceQueryTokenizer(
        model_dim=16,
        image_feature_dim=8,
        hidden_dim=12,
        grid_hw=(2, 2),
    )
    with pytest.raises(ValueError, match="shape differs"):
        module(torch.zeros(1, 6, 3, 2, 8))
    with pytest.raises(ValueError, match="shape differs"):
        module(torch.zeros(1, 6, 2, 2, 9))
    with pytest.raises(ValueError, match="finite"):
        value = torch.zeros(1, 6, 2, 2, 8)
        value[0, 0, 0, 0, 0] = float("nan")
        module(value)
