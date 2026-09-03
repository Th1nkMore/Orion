import torch

from scripts.train_stage2l_v101_view_aligned_phase_a import (
    _average_precision,
    _copy_v10_query_state,
)
from uq_estimator.uq_relevance_tokenizer import (
    SpatialTaskRelevanceQueryTokenizer,
    ViewAlignedTaskRelevanceQueryTokenizer,
)


def test_v101_warm_start_copies_old_queries_but_not_new_evidence_path():
    torch.manual_seed(3)
    old = SpatialTaskRelevanceQueryTokenizer(
        model_dim=16, hidden_dim=8, grid_hw=(2, 2), max_views=3
    )
    new = ViewAlignedTaskRelevanceQueryTokenizer(
        model_dim=16,
        image_feature_dim=12,
        hidden_dim=8,
        grid_hw=(2, 2),
        max_views=3,
    )
    evidence_before = new.evidence_projection[0].weight.detach().clone()
    copied = _copy_v10_query_state(new, old.state_dict())
    assert set(copied) == set(old.state_dict())
    assert torch.equal(new.base_query, old.base_query)
    assert torch.equal(new.view_embedding.weight, old.view_embedding.weight)
    assert torch.equal(new.evidence_projection[0].weight, evidence_before)


def test_average_precision_exceeds_prevalence_for_correct_ordering():
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1]).numpy()
    truth = torch.tensor([1, 0, 1, 0], dtype=torch.bool).numpy()
    assert _average_precision(scores, truth) > truth.mean()
