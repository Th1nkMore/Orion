import torch

from scripts.train_stage2l_v6_route151_smoke import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    _expanded_labels,
    _same_variant_answers,
)
from uq_estimator.stage2l_matched_objective import (
    MATCHED_VARIANTS,
    QUESTION_FAMILIES,
    hard_language_supervision_allowed,
    partition_complete_matched_groups,
)


def _group():
    rows = []
    for variant in MATCHED_VARIANTS:
        for family in QUESTION_FAMILIES:
            rows.append(
                {
                    "sample_id": "%s/%s" % (variant, family),
                    "event_id": "route151_step218",
                    "split": "train",
                    "counterfactual": {
                        "group_id": "route151_step218_saved_0018",
                        "variant": variant,
                    },
                    "question_family": family,
                    "conversation": [
                        {"from": "human", "value": "%s question" % family},
                        {"from": "gpt", "value": "%s answer" % family},
                    ],
                }
            )
    return partition_complete_matched_groups(rows)[0]


def test_expanded_labels_replace_only_the_image_token():
    input_ids = torch.tensor([1, IMAGE_TOKEN_INDEX, 7, 8])
    labels = torch.tensor([IGNORE_INDEX, IGNORE_INDEX, 7, 8])
    expanded = _expanded_labels(
        labels, input_ids, visual_token_count=5
    )
    assert expanded.tolist() == [
        IGNORE_INDEX,
        *([IGNORE_INDEX] * 5),
        7,
        8,
    ]


def test_v6_group_has_eighteen_language_anchors_and_full_negatives():
    group = _group()
    anchors = [
        row
        for row in group
        if hard_language_supervision_allowed(
            row["counterfactual"]["variant"], row["question_family"]
        )
    ]
    assert len(anchors) == 18
    for row in anchors:
        answers = _same_variant_answers(group, row)
        assert answers[0] == "%s answer" % row["question_family"]
        assert len(answers) == 4
        assert len(set(answers)) == 4
