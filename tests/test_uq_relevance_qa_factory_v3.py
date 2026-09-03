import json
from pathlib import Path

from scripts.uq_relevance_qa_factory_v3_lib import (
    RECORD_SCHEMA,
    audit_v3_records,
    upgrade_records,
)
from scripts.upgrade_stage2l_v7_qa_records import (
    _rebase_relative_sidecar_references,
)
from uq_estimator.stage2l_matched_objective import (
    MATCHED_VARIANTS,
    QUESTION_FAMILIES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config():
    return json.loads(
        (
            PROJECT_ROOT
            / "configs/scenario_factory/qa_factory_v3_calibrated_semantics.json"
        ).read_text()
    )


def _answer(variant, family):
    if family == "observation_semantics":
        return "Observation uncertainty is high at %s." % variant
    if family == "epistemic_limitation":
        return "The uncertain evidence is at %s; hidden facts remain unknown." % variant
    if family == "task_relevance":
        level = "high" if variant == "on_path_uq" else "low"
        return "<task_relevance_map> Task relevance is %s." % level
    stance = "prepare_to_yield" if variant == "on_path_uq" else "maintain"
    return (
        "<task_relevance_map> The uncertainty-aware planning stance is %s. "
        "This is a planning implication, not a direct brake or steering command."
        % stance
    )


def _group():
    rows = []
    for variant in MATCHED_VARIANTS:
        for family in QUESTION_FAMILIES:
            stance = "prepare_to_yield" if variant == "on_path_uq" else "maintain"
            answer = _answer(variant, family)
            rows.append(
                {
                    "schema": "orion.uq_relevance_qa_record.v1",
                    "sample_id": "event/frame/%s/%s" % (variant, family),
                    "event_id": "event",
                    "counterfactual": {
                        "group_id": "event/frame",
                        "variant": variant,
                    },
                    "question_family": family,
                    "conversation": [
                        {"from": "human", "value": "question"},
                        {"from": "gpt", "value": answer},
                    ],
                    "target": {
                        "rendered_answer": answer,
                        "structured_summary": {
                            "planning_implication": {"stance": stance}
                        },
                    },
                    "provenance": {},
                }
            )
    return rows


def test_v3_upgrade_adds_unique_tags_and_disables_cross_family_negatives():
    upgraded, audit = upgrade_records(_group(), config=_config())
    assert len(upgraded) == 20
    assert audit["passed"]
    assert audit["matched_group_count"] == 1
    assert audit["same_family_preference_anchor_count"] > 0
    assert all(row["schema"] == RECORD_SCHEMA for row in upgraded)
    driving = next(
        row
        for row in upgraded
        if row["counterfactual"]["variant"] == "on_path_uq"
        and row["question_family"] == "driving_implication"
    )
    assert driving["conversation"][1]["value"].startswith("<planning_stance>")
    assert driving["loss_policy"]["cross_family_preference_anchor"] is False


def test_v3_audit_fails_tampered_family_tag_and_stance():
    upgraded, _ = upgrade_records(_group(), config=_config())
    upgraded[0]["conversation"][1]["value"] = "<planning_stance> wrong"
    audit = audit_v3_records(upgraded, config=_config())
    assert not audit["checks"]["all_family_tags_parse_and_match"]
    assert not audit["checks"]["rendered_answers_are_hashable_v3_targets"]


def test_v3_rebases_relative_sidecars_when_records_move(tmp_path):
    source_dir = tmp_path / "qa_dataset"
    output_dir = tmp_path / "qa_dataset_v3"
    sidecar = source_dir / "map_sidecars" / "sample.npz"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(b"sidecar")
    output_dir.mkdir()
    records = [{"target": {"map_sidecar": {"path": "map_sidecars/sample.npz"}}}]
    assert _rebase_relative_sidecar_references(
        records, source_dir=source_dir, output_dir=output_dir
    ) == 1
    resolved = (output_dir / records[0]["target"]["map_sidecar"]["path"]).resolve()
    assert resolved == sidecar.resolve()
