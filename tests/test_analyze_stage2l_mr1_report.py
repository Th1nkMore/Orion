from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_stage2l_mr1_report import _mean, _metric_snapshot, _target_distributions


def test_mean_is_exact_for_short_windows():
    assert _mean([1.0, 2.0, 3.0]) == 2.0


def test_metric_snapshot_keeps_release_relevant_fields():
    split = {
        "relevance_support": {
            "foreground_recall": 0.8,
            "background_false_positive_rate": 0.1,
        },
        "ranking": {"positive_order_fraction": 0.7, "minimum_attained_fraction": 0.2},
        "task_fields": {
            "overall_accuracy": 0.6,
            "supported_class_macro_recall": 0.5,
            "per_field_accuracy": {"stance": 0.9},
            "zero_uq_complete_field_accuracy": 1.0,
        },
        "deterministic_render": {
            "semantic_answer_exact_match": 0.75,
            "semantic_field_accuracy": 0.8,
        },
    }
    result = _metric_snapshot(split)
    assert result["positive_order_fraction"] == 0.7
    assert result["stance_accuracy"] == 0.9
    assert result["zero_uq_complete_field_accuracy"] == 1.0


def test_target_distribution_uses_only_task_field_qa_rows(tmp_path):
    base = {
        "split": "train",
        "counterfactual": {"group_id": "g1", "variant": "observed"},
        "target": {
            "vlm_task_field_targets": {},
            "structured_summary": {"planning_implication": {"stance": "maintain"}},
        },
    }
    relevance_row = json.loads(json.dumps(base))
    relevance_row["target"]["vlm_task_field_targets"] = {
        "relevance_level": "low",
        "risk_level": "low",
    }
    stance_row = json.loads(json.dumps(base))
    stance_row["target"]["vlm_task_field_targets"] = {"stance": "maintain"}
    records = tmp_path / "records.jsonl"
    records.write_text(
        "\n".join(json.dumps(row) for row in (base, relevance_row, stance_row)) + "\n",
        encoding="utf-8",
    )

    result = _target_distributions(records)

    assert result["unique_task_field_variant_count"] == 1
    assert result["ignored_non_task_field_qa_rows"] == 1
    assert result["by_split"]["train"]["relevance_level"] == {"low": 1}
    assert result["by_split"]["train"]["stance"] == {"maintain": 1}
