from __future__ import annotations

import json
from pathlib import Path

from scripts.finalize_uq_relevance_multiframe_event import _write_v5_qa_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "observed",
    "zero_uq",
    "on_path_uq",
    "off_path_uq",
    "view_shuffled_uq",
)
FAMILIES = (
    "observation_semantics",
    "epistemic_limitation",
    "task_relevance",
    "driving_implication",
)


def _summary(variant: str) -> dict:
    zero = variant == "zero_uq"
    no_risk = variant in {"zero_uq", "off_path_uq"}
    return {
        "observation_uncertainty": {
            "level": "low" if zero else "high",
            "peak_score": 0.0 if zero else 0.9,
            "peak_view": "CAM_FRONT",
            "peak_region": "upper_left" if zero else "lower_center",
            "temporal_trend": "stable" if zero else "rising",
            "temporal_peak_region_delta": 0.0 if zero else 0.5,
        },
        "relevance_at_most_uncertain_region": {
            "level": "low" if no_risk else "high",
            "score": 0.0 if no_risk else 0.7,
        },
        "task_risk": {
            "level": "low" if no_risk else "medium",
            "peak_score": 0.0 if no_risk else 0.6,
            "peak_view": "CAM_FRONT",
            "peak_region": "upper_left" if no_risk else "lower_center",
        },
        "planning_implication": {
            "stance": "prepare_to_yield" if variant == "on_path_uq" else "maintain",
            "risk_bearing": "forward_or_crossing",
            "is_direct_control_command": False,
        },
    }


def _base_records() -> list[dict]:
    rows = []
    for variant in VARIANTS:
        for family in FAMILIES:
            rows.append(
                {
                    "schema": "orion.uq_relevance_qa_record.v1",
                    "sample_id": f"g0/{variant}/{family}",
                    "question_family": family,
                    "counterfactual": {"group_id": "g0", "variant": variant},
                    "conversation": [
                        {"from": "human", "value": "question"},
                        {"from": "gpt", "value": "legacy answer"},
                    ],
                    "target": {
                        "structured_summary": _summary(variant),
                        "map_sidecar": {"path": "map_sidecars/g0.npz"},
                    },
                    "model_input": {"unchanged": True},
                }
            )
    return rows


def test_write_v5_dataset_preserves_sidecars_and_freezes_task_fields(tmp_path: Path) -> None:
    base = tmp_path / "base"
    (base / "map_sidecars").mkdir(parents=True)
    (base / "map_sidecars" / "g0.npz").write_bytes(b"sidecar")
    (base / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in _base_records()),
        encoding="utf-8",
    )
    output = tmp_path / "v5"
    config = (
        PROJECT_ROOT
        / "configs"
        / "scenario_factory"
        / "qa_factory_v5_vlm_task_fields.json"
    )

    result = _write_v5_qa_dataset(
        base_qa_root=base,
        output_root=output,
        qa_factory_config=config,
    )

    records = [
        json.loads(line)
        for line in result["records_path"].read_text(encoding="utf-8").splitlines()
    ]
    audit = json.loads(result["audit_path"].read_text(encoding="utf-8"))
    dataset = json.loads(result["dataset_path"].read_text(encoding="utf-8"))
    assert result["record_count"] == 20
    assert audit["passed"] is True
    assert all(row["schema"] == "orion.uq_relevance_qa_record.v5" for row in records)
    assert (output / "map_sidecars" / "g0.npz").read_bytes() == b"sidecar"
    assert dataset["schema"] == "orion.uq_relevance_qa_dataset.v5"
    assert dataset["formal_training_ready"] is False
    assert dataset["qa_contract"] == "orion.stage2l_qa_contract.v5"
