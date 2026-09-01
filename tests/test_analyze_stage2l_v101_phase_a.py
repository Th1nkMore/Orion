import json
import numpy as np
import torch

from scripts.analyze_stage2l_v101_phase_a import analyze, group_view_diagnostics


def test_group_view_diagnostic_detects_correct_and_wrong_view():
    target = np.zeros((1, 6, 10, 10), dtype=np.float32)
    target[0, 2, 4, 5] = 1.0
    correct = np.zeros_like(target)
    correct[0, 2, 4, 5] = 0.9
    row = group_view_diagnostics(correct, target, support_fraction=0.1)
    assert row["target_supported_views"] == ["CAM_FRONT_RIGHT"]
    assert row["prediction_dominant_view"] == "CAM_FRONT_RIGHT"
    assert row["prediction_top1_hits_any_target_supported_view"] is True

    wrong = np.zeros_like(target)
    wrong[0, 0, 4, 5] = 0.9
    row = group_view_diagnostics(wrong, target, support_fraction=0.1)
    assert row["prediction_dominant_view"] == "CAM_FRONT"
    assert row["prediction_top1_hits_any_target_supported_view"] is False


def test_terminal_analysis_compares_v10_and_keeps_later_stages_locked(tmp_path):
    report = {
        "schema": "orion.stage2l_v101_view_aligned_phase_a.v1",
        "status": "phase_a_stopped_without_gate_pass",
        "phase_a_only": True,
        "formal_stage2l_ready": False,
        "stage2p_ready": False,
        "optimizer_steps": 40,
        "stop_reason": "maximum_steps_reached",
        "locks": {"stage1_uq_loaded": False},
        "final_metrics": {
            "train": {"average_precision": 0.70, "per_event": {"train_event": {}}},
            "dev": {"average_precision": 0.35, "per_event": {"dev_event": {}}},
        },
        "final_checks": {"train_foreground_recall": False, "dev_background_fpr": True},
        "evaluations": [
            {
                "optimizer_step": 40,
                "metrics": {
                    "train": {"average_precision": 0.70},
                    "dev": {"average_precision": 0.35},
                },
                "passed": False,
            }
        ],
    }
    baseline = {
        "schema": "orion.stage2l_v10_phase_a_replay_analysis.v1",
        "split_metrics": {
            "train": {"average_precision": 0.44},
            "dev": {"average_precision": 0.23},
        },
    }
    maps = {}
    for event_id in ("train_event", "dev_event"):
        for index in range(40):
            target = torch.zeros(1, 6, 10, 10)
            probability = torch.zeros_like(target)
            target[0, 2, 4, 5] = 1.0
            probability[0, 2, 4, 5] = 0.9
            maps[f"{event_id}_saved_{index:04d}"] = {
                "target": target,
                "probability": probability,
            }
    report_path = tmp_path / "report.json"
    baseline_path = tmp_path / "baseline.json"
    maps_path = tmp_path / "spatial_maps_step040.pt"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    torch.save(maps, maps_path)
    result = analyze(
        report=report,
        baseline=baseline,
        maps=maps,
        report_path=report_path,
        baseline_path=baseline_path,
        maps_path=maps_path,
        support_fraction=0.1,
    )
    comparison = result["average_precision_comparison"]
    assert comparison["train_delta"] == 0.70 - 0.44
    assert comparison["dev_delta"] == 0.35 - 0.23
    assert result["view_binding_metrics"]["per_split"]["dev"][
        "top1_hits_any_target_supported_view_fraction"
    ] == 1.0
    assert result["decision"]["phase_b_automatically_authorized"] is False
    assert result["decision"]["formal_stage2l"] is False
