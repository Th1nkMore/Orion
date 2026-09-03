import json

from scripts.evaluate_pairwise_closedloop_trace import evaluate


def test_frozen_pairwise_trace_gate_accepts_selective_timely_signal(tmp_path):
    run_dir = tmp_path / "run"
    trace_dir = run_dir / "records_x" / "route"
    trace_dir.mkdir(parents=True)
    checkpoint_sha = "a" * 64
    (run_dir / "manifest.json").write_text(json.dumps({
        "pilot_condition": "front_corrupt_transient_pairwise_trace",
        "orion_closedloop_risk_mode": "off",
        "orion_observation_uq_checkpoint_sha256": checkpoint_sha,
    }))
    prereg = tmp_path / "prereg.json"
    prereg.write_text(json.dumps({
        "signal_checkpoint": {"sha256": checkpoint_sha},
        "trace_gate": {
            "minimum_baseline_frames": 40,
            "pre_event_sustained_false_trigger_max_seconds": 0.25,
            "trigger_threshold": 0.5,
            "maximum_detection_latency_seconds": 0.5,
            "minimum_event_trigger_coverage": 0.8,
            "minimum_event_median_score": 0.8,
            "maximum_post_event_recovery_seconds": 1.5,
        },
    }))
    records = []
    for step in range(241):
        current_time = step * 0.05
        active = 8.0 <= current_time < 10.0
        score = 0.99 if active else 0.05
        view_scores = [2.0, 1.1, 1.0, 1.0, 1.0, 1.0] if active else [1.0] * 6
        records.append({
            "sim_time_seconds": current_time,
            "corruption_active": active,
            "raw_uq_score": score,
            "risk": {"mode": "off", "intensity": 0.0},
            "observation_uq": {
                "front_view_index": 0,
                "aggregate": {"view_raw_scores": view_scores},
                "calibration": {
                    "baseline_frozen": current_time >= 4.0,
                    "baseline_count": min(max(step - 20, 0), 60),
                },
            },
        })
    (trace_dir / "control_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    report = evaluate(run_dir, prereg)
    assert report["gate_passed"]
    assert report["decision"] == "submit_single_pairwise_controlled_stop"
