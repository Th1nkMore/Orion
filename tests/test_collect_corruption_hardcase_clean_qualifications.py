import hashlib
import json

import pytest

from scripts.collect_corruption_hardcase_clean_qualifications import collect


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "schema": "orion.corruption_hardcase_wave1_clean_qualification.v1",
        "selection": {"routes": [158, 164, 185, 207]},
    }))
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    for route in (158, 164, 185, 207):
        passed = route in {158, 185}
        (analysis / ("route%d_clean_q1_qualification.json" % route)).write_text(
            json.dumps({
                "schema": "orion.corruption_hardcase_clean_qualification.v1",
                "phase": "q1",
                "route_index": route,
                "protocol": {"sha256": _sha(protocol)},
                "status": "clean_qualified" if passed else "clean_rejected",
                "qualified_for_next_clean_repeat": passed,
                "failed_checks": [] if passed else ["zero_hard_infractions"],
                "route_completion_percent": 100.0,
                "longest_low_speed_interval": {"duration_seconds": 1.0},
                "hard_infraction_counts": {"collisions_vehicle": 0 if passed else 1},
            })
        )
    return protocol, analysis


def test_collects_exact_q1_passers_without_unlocking_q2(tmp_path):
    protocol, analysis = _fixture(tmp_path)
    report = collect(analysis_root=analysis, protocol_path=protocol)
    assert report["q1_pass_routes"] == [158, 185]
    assert report["q1_rejected_routes"] == [164, 207]
    assert not report["next_stage"]["q2_automatically_authorized"]
    assert not report["next_stage"]["corruption_automatically_authorized"]


def test_requires_all_frozen_reports(tmp_path):
    protocol, analysis = _fixture(tmp_path)
    (analysis / "route207_clean_q1_qualification.json").unlink()
    with pytest.raises(FileNotFoundError):
        collect(analysis_root=analysis, protocol_path=protocol)
