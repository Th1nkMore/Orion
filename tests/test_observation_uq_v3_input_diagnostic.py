"""CLI check for the non-training v3 input diagnostic."""

from __future__ import annotations

import json
import subprocess
import sys

def test_diagnostic_reports_backbone_change_without_using_actual_target(tmp_path):
    extraction = tmp_path / "paired.pt"
    subprocess.run(
        [
            sys.executable,
            "scripts/extract_paired_spatial_features.py",
            "--mock",
            "--output",
            str(extraction),
            "--mock-samples",
            "4",
            "--corruption",
            "local_dark",
            "--severities",
            "1",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    routes = ["mock_route_%03d" % index for index in range(4)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "spatial-uq-route-manifest/v1",
                "splits": {
                    "train": {"route_ids": routes[:2]},
                    "validation": {"route_ids": routes[2:3]},
                    "held_out": {"route_ids": routes[3:]},
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "diagnostic.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/diagnose_observation_uq_v3_inputs.py",
            "--records",
            str(extraction),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["claim_boundary"]["training_performed"] is False
    assert report["record_count"] == 8
    assert report["by_split_family"]["train/local_dark"][
        "paired_cosine_error_inside_mask"
    ] > report["by_split_family"]["train/local_dark"][
        "paired_cosine_error_outside_mask"
    ]
