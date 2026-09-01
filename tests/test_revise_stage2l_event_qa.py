import json
from pathlib import Path

import pytest

from scripts.revise_stage2l_event_qa import (
    FACTORY_SCHEMA,
    collect_bundle_paths,
    materialize_split_overridden_bundles,
)
from scripts.scenario_factory_lib import sha256_file


VARIANTS = (
    "observed",
    "zero_uq",
    "on_path_uq",
    "off_path_uq",
    "view_shuffled_uq",
)


def _reference(path: Path, **extra):
    return {"path": str(path), "sha256": sha256_file(path), **extra}


def _source_report(tmp_path: Path):
    frame_reports = []
    bundle_paths = []
    for frame in (1, 2, 3):
        bundles = []
        for variant in VARIANTS:
            path = tmp_path / ("frame%d_%s.json" % (frame, variant))
            path.write_text(json.dumps({
                "frame_id": "saved_%04d" % frame,
                "split": "dev",
                "counterfactual": {"variant": variant},
                "model_input": {"u_map": "immutable-u"},
                "route": {"points": [[1, 2]]},
                "supervision": {"r_map": "immutable-r", "k_map": "immutable-k"},
                "provenance": {"source": "immutable-source"},
            }))
            bundle_paths.append(path)
            bundles.append(_reference(path, variant=variant))
        batch_path = tmp_path / ("frame%d_batch.json" % frame)
        batch_path.write_text(json.dumps({
            "frame_id": "saved_%04d" % frame,
            "bundles": bundles,
        }))
        frame_reports.append({
            "selected_saved_frame_index": frame,
            "frame_bundle_batch": _reference(batch_path),
        })
    return {
        "schema": FACTORY_SCHEMA,
        "frame_reports": frame_reports,
    }, bundle_paths


def test_collect_bundle_paths_verifies_complete_three_frame_inventory(tmp_path):
    report, expected = _source_report(tmp_path)

    actual = collect_bundle_paths(report)

    assert actual == expected


def test_collect_bundle_paths_rejects_tampered_reused_bundle(tmp_path):
    report, paths = _source_report(tmp_path)
    paths[4].write_text("tampered")

    with pytest.raises(ValueError, match="frame bundle SHA-256 mismatch"):
        collect_bundle_paths(report)


def test_materialize_split_override_preserves_source_and_payload(tmp_path):
    report, paths = _source_report(tmp_path)
    verified = collect_bundle_paths(report)
    source_bytes = paths[0].read_bytes()
    source = json.loads(source_bytes)

    result = materialize_split_overridden_bundles(
        verified,
        output_dir=tmp_path / "revised",
        split_override="train",
    )

    assert paths[0].read_bytes() == source_bytes
    assert result["source_splits"] == ["dev"]
    assert result["revised_split"] == "train"
    assert len(result["paths"]) == 15
    revised = json.loads(result["paths"][0].read_text())
    assert revised["split"] == "train"
    for key in ("model_input", "route", "supervision", "counterfactual", "provenance"):
        assert revised[key] == source[key]
    provenance = revised["split_only_revision"]
    assert provenance["source_split"] == "dev"
    assert provenance["revised_split"] == "train"
    assert provenance["source_bundle"]["sha256"] == sha256_file(paths[0])


def test_materialize_split_override_rejects_unknown_split(tmp_path):
    report, _ = _source_report(tmp_path)

    with pytest.raises(ValueError, match="unsupported split override"):
        materialize_split_overridden_bundles(
            collect_bundle_paths(report),
            output_dir=tmp_path / "revised",
            split_override="heldout",
        )
