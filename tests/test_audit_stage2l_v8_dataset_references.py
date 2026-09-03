import hashlib
from pathlib import Path

from scripts.audit_stage2l_v8_dataset_references import audit_references


def _reference(path: Path, with_hash=True):
    value = {"path": str(path)}
    if with_hash:
        value["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return value


def _row(tmp_path):
    files = {}
    for name in ("camera", "uq", "relevance", "geometry", "sidecar"):
        path = tmp_path / (name + ".bin")
        path.write_bytes(name.encode())
        files[name] = path
    return {
        "model_input": {
            "observation": {"camera_files": [_reference(files["camera"], False)]},
            "stage1_observation_uq": _reference(files["uq"]),
        },
        "provenance": {
            "relevance_supervision": {
                **_reference(files["relevance"]),
                "geometry_manifest": _reference(files["geometry"]),
            }
        },
        "target": {"map_sidecar": _reference(files["sidecar"])},
    }


def test_reference_audit_verifies_files_and_declared_hashes(tmp_path):
    result = audit_references([_row(tmp_path)], records_parent=tmp_path)
    assert result["passed"] is True
    assert result["unique_reference_count"] == 5
    assert result["verified_sha256_count"] == 4


def test_reference_audit_fails_closed_on_missing_file(tmp_path):
    row = _row(tmp_path)
    Path(row["target"]["map_sidecar"]["path"]).unlink()
    result = audit_references([row], records_parent=tmp_path)
    assert result["passed"] is False
    assert result["checks"]["all_referenced_files_exist"] is False
