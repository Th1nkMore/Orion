import gzip
import importlib.util
import json
from pathlib import Path
import pickle
import struct


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_qwen_visibility_grounding_source_inventory.py"
)
SPEC = importlib.util.spec_from_file_location("qwen_source_inventory", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_png_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">IIBBBBB", 1600, 900, 8, 0, 0, 0, 0)
    )


def _annotation() -> dict:
    calibration = {
        "cam2ego": [[1, 0, 0, 0]] * 4,
        "intrinsic": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "image_size_x": 1600,
        "image_size_y": 900,
    }
    value = {
        "sensors": {name: calibration for name in MODULE.REQUIRED_CAMERA_KEYS}
    }
    value.update({name: 0.0 for name in MODULE.REQUIRED_STATE_KEYS})
    return value


def test_inventory_reports_sampled_contract_and_split_capacity(tmp_path):
    dataset = tmp_path / "dataset"
    folders = {
        "train": "v1/A_Town01_Route1_Weather0",
        "validation": "v1/B_Town02_Route2_Weather0",
        "calibration": "v1/C_Town03_Route3_Weather0",
        "held_out": "v1/D_Town04_Route4_Weather0",
    }
    infos = []
    for folder in folders.values():
        infos.append({"folder": folder, "frame_idx": 0})
        for stream in MODULE.REQUIRED_CAMERA_STREAMS:
            suffix = ".jpg" if stream.startswith("rgb_") else ".png"
            path = dataset / folder / "camera" / stream / ("00000" + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            if stream.startswith("depth_"):
                _write_png_header(path)
            else:
                path.write_bytes(b"jpeg-placeholder")
        annotation = dataset / folder / "anno" / "00000.json.gz"
        annotation.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(annotation, "wt", encoding="utf-8") as handle:
            json.dump(_annotation(), handle)
        expert = dataset / folder / "expert_assessment" / "00000.npz"
        expert.parent.mkdir(parents=True, exist_ok=True)
        expert.write_bytes(b"npz-placeholder")

    infos_path = tmp_path / "infos.pkl"
    with infos_path.open("wb") as handle:
        pickle.dump(infos, handle)
    split_statistics = {
        split: {"folder_route_ids": [folder]}
        for split, folder in folders.items()
    }
    route_manifest = {
        "route_disjoint": True,
        "lineage_audit": {
            "leakage_checks": {"passed": True},
            "input_summary": {
                "scenario_types": ["A", "B", "C", "D"],
                "towns": ["Town01", "Town02", "Town03", "Town04"],
            },
            "split_statistics": split_statistics,
        },
    }
    route_manifest_path = tmp_path / "route_manifest.json"
    route_manifest_path.write_text(json.dumps(route_manifest), encoding="utf-8")
    token_manifest = {
        "frame_count": 2,
        "records": [
            {"valid_frontier_tokens": 32},
            {"valid_frontier_tokens": 31},
        ],
    }
    token_manifest_path = tmp_path / "token_manifest.json"
    token_manifest_path.write_text(json.dumps(token_manifest), encoding="utf-8")

    report = MODULE.audit(
        dataset_root=dataset,
        infos_path=infos_path,
        route_manifest_path=route_manifest_path,
        token_manifest_path=token_manifest_path,
    )

    assert report["dataset"]["route_count"] == 4
    assert report["dataset"]["frame_count"] == 4
    assert report["sample_audit"]["sampled_frame_count"] == 4
    assert report["sample_audit"]["required_stream_or_annotation_failures"] == 0
    assert report["sample_audit"]["annotation_contract_failures"] == []
    assert report["sample_audit"]["depth_png_headers"] == [
        {
            "bit_depth": 8,
            "color_type": 0,
            "height": 900,
            "sample_count": 12,
            "width": 1600,
        }
    ]
    assert report["dataset"]["splits"]["train"] == {
        "route_count": 1,
        "frame_count": 1,
        "maximum_addressable_F00_F31_row_frame_pairs": 32,
    }
    assert report["existing_qwen_visibility_tokens"][
        "frames_with_all_32_frontier_rows"
    ] == 1
    assert report["feasibility"]["same_frame_answer_equality_penalty_allowed"] is False
