#!/usr/bin/env python3
"""Audit whether installed Bench2Drive data can support cross-route U grounding.

The audit is intentionally read-only apart from its new JSON report.  It does
not generate U tokens, load Qwen, or start training.  By default it checks the
first, middle, and last indexed frame of every route so the report does not
pretend to be an exhaustive file-integrity scan.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import struct
from typing import Any, Dict, Mapping, Sequence


SCHEMA = "orion.qwen-visibility-grounding-source-inventory/v1"
REQUIRED_CAMERA_STREAMS = (
    "rgb_front",
    "rgb_front_left",
    "rgb_front_right",
    "depth_front",
    "depth_front_left",
    "depth_front_right",
)
REQUIRED_CAMERA_KEYS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
REQUIRED_CALIBRATION_KEYS = (
    "cam2ego",
    "intrinsic",
    "image_size_x",
    "image_size_y",
)
REQUIRED_STATE_KEYS = (
    "speed",
    "x",
    "y",
    "theta",
    "x_command_near",
    "y_command_near",
    "x_command_far",
    "y_command_far",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _load_infos(path: Path) -> list[Mapping[str, Any]]:
    # The trusted Bench2Drive infos pickle contains NumPy arrays.  The caller
    # must therefore use the same trusted local artifact and an environment
    # with NumPy installed; this script never accepts an untrusted pickle.
    with path.open("rb") as handle:
        value = pickle.load(handle)  # noqa: S301 - trusted experiment artifact
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("infos must contain a list of dictionaries")
    return value


def _split_by_folder(route_manifest: Mapping[str, Any]) -> Dict[str, str]:
    try:
        statistics = route_manifest["lineage_audit"]["split_statistics"]
    except KeyError as error:
        raise ValueError("route manifest lacks split statistics") from error
    result: Dict[str, str] = {}
    for split_name, split in statistics.items():
        for folder in split["folder_route_ids"]:
            if folder in result:
                raise ValueError("folder appears in more than one split: %s" % folder)
            result[str(folder)] = str(split_name)
    return result


def _png_header(path: Path) -> Dict[str, int]:
    with path.open("rb") as handle:
        header = handle.read(29)
    if len(header) != 29 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("invalid PNG signature: %s" % path)
    if header[12:16] != b"IHDR":
        raise ValueError("PNG lacks leading IHDR chunk: %s" % path)
    width, height, bit_depth, color_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", header[16:29])
    )
    return {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
        "compression": compression,
        "filtering": filtering,
        "interlace": interlace,
    }


def _selected_frames(frames: Sequence[int]) -> list[int]:
    if not frames:
        return []
    return sorted({int(frames[0]), int(frames[len(frames) // 2]), int(frames[-1])})


def _stream_path(dataset_root: Path, folder: str, stream: str, frame: int) -> Path:
    suffix = ".jpg" if stream.startswith("rgb_") else ".png"
    return dataset_root / folder / "camera" / stream / (("%05d" % frame) + suffix)


def _annotation_path(dataset_root: Path, folder: str, frame: int) -> Path:
    return dataset_root / folder / "anno" / (("%05d" % frame) + ".json.gz")


def _expert_path(dataset_root: Path, folder: str, frame: int) -> Path:
    return dataset_root / folder / "expert_assessment" / (("%05d" % frame) + ".npz")


def _annotation_contract(path: Path) -> tuple[bool, str | None]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
        sensors = value.get("sensors", {})
        if not all(
            camera in sensors
            and all(key in sensors[camera] for key in REQUIRED_CALIBRATION_KEYS)
            for camera in REQUIRED_CAMERA_KEYS
        ):
            return False, "missing_camera_calibration"
        if not all(key in value for key in REQUIRED_STATE_KEYS):
            return False, "missing_ego_or_navigation_state"
        return True, None
    except Exception as error:  # report malformed artifacts without aborting the audit
        return False, "%s: %s" % (type(error).__name__, error)


def _frame_gaps(frames: Sequence[int]) -> list[int]:
    if not frames:
        return []
    expected = set(range(int(frames[0]), int(frames[-1]) + 1))
    return sorted(expected - set(int(value) for value in frames))


def _existing_token_summary(token_manifest_path: Path) -> Dict[str, Any]:
    manifest = _read_json(token_manifest_path)
    valid = [int(row["valid_frontier_tokens"]) for row in manifest["records"]]
    return {
        "manifest": _reference(token_manifest_path),
        "frame_count": int(manifest["frame_count"]),
        "minimum_valid_frontier_rows": min(valid),
        "maximum_valid_frontier_rows": max(valid),
        "frames_with_all_32_frontier_rows": sum(value == 32 for value in valid),
        "scope": "Route151 only",
    }


def audit(
    *,
    dataset_root: Path,
    infos_path: Path,
    route_manifest_path: Path,
    token_manifest_path: Path,
) -> Dict[str, Any]:
    dataset_root = dataset_root.resolve()
    infos_path = infos_path.resolve()
    route_manifest_path = route_manifest_path.resolve()
    token_manifest_path = token_manifest_path.resolve()
    infos = _load_infos(infos_path)
    route_manifest = _read_json(route_manifest_path)
    folder_split = _split_by_folder(route_manifest)

    frames_by_folder: Dict[str, list[int]] = defaultdict(list)
    for row in infos:
        frames_by_folder[str(row["folder"])].append(int(row["frame_idx"]))
    for frames in frames_by_folder.values():
        frames.sort()
    if set(frames_by_folder) != set(folder_split):
        raise ValueError("infos folders and route-manifest folders differ")
    if route_manifest.get("route_disjoint") is not True:
        raise ValueError("route manifest is not marked route-disjoint")
    leakage = route_manifest["lineage_audit"].get("leakage_checks", {})
    if leakage.get("passed") is not True:
        raise ValueError("route-manifest leakage checks did not pass")

    missing = Counter()
    annotation_failures = []
    depth_headers = Counter()
    expert_missing = []
    sampled_rows = []
    gaps = []
    for folder in sorted(frames_by_folder):
        frames = frames_by_folder[folder]
        route_gaps = _frame_gaps(frames)
        if route_gaps:
            gaps.append({"folder": folder, "missing_frame_indices": route_gaps})
        for frame in _selected_frames(frames):
            row_missing = []
            for stream in REQUIRED_CAMERA_STREAMS:
                path = _stream_path(dataset_root, folder, stream, frame)
                if not path.is_file():
                    missing[stream] += 1
                    row_missing.append(stream)
                    continue
                if stream.startswith("depth_"):
                    header = _png_header(path)
                    depth_headers[
                        (
                            header["width"],
                            header["height"],
                            header["bit_depth"],
                            header["color_type"],
                        )
                    ] += 1
            annotation = _annotation_path(dataset_root, folder, frame)
            if not annotation.is_file():
                missing["anno"] += 1
                row_missing.append("anno")
            else:
                valid, reason = _annotation_contract(annotation)
                if not valid:
                    annotation_failures.append(
                        {"folder": folder, "frame": frame, "reason": reason}
                    )
            expert = _expert_path(dataset_root, folder, frame)
            if not expert.is_file():
                missing["expert_assessment"] += 1
                expert_missing.append(
                    {
                        "folder": folder,
                        "frame": frame,
                        "is_terminal_indexed_frame": frame == frames[-1],
                    }
                )
            sampled_rows.append(
                {
                    "folder": folder,
                    "split": folder_split[folder],
                    "frame": frame,
                    "missing_required_inputs": row_missing,
                }
            )

    split_summary = {}
    for split_name in ("train", "validation", "calibration", "held_out"):
        folders = [name for name, split in folder_split.items() if split == split_name]
        frame_count = sum(len(frames_by_folder[name]) for name in folders)
        split_summary[split_name] = {
            "route_count": len(folders),
            "frame_count": frame_count,
            "maximum_addressable_F00_F31_row_frame_pairs": frame_count * 32,
        }

    header_summary = [
        {
            "width": key[0],
            "height": key[1],
            "bit_depth": key[2],
            "color_type": key[3],
            "sample_count": count,
        }
        for key, count in sorted(depth_headers.items())
    ]
    terminal_expert_missing = sum(
        bool(row["is_terminal_indexed_frame"]) for row in expert_missing
    )
    nonterminal_expert_missing = len(expert_missing) - terminal_expert_missing
    token_summary = _existing_token_summary(token_manifest_path)
    return {
        "schema": SCHEMA,
        "claim_boundary": {
            "training_started": False,
            "u_tokens_generated_from_offline_data": False,
            "file_coverage": "first_middle_last_indexed_frame_per_route",
            "full_file_integrity_proven": False,
            "offline_depth_equivalent_to_live_24bit_oracle": False,
        },
        "inputs": {
            "dataset_root": str(dataset_root),
            "infos": _reference(infos_path),
            "route_manifest": _reference(route_manifest_path),
        },
        "dataset": {
            "route_count": len(frames_by_folder),
            "frame_count": len(infos),
            "scenario_type_count": len(
                route_manifest["lineage_audit"]["input_summary"]["scenario_types"]
            ),
            "town_count": len(route_manifest["lineage_audit"]["input_summary"]["towns"]),
            "route_disjoint": True,
            "leakage_checks_passed": True,
            "splits": split_summary,
            "frame_gaps": gaps,
        },
        "sample_audit": {
            "sampled_frame_count": len(sampled_rows),
            "required_camera_streams_per_frame": list(REQUIRED_CAMERA_STREAMS),
            "required_camera_file_checks": len(sampled_rows)
            * len(REQUIRED_CAMERA_STREAMS),
            "missing_counts": dict(sorted(missing.items())),
            "required_stream_or_annotation_failures": sum(
                missing[name] for name in REQUIRED_CAMERA_STREAMS + ("anno",)
            ),
            "annotation_contract_failures": annotation_failures,
            "depth_png_headers": header_summary,
            "expert_assessment_missing": {
                "total": len(expert_missing),
                "terminal_indexed_frames": terminal_expert_missing,
                "nonterminal_indexed_frames": nonterminal_expert_missing,
            },
        },
        "depth_contract": {
            "installed_storage": "uint8 grayscale PNG containing rounded metric metres",
            "collection_rate_hz": 10,
            "saturation_value": 255,
            "effective_near_range_quantization_m": 1.0,
            "source_provenance": (
                "Bench2Drive tools/data_collect.py calls convert_depth and writes the "
                "result with cv2.imwrite; tools/utils.py returns metric metres and the "
                "installed PNG IHDR is 8-bit grayscale"
            ),
            "accepted_use": "coarse visibility preflight after a recorded quantization policy",
            "rejected_claim": "drop-in reproduction of live CARLA 24-bit oracle depth",
        },
        "existing_qwen_visibility_tokens": token_summary,
        "feasibility": {
            "route_disjoint_split_available": True,
            "frame_disjoint_split_available_via_route_split": True,
            "random_F00_F31_query_manifest_can_be_constructed": True,
            "strict_global_target_row_identity_split_can_be_constructed": True,
            "ordinary_per_example_supervision_only": True,
            "same_frame_answer_equality_penalty_allowed": False,
            "current_status": (
                "source_ready_pending_depth_quantization_acceptance_and_"
                "offline_tokenization_preflight"
            ),
            "remaining_gates": [
                "freeze the uint8-depth interval/tolerance policy or recapture lossless depth",
                "freeze route-polyline reconstruction from navigation annotations",
                "tokenize a route-diverse pilot and measure how often all 32 frontier rows exist",
                "freeze row/frame/route query splits before any Qwen optimization",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--infos", required=True, type=Path)
    parser.add_argument("--route-manifest", required=True, type=Path)
    parser.add_argument("--token-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite inventory: %s" % output)
    report = audit(
        dataset_root=args.dataset_root,
        infos_path=args.infos,
        route_manifest_path=args.route_manifest,
        token_manifest_path=args.token_manifest,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
