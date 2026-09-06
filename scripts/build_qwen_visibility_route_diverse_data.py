#!/usr/bin/env python3
"""Build immutable route-diverse offline-U tokens and a random-row curriculum."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SCHEMA = "orion.qwen-visibility-route-diverse-data-protocol/v1"
MANIFEST_SCHEMA = "orion.qwen-visibility-grounding-manifest/v1"
CURRICULUM_SCHEMA = "orion.qwen-visibility-route-diverse-curriculum/v1"
BUILD_REPORT_SCHEMA = "orion.qwen-visibility-route-diverse-data-build/v1"
CAMERA_STREAMS = {
    "CAM_FRONT": "rgb_front",
    "CAM_FRONT_LEFT": "rgb_front_left",
    "CAM_FRONT_RIGHT": "rgb_front_right",
}


def _load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load local module from %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_preflight = _load_local_module(
    "_orion_qwen_route_diverse_preflight",
    PROJECT_ROOT / "scripts" / "preflight_qwen_visibility_offline_depth.py",
)
_visibility = _preflight._visibility


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _resolve(path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _validated_reference(reference: Mapping[str, Any]) -> Path:
    path = _resolve(str(reference["path"]))
    if _sha256(path) != str(reference["sha256"]):
        raise ValueError("input SHA-256 differs: %s" % path)
    return path


def balanced_route_label_assignments(
    folders: Sequence[str], seed: int
) -> Dict[str, str]:
    """Assign exactly half of an even route set to each route label."""

    ordered = sorted(str(folder) for folder in folders)
    if not ordered or len(ordered) % 2:
        raise ValueError("balanced label assignment requires an even route count")
    if len(set(ordered)) != len(ordered):
        raise ValueError("route folders must be unique")
    permutation = np.random.default_rng(int(seed)).permutation(len(ordered))
    on_indices = set(int(value) for value in permutation[: len(ordered) // 2])
    return {
        folder: ("ON_ROUTE" if index in on_indices else "OFF_ROUTE")
        for index, folder in enumerate(ordered)
    }


def feasible_balanced_route_label_assignments(
    availability: Mapping[str, Mapping[str, bool]], seed: int
) -> Dict[str, str]:
    """Balance labels while respecting which natural labels each route contains."""

    ordered = sorted(str(folder) for folder in availability)
    if not ordered or len(ordered) % 2:
        raise ValueError("feasible label assignment requires an even route count")
    forced_on = []
    forced_off = []
    flexible = []
    for folder in ordered:
        labels = availability[folder]
        has_on = bool(labels.get("ON_ROUTE"))
        has_off = bool(labels.get("OFF_ROUTE"))
        if not has_on and not has_off:
            raise ValueError("route has no valid query label: %s" % folder)
        if has_on and has_off:
            flexible.append(folder)
        elif has_on:
            forced_on.append(folder)
        else:
            forced_off.append(folder)
    target_on = len(ordered) // 2
    flexible_on_count = target_on - len(forced_on)
    if flexible_on_count < 0 or flexible_on_count > len(flexible):
        raise ValueError("split cannot support a balanced feasible label assignment")
    shuffled = [
        flexible[int(index)]
        for index in np.random.default_rng(int(seed)).permutation(len(flexible))
    ]
    selected_on = set(forced_on + shuffled[:flexible_on_count])
    assignments = {
        folder: ("ON_ROUTE" if folder in selected_on else "OFF_ROUTE")
        for folder in ordered
    }
    if sum(value == "ON_ROUTE" for value in assignments.values()) != target_on:
        raise RuntimeError("feasible assignment is not balanced")
    if any(
        not bool(availability[folder][label])
        for folder, label in assignments.items()
    ):
        raise RuntimeError("feasible assignment selected an unavailable label")
    return assignments


def choose_candidate_index(labels: Sequence[bool], target: str, seed: int) -> int:
    """Choose one natural row uniformly from rows carrying the assigned label."""

    if target not in {"ON_ROUTE", "OFF_ROUTE"}:
        raise ValueError("invalid target label")
    desired = target == "ON_ROUTE"
    candidates = [index for index, value in enumerate(labels) if bool(value) == desired]
    if not candidates:
        raise ValueError("route candidate window lacks assigned label %s" % target)
    position = int(np.random.default_rng(int(seed)).integers(0, len(candidates)))
    return int(candidates[position])


def _query_question(frontier: str) -> str:
    return (
        "Read the continuous visibility-belief tokens inserted after the camera "
        "block. Use the complete physical record at frontier token %s. Is "
        "route_weight_mean at least 0.2? Reply with exactly ON_ROUTE or "
        "OFF_ROUTE." % frontier
    )


def _label(tokens, row_index: int, threshold: float) -> str:
    feature_index = tokens.feature_names.index("route_weight_mean")
    value = float(tokens.frontier_tokens[row_index, feature_index])
    return "ON_ROUTE" if value >= float(threshold) else "OFF_ROUTE"


def _image_paths(dataset_root: Path, folder: str, frame: int) -> list[Path]:
    return [
        dataset_root
        / folder
        / "camera"
        / stream
        / (("%05d" % frame) + ".jpg")
        for stream in CAMERA_STREAMS.values()
    ]


def _load_infos(path: Path) -> list[Mapping[str, Any]]:
    with path.open("rb") as handle:
        value = pickle.load(handle)  # noqa: S301 - frozen trusted artifact
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("infos must contain a list of dictionaries")
    return value


def _build_tokens_for_route(
    *,
    dataset_root: Path,
    folder: str,
    frames: Sequence[int],
    qwen_config: Mapping[str, Any],
    surface_tolerance_m: float,
    depth_config: Mapping[str, Any],
    raw_hz: float,
):
    cameras = _preflight._camera_specs(qwen_config)
    spec = _preflight._grid_spec(qwen_config, surface_tolerance_m)
    memory_config = qwen_config["oracle_visibility"]["temporal_memory"]
    memory = _visibility.VisibilityObservationMemory(
        spec,
        max_age_seconds=float(memory_config["max_age_seconds"]),
        observed_ratio_threshold=float(memory_config["observed_ratio_threshold"]),
    )
    exposure_config = qwen_config["oracle_visibility"]["exposure"]
    tokenizer_config = qwen_config["oracle_visibility"]["tokenizer"]
    frontier_threshold = float(
        qwen_config["oracle_visibility"]["frontier_unknown_threshold"]
    )
    result = []
    for frame in frames:
        annotation = _preflight._load_annotation(dataset_root, folder, frame)
        route_world, pose, speed = _preflight._route_and_pose(annotation)
        stored = _preflight._load_stored_depth(dataset_root, folder, frame)
        depth = _preflight.depth_hypothesis(
            stored,
            offset_m=float(depth_config["depth_offset_m"]),
            saturated_value=int(depth_config["saturated_value"]),
            saturated_replacement_m=float(depth_config["saturated_replacement_m"]),
        )
        belief = _visibility.compute_visibility_belief(
            depth,
            cameras,
            spec,
            frontier_unknown_threshold=frontier_threshold,
        )
        memory_state = memory.update(
            belief, pose, timestamp_seconds=float(frame) / float(raw_hz)
        )
        route_ego = _visibility.carla_world_route_to_qwen_ego(
            route_world, pose, float(exposure_config["route_max_length_m"])
        )
        exposure = _visibility.compute_visibility_exposure(
            belief,
            route_ego,
            speed_mps=speed,
            reaction_time_seconds=float(exposure_config["reaction_time_seconds"]),
            safe_deceleration_mps2=float(exposure_config["safe_deceleration_mps2"]),
            route_sigma_m=float(exposure_config["route_sigma_m"]),
            stopping_transition_m=float(exposure_config["stopping_transition_m"]),
        )
        tokens = _visibility.tokenize_visibility_belief(
            belief,
            memory_state,
            exposure,
            global_grid_shape=tokenizer_config["global_grid_shape"],
            max_frontier_tokens=int(tokenizer_config["max_frontier_tokens"]),
            frontier_patch_radius_m=float(tokenizer_config["frontier_patch_radius_m"]),
            frontier_nms_radius_m=float(tokenizer_config["frontier_nms_radius_m"]),
            frontier_selection_floor=float(tokenizer_config["frontier_selection_floor"]),
            depth_confidence=float(tokenizer_config["oracle_depth_confidence"]),
        )
        result.append((int(frame), tokens))
    return result


def _write_token_artifact(
    path: Path,
    tokens,
    *,
    control_seed: int,
    provenance: Mapping[str, Any],
) -> None:
    zero = _visibility.zero_visibility_tokens(tokens)
    shuffled = _visibility.spatially_shuffle_visibility_tokens(tokens, control_seed)
    payload = {}
    for prefix, value in (
        ("visibility_tokens", tokens),
        ("visibility_tokens_zero_u", zero),
        ("visibility_tokens_spatial_shuffle", shuffled),
    ):
        payload.update(_visibility.visibility_token_npz_payload(value, prefix))
    payload["provenance_json"] = np.asarray(
        json.dumps(dict(provenance), sort_keys=True, separators=(",", ":"))
    )
    np.savez_compressed(path, **payload)


def build(protocol_path: Path, output_root: Path) -> Dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError("refusing to reuse route-diverse output: %s" % output_root)
    protocol = _read_json(protocol_path)
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected route-diverse data protocol schema")
    if protocol["claim_boundary"].get("starts_training") is not False:
        raise ValueError("data build protocol must prohibit training")

    inputs = protocol["inputs"]
    dataset_root = _resolve(inputs["dataset_root"])
    infos_path = _validated_reference(inputs["infos"])
    route_manifest_path = _validated_reference(inputs["route_manifest"])
    qwen_config_path = _validated_reference(inputs["qwen_visibility_config"])
    preflight_path = _validated_reference(inputs["accepted_depth_preflight"])
    accepted = _read_json(preflight_path)
    if accepted.get("all_engineering_gates_passed") is not True:
        raise ValueError("accepted depth preflight no longer passes")
    infos = _load_infos(infos_path)
    route_manifest = _read_json(route_manifest_path)
    qwen_config = _read_json(qwen_config_path)

    frames_by_folder: Dict[str, list[int]] = defaultdict(list)
    for row in infos:
        frames_by_folder[str(row["folder"])].append(int(row["frame_idx"]))
    frames_by_folder = {
        folder: sorted(set(frames)) for folder, frames in frames_by_folder.items()
    }
    folder_split = _preflight._folder_split(route_manifest)
    if set(frames_by_folder) != set(folder_split):
        raise ValueError("infos and route manifest folders differ")

    sampling = protocol["sampling"]
    included_splits = tuple(str(value) for value in sampling["included_splits"])
    if included_splits != ("train", "validation", "held_out"):
        raise ValueError("included split contract changed")
    expected = protocol["expected_counts"]
    split_folders = {
        split: sorted(folder for folder, value in folder_split.items() if value == split)
        for split in included_splits
    }
    for split, folders in split_folders.items():
        if len(folders) != int(expected[split]["routes"]):
            raise ValueError("route count changed for %s" % split)

    output_root.mkdir(parents=True, exist_ok=False)
    token_root = output_root / "tokens"
    token_root.mkdir()
    route_threshold = float(protocol["target"]["threshold"])
    selection_seed = int(sampling["selection_seed"])
    control_seed_base = int(protocol["controls"]["seed"])
    records = []
    candidate_summaries = []
    started = time.perf_counter()
    split_seed_offsets = {"train": 0, "validation": 100000, "held_out": 200000}
    for split in included_splits:
        folders = split_folders[split]
        route_candidates = {}
        for route_index, folder in enumerate(folders):
            frames = _preflight.select_centered_stride_window(
                frames_by_folder[folder],
                stride=int(sampling["raw_frame_stride"]),
                length=int(sampling["candidate_window_frames"]),
            )
            route_tokens = _build_tokens_for_route(
                dataset_root=dataset_root,
                folder=folder,
                frames=frames,
                qwen_config=qwen_config,
                surface_tolerance_m=float(protocol["depth"]["surface_tolerance_m"]),
                depth_config=protocol["depth"],
                raw_hz=float(sampling["raw_data_hz"]),
            )
            flat = []
            labels = []
            valid_counts = []
            for frame, tokens in route_tokens:
                valid_count = int(tokens.frontier_mask.sum())
                valid_counts.append(valid_count)
                for row_index in range(valid_count):
                    flat.append((frame, row_index, tokens))
                    labels.append(_label(tokens, row_index, route_threshold) == "ON_ROUTE")
            route_candidates[folder] = {
                "frames": frames,
                "flat": flat,
                "labels": labels,
                "valid_counts": valid_counts,
            }
        availability = {
            folder: {
                "ON_ROUTE": any(value["labels"]),
                "OFF_ROUTE": any(not label for label in value["labels"]),
            }
            for folder, value in route_candidates.items()
        }
        assignments = feasible_balanced_route_label_assignments(
            availability, selection_seed + split_seed_offsets[split]
        )
        for route_index, folder in enumerate(folders):
            candidate = route_candidates[folder]
            frames = candidate["frames"]
            flat = candidate["flat"]
            labels = candidate["labels"]
            valid_counts = candidate["valid_counts"]
            assigned = assignments[folder]
            route_seed = selection_seed + split_seed_offsets[split] + route_index + 1
            selected_flat_index = choose_candidate_index(labels, assigned, route_seed)
            frame, row_index, tokens = flat[selected_flat_index]
            if int(tokens.frontier_mask.sum()) != 32:
                raise ValueError("selected frame lacks all 32 frontier rows")
            query_frontier = "F%02d" % row_index
            sample_id = "%s-route-%03d-frame-%05d" % (split, route_index, frame)
            token_path = token_root / (sample_id + ".npz")
            control_seed = control_seed_base + split_seed_offsets[split] + route_index
            provenance = {
                "source_folder": folder,
                "source_frame": frame,
                "source_oracle_depth": False,
                "source_used_by_qwen": False,
                "source_visibility_supervision": "bench2drive_offline_uint8_depth",
                "stored_depth_units": "rounded_metric_metres",
                "surface_tolerance_m": float(protocol["depth"]["surface_tolerance_m"]),
                "future_executed_ego_motion_used": False,
                "split": split,
                "query_frontier": query_frontier,
            }
            _write_token_artifact(
                token_path,
                tokens,
                control_seed=control_seed,
                provenance=provenance,
            )
            image_paths = _image_paths(dataset_root, folder, frame)
            if not all(path.is_file() for path in image_paths):
                raise FileNotFoundError("selected frame lacks all three RGB images")
            zero = _visibility.zero_visibility_tokens(tokens)
            shuffled = _visibility.spatially_shuffle_visibility_tokens(
                tokens, control_seed
            )
            control_answers = {
                "true_u": _label(tokens, row_index, route_threshold),
                "zero_u": _label(zero, row_index, route_threshold),
                "spatial_shuffle": _label(shuffled, row_index, route_threshold),
            }
            if control_answers["true_u"] != assigned:
                raise RuntimeError("selected row label changed")
            identity = list(range(32))
            records.append(
                {
                    "sample_id": sample_id,
                    "source_folder": folder,
                    "source_frame": frame,
                    "split": split,
                    "token_artifact": str(token_path),
                    "token_sha256": _sha256(token_path),
                    "camera_images": [str(path) for path in image_paths],
                    "camera_sha256": [_sha256(path) for path in image_paths],
                    "frontier_permutation_new_to_old": identity,
                    "query_frontier": query_frontier,
                    "target": {"route": assigned},
                    "canonical_answer": assigned,
                    "control_expected_answers": control_answers,
                    "selection": {
                        "assigned_label": assigned,
                        "route_seed": route_seed,
                        "candidate_window_frames": frames,
                        "eligible_ON_ROUTE_rows": int(sum(labels)),
                        "eligible_OFF_ROUTE_rows": int(len(labels) - sum(labels)),
                        "selected_flat_candidate_index": selected_flat_index,
                    },
                }
            )
            candidate_summaries.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "source_folder": folder,
                    "minimum_valid_rows_per_candidate_frame": min(valid_counts),
                    "maximum_valid_rows_per_candidate_frame": max(valid_counts),
                    "eligible_ON_ROUTE_rows": int(sum(labels)),
                    "eligible_OFF_ROUTE_rows": int(len(labels) - sum(labels)),
                }
            )

    records.sort(key=lambda row: row["sample_id"])
    if len({row["sample_id"] for row in records}) != len(records):
        raise RuntimeError("duplicate sample ids")
    if len({(row["source_folder"], row["source_frame"], row["query_frontier"]) for row in records}) != len(records):
        raise RuntimeError("duplicate route/frame/Fxx query keys")
    split_counts = {
        split: {
            "routes": sum(row["split"] == split for row in records),
            **dict(
                sorted(
                    Counter(
                        row["target"]["route"]
                        for row in records
                        if row["split"] == split
                    ).items()
                )
            ),
        }
        for split in included_splits
    }
    if split_counts != expected:
        raise RuntimeError("constructed split counts differ from protocol")

    manifest_path = output_root / "manifest.json"
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "purpose": "bounded route-disjoint random-row route grounding baseline",
        "reportable_generalization": True,
        "bounded_route_disjoint_grounding_baseline_only": True,
        "oracle_depth": False,
        "visibility_source": "bench2drive_offline_uint8_depth",
        "hidden_actor_labels_used": False,
        "controls_used_for_optimizer": False,
        "planning_expert_used_for_optimizer": False,
        "image_profile": "three native 1600x900 current RGB views; official Qwen preprocessing",
        "token_root": str(token_root),
        "protocol": _reference(protocol_path),
        "system_prompt": (
            "You ground an explicit metric visibility-belief input for driving. "
            "The continuous tokens describe observed free/occupied space, occluded "
            "unknown space, observation age, and deterministic route/stopping "
            "exposure. They do not assert that a hidden actor exists."
        ),
        "record_count": len(records),
        "split_counts": split_counts,
        "records": records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    examples = []
    for record in records:
        frontier = record["query_frontier"]
        examples.append(
            {
                "example_id": "route-query-" + record["sample_id"],
                "sample_id": record["sample_id"],
                "source_folder": record["source_folder"],
                "split": record["split"],
                "task_field": "route",
                "query_frontier": frontier,
                "question": _query_question(frontier),
                "expected_answer": record["target"]["route"],
                "control_expected_answers": record["control_expected_answers"],
                "sequence_permutation_new_to_manifest": list(range(32)),
            }
        )
    by_id = {row["example_id"]: row for row in examples}
    training_ids = sorted(
        row["example_id"] for row in examples if row["split"] == "train"
    )
    validation_ids = sorted(
        row["example_id"] for row in examples if row["split"] == "validation"
    )
    held_out_ids = sorted(
        row["example_id"] for row in examples if row["split"] == "held_out"
    )
    pools = {
        label: [
            value
            for value in training_ids
            if by_id[value]["expected_answer"] == label
        ]
        for label in ("ON_ROUTE", "OFF_ROUTE")
    }
    optimizer_steps = int(protocol["curriculum"]["optimizer_steps"])
    schedule = []
    for round_index in range(optimizer_steps // 2):
        for label in ("ON_ROUTE", "OFF_ROUTE"):
            pool = pools[label]
            schedule.append(pool[round_index % len(pool)])
    curriculum_path = output_root / "curriculum.json"
    curriculum = {
        "schema": CURRICULUM_SCHEMA,
        "purpose": "one natural random addressed route query per physical route",
        "base_manifest_path": str(manifest_path),
        "base_manifest_sha256": _sha256(manifest_path),
        "reportable_generalization": True,
        "bounded_route_disjoint_grounding_baseline_only": True,
        "oracle_depth": False,
        "ordinary_per_example_supervision_only": True,
        "same_frame_answer_equality_penalty": False,
        "pair_flip_loss": False,
        "controls_used_for_optimizer": False,
        "spatial_shuffle_examples_used_for_optimizer": False,
        "planning_expert_used_for_optimizer": False,
        "natural_row_order_only": True,
        "one_query_per_route": True,
        "route_disjoint_splits": True,
        "threshold": route_threshold,
        "example_count": len(examples),
        "training_example_ids": training_ids,
        "validation_example_ids": validation_ids,
        "held_out_example_ids": held_out_ids,
        "evaluation_example_ids": validation_ids + held_out_ids,
        "split_counts": split_counts,
        "optimizer_steps": optimizer_steps,
        "optimizer_label_counts": dict(
            sorted(Counter(by_id[value]["expected_answer"] for value in schedule).items())
        ),
        "training_schedule": schedule,
        "examples": examples,
    }
    expected_optimizer = {
        "ON_ROUTE": int(protocol["curriculum"]["optimizer_ON_ROUTE"]),
        "OFF_ROUTE": int(protocol["curriculum"]["optimizer_OFF_ROUTE"]),
    }
    if curriculum["optimizer_label_counts"] != expected_optimizer:
        raise RuntimeError("optimizer label counts differ from protocol")
    curriculum_path.write_text(
        json.dumps(curriculum, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = {
        "schema": BUILD_REPORT_SCHEMA,
        "status": "complete",
        "implementation": _reference(Path(__file__)),
        "protocol": _reference(protocol_path),
        "inputs": {
            "infos": _reference(infos_path),
            "route_manifest": _reference(route_manifest_path),
            "qwen_visibility_config": _reference(qwen_config_path),
            "accepted_depth_preflight": _reference(preflight_path),
        },
        "manifest": _reference(manifest_path),
        "curriculum": _reference(curriculum_path),
        "split_counts": split_counts,
        "query_frontier_counts": dict(
            sorted(Counter(row["query_frontier"] for row in records).items())
        ),
        "candidate_summary": {
            "route_count": len(candidate_summaries),
            "minimum_valid_rows_per_candidate_frame": min(
                row["minimum_valid_rows_per_candidate_frame"]
                for row in candidate_summaries
            ),
            "minimum_ON_ROUTE_candidates_per_route": min(
                row["eligible_ON_ROUTE_rows"] for row in candidate_summaries
            ),
            "minimum_OFF_ROUTE_candidates_per_route": min(
                row["eligible_OFF_ROUTE_rows"] for row in candidate_summaries
            ),
        },
        "audit": {
            "duplicate_sample_ids": False,
            "duplicate_route_frame_Fxx_queries": False,
            "one_query_per_route": len(records) == len(set(row["source_folder"] for row in records)),
            "route_disjoint_by_frozen_manifest": True,
            "all_selected_frames_have_32_rows": True,
            "all_artifact_and_image_hashes_recorded": True,
            "ordinary_per_example_supervision_only": True,
            "controls_used_for_optimizer": False,
            "training_started": False,
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    report_path = output_root / "build_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    report = build(args.protocol, args.output_root)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
