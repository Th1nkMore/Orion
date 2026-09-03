"""Fail-closed planning for a chronological frozen-ORION target smoke.

This module deliberately does not import ORION, MMCV, CARLA, or CUDA.  It
turns one route in a persisted B2D pilot manifest into an exact replay
contract, optionally verifies the source info/image files, and evaluates a
runtime attestation emitted by a future real exporter.  Merely building a
plan can never pass G1.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

try:  # Package import in tests and direct sibling import in the pure CLI.
    from .b2d_route_manifest import load_b2d_infos, normalize_folder
except ImportError:  # pragma: no cover - exercised by CLI subprocess tests.
    from b2d_route_manifest import load_b2d_infos, normalize_folder  # type: ignore


REPLAY_PLAN_SCHEMA_VERSION = "orion-chronological-replay-smoke-plan/v1"
RUNTIME_ATTESTATION_SCHEMA_VERSION = "orion-replay-runtime-attestation/v1"
SUPPORTED_PILOT_SCHEMA = "spatial-uq-pilot-submanifest/v1"

DEFAULT_ROUTE_KEY = "Town04/Route214"
DEFAULT_PREFIX_END = 63
REQUIRED_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

_GLOBAL_G1_CHECKS = (
    "source_info_sha256_verified",
    "camera_files_exist_for_all_frames",
    "annotation_files_exist_for_all_frames",
    "with_light_state_enabled",
    "traffic_state_not_overwritten_by_mask",
    "post_augmentation_matrices_verified",
    "camera_order_verified",
    "decoded_output_adapter_ready",
    "actual_target_adapter_ready",
    "decoder_parity_passed",
    "selected_motion_mode_passed",
    "projection_overlay_passed",
)


class OrionReplayPlanError(ValueError):
    """Raised when a chronological replay or attestation fails closed."""


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OrionReplayPlanError("%s must be an object" % field)
    return value


def _require_bool(mapping: Mapping[str, Any], field: str) -> bool:
    value = mapping.get(field)
    if not isinstance(value, bool):
        raise OrionReplayPlanError("%s must be an explicit boolean" % field)
    return value


def _frame_index(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise OrionReplayPlanError("%s must be an integer" % field)
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise OrionReplayPlanError("%s must be an integer" % field) from exc
    try:
        if float(value) != float(result):
            raise OrionReplayPlanError("%s must be integral" % field)
    except (TypeError, ValueError) as exc:
        raise OrionReplayPlanError("%s must be an integer" % field) from exc
    if result < 0:
        raise OrionReplayPlanError("%s must be non-negative" % field)
    return result


def load_pilot_manifest(path: Union[Path, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    path = Path(path)
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise OrionReplayPlanError("pilot manifest root must be an object")
    if payload.get("schema_version") != SUPPORTED_PILOT_SCHEMA:
        raise OrionReplayPlanError("unsupported pilot manifest schema")
    return payload, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _find_route_folder(
    pilot: Mapping[str, Any], canonical_route_key: str
) -> Tuple[str, Mapping[str, Any], Mapping[str, Any]]:
    route_statistics = _require_mapping(pilot.get("route_statistics"), "route_statistics")
    if canonical_route_key not in route_statistics:
        raise OrionReplayPlanError(
            "canonical route %r is absent from the pilot" % canonical_route_key
        )
    route = _require_mapping(
        route_statistics[canonical_route_key],
        "route_statistics[%s]" % canonical_route_key,
    )
    folders = route.get("folders")
    if not isinstance(folders, list) or len(folders) != 1:
        raise OrionReplayPlanError(
            "smoke route must resolve to exactly one concrete folder"
        )
    folder = normalize_folder(folders[0])
    temporal = _require_mapping(
        pilot.get("temporal_execution_contract"), "temporal_execution_contract"
    )
    if _require_bool(temporal, "chronological_full_folder_replay_required") is not True:
        raise OrionReplayPlanError("pilot does not require chronological replay")
    if _require_bool(temporal, "frame_independent_inference_forbidden") is not True:
        raise OrionReplayPlanError("pilot does not forbid frame-independent inference")
    if _require_bool(temporal, "memory_reset_between_folders_required") is not True:
        raise OrionReplayPlanError("pilot does not require resets between folders")
    if (
        _require_bool(temporal, "memory_reset_between_clean_and_observed_passes_required")
        is not True
    ):
        raise OrionReplayPlanError("pilot does not require resets between branches")
    folder_statistics = _require_mapping(
        temporal.get("folder_replay_statistics"),
        "temporal_execution_contract.folder_replay_statistics",
    )
    if folder not in folder_statistics:
        raise OrionReplayPlanError("route folder lacks temporal replay statistics")
    replay = _require_mapping(folder_statistics[folder], "folder replay statistics")
    if replay.get("canonical_route_key") != canonical_route_key:
        raise OrionReplayPlanError("folder route identity disagrees with route statistics")
    if replay.get("chronologically_contiguous") is not True:
        raise OrionReplayPlanError("folder is not declared chronologically contiguous")
    start = _frame_index(replay.get("replay_frame_start"), "replay_frame_start")
    end = _frame_index(replay.get("replay_frame_end"), "replay_frame_end")
    count = _frame_index(replay.get("replay_frame_count"), "replay_frame_count")
    if start != 0 or end < start or count != end + 1:
        raise OrionReplayPlanError("folder must be an exact frame-0-through-end replay")

    per_folder = _require_mapping(route.get("folder_statistics"), "folder_statistics")
    route_replay = _require_mapping(per_folder.get(folder), "route folder statistics")
    if (
        _frame_index(route_replay.get("replay_frame_start"), "route replay start") != start
        or _frame_index(route_replay.get("replay_frame_end"), "route replay end") != end
        or _frame_index(route_replay.get("replay_frame_count"), "route replay count") != count
    ):
        raise OrionReplayPlanError("route and temporal replay statistics disagree")
    return folder, route, replay


def _measurement_samples(
    pilot: Mapping[str, Any], canonical_route_key: str, folder: str, prefix_end: int
) -> List[Mapping[str, Any]]:
    samples = pilot.get("samples")
    if not isinstance(samples, list):
        raise OrionReplayPlanError("pilot samples must be a list")
    selected: List[Mapping[str, Any]] = []
    seen = set()
    for raw in samples:
        sample = _require_mapping(raw, "one pilot sample")
        if sample.get("canonical_route_key") != canonical_route_key:
            continue
        if normalize_folder(sample.get("folder")) != folder:
            raise OrionReplayPlanError("route sample points at an unexpected folder")
        frame = _frame_index(sample.get("frame_idx"), "sample.frame_idx")
        if frame in seen:
            raise OrionReplayPlanError("duplicate measurement frame %d" % frame)
        seen.add(frame)
        if frame <= prefix_end:
            selected.append(sample)
    selected.sort(key=lambda item: _frame_index(item.get("frame_idx"), "sample.frame_idx"))
    if not selected:
        raise OrionReplayPlanError("prefix contains no measurement frame")
    if not any(
        item.get("annotation_stratum") == "safety_visible_candidate_annotation_only"
        for item in selected
    ):
        raise OrionReplayPlanError(
            "prefix contains no annotation candidate; it cannot smoke object projection"
        )
    return selected


def verify_source_infos(
    infos_path: Union[Path, str],
    pilot: Mapping[str, Any],
    folder: str,
    prefix_end: int,
    *,
    dataset_root: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Verify exact source lineage, frame continuity, cameras, and disk files.

    Loading the trusted pickle is opt-in.  If ``dataset_root`` is omitted, the
    metadata is checked but file-existence gates remain false.
    """

    infos, lineage = load_b2d_infos(infos_path)
    parent = _require_mapping(pilot.get("parent_manifest"), "parent_manifest")
    expected_source = _require_mapping(parent.get("info_source"), "parent info_source")
    expected_sha = expected_source.get("sha256")
    sha_verified = isinstance(expected_sha, str) and lineage.get("sha256") == expected_sha
    if not sha_verified:
        raise OrionReplayPlanError("source info SHA-256 disagrees with pilot lineage")

    by_frame: Dict[int, Mapping[str, Any]] = {}
    for raw in infos:
        if not isinstance(raw, Mapping) or raw.get("folder") is None:
            continue
        if normalize_folder(raw["folder"]) != folder:
            continue
        frame = _frame_index(raw.get("frame_idx"), "source frame_idx")
        if frame in by_frame:
            raise OrionReplayPlanError("source infos contain a duplicate route frame")
        by_frame[frame] = raw
    expected_frames = list(range(prefix_end + 1))
    present = sorted(frame for frame in by_frame if frame <= prefix_end)
    if present != expected_frames:
        missing = sorted(set(expected_frames) - set(present))
        raise OrionReplayPlanError(
            "source infos are not an exact frame-0 prefix; missing=%s" % missing[:10]
        )

    camera_metadata_ok = True
    camera_files_ok = dataset_root is not None
    annotation_files_ok = dataset_root is not None
    missing_camera_files: List[str] = []
    missing_annotation_files: List[str] = []
    root = Path(dataset_root) if dataset_root is not None else None
    for frame in expected_frames:
        sensors = by_frame[frame].get("sensors")
        if not isinstance(sensors, Mapping):
            raise OrionReplayPlanError("source frame %d lacks sensors" % frame)
        camera_names = tuple(name for name in sensors if str(name).startswith("CAM"))
        if camera_names != REQUIRED_CAMERAS:
            raise OrionReplayPlanError(
                "source frame %d camera insertion order is not the canonical ORION order"
                % frame
            )
        for camera in REQUIRED_CAMERAS:
            camera_info = _require_mapping(sensors[camera], "camera metadata")
            path_value = camera_info.get("data_path")
            if not isinstance(path_value, str) or not path_value.strip():
                camera_metadata_ok = False
                raise OrionReplayPlanError("camera data_path is missing")
            if root is not None:
                image_path = Path(path_value)
                if not image_path.is_absolute():
                    image_path = root / image_path
                if not image_path.is_file():
                    camera_files_ok = False
                    missing_camera_files.append(str(image_path))
        if root is not None:
            first_path = Path(str(_require_mapping(sensors[REQUIRED_CAMERAS[0]], "camera metadata")["data_path"]))
            if not first_path.is_absolute():
                first_path = root / first_path
            text_path = str(first_path)
            if "camera" not in text_path:
                annotation_files_ok = False
                missing_annotation_files.append("unresolvable-from-camera:%s" % text_path)
            else:
                annotation_root = Path(text_path.split("camera", 1)[0])
                annotation_path = annotation_root / "anno" / ("%05d.json.gz" % frame)
                if not annotation_path.is_file():
                    annotation_files_ok = False
                    missing_annotation_files.append(str(annotation_path))

    return {
        "performed": True,
        "info_path": str(Path(infos_path).resolve()),
        "info_sha256": lineage.get("sha256"),
        "source_info_sha256_verified": sha_verified,
        "exact_frame_zero_prefix_verified": True,
        "verified_frame_count": len(expected_frames),
        "canonical_six_camera_metadata_verified": camera_metadata_ok,
        "camera_files_exist_for_all_frames": bool(camera_files_ok),
        "annotation_files_exist_for_all_frames": bool(annotation_files_ok),
        "dataset_root": str(root.resolve()) if root is not None else None,
        "missing_camera_file_count": len(missing_camera_files),
        "missing_camera_files_first10": missing_camera_files[:10],
        "missing_annotation_file_count": len(missing_annotation_files),
        "missing_annotation_files_first10": missing_annotation_files[:10],
    }


def _plan_id(core: Mapping[str, Any]) -> str:
    rendered = json.dumps(core, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def build_replay_smoke_plan(
    pilot: Mapping[str, Any],
    pilot_lineage: Mapping[str, Any],
    *,
    canonical_route_key: str = DEFAULT_ROUTE_KEY,
    prefix_end: int = DEFAULT_PREFIX_END,
    corruption_family: str = "local_occlusion",
    severity: int = 2,
    seed: int = 20260826,
    source_verification: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    prefix_end = _frame_index(prefix_end, "prefix_end")
    if not corruption_family.strip() or corruption_family == "clean":
        raise OrionReplayPlanError("observed corruption family must be explicit")
    if isinstance(severity, bool) or int(severity) not in (1, 2, 3):
        raise OrionReplayPlanError("severity must be one of 1,2,3")
    folder, route, replay = _find_route_folder(pilot, canonical_route_key)
    full_end = _frame_index(replay.get("replay_frame_end"), "replay_frame_end")
    if prefix_end > full_end:
        raise OrionReplayPlanError("prefix_end exceeds the available route")
    samples = _measurement_samples(pilot, canonical_route_key, folder, prefix_end)
    measurement_frames = [
        _frame_index(item.get("frame_idx"), "sample.frame_idx") for item in samples
    ]
    candidate_count = sum(
        item.get("annotation_stratum") == "safety_visible_candidate_annotation_only"
        for item in samples
    )
    expected_frames = list(range(prefix_end + 1))
    core = {
        "pilot_manifest_sha256": str(pilot_lineage.get("sha256", "")),
        "canonical_route_key": canonical_route_key,
        "folder": folder,
        "prefix_start": 0,
        "prefix_end": prefix_end,
        "measurement_frames": measurement_frames,
        "corruption": {
            "family": corruption_family,
            "severity": int(severity),
            "seed": int(seed),
            "event_window_frames_inclusive": [0, prefix_end],
        },
    }
    identifier = _plan_id(core)
    source = dict(source_verification or {})
    if not source:
        source = {
            "performed": False,
            "source_info_sha256_verified": False,
            "exact_frame_zero_prefix_verified": False,
            "canonical_six_camera_metadata_verified": False,
            "camera_files_exist_for_all_frames": False,
            "annotation_files_exist_for_all_frames": False,
        }
    plan = {
        "schema_version": REPLAY_PLAN_SCHEMA_VERSION,
        "plan_id": identifier,
        "purpose": "G1 actual-target chronological replay smoke only",
        "pilot_manifest": dict(pilot_lineage),
        "route": {
            "canonical_route_key": canonical_route_key,
            "folder": folder,
            "town": route.get("town"),
            "scenario_types": route.get("scenario_types"),
            "parent_split": route.get("parent_split"),
            "full_route_frame_range_inclusive": [0, full_end],
            "full_route_frame_count": full_end + 1,
            "smoke_prefix_frame_range_inclusive": [0, prefix_end],
            "smoke_prefix_frame_count": len(expected_frames),
        },
        "corruption": {
            "family": corruption_family,
            "severity": int(severity),
            "seed": int(seed),
            "event_window_frames_inclusive": [0, prefix_end],
            "scope": "diagnostic_full_prefix_not_formal_time_alignment_evidence",
        },
        "execution": {
            "branch_order": ["clean", "observed"],
            "branches_are_separate_full_replays": True,
            "expected_frames_each_branch": expected_frames,
            "expected_forward_count_each_branch": len(expected_frames),
            "expected_forward_count_total": 2 * len(expected_frames),
            "measurement_frames": measurement_frames,
            "measurement_frame_count": len(measurement_frames),
            "persist_only_measurement_frames": True,
            "expected_paired_target_record_count": len(measurement_frames),
            "expected_branch_decoded_record_count": 2 * len(measurement_frames),
            "warmup_or_unscored_frame_count_each_branch": (
                len(expected_frames) - len(measurement_frames)
            ),
            "annotation_candidate_measurement_count": candidate_count,
            "background_measurement_count": len(measurement_frames) - candidate_count,
            "reset_before_each_branch": True,
            "required_reset_calls": [
                "model.pts_bbox_head.reset_memory()",
                "model.map_head.reset_memory() if map_head participates",
            ],
            "assert_reset_memory_fields_are_none": [
                "memory_embedding",
                "memory_reference_point",
                "memory_timestamp",
                "memory_egopose",
                "memory_velo",
                "sample_time",
                "memory_canbus",
                "memory_scene_tokens",
                "his_memory_canbus_len",
                "memory_scene_query if enabled",
                "scene_memory_timestamp if enabled",
            ],
            "batch_size": 1,
            "shuffle": False,
            "drop_last": False,
        },
        "source_verification": source,
        "runtime_attestation_contract": {
            "schema_version": RUNTIME_ATTESTATION_SCHEMA_VERSION,
            "required_global_g1_checks": list(_GLOBAL_G1_CHECKS),
            "required_branch_reports": ["clean", "observed"],
            "required_per_frame_audit": True,
            "per_frame_required_fields": [
                "frame_idx",
                "scene_token",
                "model_forward_completed",
                "six_camera_images_loaded",
                "traffic_state_shape_n_by_2",
                "traffic_state_mask_matches_objects",
                "post_augmentation_lidar2img_count_is_6",
                "processed_image_shape_present",
                "decoded_output_adapter_ready",
                "actual_target_adapter_ready",
                "persisted",
            ],
        },
        "known_static_blockers": [
            {
                "code": "agent_test_pipeline_light_state_disabled",
                "path": "adzoo/orion/configs/orion_stage3_agent.py",
                "requirement": "actual-target export config must use with_light_state=True",
            },
            {
                "code": "real_decoded_target_hook_not_connected",
                "path": "uq_estimator/decoded_actual_target_export.py",
                "requirement": "real frozen-ORION decoded-output and target adapter must be wired",
            },
        ],
        "resolved_code_findings_requiring_runtime_attestation": [
            {
                "code": "traffic_state_formatting_overwrite",
                "path": "mmcv/datasets/pipelines/formating.py",
                "local_fix": (
                    "filter [N,2] traffic_state and [N] validity together from the "
                    "same original GT-box mask"
                ),
                "regression_test": "tests/test_traffic_state_alignment.py",
                "runtime_attestation_still_required": True,
            }
        ],
        "resources": {
            "gpu": "1 x NVIDIA A800 80GB",
            "known_full_orion_host_memory_envelope_gb": 192,
            "recommended_slurm_memory_gb": 220,
            "recommended_cpus_per_task": 8,
            "recommended_time_limit": "02:00:00",
            "walltime_status": "unknown_until_first_real_measured_smoke",
            "job_submitted": False,
            "carla_required": False,
            "training_performed": False,
        },
        "full_route_comparison": {
            "frame_count_each_branch": full_end + 1,
            "minimum_forward_count_clean_plus_observed": 2 * (full_end + 1),
            "paired_measurement_record_count": int(replay.get("measurement_frame_count")),
        },
        "gates": {
            "g0_manifest_contract_passed": True,
            "g1_passed": False,
            "g1_status": "not_run",
            "execution_ready": False,
            "reason": (
                "A plan is not a real ORION run. G1 requires source/file preflight "
                "plus a complete runtime attestation."
            ),
        },
        "claim_boundary": {
            "formal_closed_loop_result": False,
            "stage1_training_authorized": False,
            "temporal_alignment_claim_supported": False,
            "g1_can_be_inferred_from_process_exit_code": False,
            "statement": (
                "This contract only bounds a diagnostic actual-target exporter smoke. "
                "It does not run ORION, pass G1, train an adapter, or support safety claims."
            ),
        },
    }
    return plan


def evaluate_runtime_attestation(
    plan: Mapping[str, Any], attestation: Mapping[str, Any]
) -> Dict[str, Any]:
    """Evaluate a real runner's exhaustive attestation; any omission fails G1."""

    if plan.get("schema_version") != REPLAY_PLAN_SCHEMA_VERSION:
        raise OrionReplayPlanError("unsupported replay plan schema")
    if attestation.get("schema_version") != RUNTIME_ATTESTATION_SCHEMA_VERSION:
        raise OrionReplayPlanError("unsupported runtime attestation schema")
    if attestation.get("plan_id") != plan.get("plan_id"):
        raise OrionReplayPlanError("runtime attestation plan_id mismatch")
    source = _require_mapping(plan.get("source_verification"), "source_verification")
    source_checks = {
        name: bool(source.get(name))
        for name in (
            "source_info_sha256_verified",
            "exact_frame_zero_prefix_verified",
            "canonical_six_camera_metadata_verified",
            "camera_files_exist_for_all_frames",
            "annotation_files_exist_for_all_frames",
        )
    }
    global_checks_raw = _require_mapping(
        attestation.get("global_checks"), "attestation.global_checks"
    )
    global_checks: Dict[str, bool] = {}
    for name in _GLOBAL_G1_CHECKS:
        global_checks[name] = _require_bool(global_checks_raw, name)

    execution = _require_mapping(plan.get("execution"), "execution")
    expected_frames = list(execution.get("expected_frames_each_branch", []))
    measurement_frames = list(execution.get("measurement_frames", []))
    expected_folder = _require_mapping(plan.get("route"), "route").get("folder")
    branches_raw = _require_mapping(attestation.get("branches"), "attestation.branches")
    if set(branches_raw) != {"clean", "observed"}:
        raise OrionReplayPlanError("attestation must contain exactly clean and observed")

    branch_checks: Dict[str, Dict[str, Any]] = {}
    paired_replay_ids = set()
    branch_history_ids = set()
    for branch_name in ("clean", "observed"):
        branch = _require_mapping(branches_raw[branch_name], "branch %s" % branch_name)
        processed = branch.get("frames_processed")
        persisted = branch.get("measurement_frames_persisted")
        per_frame = branch.get("per_frame_audit")
        if not isinstance(processed, list) or not isinstance(persisted, list):
            raise OrionReplayPlanError("branch frame lists must be explicit lists")
        if not isinstance(per_frame, list):
            raise OrionReplayPlanError("per_frame_audit must be a list")
        exact_processed = processed == expected_frames
        exact_persisted = persisted == measurement_frames
        exact_per_frame_count = len(per_frame) == len(expected_frames)
        frame_audits_pass = exact_per_frame_count
        audited_indices: List[int] = []
        for expected, raw_frame in zip(expected_frames, per_frame):
            frame = _require_mapping(raw_frame, "one per-frame audit")
            index = _frame_index(frame.get("frame_idx"), "attested frame_idx")
            audited_indices.append(index)
            if index != expected or frame.get("scene_token") != expected_folder:
                frame_audits_pass = False
            for field in plan["runtime_attestation_contract"]["per_frame_required_fields"]:
                if field in ("frame_idx", "scene_token", "persisted"):
                    continue
                if not isinstance(frame.get(field), bool) or frame.get(field) is not True:
                    frame_audits_pass = False
            if frame.get("persisted") is not (expected in set(measurement_frames)):
                frame_audits_pass = False
        reset_called = _require_bool(branch, "reset_called_before_frame_zero")
        reset_verified = _require_bool(branch, "reset_state_verified_empty")
        no_interleaving = _require_bool(branch, "no_other_branch_interleaving")
        paired_replay_id = branch.get("paired_replay_id")
        branch_history_id = branch.get("branch_history_id")
        if not isinstance(paired_replay_id, str) or not paired_replay_id.strip():
            raise OrionReplayPlanError("paired_replay_id must be non-empty")
        if not isinstance(branch_history_id, str) or not branch_history_id.strip():
            raise OrionReplayPlanError("branch_history_id must be non-empty")
        paired_replay_ids.add(paired_replay_id)
        branch_history_ids.add(branch_history_id)
        branch_checks[branch_name] = {
            "frames_processed_exactly_0_through_n": exact_processed,
            "measurement_frames_persisted_exactly": exact_persisted,
            "per_frame_audit_count_exact": exact_per_frame_count,
            "per_frame_checks_all_pass": frame_audits_pass,
            "reset_called_before_frame_zero": reset_called,
            "reset_state_verified_empty": reset_verified,
            "no_other_branch_interleaving": no_interleaving,
            "audited_frame_indices_exact": audited_indices == expected_frames,
        }

    temporal_pairing = {
        "shared_paired_replay_id": len(paired_replay_ids) == 1,
        "distinct_branch_history_ids": len(branch_history_ids) == 2,
    }
    failures: List[str] = []
    failures.extend("source:%s" % key for key, value in source_checks.items() if not value)
    failures.extend("global:%s" % key for key, value in global_checks.items() if not value)
    for branch_name, checks in branch_checks.items():
        failures.extend(
            "branch:%s:%s" % (branch_name, key)
            for key, value in checks.items()
            if not value
        )
    failures.extend(
        "temporal:%s" % key for key, value in temporal_pairing.items() if not value
    )
    g1_passed = not failures
    return {
        "schema_version": "orion-replay-g1-evaluation/v1",
        "plan_id": plan.get("plan_id"),
        "source_checks": source_checks,
        "global_checks": global_checks,
        "branch_checks": branch_checks,
        "temporal_pairing_checks": temporal_pairing,
        "g1_passed": g1_passed,
        "g1_status": "passed" if g1_passed else "failed_closed",
        "failure_count": len(failures),
        "failures": failures,
    }


def replay_plan_summary(plan: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": plan["schema_version"],
        "plan_id": plan["plan_id"],
        "route": plan["route"],
        "corruption": plan["corruption"],
        "execution": plan["execution"],
        "source_verification": plan["source_verification"],
        "known_static_blockers": plan["known_static_blockers"],
        "resources": plan["resources"],
        "full_route_comparison": plan["full_route_comparison"],
        "gates": plan["gates"],
        "claim_boundary": plan["claim_boundary"],
    }
