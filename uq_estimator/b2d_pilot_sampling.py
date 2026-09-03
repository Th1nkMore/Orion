"""Auditable metadata-only pilot sampling under a parent B2D route manifest.

The sampler selects complete canonical route groups from the persisted parent
split, then chooses unique ``(folder, frame_idx)`` states inside those groups.
GT names/boxes are used only to stratify an *annotation candidate* for a nearby
forward road actor versus background.  This is not camera visibility: G1
projection overlays remain mandatory before a frame can be called visible.
"""

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Set, Tuple, Union

try:  # Package import in tests and direct sibling import in the pure CLI.
    from .b2d_route_manifest import (
        REQUIRED_SPLITS,
        B2DManifestError,
        canonicalize_b2d_folder,
        load_b2d_infos,
        normalize_folder,
    )
except ImportError:  # pragma: no cover - exercised by CLI subprocess tests.
    from b2d_route_manifest import (  # type: ignore
        REQUIRED_SPLITS,
        B2DManifestError,
        canonicalize_b2d_folder,
        load_b2d_infos,
        normalize_folder,
    )


PILOT_MANIFEST_SCHEMA_VERSION = "spatial-uq-pilot-submanifest/v1"
PILOT_SELECTION_VERSION = "b2d-annotation-stratified-frame-sampling/v1"

_CRITICAL_NORMALIZED_CLASSES = {
    "car",
    "van",
    "truck",
    "bus",
    "construction_vehicle",
    "trailer",
    "motorcycle",
    "bicycle",
    "pedestrian",
}


class B2DPilotSamplingError(B2DManifestError):
    """Raised when parent lineage or pilot sampling fails closed."""


class AnnotationCandidate(NamedTuple):
    candidate: bool
    candidate_object_count: int
    candidate_classes: Tuple[str, ...]
    nearest_distance_m: Optional[float]


class PilotFrame(NamedTuple):
    sample_id: str
    folder: str
    frame_idx: int
    canonical_route_key: str
    parent_split: str
    town: str
    scenario: str
    weather: Optional[int]
    repetition: Optional[int]
    annotation: AnnotationCandidate


def _as_list(value: Any, field: str) -> List[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise B2DPilotSamplingError(f"{field} must be a list-like annotation")
    return value


def _normalized_actor_class(raw_name: Any) -> Optional[str]:
    name = str(raw_name).strip().lower()
    if name in _CRITICAL_NORMALIZED_CLASSES:
        return name
    if name.startswith("walker.") or "pedestrian" in name:
        return "pedestrian"
    if any(token in name for token in ("bicycle", "crossbike", "diamondback", "gazelle")):
        return "bicycle"
    if "motorcycle" in name or "motorbike" in name:
        return "motorcycle"
    if "ambulance" in name:
        return "van"
    if "firetruck" in name or "truck" in name:
        return "truck"
    if name.startswith("vehicle."):
        return "car"
    return None


def annotation_candidate_from_info(
    info: Mapping[str, Any],
    *,
    max_distance_m: float = 50.0,
    max_lateral_m: float = 15.0,
) -> AnnotationCandidate:
    """Classify a frame using lidar-frame GT annotations only.

    A candidate has a mapped road actor with nonzero points (when that field is
    available), a finite center in front of ego, and a bounded radial/lateral
    distance.  No camera frustum, projection, occlusion, or pixel support is
    checked here, so the result must never be interpreted as true visibility.
    """

    if max_distance_m <= 0 or max_lateral_m <= 0:
        raise B2DPilotSamplingError("candidate distance gates must be positive")
    if "gt_names" not in info or "gt_boxes" not in info:
        raise B2DPilotSamplingError(
            "every pilot info needs gt_names and gt_boxes; missing GT is not background"
        )
    names = _as_list(info["gt_names"], "gt_names")
    boxes = _as_list(info["gt_boxes"], "gt_boxes")
    if len(names) != len(boxes):
        raise B2DPilotSamplingError("gt_names and gt_boxes lengths disagree")
    point_counts: Optional[List[Any]] = None
    if "num_points" in info and info["num_points"] is not None:
        point_counts = _as_list(info["num_points"], "num_points")
        if len(point_counts) != len(names):
            raise B2DPilotSamplingError("num_points and gt_names lengths disagree")

    matched: List[Tuple[str, float]] = []
    for index, (name, box_value) in enumerate(zip(names, boxes)):
        actor_class = _normalized_actor_class(name)
        if actor_class is None:
            continue
        if point_counts is not None:
            try:
                if float(point_counts[index]) <= 0:
                    continue
            except (TypeError, ValueError) as exc:
                raise B2DPilotSamplingError("num_points must be numeric") from exc
        box = _as_list(box_value, "one gt_boxes row")
        if len(box) < 2:
            raise B2DPilotSamplingError("each gt_boxes row must contain at least x,y")
        try:
            forward = float(box[0])
            lateral = float(box[1])
        except (TypeError, ValueError) as exc:
            raise B2DPilotSamplingError("gt box centers must be numeric") from exc
        if not math.isfinite(forward) or not math.isfinite(lateral):
            raise B2DPilotSamplingError("gt box centers must be finite")
        distance = math.hypot(forward, lateral)
        if 0.0 <= forward <= max_distance_m and abs(lateral) <= max_lateral_m and distance <= max_distance_m:
            matched.append((actor_class, distance))

    return AnnotationCandidate(
        candidate=bool(matched),
        candidate_object_count=len(matched),
        candidate_classes=tuple(sorted({name for name, _ in matched})),
        nearest_distance_m=min((distance for _, distance in matched), default=None),
    )


def _required_frame_idx(info: Mapping[str, Any], index: int) -> int:
    value = None
    for key in ("frame_idx", "frame_id", "frame"):
        if key in info and info[key] is not None:
            value = info[key]
            break
    if value is None or isinstance(value, bool):
        raise B2DPilotSamplingError(f"info record {index} lacks integer frame identity")
    try:
        frame = int(value)
    except (TypeError, ValueError) as exc:
        raise B2DPilotSamplingError(f"invalid frame identity at info record {index}") from exc
    try:
        if float(value) != float(frame):
            raise B2DPilotSamplingError(f"non-integral frame identity at info record {index}")
    except (TypeError, ValueError) as exc:
        raise B2DPilotSamplingError(f"invalid frame identity at info record {index}") from exc
    if frame < 0:
        raise B2DPilotSamplingError("frame identity must be non-negative")
    return frame


def _parent_folder_owners(parent: Mapping[str, Any]) -> Dict[str, str]:
    if parent.get("route_disjoint") is not True:
        raise B2DPilotSamplingError("parent manifest must assert route_disjoint=true")
    raw_splits = parent.get("splits")
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != set(REQUIRED_SPLITS):
        raise B2DPilotSamplingError("parent manifest must contain exactly four splits")
    owners: Dict[str, str] = {}
    for split in REQUIRED_SPLITS:
        split_payload = raw_splits[split]
        if not isinstance(split_payload, Mapping):
            raise B2DPilotSamplingError(f"parent split {split} is malformed")
        folders = split_payload.get("route_ids")
        if not isinstance(folders, list) or not folders:
            raise B2DPilotSamplingError(f"parent split {split} has no route_ids")
        for raw_folder in folders:
            folder = normalize_folder(raw_folder)
            if folder in owners:
                raise B2DPilotSamplingError(
                    f"parent folder {folder!r} occurs in {owners[folder]} and {split}"
                )
            owners[folder] = split
    return owners


def load_parent_manifest(path: Union[Path, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    path = Path(path)
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise B2DPilotSamplingError("parent manifest root must be an object")
    _parent_folder_owners(payload)
    return payload, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "schema_version": payload.get("schema_version"),
        "seed": payload.get("seed"),
    }


def index_pilot_frames(
    infos: Sequence[Mapping[str, Any]],
    parent: Mapping[str, Any],
    *,
    max_distance_m: float = 50.0,
    max_lateral_m: float = 15.0,
) -> Dict[str, List[PilotFrame]]:
    """Index all info states by canonical route while enforcing parent lineage."""

    owners = _parent_folder_owners(parent)
    by_route: Dict[str, List[PilotFrame]] = defaultdict(list)
    seen: Set[Tuple[str, int]] = set()
    seen_folders: Set[str] = set()
    route_owner: Dict[str, str] = {}
    for index, info in enumerate(infos):
        if not isinstance(info, Mapping):
            raise B2DPilotSamplingError(f"info record {index} is not a mapping")
        if "folder" not in info:
            raise B2DPilotSamplingError(f"info record {index} lacks folder")
        parsed = canonicalize_b2d_folder(info["folder"])
        if parsed.folder not in owners:
            raise B2DPilotSamplingError(
                f"info folder {parsed.folder!r} is absent from parent manifest"
            )
        split = owners[parsed.folder]
        previous_split = route_owner.setdefault(parsed.canonical_route_key, split)
        if previous_split != split:
            raise B2DPilotSamplingError(
                f"weather/repetition variants of {parsed.canonical_route_key} "
                f"cross parent splits {previous_split}/{split}"
            )
        frame_idx = _required_frame_idx(info, index)
        identity = (parsed.folder, frame_idx)
        if identity in seen:
            raise B2DPilotSamplingError(f"duplicate route/frame identity {identity!r}")
        seen.add(identity)
        seen_folders.add(parsed.folder)
        annotation = annotation_candidate_from_info(
            info,
            max_distance_m=max_distance_m,
            max_lateral_m=max_lateral_m,
        )
        by_route[parsed.canonical_route_key].append(
            PilotFrame(
                sample_id=f"{parsed.folder}__frame_{frame_idx:06d}",
                folder=parsed.folder,
                frame_idx=frame_idx,
                canonical_route_key=parsed.canonical_route_key,
                parent_split=split,
                town=parsed.town,
                scenario=parsed.scenario,
                weather=parsed.weather,
                repetition=parsed.repetition,
                annotation=annotation,
            )
        )
    if not by_route:
        raise B2DPilotSamplingError("infos contain no pilot frames")
    missing_parent_folders = set(owners) - seen_folders
    if missing_parent_folders:
        raise B2DPilotSamplingError(
            "parent manifest folders are absent from infos: "
            + ", ".join(sorted(missing_parent_folders)[:10])
        )
    for frames in by_route.values():
        frames.sort(key=lambda item: (item.frame_idx, item.folder))
    return dict(by_route)


def _stable_tie(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _route_quotas(
    routes_by_split: Mapping[str, Sequence[str]],
    route_count: int,
    minimum_per_split: int,
) -> Dict[str, int]:
    if not 8 <= route_count <= 12:
        raise B2DPilotSamplingError("pilot route_count must lie in [8, 12]")
    if minimum_per_split < 1 or route_count < minimum_per_split * 4:
        raise B2DPilotSamplingError("route_count cannot satisfy minimum routes per split")
    quotas = {split: minimum_per_split for split in REQUIRED_SPLITS}
    if any(len(routes_by_split[split]) < quotas[split] for split in REQUIRED_SPLITS):
        raise B2DPilotSamplingError("a parent split has too few canonical routes for pilot")
    remaining = route_count - sum(quotas.values())
    while remaining:
        candidates = [
            split
            for split in REQUIRED_SPLITS
            if quotas[split] < len(routes_by_split[split])
        ]
        if not candidates:
            raise B2DPilotSamplingError("parent manifest cannot supply route_count")
        # Allocate extras toward the parent route proportions; with the current
        # 35/5/5/5 manifest this intentionally puts pilot extras in train.
        chosen = max(
            candidates,
            key=lambda split: (
                len(routes_by_split[split]) / float(quotas[split] + 1),
                -REQUIRED_SPLITS.index(split),
            ),
        )
        quotas[chosen] += 1
        remaining -= 1
    return quotas


def _select_routes(
    frames_by_route: Mapping[str, Sequence[PilotFrame]],
    *,
    route_count: int,
    target_states: int,
    minimum_per_split: int,
    seed: int,
) -> Tuple[List[str], Dict[str, int]]:
    routes_by_split: Dict[str, List[str]] = {split: [] for split in REQUIRED_SPLITS}
    for key, frames in frames_by_route.items():
        owners = {frame.parent_split for frame in frames}
        if len(owners) != 1:
            raise B2DPilotSamplingError(f"canonical route {key} crosses parent splits")
        routes_by_split[next(iter(owners))].append(key)
    quotas = _route_quotas(routes_by_split, route_count, minimum_per_split)
    approximate_states = int(math.ceil(target_states / float(route_count)))
    selected: List[str] = []
    for split in REQUIRED_SPLITS:
        chosen: List[str] = []
        used_towns: Set[str] = set()
        used_scenarios: Set[str] = set()
        candidates = list(routes_by_split[split])
        while len(chosen) < quotas[split]:
            if not candidates:
                raise B2DPilotSamplingError(f"cannot fill route quota for {split}")
            def score(key: str) -> Tuple[Any, ...]:
                frames = frames_by_route[key]
                candidate_count = sum(frame.annotation.candidate for frame in frames)
                background_count = len(frames) - candidate_count
                desired_candidate = approximate_states // 2
                desired_background = approximate_states - desired_candidate
                balanced_capacity = min(candidate_count, desired_candidate) + min(
                    background_count, desired_background
                )
                towns = {frame.town for frame in frames}
                scenarios = {frame.scenario for frame in frames}
                return (
                    min(len(frames), approximate_states),
                    balanced_capacity,
                    int(bool(towns - used_towns)),
                    int(bool(scenarios - used_scenarios)),
                    -abs(candidate_count / float(len(frames)) - 0.5),
                    -_stable_tie(seed, key),
                )
            best = max(candidates, key=score)
            candidates.remove(best)
            chosen.append(best)
            used_towns.update(frame.town for frame in frames_by_route[best])
            used_scenarios.update(frame.scenario for frame in frames_by_route[best])
        selected.extend(chosen)
    return selected, quotas


def _state_allocations(
    selected_routes: Sequence[str],
    frames_by_route: Mapping[str, Sequence[PilotFrame]],
    target_states: int,
) -> Dict[str, int]:
    allocations = {key: target_states // len(selected_routes) for key in selected_routes}
    for key in selected_routes[: target_states % len(selected_routes)]:
        allocations[key] += 1
    deficit = 0
    for key in selected_routes:
        capacity = len(frames_by_route[key])
        if allocations[key] > capacity:
            deficit += allocations[key] - capacity
            allocations[key] = capacity
    while deficit:
        progressed = False
        for key in selected_routes:
            if allocations[key] < len(frames_by_route[key]):
                allocations[key] += 1
                deficit -= 1
                progressed = True
                if not deficit:
                    break
        if not progressed:
            raise B2DPilotSamplingError(
                f"selected routes contain fewer than target_states={target_states} unique states"
            )
    return allocations


def _even_sample(frames: Sequence[PilotFrame], count: int, seed: int, key: str) -> List[PilotFrame]:
    if count < 0 or count > len(frames):
        raise B2DPilotSamplingError("invalid within-stratum sample count")
    if count == len(frames):
        return list(frames)
    if count == 0:
        return []
    ordered = sorted(frames, key=lambda item: (item.frame_idx, item.folder))
    selected: List[PilotFrame] = []
    for index in range(count):
        start = int(math.floor(index * len(ordered) / float(count)))
        stop = int(math.floor((index + 1) * len(ordered) / float(count)))
        stop = max(start + 1, stop)
        offset = _stable_tie(seed, f"{key}|{index}") % (stop - start)
        selected.append(ordered[start + offset])
    return selected


def _sample_route_frames(
    frames: Sequence[PilotFrame], count: int, seed: int, route_key: str
) -> List[PilotFrame]:
    candidate = [frame for frame in frames if frame.annotation.candidate]
    background = [frame for frame in frames if not frame.annotation.candidate]
    target_candidate = count // 2
    take_candidate = min(len(candidate), target_candidate)
    take_background = min(len(background), count - take_candidate)
    remaining = count - take_candidate - take_background
    if remaining:
        candidate_spare = len(candidate) - take_candidate
        extra = min(remaining, candidate_spare)
        take_candidate += extra
        remaining -= extra
    if remaining:
        background_spare = len(background) - take_background
        extra = min(remaining, background_spare)
        take_background += extra
        remaining -= extra
    if remaining:
        raise B2DPilotSamplingError(f"route {route_key} lacks enough unique states")
    sampled = _even_sample(candidate, take_candidate, seed, route_key + "/candidate")
    sampled += _even_sample(background, take_background, seed, route_key + "/background")
    sampled.sort(key=lambda item: (item.frame_idx, item.folder))
    return sampled


def _frame_to_dict(frame: PilotFrame) -> Dict[str, Any]:
    return {
        "sample_id": frame.sample_id,
        "folder": frame.folder,
        "frame_idx": frame.frame_idx,
        "canonical_route_key": frame.canonical_route_key,
        "parent_split": frame.parent_split,
        "town": frame.town,
        "scenario": frame.scenario,
        "weather": frame.weather,
        "repetition": frame.repetition,
        "annotation_stratum": (
            "safety_visible_candidate_annotation_only"
            if frame.annotation.candidate
            else "background_annotation"
        ),
        "candidate_object_count": frame.annotation.candidate_object_count,
        "candidate_classes": list(frame.annotation.candidate_classes),
        "nearest_candidate_distance_m": (
            round(frame.annotation.nearest_distance_m, 4)
            if frame.annotation.nearest_distance_m is not None
            else None
        ),
        "camera_visibility_confirmed": False,
        "g1_overlay_required": True,
    }


def _route_stats(
    route_key: str,
    available: Sequence[PilotFrame],
    sampled: Sequence[PilotFrame],
) -> Dict[str, Any]:
    folders = sorted({frame.folder for frame in available})
    return {
        "canonical_route_key": route_key,
        "parent_split": available[0].parent_split,
        "town": available[0].town,
        "scenario_types": sorted({frame.scenario for frame in available}),
        "folders": folders,
        "weather_variants": sorted(
            {frame.weather for frame in available if frame.weather is not None}
        ),
        "repetition_variants": sorted(
            {frame.repetition for frame in available if frame.repetition is not None}
        ),
        "available_states": len(available),
        "available_annotation_candidates": sum(
            frame.annotation.candidate for frame in available
        ),
        "available_background": sum(
            not frame.annotation.candidate for frame in available
        ),
        "sampled_states": len(sampled),
        "sampled_annotation_candidates": sum(
            frame.annotation.candidate for frame in sampled
        ),
        "sampled_background": sum(not frame.annotation.candidate for frame in sampled),
        "folder_statistics": {
            folder: {
                "available_frame_count": len(
                    [frame for frame in available if frame.folder == folder]
                ),
                "chronological_replay_required": True,
                "replay_frame_count": len(
                    [frame for frame in available if frame.folder == folder]
                ),
                "replay_frame_start": min(
                    frame.frame_idx for frame in available if frame.folder == folder
                ),
                "replay_frame_end": max(
                    frame.frame_idx for frame in available if frame.folder == folder
                ),
                "sampled_frame_count": sum(frame.folder == folder for frame in sampled),
                "sampled_annotation_candidate_count": sum(
                    frame.folder == folder and frame.annotation.candidate
                    for frame in sampled
                ),
                "sampled_background_count": sum(
                    frame.folder == folder and not frame.annotation.candidate
                    for frame in sampled
                ),
            }
            for folder in folders
        },
        "sampled_frame_ranges_by_folder": {
            folder: {
                "count": len(values),
                "min": min(frame.frame_idx for frame in values),
                "max": max(frame.frame_idx for frame in values),
            }
            for folder, values in sorted(
                (
                    (folder, [frame for frame in sampled if frame.folder == folder])
                    for folder in {frame.folder for frame in sampled}
                ),
                key=lambda item: item[0],
            )
        },
    }


def build_b2d_pilot_submanifest(
    infos: Sequence[Mapping[str, Any]],
    parent: Mapping[str, Any],
    parent_lineage: Mapping[str, Any],
    *,
    seed: int = 20260826,
    route_count: int = 10,
    target_states: int = 900,
    minimum_routes_per_split: int = 2,
    max_candidate_distance_m: float = 50.0,
    max_candidate_lateral_m: float = 15.0,
    minimum_candidate_fraction: float = 0.25,
    minimum_background_fraction: float = 0.25,
) -> Dict[str, Any]:
    """Build a bounded exploratory pilot without mutating parent split labels."""

    if not 800 <= target_states <= 1000:
        raise B2DPilotSamplingError("pilot target_states must lie in [800, 1000]")
    if not 0 <= minimum_candidate_fraction <= 1 or not 0 <= minimum_background_fraction <= 1:
        raise B2DPilotSamplingError("minimum stratum fractions must lie in [0,1]")
    if minimum_candidate_fraction + minimum_background_fraction > 1:
        raise B2DPilotSamplingError("minimum candidate/background fractions exceed one")
    frames_by_route = index_pilot_frames(
        infos,
        parent,
        max_distance_m=max_candidate_distance_m,
        max_lateral_m=max_candidate_lateral_m,
    )
    selected_routes, quotas = _select_routes(
        frames_by_route,
        route_count=route_count,
        target_states=target_states,
        minimum_per_split=minimum_routes_per_split,
        seed=seed,
    )
    allocations = _state_allocations(selected_routes, frames_by_route, target_states)
    sampled_by_route = {
        key: _sample_route_frames(frames_by_route[key], allocations[key], seed, key)
        for key in selected_routes
    }
    sampled = [frame for key in selected_routes for frame in sampled_by_route[key]]
    sample_ids = [frame.sample_id for frame in sampled]
    if len(sample_ids) != len(set(sample_ids)):
        raise B2DPilotSamplingError("pilot contains duplicate route/frame identities")
    candidate_count = sum(frame.annotation.candidate for frame in sampled)
    background_count = len(sampled) - candidate_count
    if candidate_count / float(len(sampled)) < minimum_candidate_fraction:
        raise B2DPilotSamplingError("pilot cannot meet minimum annotation-candidate fraction")
    if background_count / float(len(sampled)) < minimum_background_fraction:
        raise B2DPilotSamplingError("pilot cannot meet minimum background fraction")

    route_stats = {
        key: _route_stats(key, frames_by_route[key], sampled_by_route[key])
        for key in selected_routes
    }
    folder_replay_statistics: Dict[str, Dict[str, Any]] = {}
    for key in selected_routes:
        available = frames_by_route[key]
        route_sampled = sampled_by_route[key]
        for folder in sorted({frame.folder for frame in available}):
            folder_frames = sorted(
                frame.frame_idx for frame in available if frame.folder == folder
            )
            if not folder_frames or folder_frames[0] != 0:
                raise B2DPilotSamplingError(
                    f"selected folder {folder!r} cannot replay from frame zero"
                )
            expected = list(range(folder_frames[-1] + 1))
            if folder_frames != expected:
                raise B2DPilotSamplingError(
                    f"selected folder {folder!r} is not chronologically contiguous"
                )
            measurement_frames = sorted(
                frame.frame_idx for frame in route_sampled if frame.folder == folder
            )
            folder_replay_statistics[folder] = {
                "canonical_route_key": key,
                "parent_split": available[0].parent_split,
                "replay_frame_start": 0,
                "replay_frame_end": folder_frames[-1],
                "replay_frame_count": len(folder_frames),
                "chronologically_contiguous": True,
                "measurement_frame_count": len(measurement_frames),
                "measurement_frame_min": (
                    min(measurement_frames) if measurement_frames else None
                ),
                "measurement_frame_max": (
                    max(measurement_frames) if measurement_frames else None
                ),
                "warmup_or_unscored_frame_count": len(folder_frames)
                - len(measurement_frames),
            }
    replay_frame_count = sum(
        stats["replay_frame_count"] for stats in folder_replay_statistics.values()
    )
    minimum_forward_count = 2 * replay_frame_count
    split_stats: Dict[str, Dict[str, Any]] = {}
    for split in REQUIRED_SPLITS:
        keys = sorted(
            key for key in selected_routes if frames_by_route[key][0].parent_split == split
        )
        split_frames = [frame for key in keys for frame in sampled_by_route[key]]
        split_folders = {
            frame.folder for key in keys for frame in frames_by_route[key]
        }
        split_replay_frames = sum(
            folder_replay_statistics[folder]["replay_frame_count"]
            for folder in split_folders
        )
        split_stats[split] = {
            "canonical_route_count": len(keys),
            "canonical_route_keys": keys,
            "folder_count": len(split_folders),
            "sampled_state_count": len(split_frames),
            "chronological_replay_frame_count": split_replay_frames,
            "estimated_minimum_forward_count_clean_plus_observed": (
                2 * split_replay_frames
            ),
            "annotation_candidate_count": sum(
                frame.annotation.candidate for frame in split_frames
            ),
            "background_count": sum(
                not frame.annotation.candidate for frame in split_frames
            ),
            "towns": sorted({frame.town for frame in split_frames}),
            "scenario_types": sorted({frame.scenario for frame in split_frames}),
        }
        if len(keys) != quotas[split]:
            raise B2DPilotSamplingError("selected route quota changed unexpectedly")
        if split_stats[split]["annotation_candidate_count"] == 0:
            raise B2DPilotSamplingError(
                f"parent split {split} has no sampled dynamic safety-object annotation candidate"
            )

    parent_sha = str(parent_lineage.get("sha256", ""))
    if len(parent_sha) != 64:
        raise B2DPilotSamplingError("parent lineage requires a SHA-256 digest")
    payload = {
        "schema_version": PILOT_MANIFEST_SCHEMA_VERSION,
        "selection_version": PILOT_SELECTION_VERSION,
        "parent_manifest": dict(parent_lineage),
        "selection": {
            "seed": int(seed),
            "route_count": len(selected_routes),
            "target_state_count": target_states,
            "sampled_state_count": len(sampled),
            "route_quota_by_parent_split": quotas,
            "candidate_gate": {
                "source": "gt_names+gt_boxes with optional num_points filtering",
                "forward_only": True,
                "max_distance_m": max_candidate_distance_m,
                "max_lateral_m": max_candidate_lateral_m,
                "target_role": "sampling_stratification_only",
                "camera_visibility_confirmed": False,
                "g1_overlay_required": True,
            },
            "annotation_candidate_count": candidate_count,
            "background_count": background_count,
            "annotation_candidate_fraction": candidate_count / float(len(sampled)),
            "background_fraction": background_count / float(len(sampled)),
        },
        "split_statistics": split_stats,
        "route_statistics": route_stats,
        "temporal_execution_contract": {
            "chronological_full_folder_replay_required": True,
            "frame_independent_inference_forbidden": True,
            "measurement_frames_role": "target_scoring_and_persistence_only",
            "measurement_frame_count": len(sampled),
            "replay_scope": (
                "every available parent-info frame from frame 0 through the end "
                "of every folder in each selected canonical route"
            ),
            "selected_folder_count": len(folder_replay_statistics),
            "chronological_replay_frame_count": replay_frame_count,
            "minimum_passes_per_replay_frame": {
                "clean": 1,
                "observed": 1,
                "total": 2,
            },
            "memory_reset_between_folders_required": True,
            "memory_reset_between_clean_and_observed_passes_required": True,
            "model_state_may_carry_across_measurement_frames_only_via_replay": True,
            "estimated_minimum_forward_count_clean_plus_observed": (
                minimum_forward_count
            ),
            "forward_count_formula": (
                "chronological_replay_frame_count * (1 clean + at least 1 observed)"
            ),
            "additional_observed_conditions_require_additional_passes": True,
            "folder_replay_statistics": folder_replay_statistics,
        },
        "samples": sorted(
            (_frame_to_dict(frame) for frame in sampled),
            key=lambda item: (
                REQUIRED_SPLITS.index(item["parent_split"]),
                item["canonical_route_key"],
                item["frame_idx"],
                item["folder"],
            ),
        ),
        "audit": {
            "parent_sha256_recorded": parent_sha,
            "parent_split_preserved": True,
            "weather_repetition_variants_selected_as_canonical_groups": True,
            "route_frame_identity_unique": True,
            "chronological_full_folder_replay_required": True,
            "frame_independent_inference_forbidden": True,
            "measurement_frames_are_not_forward_pass_count": True,
            "orion_temporal_memory_preserved_by_contract": True,
            "all_four_parent_splits_represented": all(
                split_stats[split]["canonical_route_count"] > 0
                for split in REQUIRED_SPLITS
            ),
            "dynamic_safety_annotation_candidate_present_in_each_split": all(
                split_stats[split]["annotation_candidate_count"] > 0
                for split in REQUIRED_SPLITS
            ),
            "closed_loop_carla_map_filter_applied": False,
            "offline_town12_allowed": True,
            "image_files_opened": False,
            "model_loaded": False,
            "gpu_used": False,
            "carla_used": False,
            "scheduler_job_submitted": False,
        },
        "claim_boundary": {
            "purpose": "exploratory G0-G3 pipeline pilot subset",
            "independent_test_set": False,
            "unbiased_validation_metrics_supported": False,
            "training_heldout_beyond_parent_manifest_supported": False,
            "closed_loop_heldout_supported": False,
            "gt_annotation_used_for_selection": True,
            "sampled_states_are_measurement_frames_only": True,
            "frame_independent_orion_inference_permitted": False,
            "orion_temporal_memory_requirement": (
                "replay every selected folder chronologically from frame 0; "
                "persist/score targets only on declared measurement frames"
            ),
            "annotation_candidate_equals_camera_visible": False,
            "visibility_requirement": (
                "G1 six-view projection/occlusion overlay must confirm camera patch support"
            ),
            "installed_carla_runtime_compatibility_required": False,
            "town12_policy": (
                "Town12 is valid for this offline annotation/target pilot and is not "
                "filtered merely because the installed closed-loop CARLA lacks Town12"
            ),
            "statement": (
                "This is an annotation-stratified subset of the exploratory B2D "
                "validation infos. It is suitable for bounded exporter and target QA, "
                "not for final generalization, safety, or closed-loop claims."
            ),
        },
    }
    return payload


def pilot_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a compact, deterministic summary for dry-run terminals."""

    return {
        "schema_version": payload["schema_version"],
        "parent_manifest_sha256": payload["parent_manifest"]["sha256"],
        "selection": payload["selection"],
        "split_statistics": payload["split_statistics"],
        "route_statistics": payload["route_statistics"],
        "temporal_execution_contract": payload["temporal_execution_contract"],
        "audit": payload["audit"],
        "claim_boundary": payload["claim_boundary"],
        "writes_performed": payload.get("writes_performed", False),
    }
