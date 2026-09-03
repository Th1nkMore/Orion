"""Build an auditable route-disjoint split from Bench2Drive info metadata.

The builder deliberately reads only the lightweight ``folder``, town, and
frame fields in an info file.  It never opens camera data or imports an ORION
model.  Weather and repetition variants of the same physical route are grouped
under one canonical key before any split is made.

The emitted payload remains compatible with
``spatial_training.RouteDisjointManifest``: the additional ``lineage_audit``
field is descriptive and ignored by the existing loader.
"""

import hashlib
import itertools
import json
import math
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

CANONICALIZATION_VERSION = "b2d-physical-route/v1"
LINEAGE_AUDIT_SCHEMA_VERSION = "spatial-uq-route-lineage-audit/v1"
ROUTE_MANIFEST_SCHEMA_VERSION = "spatial-uq-route-manifest/v1"
REQUIRED_SPLITS = ("train", "validation", "calibration", "held_out")

_FOLDER_RE = re.compile(
    r"^(?P<scenario>.+?)_"
    r"(?P<town>Town\d+(?:HD)?)_"
    r"(?P<route>Route\d+)"
    r"(?P<suffix>(?:_.*)?)$",
    re.IGNORECASE,
)
_WEATHER_RE = re.compile(r"(?:^|_)Weather(?P<value>\d+)(?:_|$)", re.IGNORECASE)
_REPETITION_RE = re.compile(
    r"(?:^|_)(?:Repetition|Rep|Repeat|Run|Seed)(?P<value>\d+)(?:_|$)",
    re.IGNORECASE,
)
_BARE_REPETITION_AFTER_WEATHER_RE = re.compile(
    r"(?:^|_)Weather\d+_(?P<value>\d+)(?:_|$)", re.IGNORECASE
)


class B2DManifestError(ValueError):
    """Raised when B2D metadata cannot support an auditable split."""


class ParsedB2DFolder(NamedTuple):
    """Canonical identity parsed from one B2D folder name."""

    folder: str
    basename: str
    canonical_route_key: str
    town: str
    route_token: str
    scenario: str
    weather: Optional[int]
    repetition: Optional[int]


class B2DFrameMetadata(NamedTuple):
    """The only per-frame fields consumed by the split builder."""

    folder: str
    canonical_route_key: str
    town: str
    route_token: str
    scenario: str
    weather: Optional[int]
    repetition: Optional[int]
    frame_idx: int


class B2DRouteGroup(NamedTuple):
    """All weather/repetition folders belonging to one physical route."""

    canonical_route_key: str
    town: str
    route_token: str
    scenarios: Tuple[str, ...]
    folders: Tuple[str, ...]
    weather_variants: Tuple[int, ...]
    repetition_variants: Tuple[int, ...]
    frame_count: int
    frame_min: int
    frame_max: int

    @property
    def stratum_scenario(self) -> str:
        # Scenario disagreement is retained in the audit while the physical
        # route still remains indivisible for leakage prevention.
        return "+".join(self.scenarios)


def normalize_folder(folder: Any) -> str:
    """Normalize a metadata folder without resolving it on the filesystem."""

    value = str(folder).strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    value = value.rstrip("/")
    if not value:
        raise B2DManifestError("folder metadata must be non-empty")
    return str(PurePosixPath(value))


def canonicalize_b2d_folder(folder: Any) -> ParsedB2DFolder:
    """Remove weather/repetition variants from a B2D physical-route key.

    The scenario prefix is intentionally *not* part of the canonical key.
    Bench2Drive route numbers are interpreted only together with their town,
    yielding keys such as ``Town04/Route166``.
    """

    normalized = normalize_folder(folder)
    basename = PurePosixPath(normalized).name
    match = _FOLDER_RE.fullmatch(basename)
    if match is None:
        raise B2DManifestError(
            "cannot parse B2D folder; expected "
            "<Scenario>_<Town>_<Route>[_WeatherN][_RepN], got "
            f"{normalized!r}"
        )

    town_raw = match.group("town")
    route_raw = match.group("route")
    town_suffix = town_raw[4:]
    town = "Town" + (town_suffix[:-2] + "HD" if town_suffix.upper().endswith("HD") else town_suffix)
    route = "Route" + route_raw[5:]
    suffix = match.group("suffix") or ""
    weather_match = _WEATHER_RE.search(suffix)
    repetition_match = _REPETITION_RE.search(suffix)
    if repetition_match is None:
        repetition_match = _BARE_REPETITION_AFTER_WEATHER_RE.search(suffix)

    return ParsedB2DFolder(
        folder=normalized,
        basename=basename,
        canonical_route_key=f"{town}/{route}",
        town=town,
        route_token=route,
        scenario=match.group("scenario"),
        weather=int(weather_match.group("value")) if weather_match else None,
        repetition=(
            int(repetition_match.group("value")) if repetition_match else None
        ),
    )


def _first_present(info: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in info and info[name] is not None:
            return info[name]
    return None


def extract_b2d_frame_metadata(
    infos: Sequence[Mapping[str, Any]],
) -> List[B2DFrameMetadata]:
    """Extract and validate folder/town/frame metadata from B2D infos."""

    if not infos:
        raise B2DManifestError("B2D infos contain no records")

    frames: List[B2DFrameMetadata] = []
    seen: Set[Tuple[str, int]] = set()
    for index, info in enumerate(infos):
        if not isinstance(info, Mapping):
            raise B2DManifestError(f"info record {index} is not a mapping")
        folder_value = _first_present(info, ("folder", "route_folder"))
        if folder_value is None:
            raise B2DManifestError(f"info record {index} has no folder metadata")
        parsed = canonicalize_b2d_folder(folder_value)

        explicit_town = _first_present(info, ("town_name", "town"))
        if explicit_town is not None:
            explicit_town = str(explicit_town).strip()
            if explicit_town and explicit_town.lower() != parsed.town.lower():
                raise B2DManifestError(
                    f"info record {index} town {explicit_town!r} conflicts with "
                    f"folder town {parsed.town!r}"
                )

        frame_value = _first_present(info, ("frame_idx", "frame_id", "frame"))
        if frame_value is None or isinstance(frame_value, bool):
            raise B2DManifestError(f"info record {index} has no integer frame metadata")
        try:
            frame_idx = int(frame_value)
        except (TypeError, ValueError) as exc:
            raise B2DManifestError(
                f"info record {index} has invalid frame metadata {frame_value!r}"
            ) from exc
        if str(frame_value).strip() not in {str(frame_idx), f"{frame_idx}.0"}:
            # Reject silent truncation such as frame 1.5 -> 1.
            try:
                if float(frame_value) != float(frame_idx):
                    raise B2DManifestError(
                        f"info record {index} frame metadata is not integral"
                    )
            except (TypeError, ValueError) as exc:
                raise B2DManifestError(
                    f"info record {index} frame metadata is not integral"
                ) from exc
        if frame_idx < 0:
            raise B2DManifestError(f"info record {index} has negative frame index")
        identity = (parsed.folder, frame_idx)
        if identity in seen:
            raise B2DManifestError(
                f"duplicate folder/frame metadata encountered: {identity!r}"
            )
        seen.add(identity)
        frames.append(
            B2DFrameMetadata(
                folder=parsed.folder,
                canonical_route_key=parsed.canonical_route_key,
                town=parsed.town,
                route_token=parsed.route_token,
                scenario=parsed.scenario,
                weather=parsed.weather,
                repetition=parsed.repetition,
                frame_idx=frame_idx,
            )
        )
    return frames


def group_b2d_routes(frames: Sequence[B2DFrameMetadata]) -> List[B2DRouteGroup]:
    grouped: Dict[str, List[B2DFrameMetadata]] = defaultdict(list)
    for frame in frames:
        grouped[frame.canonical_route_key].append(frame)

    routes: List[B2DRouteGroup] = []
    for key, members in sorted(grouped.items()):
        towns = {member.town for member in members}
        route_tokens = {member.route_token for member in members}
        if len(towns) != 1 or len(route_tokens) != 1:
            raise B2DManifestError(f"canonical route {key!r} has inconsistent identity")
        frame_indices = [member.frame_idx for member in members]
        routes.append(
            B2DRouteGroup(
                canonical_route_key=key,
                town=next(iter(towns)),
                route_token=next(iter(route_tokens)),
                scenarios=tuple(sorted({member.scenario for member in members})),
                folders=tuple(sorted({member.folder for member in members})),
                weather_variants=tuple(
                    sorted({member.weather for member in members if member.weather is not None})
                ),
                repetition_variants=tuple(
                    sorted(
                        {
                            member.repetition
                            for member in members
                            if member.repetition is not None
                        }
                    )
                ),
                frame_count=len(members),
                frame_min=min(frame_indices),
                frame_max=max(frame_indices),
            )
        )
    return routes


def _target_ratios(ratios: Mapping[str, float]) -> Dict[str, float]:
    required = REQUIRED_SPLITS
    if set(ratios) != set(required):
        raise B2DManifestError(
            f"split ratios must contain exactly {list(required)}, got {sorted(ratios)}"
        )
    cleaned = {name: float(ratios[name]) for name in required}
    if any(not math.isfinite(value) or value <= 0 for value in cleaned.values()):
        raise B2DManifestError("all split ratios must be finite and positive")
    total = sum(cleaned.values())
    return {name: value / total for name, value in cleaned.items()}


def _normalize_subset_ratios(ratios: Mapping[str, float]) -> Dict[str, float]:
    """Normalize a non-empty subset of the four public split names."""

    if not ratios or not set(ratios).issubset(REQUIRED_SPLITS):
        raise B2DManifestError("internal split ratios contain unknown or no splits")
    cleaned = {name: float(value) for name, value in ratios.items()}
    if any(not math.isfinite(value) or value <= 0 for value in cleaned.values()):
        raise B2DManifestError("all split ratios must be finite and positive")
    total = sum(cleaned.values())
    return {name: value / total for name, value in cleaned.items()}


def _heldout_town_subset(
    routes: Sequence[B2DRouteGroup],
    heldout_ratio: float,
    min_routes_per_split: int,
) -> Tuple[Tuple[str, ...], str]:
    by_town: Dict[str, List[B2DRouteGroup]] = defaultdict(list)
    for route in routes:
        by_town[route.town].append(route)
    towns = sorted(by_town)
    total_frames = sum(route.frame_count for route in routes)
    total_routes = len(routes)
    candidates: List[Tuple[float, int, Tuple[str, ...]]] = []
    max_towns = min(3, max(0, len(towns) - 1))
    for size in range(1, max_towns + 1):
        for subset in itertools.combinations(towns, size):
            held = [route for town in subset for route in by_town[town]]
            remaining = total_routes - len(held)
            if len(held) < min_routes_per_split:
                continue
            if remaining < 3 * min_routes_per_split:
                continue
            frame_fraction = sum(route.frame_count for route in held) / total_frames
            route_fraction = len(held) / total_routes
            # Town-disjointness is primary; among feasible choices, match both
            # frame mass and route count while preferring fewer held-out towns.
            score = (
                2.0 * abs(frame_fraction - heldout_ratio)
                + abs(route_fraction - heldout_ratio)
                + 0.01 * (size - 1)
            )
            candidates.append((score, size, subset))
    if not candidates:
        return (), "no town subset leaves enough routes for all four splits"
    return min(candidates)[2], "selected whole town subset closest to held-out target"


def _assignment_cost(
    assignment: Mapping[str, Sequence[B2DRouteGroup]],
    ratios: Mapping[str, float],
) -> float:
    names = tuple(ratios)
    all_routes = [route for name in names for route in assignment[name]]
    total_frames = max(1, sum(route.frame_count for route in all_routes))
    total_routes = max(1, len(all_routes))
    scenario_frames: Counter[str] = Counter()
    town_frames: Counter[str] = Counter()
    for route in all_routes:
        scenario_frames[route.stratum_scenario] += route.frame_count
        town_frames[route.town] += route.frame_count

    cost = 0.0
    for name in names:
        members = assignment[name]
        frame_fraction = sum(route.frame_count for route in members) / total_frames
        route_fraction = len(members) / total_routes
        cost += 3.0 * abs(frame_fraction - ratios[name])
        cost += 0.7 * abs(route_fraction - ratios[name])

        per_scenario: Counter[str] = Counter()
        per_town: Counter[str] = Counter()
        for route in members:
            per_scenario[route.stratum_scenario] += route.frame_count
            per_town[route.town] += route.frame_count
        for scenario, count in scenario_frames.items():
            weight = count / total_frames
            cost += 0.8 * weight * abs(per_scenario[scenario] / count - ratios[name])
        for town, count in town_frames.items():
            weight = count / total_frames
            cost += 0.35 * weight * abs(per_town[town] / count - ratios[name])
    return cost


def _stratified_assign(
    routes: Sequence[B2DRouteGroup],
    ratios: Mapping[str, float],
    min_routes_per_split: int,
    seed: int,
    trials: int = 256,
) -> Dict[str, List[B2DRouteGroup]]:
    names = tuple(ratios)
    if len(routes) < len(names) * min_routes_per_split:
        raise B2DManifestError(
            f"{len(routes)} routes cannot provide at least {min_routes_per_split} "
            f"routes to each of {len(names)} splits"
        )
    normalized = _normalize_subset_ratios(ratios)
    total_frames = sum(route.frame_count for route in routes)
    total_routes = len(routes)
    scenario_totals: Counter[str] = Counter()
    town_totals: Counter[str] = Counter()
    for route in routes:
        scenario_totals[route.stratum_scenario] += route.frame_count
        town_totals[route.town] += route.frame_count

    best: Optional[Tuple[float, Tuple[Tuple[str, Tuple[str, ...]], ...], Dict[str, List[B2DRouteGroup]]]] = None
    for trial in range(max(1, trials)):
        rng = random.Random(seed * 1_000_003 + trial)
        ordered = list(routes)
        # Large and rare strata are placed early; controlled jitter explores
        # alternatives without losing determinism.
        ordered.sort(
            key=lambda route: (
                -route.frame_count
                * (0.75 + 0.5 * rng.random())
                / math.sqrt(max(1, scenario_totals[route.stratum_scenario])),
                route.canonical_route_key,
            )
        )
        assignment: Dict[str, List[B2DRouteGroup]] = {name: [] for name in names}
        split_frames: Counter[str] = Counter()
        split_scenarios: Dict[str, Counter[str]] = {
            name: Counter() for name in names
        }
        split_towns: Dict[str, Counter[str]] = {name: Counter() for name in names}

        for index, route in enumerate(ordered):
            remaining_after = len(ordered) - index - 1
            unmet = {
                name: max(0, min_routes_per_split - len(assignment[name]))
                for name in names
            }
            must_fill = [
                name
                for name in names
                if unmet[name] > 0 and sum(unmet.values()) > remaining_after
            ]
            candidates = must_fill or list(names)
            ranked: List[Tuple[float, float, str]] = []
            for name in candidates:
                ratio = normalized[name]
                target_frames = max(1.0, ratio * total_frames)
                target_routes = max(1.0, ratio * total_routes)
                frame_deficit = (target_frames - split_frames[name]) / target_frames
                route_deficit = (target_routes - len(assignment[name])) / target_routes
                scenario_target = max(
                    1.0, ratio * scenario_totals[route.stratum_scenario]
                )
                scenario_deficit = (
                    scenario_target
                    - split_scenarios[name][route.stratum_scenario]
                ) / scenario_target
                town_target = max(1.0, ratio * town_totals[route.town])
                town_deficit = (
                    town_target - split_towns[name][route.town]
                ) / town_target
                score = (
                    3.0 * frame_deficit
                    + 0.7 * route_deficit
                    + 0.8 * scenario_deficit
                    + 0.35 * town_deficit
                )
                ranked.append((score, rng.random(), name))
            chosen = max(ranked)[2]
            assignment[chosen].append(route)
            split_frames[chosen] += route.frame_count
            split_scenarios[chosen][route.stratum_scenario] += route.frame_count
            split_towns[chosen][route.town] += route.frame_count

        if any(len(assignment[name]) < min_routes_per_split for name in names):
            continue
        cost = _assignment_cost(assignment, normalized)
        signature = tuple(
            (name, tuple(sorted(route.canonical_route_key for route in assignment[name])))
            for name in names
        )
        candidate = (cost, signature, assignment)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise B2DManifestError("unable to construct a split satisfying minimum sizes")
    return best[2]


def _split_stats(routes: Sequence[B2DRouteGroup]) -> Dict[str, Any]:
    scenario_frames: Counter[str] = Counter()
    town_frames: Counter[str] = Counter()
    for route in routes:
        scenario_frames[route.stratum_scenario] += route.frame_count
        town_frames[route.town] += route.frame_count
    return {
        "canonical_route_count": len(routes),
        "canonical_route_keys": [
            route.canonical_route_key
            for route in sorted(routes, key=lambda item: item.canonical_route_key)
        ],
        "folder_count": sum(len(route.folders) for route in routes),
        "folder_route_ids": sorted(
            folder for route in routes for folder in route.folders
        ),
        "frame_count": sum(route.frame_count for route in routes),
        "towns": sorted({route.town for route in routes}),
        "scenario_types": sorted(
            {scenario for route in routes for scenario in route.scenarios}
        ),
        "frames_by_town": dict(sorted(town_frames.items())),
        "frames_by_scenario_stratum": dict(sorted(scenario_frames.items())),
    }


def _assert_no_leakage(
    assignment: Mapping[str, Sequence[B2DRouteGroup]],
) -> Dict[str, Any]:
    route_owner: Dict[str, str] = {}
    folder_owner: Dict[str, str] = {}
    route_overlap: List[Dict[str, str]] = []
    folder_overlap: List[Dict[str, str]] = []
    for split, routes in assignment.items():
        for route in routes:
            previous = route_owner.setdefault(route.canonical_route_key, split)
            if previous != split:
                route_overlap.append(
                    {"canonical_route_key": route.canonical_route_key, "a": previous, "b": split}
                )
            for folder in route.folders:
                previous_folder = folder_owner.setdefault(folder, split)
                if previous_folder != split:
                    folder_overlap.append(
                        {"folder": folder, "a": previous_folder, "b": split}
                    )
    if route_overlap or folder_overlap:
        raise B2DManifestError("split construction leaked a route or folder across splits")
    return {
        "canonical_route_overlap": route_overlap,
        "folder_overlap": folder_overlap,
        "weather_repetition_siblings_grouped_before_split": True,
        "passed": True,
    }


def build_b2d_route_manifest(
    infos: Sequence[Mapping[str, Any]],
    *,
    seed: int = 0,
    ratios: Optional[Mapping[str, float]] = None,
    exclude_folders: Sequence[str] = (),
    allow_unmatched_excludes: bool = False,
    minimum_canonical_routes: int = 12,
    min_routes_per_split: int = 2,
    prefer_town_disjoint_heldout: bool = True,
    closed_loop_development_routes: Sequence[str] = (),
    closed_loop_headline_routes: Sequence[str] = (),
    source_lineage: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a compatible manifest payload plus a detailed lineage audit."""

    if minimum_canonical_routes < 4:
        raise B2DManifestError("minimum_canonical_routes must be at least four")
    if min_routes_per_split < 1:
        raise B2DManifestError("min_routes_per_split must be positive")
    normalized_ratios = _target_ratios(
        ratios
        or {"train": 0.7, "validation": 0.1, "calibration": 0.1, "held_out": 0.1}
    )
    frames = extract_b2d_frame_metadata(infos)
    input_groups = group_b2d_routes(frames)

    requested_excludes = tuple(sorted({normalize_folder(value) for value in exclude_folders}))
    input_folders = {frame.folder for frame in frames}
    excluded_keys: Set[str] = set()
    matched_folders: Set[str] = set()
    unmatched: List[str] = []
    for excluded in requested_excludes:
        exact = excluded in input_folders
        try:
            parsed = canonicalize_b2d_folder(excluded)
            canonical_exists = any(
                group.canonical_route_key == parsed.canonical_route_key
                for group in input_groups
            )
        except B2DManifestError:
            parsed = None
            canonical_exists = False
        if exact:
            matched_folders.add(excluded)
        if exact or canonical_exists:
            if parsed is None:
                parsed = canonicalize_b2d_folder(excluded)
            excluded_keys.add(parsed.canonical_route_key)
            matched_folders.update(
                folder
                for group in input_groups
                if group.canonical_route_key == parsed.canonical_route_key
                for folder in group.folders
            )
        else:
            unmatched.append(excluded)
    if unmatched and not allow_unmatched_excludes:
        raise B2DManifestError(
            "exclude entries matched no input folder/canonical route: "
            + ", ".join(unmatched)
        )

    routes = [
        route for route in input_groups if route.canonical_route_key not in excluded_keys
    ]
    if len(routes) < minimum_canonical_routes:
        raise B2DManifestError(
            f"only {len(routes)} canonical routes remain; fail-closed minimum is "
            f"{minimum_canonical_routes}"
        )
    if len(routes) < 4 * min_routes_per_split:
        raise B2DManifestError(
            f"only {len(routes)} routes remain, fewer than four splits x "
            f"{min_routes_per_split} routes"
        )

    heldout_towns: Tuple[str, ...] = ()
    town_reason = "town-disjoint held-out selection disabled"
    if prefer_town_disjoint_heldout:
        heldout_towns, town_reason = _heldout_town_subset(
            routes, normalized_ratios["held_out"], min_routes_per_split
        )

    if heldout_towns:
        heldout = [route for route in routes if route.town in heldout_towns]
        remaining = [route for route in routes if route.town not in heldout_towns]
        remaining_ratio_total = sum(
            normalized_ratios[name] for name in ("train", "validation", "calibration")
        )
        remaining_ratios = {
            name: normalized_ratios[name] / remaining_ratio_total
            for name in ("train", "validation", "calibration")
        }
        assignment = _stratified_assign(
            remaining,
            remaining_ratios,
            min_routes_per_split,
            seed,
        )
        assignment["held_out"] = heldout
    else:
        assignment = _stratified_assign(
            routes,
            normalized_ratios,
            min_routes_per_split,
            seed,
        )

    # Normalize ordering both for audit readability and deterministic output.
    assignment = {
        split: sorted(assignment[split], key=lambda route: route.canonical_route_key)
        for split in REQUIRED_SPLITS
    }
    leakage = _assert_no_leakage(assignment)
    # Existing paired-feature records use the annotation folder as route_id.
    # Expand each already-assigned canonical group here so the persisted
    # manifest works without rewriting those records.
    manifest_splits = {
        split: tuple(
            sorted(folder for route in assignment[split] for folder in route.folders)
        )
        for split in REQUIRED_SPLITS
    }

    all_selected_folders = {
        folder for split in assignment.values() for route in split for folder in route.folders
    }
    if all_selected_folders & matched_folders:
        raise B2DManifestError("excluded folder survived canonical-route exclusion")
    heldout_town_set = set(_split_stats(assignment["held_out"])["towns"])
    nonheld_town_set = {
        route.town
        for name in ("train", "validation", "calibration")
        for route in assignment[name]
    }
    heldout_town_disjoint = not bool(heldout_town_set & nonheld_town_set)

    route_catalog = {
        route.canonical_route_key: {
            "town": route.town,
            "route_token": route.route_token,
            "scenario_types": list(route.scenarios),
            "folders": list(route.folders),
            "weather_variants": list(route.weather_variants),
            "repetition_variants": list(route.repetition_variants),
            "frame_count": route.frame_count,
            "frame_min": route.frame_min,
            "frame_max": route.frame_max,
        }
        for route in routes
    }
    # Keep this pure-stdlib module importable on a login node without PyTorch.
    # The shape below is exercised against RouteDisjointManifest.from_dict in
    # the local test suite.
    payload = {
        "schema_version": ROUTE_MANIFEST_SCHEMA_VERSION,
        "split_unit": "route_id",
        "seed": int(seed),
        "route_disjoint": True,
        "splits": {
            split: {"route_ids": list(manifest_splits[split])}
            for split in REQUIRED_SPLITS
        },
    }
    payload["lineage_audit"] = {
        "schema_version": LINEAGE_AUDIT_SCHEMA_VERSION,
        "source": dict(source_lineage or {"kind": "in_memory_b2d_infos"}),
        "canonicalization": {
            "version": CANONICALIZATION_VERSION,
            "key_format": "<Town>/<RouteN>",
            "manifest_route_ids": "original B2D folder strings",
            "split_assignment_unit": "canonical physical-route group",
            "scenario_in_key": False,
            "weather_in_key": False,
            "repetition_in_key": False,
            "guarantee": (
                "all observed weather/repetition folders for one Town/RouteN key "
                "are assigned as one indivisible group"
            ),
        },
        "input_summary": {
            "raw_frame_records": len(frames),
            "folder_count": len(input_folders),
            "canonical_route_count": len(input_groups),
            "towns": sorted({route.town for route in input_groups}),
            "scenario_types": sorted(
                {scenario for route in input_groups for scenario in route.scenarios}
            ),
        },
        "exclusions": {
            "scope": "entire_canonical_route_for_each_matched_folder",
            "requested": list(requested_excludes),
            "matched_input_folders": sorted(matched_folders),
            "expanded_canonical_route_keys": sorted(excluded_keys),
            "unmatched": unmatched,
            "allow_unmatched": bool(allow_unmatched_excludes),
            "excluded_frame_count": sum(
                route.frame_count
                for route in input_groups
                if route.canonical_route_key in excluded_keys
            ),
        },
        "selection": {
            "seed": seed,
            "requested_ratios": normalized_ratios,
            "minimum_canonical_routes": minimum_canonical_routes,
            "min_routes_per_split": min_routes_per_split,
            "stratification_fields": ["scenario_type", "town", "frame_count"],
            "town_disjoint_heldout_preferred": bool(prefer_town_disjoint_heldout),
            "town_disjoint_heldout_achieved": heldout_town_disjoint,
            "heldout_towns": sorted(heldout_town_set),
            "town_selection_reason": town_reason,
        },
        "split_statistics": {
            split: _split_stats(assignment[split])
            for split in REQUIRED_SPLITS
        },
        "leakage_checks": leakage,
        "closed_loop_route_semantics": {
            "development_route_labels": sorted(
                {str(value) for value in closed_loop_development_routes}
            ),
            "headline_route_labels": sorted(
                {str(value) for value in closed_loop_headline_routes}
            ),
            "used_for_offline_id_matching": False,
            "offline_folder_route_id_equals_closed_loop_xml_or_index_id": False,
            "training_heldout_claim_supported_by_these_labels": False,
            "statement": (
                "Closed-loop route labels are external experiment annotations. "
                "They are not equated with B2D offline Town/RouteN folder keys and "
                "cannot establish training-held-out status. Explicit offline folder "
                "exclusions and upstream pretraining lineage are required."
            ),
        },
        "claim_boundary": {
            "held_out_means": "disjoint within this input info file after declared exclusions",
            "proves_absent_from_orion_pretraining": False,
            "proves_closed_loop_route_is_training_heldout": False,
        },
        "route_catalog": route_catalog,
    }
    return payload


def load_b2d_infos(
    path: Union[Path, str]
) -> Tuple[List[Mapping[str, Any]], Dict[str, Any]]:
    """Load a trusted B2D info list and return source-lineage metadata.

    Pickle loading is necessarily unsafe for untrusted files.  The CLI states
    this explicitly; expected inputs are locally generated Bench2Drive infos.
    """

    path = Path(path)
    if not path.is_file():
        raise B2DManifestError(f"B2D infos file does not exist: {path}")
    raw = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(raw.decode("utf-8"))
        source_format = "json"
    elif suffix in {".pkl", ".pickle"}:
        payload = pickle.loads(raw)
        source_format = "trusted_pickle"
    else:
        raise B2DManifestError("B2D infos must be .pkl, .pickle, or .json")

    if isinstance(payload, Mapping):
        for key in ("infos", "data_infos", "data_list"):
            if isinstance(payload.get(key), (list, tuple)):
                payload = payload[key]
                break
    if not isinstance(payload, (list, tuple)):
        raise B2DManifestError("B2D infos payload must be a list or a dict containing infos")
    infos = list(payload)
    lineage = {
        "kind": "b2d_info_metadata",
        "path": str(path.resolve()),
        "format": source_format,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "image_files_opened": False,
        "model_loaded": False,
    }
    return infos, lineage


def load_exclude_list(path: Union[Path, str]) -> List[str]:
    """Load exclusions from a JSON list or a newline-delimited text file."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, Mapping):
            payload = payload.get("folders")
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise B2DManifestError("JSON exclude list must be a string list or {folders: [...]}")
        return payload
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
