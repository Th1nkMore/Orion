"""Auditable helpers for paired clean/corrupt EVAViT feature extraction.

This module deliberately contains no MMCV or ORION imports.  It defines the
shape-preserving conversion and metadata checks used by the standalone
extractor, and can therefore be validated on CPU.  The extracted target is
*only* paired frozen-representation error; it is not an actual perception
failure target and is not semantic uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from uq_estimator.spatial_training import (
    TARGET_REPRESENTATION_PROXY,
    PairedSpatialFeatureRecord,
)


PAIRED_EXTRACTION_SCHEMA_VERSION = "orion.spatial-paired-extraction/v1"


class PairedFeatureExtractionError(ValueError):
    """Raised when source identity or tensor shapes are not auditable."""


def select_contiguous_route_balanced_infos(
    infos: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    split_route_quotas: Mapping[str, int],
    samples_per_route: int,
) -> list[Mapping[str, Any]]:
    """Select deterministic contiguous frames across manifest route splits.

    The selector is metadata-only so route balance and temporal adjacency can
    be verified on CPU before a GPU job starts.
    """

    if manifest.get("schema_version") != "spatial-uq-route-manifest/v1":
        raise PairedFeatureExtractionError("unsupported route manifest schema")
    if samples_per_route <= 0:
        raise PairedFeatureExtractionError("samples_per_route must be positive")
    if not split_route_quotas or any(
        int(value) <= 0 for value in split_route_quotas.values()
    ):
        raise PairedFeatureExtractionError("split route quotas must be positive")
    infos_by_route: dict[str, list[Mapping[str, Any]]] = {}
    for info in infos:
        route_id = str(info.get("folder", info.get("route_folder", ""))).strip()
        frame = info.get("frame_idx", info.get("frame_index"))
        if not route_id or frame is None:
            raise PairedFeatureExtractionError(
                "balanced selection requires folder and frame_idx metadata"
            )
        infos_by_route.setdefault(route_id, []).append(info)

    selected: list[Mapping[str, Any]] = []
    selected_routes: set[str] = set()
    for split, route_quota in split_route_quotas.items():
        split_payload = manifest.get("splits", {}).get(split)
        if not isinstance(split_payload, Mapping):
            raise PairedFeatureExtractionError(
                "route manifest has no requested split %s" % split
            )
        candidates = [
            str(route_id)
            for route_id in split_payload.get("route_ids", [])
            if str(route_id) in infos_by_route
        ]
        viable = []
        for route_id in candidates:
            route_infos = sorted(
                infos_by_route[route_id],
                key=lambda item: int(
                    item.get("frame_idx", item.get("frame_index"))
                ),
            )
            run: list[Mapping[str, Any]] = []
            previous_frame = None
            for info in route_infos:
                frame = int(info.get("frame_idx", info.get("frame_index")))
                if previous_frame is None or frame == previous_frame + 1:
                    run.append(info)
                else:
                    run = [info]
                previous_frame = frame
                if len(run) == samples_per_route:
                    viable.append((route_id, list(run)))
                    break
        if len(viable) < int(route_quota):
            raise PairedFeatureExtractionError(
                "split %s has %d viable routes, fewer than quota %d"
                % (split, len(viable), int(route_quota))
            )
        for route_id, route_infos in viable[: int(route_quota)]:
            if route_id in selected_routes:
                raise PairedFeatureExtractionError(
                    "route selected across multiple splits"
                )
            selected_routes.add(route_id)
            selected.extend(route_infos)
    return selected


@dataclass(frozen=True)
class RouteFrameIdentity:
    """Route/frame identity resolved from annotations and collected tokens."""

    route_id: str
    town: str
    frame_idx: int
    sample_token: str


def _required_text(mapping: Mapping[str, Any], key: str, source: str) -> str:
    value = mapping.get(key)
    if value is None or not str(value).strip():
        raise PairedFeatureExtractionError(
            f"{source} is missing a reliable non-empty {key!r}"
        )
    return str(value).strip()


def _required_frame_idx(mapping: Mapping[str, Any], source: str) -> int:
    value = mapping.get("frame_idx")
    if isinstance(value, bool) or value is None:
        raise PairedFeatureExtractionError(
            f"{source} is missing a reliable integer 'frame_idx'"
        )
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise PairedFeatureExtractionError(
            f"{source} has an invalid frame_idx {value!r}"
        ) from error
    if str(value).strip() != str(result) and not isinstance(value, int):
        # Refuse lossy conversions such as 1.5 -> 1 while allowing numpy ints.
        try:
            if float(value) != float(result):
                raise PairedFeatureExtractionError(
                    f"{source} has a non-integral frame_idx {value!r}"
                )
        except (TypeError, ValueError) as error:
            raise PairedFeatureExtractionError(
                f"{source} has an invalid frame_idx {value!r}"
            ) from error
    if result < 0:
        raise PairedFeatureExtractionError(f"{source} frame_idx must be non-negative")
    return result


def resolve_route_frame_identity(
    info: Mapping[str, Any],
    image_meta: Mapping[str, Any],
) -> RouteFrameIdentity:
    """Resolve identity without falling back to filenames or batch offsets.

    Bench2Drive annotations identify a route with ``folder``.  ORION's
    collection pipeline independently exposes that value as ``scene_token``
    (and normally ``folder`` as well).  Both the route token and frame index
    must agree.  This fail-closed rule prevents route-disjoint splits from
    silently becoming frame-random splits when pipeline metadata changes.
    """

    if not isinstance(info, Mapping) or not isinstance(image_meta, Mapping):
        raise PairedFeatureExtractionError("info and image_meta must be mappings")
    info_route = _required_text(info, "folder", "data_infos record")
    meta_route = _required_text(image_meta, "scene_token", "img_metas record")
    if info_route != meta_route:
        raise PairedFeatureExtractionError(
            f"route mismatch: data_infos={info_route!r}, img_metas={meta_route!r}"
        )
    if "folder" in image_meta and str(image_meta["folder"]).strip() != info_route:
        raise PairedFeatureExtractionError(
            "img_metas folder disagrees with its scene_token/data_infos folder"
        )

    info_frame = _required_frame_idx(info, "data_infos record")
    meta_frame = _required_frame_idx(image_meta, "img_metas record")
    if info_frame != meta_frame:
        raise PairedFeatureExtractionError(
            f"frame mismatch: data_infos={info_frame}, img_metas={meta_frame}"
        )
    town = _required_text(info, "town_name", "data_infos record")
    return RouteFrameIdentity(
        route_id=info_route,
        town=town,
        frame_idx=info_frame,
        sample_token=f"{info_route}__frame_{info_frame:06d}",
    )


def build_info_identity_index(
    data_infos: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    """Index annotations by the same route/frame keys collected by ORION."""

    index: dict[tuple[str, int], Mapping[str, Any]] = {}
    for info in data_infos:
        if not isinstance(info, Mapping):
            raise PairedFeatureExtractionError("every data_infos item must be a mapping")
        route = _required_text(info, "folder", "data_infos record")
        frame = _required_frame_idx(info, "data_infos record")
        _required_text(info, "town_name", "data_infos record")
        key = (route, frame)
        if key in index:
            raise PairedFeatureExtractionError(
                f"duplicate data_infos route/frame identity: {key!r}"
            )
        index[key] = info
    if not index:
        raise PairedFeatureExtractionError("data_infos must not be empty")
    return index


def camera_view_names_from_info(info: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the camera order used by ``B2DOrionDataset.get_data_info``.

    The dataset iterates ``info['sensors']`` in insertion order and retains
    keys containing ``CAM``.  Recording those names avoids silently calling
    numeric view 0 "front" when an annotation file uses a different order.
    """

    sensors = info.get("sensors")
    if not isinstance(sensors, Mapping):
        raise PairedFeatureExtractionError(
            "data_infos record is missing an ordered sensors mapping"
        )
    names = tuple(str(name) for name in sensors if "CAM" in str(name))
    if not names:
        raise PairedFeatureExtractionError("data_infos record contains no camera sensors")
    return names


def find_info_for_image_meta(
    info_index: Mapping[tuple[str, int], Mapping[str, Any]],
    image_meta: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Find the exact annotation record for one collected image metadata item."""

    route = _required_text(image_meta, "scene_token", "img_metas record")
    frame = _required_frame_idx(image_meta, "img_metas record")
    key = (route, frame)
    if key not in info_index:
        raise PairedFeatureExtractionError(
            f"img_metas route/frame identity is absent from data_infos: {key!r}"
        )
    return info_index[key]


def feature_map_to_patch_tokens(
    feature_map: torch.Tensor,
    batch_size: int,
    views: int,
) -> torch.Tensor:
    """Convert ``[B*V,D,Hp,Wp]`` to ``[B,V,Hp*Wp,D]`` without pooling."""

    if feature_map.ndim != 4:
        raise PairedFeatureExtractionError(
            "backbone feature map must have shape [B*V,D,Hp,Wp]"
        )
    if batch_size <= 0 or views <= 0:
        raise PairedFeatureExtractionError("batch_size and views must be positive")
    if feature_map.shape[0] != batch_size * views:
        raise PairedFeatureExtractionError(
            "backbone feature batch does not equal batch_size * views"
        )
    _, feature_dim, patch_height, patch_width = feature_map.shape
    if min(feature_dim, patch_height, patch_width) <= 0:
        raise PairedFeatureExtractionError("backbone feature dimensions must be non-empty")
    return (
        feature_map.reshape(
            batch_size, views, feature_dim, patch_height, patch_width
        )
        .permute(0, 1, 3, 4, 2)
        .reshape(batch_size, views, patch_height * patch_width, feature_dim)
    )


def exact_mask_to_patch_coverage(
    pixel_mask: torch.Tensor,
    patch_height: int,
    patch_width: int,
) -> torch.Tensor:
    """Project the exact pixel mask to fractional patch coverage.

    Non-overlapping area pooling computes the affected-pixel fraction per
    backbone patch.  The result is ``[B,V,Hp*Wp]`` in ``[0,1]`` and retains
    partial edge patches instead of rounding a corruption rectangle to a
    binary patch box.
    """

    if pixel_mask.ndim != 5 or pixel_mask.shape[2] != 1:
        raise PairedFeatureExtractionError(
            "pixel_mask must have shape [B,V,1,H,W]"
        )
    if pixel_mask.dtype != torch.bool:
        raise PairedFeatureExtractionError("pixel_mask must be the exact boolean mask")
    if patch_height <= 0 or patch_width <= 0:
        raise PairedFeatureExtractionError("patch dimensions must be positive")
    batch, views, _, height, width = pixel_mask.shape
    if height <= 0 or width <= 0:
        raise PairedFeatureExtractionError("pixel mask must be spatially non-empty")
    if height % patch_height or width % patch_width:
        raise PairedFeatureExtractionError(
            "pixel dimensions must divide evenly into the EVAViT patch grid"
        )
    kernel = (height // patch_height, width // patch_width)
    coverage = F.avg_pool2d(
        pixel_mask.reshape(batch * views, 1, height, width).float(),
        kernel_size=kernel,
        stride=kernel,
    )
    return coverage.reshape(batch, views, patch_height * patch_width)


def make_representation_proxy_record(
    *,
    identity: RouteFrameIdentity,
    corruption: str,
    severity: int,
    clean_patch_features: torch.Tensor,
    corrupt_patch_features: torch.Tensor,
    patch_corruption_coverage: torch.Tensor,
    corruption_metadata: Mapping[str, Any],
    backbone_metadata: Mapping[str, Any],
) -> PairedSpatialFeatureRecord:
    """Build a v2 proxy record with no invented failure-event label."""

    if severity not in (1, 2, 3):
        raise PairedFeatureExtractionError("severity must be 1, 2, or 3")
    if clean_patch_features.ndim != 3:
        raise PairedFeatureExtractionError(
            "per-sample patch features must have shape [V,P,D]"
        )
    if corrupt_patch_features.shape != clean_patch_features.shape:
        raise PairedFeatureExtractionError(
            "clean and corrupt patch features must have identical shapes"
        )
    if patch_corruption_coverage.shape != clean_patch_features.shape[:-1]:
        raise PairedFeatureExtractionError(
            "patch corruption coverage must have shape [V,P]"
        )
    if patch_corruption_coverage.dtype == torch.bool:
        patch_corruption_coverage = patch_corruption_coverage.float()
    if not patch_corruption_coverage.is_floating_point():
        raise PairedFeatureExtractionError(
            "patch corruption coverage must be floating point"
        )
    if torch.any(patch_corruption_coverage < 0) or torch.any(
        patch_corruption_coverage > 1
    ):
        raise PairedFeatureExtractionError("patch coverage must lie in [0,1]")

    pair_id = f"{identity.sample_token}/{corruption}"
    metadata = {
        "extraction_schema_version": PAIRED_EXTRACTION_SCHEMA_VERSION,
        "source_identity": {
            "route_id": identity.route_id,
            "town": identity.town,
            "frame_idx": identity.frame_idx,
            "sample_token": identity.sample_token,
            "resolved_from": (
                "data_infos.folder+frame_idx verified_against_"
                "img_metas.scene_token+frame_idx"
            ),
            "filename_fallback_used": False,
        },
        "backbone": dict(backbone_metadata),
        "corruption": dict(corruption_metadata),
        "mask_projection": {
            "source": "exact_boolean_pixel_mask",
            "method": "nonoverlapping_patch_area_fraction",
            "partial_patch_coverage_preserved": True,
        },
        "target_contract": {
            "target_provenance": TARGET_REPRESENTATION_PROXY,
            "actual_perception_failure": False,
            "semantic_uncertainty": False,
            "supports_closed_loop_safety_claim": False,
            "supports_llm_understanding_claim": False,
        },
    }
    return PairedSpatialFeatureRecord(
        record_id=f"{pair_id}/severity_{severity}",
        pair_id=pair_id,
        route_id=identity.route_id,
        town=identity.town,
        severity=float(severity),
        observed_patch_features=corrupt_patch_features.detach().cpu().float(),
        clean_patch_features=clean_patch_features.detach().cpu().float(),
        error_severity_target=None,
        failure_event_target=None,
        target_valid_mask=None,
        clean_error_severity_target=None,
        clean_failure_event_target=None,
        clean_target_valid_mask=None,
        corruption_mask=patch_corruption_coverage.detach().cpu().float(),
        ensemble_teacher_variance=None,
        metadata=metadata,
    )
