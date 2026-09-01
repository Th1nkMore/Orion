"""Route-balanced clean-manifold candidates for persistent appearance loss.

The candidate never predicts a patch from the query frame.  It compares each
query patch with the multimodal clean feature bank at the same camera and grid
position.  Reference frames first collapse within route, then the closest
distinct routes are averaged so one nearly duplicated trajectory cannot
dominate the score.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from uq_estimator.native_appearance_audit import audit_native_appearance_score_maps
from uq_estimator.observation_uq_v3 import ObservationUQError


NATIVE_MANIFOLD_AUDIT_SCHEMA_VERSION = "orion.native-manifold-audit/v1"
MANIFOLD_CANDIDATE_TAILS = {
    "appearance_route_knn_cosine_z": "positive",
    "appearance_route_knn_standardized_l2_z": "positive",
}


def _route_index(route_ids: Sequence[str]) -> Tuple[torch.Tensor, Tuple[str, ...]]:
    if not route_ids:
        raise ObservationUQError("clean manifold requires route identities")
    routes = tuple(sorted({str(value) for value in route_ids}))
    lookup = {route: index for index, route in enumerate(routes)}
    return torch.tensor([lookup[str(value)] for value in route_ids]), routes


def _validate_features(
    clean: torch.Tensor, queries: torch.Tensor, clean_route_ids: Sequence[str]
) -> None:
    if clean.ndim != 5 or queries.ndim != 5:
        raise ObservationUQError("manifold features must have [N,V,H,W,D] shape")
    if tuple(clean.shape[1:]) != tuple(queries.shape[1:]):
        raise ObservationUQError("clean and query manifold feature shapes disagree")
    if clean.shape[0] != len(clean_route_ids):
        raise ObservationUQError("clean route identities do not match clean features")
    if not clean.is_floating_point() or not queries.is_floating_point():
        raise ObservationUQError("manifold features must be floating point")


def _route_balanced_distance(
    query: torch.Tensor,
    bank: torch.Tensor,
    bank_route_index: torch.Tensor,
    route_count: int,
    nearest_route_count: int,
    metric: str,
    excluded_query_route_index: Optional[torch.Tensor],
) -> torch.Tensor:
    """Return [P,Q] distance for one position chunk.

    ``query`` is [P,Q,D] and ``bank`` is [P,N,D].  All operations stay on the
    requested compute device; route metadata never includes corruption labels.
    """

    if metric == "cosine":
        query_value = F.normalize(query.float(), dim=-1, eps=1e-6)
        bank_value = F.normalize(bank.float(), dim=-1, eps=1e-6)
        sample_distance = 1.0 - torch.bmm(
            query_value, bank_value.transpose(1, 2)
        ).clamp(-1.0, 1.0)
    elif metric == "standardized_l2":
        query_value = query.float()
        bank_value = bank.float()
        query_square = query_value.square().mean(dim=-1, keepdim=True)
        bank_square = bank_value.square().mean(dim=-1).unsqueeze(1)
        cross = torch.bmm(query_value, bank_value.transpose(1, 2)) / float(
            query_value.shape[-1]
        )
        sample_distance = (query_square + bank_square - 2.0 * cross).clamp_min(0.0).sqrt()
    else:  # pragma: no cover - internal caller freezes the metric set
        raise ObservationUQError("unsupported clean manifold metric")

    per_route = []
    bank_route_index = bank_route_index.to(device=sample_distance.device)
    for route in range(route_count):
        member = bank_route_index.eq(route)
        if not bool(member.any()):
            raise ObservationUQError("clean route has no reference features")
        per_route.append(sample_distance[..., member].amin(dim=-1))
    route_distance = torch.stack(per_route, dim=-1)
    if excluded_query_route_index is not None:
        excluded = excluded_query_route_index.to(device=route_distance.device)
        if excluded.shape != (route_distance.shape[1],):
            raise ObservationUQError("excluded query route index has the wrong shape")
        if bool((excluded < -1).any()) or bool((excluded >= route_count).any()):
            raise ObservationUQError("excluded query route index is out of range")
        route_mask = torch.zeros(
            excluded.shape[0], route_count, dtype=torch.bool, device=route_distance.device
        )
        valid_exclusion = excluded.ge(0)
        if bool(valid_exclusion.any()):
            route_mask[valid_exclusion] = F.one_hot(
                excluded[valid_exclusion], num_classes=route_count
            ).bool()
        route_distance = route_distance.masked_fill(route_mask[None], float("inf"))
        available_routes = route_count - int(bool(valid_exclusion.any()))
    else:
        available_routes = route_count
    if nearest_route_count <= 0 or nearest_route_count > available_routes:
        raise ObservationUQError("nearest route count exceeds available clean routes")
    return route_distance.topk(
        nearest_route_count, dim=-1, largest=False, sorted=False
    ).values.mean(dim=-1)


def route_balanced_manifold_raw_maps(
    clean: torch.Tensor,
    queries: torch.Tensor,
    clean_route_ids: Sequence[str],
    nearest_route_count: int = 5,
    position_chunk_size: int = 16,
    query_route_ids: Optional[Sequence[str]] = None,
    leave_query_route_out: bool = False,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Compute both frozen clean-manifold candidates.

    When calibrating on the clean bank itself, ``leave_query_route_out`` must
    be true.  Native queries have no matching reference route and use the full
    bank.  Output tensors are CPU [Q,V,H,W] maps.
    """

    _validate_features(clean, queries, clean_route_ids)
    if position_chunk_size <= 0:
        raise ObservationUQError("position chunk size must be positive")
    clean_route_index, route_names = _route_index(clean_route_ids)
    excluded = None
    if leave_query_route_out:
        if query_route_ids is None or len(query_route_ids) != queries.shape[0]:
            raise ObservationUQError("leave-route-out queries require route identities")
        route_lookup = {route: index for index, route in enumerate(route_names)}
        try:
            excluded = torch.tensor(
                [
                    -1 if value is None else route_lookup[str(value)]
                    for value in query_route_ids
                ]
            )
        except KeyError as error:
            raise ObservationUQError("query route is absent from clean bank") from error
    elif query_route_ids is not None:
        raise ObservationUQError("query route ids are only valid for leave-route-out scoring")

    target = torch.device(device)
    query_count, views, height, width, _ = queries.shape
    position_count = height * width
    outputs = {
        name: torch.empty(query_count, views, height, width, dtype=torch.float32)
        for name in MANIFOLD_CANDIDATE_TAILS
    }
    for view in range(views):
        bank_view = clean[:, view].permute(1, 2, 0, 3).reshape(
            position_count, clean.shape[0], clean.shape[-1]
        )
        query_view = queries[:, view].permute(1, 2, 0, 3).reshape(
            position_count, query_count, queries.shape[-1]
        )
        bank_float = bank_view.float()
        center = bank_float.mean(dim=1, keepdim=True)
        scale = bank_float.std(dim=1, keepdim=True, unbiased=False)
        del bank_float
        channel_floor = torch.quantile(
            scale.reshape(position_count, -1), 0.10, dim=-1
        ).clamp_min(1e-3)
        scale = torch.maximum(scale, channel_floor[:, None, None])
        for start in range(0, position_count, position_chunk_size):
            stop = min(start + position_chunk_size, position_count)
            bank_chunk = bank_view[start:stop].to(target)
            query_chunk = query_view[start:stop].to(target)
            cosine = _route_balanced_distance(
                query_chunk,
                bank_chunk,
                clean_route_index,
                len(route_names),
                nearest_route_count,
                "cosine",
                excluded,
            )
            standardized_bank = (
                bank_chunk.float() - center[start:stop].to(target)
            ) / scale[start:stop].to(target)
            standardized_query = (
                query_chunk.float() - center[start:stop].to(target)
            ) / scale[start:stop].to(target)
            standardized_l2 = _route_balanced_distance(
                standardized_query,
                standardized_bank,
                clean_route_index,
                len(route_names),
                nearest_route_count,
                "standardized_l2",
                excluded,
            )
            outputs["appearance_route_knn_cosine_z"][:, view].view(
                query_count, position_count
            )[:, start:stop] = cosine.transpose(0, 1).cpu()
            outputs["appearance_route_knn_standardized_l2_z"][:, view].view(
                query_count, position_count
            )[:, start:stop] = standardized_l2.transpose(0, 1).cpu()
    return outputs


def audit_native_manifold_score_maps(
    payload: Mapping[str, Any],
    scores_by_candidate: Mapping[str, Mapping[str, torch.Tensor]],
) -> Dict[str, Any]:
    if set(scores_by_candidate) != set(MANIFOLD_CANDIDATE_TAILS):
        raise ObservationUQError("native manifold candidate set changed")
    report = audit_native_appearance_score_maps(
        payload,
        scores_by_candidate,
        candidate_tails=MANIFOLD_CANDIDATE_TAILS,
    )
    report["schema_version"] = NATIVE_MANIFOLD_AUDIT_SCHEMA_VERSION
    report["data_attestation"].update(
        {
            "query_current_neighbourhood_used_as_prediction_context": False,
            "query_previous_frame_used": False,
            "clean_route_balanced_reference_only": True,
        }
    )
    return report


__all__ = [
    "MANIFOLD_CANDIDATE_TAILS",
    "NATIVE_MANIFOLD_AUDIT_SCHEMA_VERSION",
    "audit_native_manifold_score_maps",
    "route_balanced_manifold_raw_maps",
]
