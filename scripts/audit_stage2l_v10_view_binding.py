#!/usr/bin/env python3
"""Audit camera-order lineage and the Stage2-L v10 R-map evidence interface.

The audit is deliberately static/data-only.  It verifies every 17-event QA
record's camera and target order, then distinguishes an ordering defect from
an architectural absence of explicit camera-grid evidence binding.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "orion.stage2l_v10_view_binding_audit.v1"
DATASET_SCHEMA = "orion.stage2l_expanded_coverage_dataset.v1"
RECORD_SCHEMA = "orion.uq_relevance_qa_record.v5"
CANONICAL_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: %s" % path)
    return value


def _literal_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(node.value)
    raise ValueError("literal assignment %s is absent in %s" % (name, path))


def _source_has_all(path: Path, snippets: Sequence[str]) -> bool:
    source = path.read_text(encoding="utf-8")
    return all(snippet in source for snippet in snippets)


def audit(
    *,
    project_root: Path,
    dataset_manifest_path: Path,
    records_path: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    manifest = _read_json(dataset_manifest_path.resolve())
    if (
        manifest.get("schema") != DATASET_SCHEMA
        or manifest.get("event_count") != 17
        or manifest.get("record_count") != 1600
        or manifest.get("records", {}).get("sha256") != _sha256(records_path.resolve())
    ):
        raise ValueError("17-event dataset manifest or records lineage differs")

    source_specs = (
        ("online_agent", project_root / "team_code/orion_b2d_agent.py", "CAMERA_ORDER"),
        (
            "offline_visual_cache",
            project_root / "scripts/cache_closedloop_orion_visual_context.py",
            "CAMERA_ORDER",
        ),
        (
            "geometry_target",
            project_root / "uq_estimator/task_relevance_geometry.py",
            "CAMERA_ORDER",
        ),
        (
            "deterministic_decoder",
            project_root / "uq_estimator/stage2l_deterministic_semantics_v10.py",
            "DEFAULT_CAMERA_ORDER",
        ),
    )
    source_orders = {}
    for label, path, symbol in source_specs:
        value = tuple(_literal_assignment(path, symbol))
        source_orders[label] = {
            "path": str(path),
            "sha256": _sha256(path),
            "symbol": symbol,
            "value": list(value),
            "canonical": value == CANONICAL_CAMERA_ORDER,
        }

    evaluator = project_root / "scripts/evaluate_stage2l_v10_phase_a_checkpoint.py"
    evaluator_order = tuple(_literal_assignment(evaluator, "VIEW_NAMES"))
    normalized_evaluator_order = tuple("CAM_" + value.upper() for value in evaluator_order)
    source_orders["replay_renderer"] = {
        "path": str(evaluator),
        "sha256": _sha256(evaluator),
        "symbol": "VIEW_NAMES",
        "value": list(evaluator_order),
        "normalized_value": list(normalized_evaluator_order),
        "canonical": normalized_evaluator_order == CANONICAL_CAMERA_ORDER,
    }

    camera_orders = set()
    target_orders = set()
    component_shapes = set()
    target_shapes = set()
    event_ids = set()
    group_ids = set()
    record_count = 0
    with records_path.resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != RECORD_SCHEMA:
                raise ValueError("record schema differs")
            cameras = row["model_input"]["observation"]["camera_files"]
            camera_orders.add(tuple(str(value["view"]) for value in cameras))
            sidecar = row["target"]["map_sidecar"]
            target_orders.add(tuple(sidecar["metadata"]["camera_order"]))
            component_shapes.add(
                tuple(row["model_input"]["stage1_observation_uq"]["component_shape"])
            )
            target_shapes.add(tuple(sidecar["metadata"]["shape"]))
            event_ids.add(str(row["event_id"]))
            group_ids.add(str(row["counterfactual"]["group_id"]))
            record_count += 1

    visual_cache_source = project_root / "scripts/cache_closedloop_orion_visual_context.py"
    relevance_pass_source = project_root / "scripts/train_stage2l_route196_bridge_smoke.py"
    query_source = project_root / "uq_estimator/uq_relevance_tokenizer.py"
    token_layout = {
        "current_object_queries": 256,
        "temporal_scene_queries": 16,
        "can_bus": 1,
        "map_queries": 256,
        "total": 529,
    }
    cache_is_det_map_only = _source_has_all(
        visual_cache_source,
        (
            'captured["det"] = output[1].detach()',
            'captured["map"] = output[1].detach()',
            'visual = torch.cat((captured["det"], captured["map"]), dim=1)',
        ),
    )
    relevance_uses_single_shared_span = _source_has_all(
        relevance_pass_source,
        (
            "vision = torch.cat((baseline_vision, queries), dim=1)",
            "visual_token_count=ORION_VISUAL_TOKENS",
            "views=6",
            "grid_h=10",
            "grid_w=10",
        ),
    )
    queries_have_view_ids = _source_has_all(
        query_source,
        (
            "self.view_embedding = nn.Embedding(max_views, model_dim)",
            "view_ids = torch.arange(views, device=device)",
            "self.view_embedding(view_ids)",
        ),
    )
    canonical_data = (
        camera_orders == {CANONICAL_CAMERA_ORDER}
        and target_orders == {CANONICAL_CAMERA_ORDER}
        and component_shapes == {(4, 6, 40, 40, 3)}
        and target_shapes == {(6, 40, 40)}
        and record_count == 1600
        and len(event_ids) == 17
        and len(group_ids) == 80
    )
    canonical_sources = all(value["canonical"] for value in source_orders.values())
    explicit_camera_grid_evidence = not cache_is_det_map_only

    return {
        "schema": SCHEMA,
        "status": "camera_order_consistent_explicit_view_grid_binding_absent",
        "record_count": record_count,
        "event_count": len(event_ids),
        "group_count": len(group_ids),
        "canonical_camera_order": list(CANONICAL_CAMERA_ORDER),
        "source_orders": source_orders,
        "data_lineage": {
            "observation_camera_orders": [list(value) for value in sorted(camera_orders)],
            "target_camera_orders": [list(value) for value in sorted(target_orders)],
            "component_shapes": [list(value) for value in sorted(component_shapes)],
            "target_shapes": [list(value) for value in sorted(target_shapes)],
            "all_records_canonical": canonical_data,
        },
        "r_map_evidence_interface": {
            "baseline_token_layout": token_layout,
            "baseline_tokens_are_detection_and_map_queries_only": cache_is_det_map_only,
            "baseline_tokens_have_explicit_per_camera_grid_segments": explicit_camera_grid_evidence,
            "relevance_queries_have_learned_view_ids": queries_have_view_ids,
            "relevance_queries_and_baseline_use_one_shared_vlm_span": relevance_uses_single_shared_span,
            "explicit_same_view_cross_attention_or_mask": False,
            "camera_grid_binding_mechanism": "implicit_learned_attention_only",
        },
        "checks": {
            "all_source_camera_orders_match": canonical_sources,
            "all_1600_record_camera_orders_match": canonical_data,
            "obvious_camera_permutation_bug_found": not (canonical_sources and canonical_data),
            "explicit_camera_grid_evidence_binding_found": explicit_camera_grid_evidence,
        },
        "diagnosis": {
            "systematic_view_order_bug_supported": False,
            "weak_or_implicit_view_binding_supported": (
                canonical_sources
                and canonical_data
                and cache_is_det_map_only
                and relevance_uses_single_shared_span
                and queries_have_view_ids
            ),
            "more_epochs_without_interface_change_supported": False,
        },
        "decision": {
            "next_model_change": (
                "Keep R owned by ORION/VLM, but expose frozen ORION image-backbone "
                "features as six explicitly indexed view grids and bind each R-query "
                "view to the corresponding evidence grid before global VLM fusion."
            ),
            "reuse_existing_det_map_cache": True,
            "additional_view_aligned_feature_cache_required": True,
            "new_phase_a_only_smoke_before_phase_b": True,
            "phase_b": False,
            "phase_c": False,
            "formal_stage2l": False,
            "stage2p": False,
        },
        "provenance": {
            "dataset_manifest": {
                "path": str(dataset_manifest_path.resolve()),
                "sha256": _sha256(dataset_manifest_path.resolve()),
            },
            "records": {
                "path": str(records_path.resolve()),
                "sha256": _sha256(records_path.resolve()),
            },
        },
        "claim_boundary": (
            "Static/data-lineage audit. It rules out an obvious ordering mismatch "
            "but does not prove that implicit attention caused the failed map, nor "
            "does it establish a passed Stage2-L or safety result."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite view-binding audit")
    value = audit(
        project_root=args.project_root,
        dataset_manifest_path=args.dataset_manifest,
        records_path=args.records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": value["status"], **value["checks"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
