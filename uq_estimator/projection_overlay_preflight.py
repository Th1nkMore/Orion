"""CPU-only Route214 six-view projection-overlay preflight.

The module consumes frames produced by the dedicated actual-target data
pipeline and writes auditable JSON/PNG artifacts.  It never constructs ORION,
uses CUDA, submits Slurm, starts CARLA, or marks visual G1 review as passed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from uq_estimator.projected_visible_support import (
    ORION_CAMERA_ORDER,
    make_projection_overlay_data,
    project_boxes_to_visible_patch_support,
    render_projection_overlay_image,
)


OVERLAY_PREFLIGHT_SCHEMA_VERSION = "orion.route214-projection-overlay-preflight/v1"
OVERLAY_FRAME_SCHEMA_VERSION = "orion.route214-projection-overlay-frame/v1"
ROUTE_KEY = "Town04/Route214"
ROUTE_FOLDER = "v1/OppositeVehicleTakingPriority_Town04_Route214_Weather6"
DEFAULT_FRAMES = (0, 39)
DEFAULT_CANDIDATE_FRAME = 39
GT_BOX_Z_ORIGIN = "bottom"
PATCH_HW = (40, 40)


class ProjectionOverlayPreflightError(RuntimeError):
    """Raised when an overlay artifact would require guessing alignment."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_cpu_tensor(
    value: torch.Tensor,
    name: str,
    *,
    ndim: int,
    floating: Optional[bool] = None,
) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ProjectionOverlayPreflightError(
            "%s must be a %d-dimensional tensor" % (name, ndim)
        )
    if value.device.type != "cpu":
        raise ProjectionOverlayPreflightError("%s must be on CPU" % name)
    if floating is True and not value.is_floating_point():
        raise ProjectionOverlayPreflightError("%s must be floating point" % name)
    if floating is False and (
        value.dtype == torch.bool or value.is_floating_point() or value.is_complex()
    ):
        raise ProjectionOverlayPreflightError("%s must use an integer dtype" % name)
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ProjectionOverlayPreflightError("%s must be finite" % name)


@dataclass(frozen=True)
class Route214ProjectionFrameV1:
    """One real or mock dedicated-pipeline frame ready for CPU projection."""

    frame_idx: int
    processed_rgb: torch.Tensor
    post_augmentation_lidar2img: torch.Tensor
    gt_boxes_lidar: torch.Tensor
    gt_classes: torch.Tensor
    gt_actor_ids: torch.Tensor
    source_image_paths: tuple[str, ...]
    image_transform_id: str
    pipeline_audit: Mapping[str, Any]
    route_key: str = ROUTE_KEY
    folder: str = ROUTE_FOLDER
    camera_order: tuple[str, ...] = ORION_CAMERA_ORDER
    box_z_origin: str = GT_BOX_Z_ORIGIN
    schema_version: str = OVERLAY_FRAME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OVERLAY_FRAME_SCHEMA_VERSION:
            raise ProjectionOverlayPreflightError("unsupported frame schema")
        if self.route_key != ROUTE_KEY or self.folder != ROUTE_FOLDER:
            raise ProjectionOverlayPreflightError("frame is not the frozen Route214 folder")
        if isinstance(self.frame_idx, bool) or not isinstance(self.frame_idx, int) or self.frame_idx < 0:
            raise ProjectionOverlayPreflightError("frame_idx must be non-negative int")
        if tuple(self.camera_order) != ORION_CAMERA_ORDER:
            raise ProjectionOverlayPreflightError("camera order is not canonical ORION order")
        if self.box_z_origin != GT_BOX_Z_ORIGIN:
            raise ProjectionOverlayPreflightError("Route214 GT boxes must use bottom z-origin")
        if not str(self.image_transform_id).strip():
            raise ProjectionOverlayPreflightError("image_transform_id must be non-empty")
        _require_cpu_tensor(self.processed_rgb, "processed_rgb", ndim=4)
        if self.processed_rgb.dtype != torch.uint8 or self.processed_rgb.shape[0] != 6 or self.processed_rgb.shape[-1] != 3:
            raise ProjectionOverlayPreflightError(
                "processed_rgb must be CPU uint8 [6,H,W,3]"
            )
        if self.processed_rgb.shape[1] <= 0 or self.processed_rgb.shape[2] <= 0:
            raise ProjectionOverlayPreflightError("processed image shape must be positive")
        _require_cpu_tensor(
            self.post_augmentation_lidar2img,
            "post_augmentation_lidar2img",
            ndim=3,
            floating=True,
        )
        if tuple(self.post_augmentation_lidar2img.shape) != (6, 4, 4):
            raise ProjectionOverlayPreflightError(
                "post_augmentation_lidar2img must be [6,4,4]"
            )
        _require_cpu_tensor(self.gt_boxes_lidar, "gt_boxes_lidar", ndim=2, floating=True)
        if self.gt_boxes_lidar.shape[1] < 7:
            raise ProjectionOverlayPreflightError("gt_boxes_lidar must be [G,D>=7]")
        count = self.gt_boxes_lidar.shape[0]
        _require_cpu_tensor(self.gt_classes, "gt_classes", ndim=1, floating=False)
        _require_cpu_tensor(self.gt_actor_ids, "gt_actor_ids", ndim=1, floating=False)
        if self.gt_classes.shape != (count,) or self.gt_actor_ids.shape != (count,):
            raise ProjectionOverlayPreflightError("GT boxes/classes/actor IDs are misaligned")
        if self.gt_actor_ids.unique().numel() != count:
            raise ProjectionOverlayPreflightError("GT actor IDs must be unique")
        if len(self.source_image_paths) != 6 or any(
            not str(path).strip() for path in self.source_image_paths
        ):
            raise ProjectionOverlayPreflightError("six source image paths are required")
        required_audit = (
            "dedicated_target_pipeline",
            "with_light_state",
            "with_actor_ids",
            "post_augmentation_geometry",
            "processed_rgb_from_normalized_tensor",
        )
        if not isinstance(self.pipeline_audit, Mapping) or any(
            self.pipeline_audit.get(key) is not True for key in required_audit
        ):
            raise ProjectionOverlayPreflightError(
                "pipeline audit does not prove the dedicated post-augmentation path"
            )

    @property
    def processed_image_hw(self) -> tuple[int, int]:
        return int(self.processed_rgb.shape[1]), int(self.processed_rgb.shape[2])


def _save_rgb_png(path: Path, image: torch.Tensor) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ProjectionOverlayPreflightError("Pillow is required for overlay PNGs") from exc
    Image.fromarray(image.numpy(), mode="RGB").save(path)


def _render_contact_sheet(paths: Sequence[Path], output: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ProjectionOverlayPreflightError("Pillow is required for contact sheet") from exc
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    sheet = Image.new("RGB", (3 * width, 2 * (height + 24)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for index, (camera, image) in enumerate(zip(ORION_CAMERA_ORDER, images)):
        row, column = divmod(index, 3)
        x, y = column * width, row * (height + 24)
        sheet.paste(image, (x, y + 24))
        draw.text((x + 6, y + 5), camera, fill=(255, 255, 255))
    sheet.save(output)
    for image in images:
        image.close()


def generate_route214_projection_overlays(
    frames: Sequence[Route214ProjectionFrameV1],
    output_dir: Path,
    *,
    candidate_frame: int = DEFAULT_CANDIDATE_FRAME,
) -> dict[str, Any]:
    """Generate per-camera overlays and a fail-closed audit manifest."""

    if not frames:
        raise ProjectionOverlayPreflightError("at least one frame is required")
    by_index = {frame.frame_idx: frame for frame in frames}
    if len(by_index) != len(frames):
        raise ProjectionOverlayPreflightError("duplicate frame_idx")
    if 0 not in by_index or candidate_frame not in by_index:
        raise ProjectionOverlayPreflightError(
            "frame 0 and candidate frame %d are both required" % candidate_frame
        )
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ProjectionOverlayPreflightError(
            "output directory already exists and is non-empty; refusing overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_audits = []
    for frame_idx in sorted(by_index):
        frame = by_index[frame_idx]
        image_h, image_w = frame.processed_image_hw
        projected = project_boxes_to_visible_patch_support(
            frame.gt_boxes_lidar.float(),
            frame.post_augmentation_lidar2img.float(),
            [(image_h, image_w)] * 6,
            camera_order=frame.camera_order,
            matrix_camera_order=frame.camera_order,
            image_shape_camera_order=frame.camera_order,
            image_transform_id=frame.image_transform_id,
            box_z_origin=frame.box_z_origin,
            patch_hw=PATCH_HW,
            expected_camera_order=ORION_CAMERA_ORDER,
            expected_patch_hw=PATCH_HW,
        )
        frame_dir = output_dir / ("frame_%06d" % frame_idx)
        frame_dir.mkdir(parents=True, exist_ok=False)
        view_audits = []
        overlay_paths = []
        for view_index, camera in enumerate(ORION_CAMERA_ORDER):
            overlay_data = make_projection_overlay_data(projected, view_index)
            for object_row in overlay_data["objects"]:
                object_index = object_row["object_index"]
                object_row["gt_class"] = int(frame.gt_classes[object_index].item())
                object_row["gt_actor_id"] = int(frame.gt_actor_ids[object_index].item())
            overlay_json = frame_dir / (camera + ".overlay.json")
            overlay_json.write_text(
                json.dumps(overlay_data, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            rendered = render_projection_overlay_image(
                projected, view_index, image=frame.processed_rgb[view_index]
            )
            overlay_png = frame_dir / (camera + ".overlay.png")
            _save_rgb_png(overlay_png, rendered)
            overlay_paths.append(overlay_png)
            support = projected.support[view_index]
            nonzero = int((support > 0).sum().item())
            visible_count = int(projected.object_visible_mask[view_index].sum().item())
            view_audits.append(
                {
                    "camera_index": view_index,
                    "camera_name": camera,
                    "source_image_path": frame.source_image_paths[view_index],
                    "visible_gt_object_count": visible_count,
                    "nonzero_object_patch_support_count": nonzero,
                    "maximum_fractional_support": float(support.max().item()) if support.numel() else 0.0,
                    "overlay_png": str(overlay_png),
                    "overlay_png_sha256": _sha256_file(overlay_png),
                    "overlay_json": str(overlay_json),
                    "overlay_json_sha256": _sha256_file(overlay_json),
                }
            )
        contact_sheet = frame_dir / "six_view_contact_sheet.png"
        _render_contact_sheet(overlay_paths, contact_sheet)
        visible_any = projected.object_visible_mask.any(dim=0)
        frame_audit = {
            "schema_version": OVERLAY_FRAME_SCHEMA_VERSION,
            "route_key": frame.route_key,
            "folder": frame.folder,
            "frame_idx": frame_idx,
            "candidate_frame": frame_idx == candidate_frame,
            "camera_order": list(frame.camera_order),
            "processed_image_hw": [image_h, image_w],
            "post_augmentation_lidar2img": frame.post_augmentation_lidar2img.tolist(),
            "image_transform_id": frame.image_transform_id,
            "box_z_origin": frame.box_z_origin,
            "patch_hw": list(PATCH_HW),
            "gt_object_count": int(frame.gt_boxes_lidar.shape[0]),
            "gt_boxes_lidar": frame.gt_boxes_lidar.tolist(),
            "gt_classes": frame.gt_classes.tolist(),
            "gt_actor_ids": frame.gt_actor_ids.tolist(),
            "objects_visible_in_any_camera": int(visible_any.sum().item()),
            "objects_invisible_in_all_cameras": torch.nonzero(
                ~visible_any, as_tuple=False
            ).flatten().tolist(),
            "nonempty_projected_support": bool(projected.support.count_nonzero()),
            "support_shape": list(projected.support.shape),
            "valid_patch_shape": list(projected.valid_patch_mask.shape),
            "support_provenance": {
                "schema_version": projected.projection_provenance.schema_version,
                "projection_matrix_kind": projected.projection_provenance.projection_matrix_kind,
                "silhouette_method": projected.projection_provenance.silhouette_method,
                "patch_pooling_method": projected.projection_provenance.patch_pooling_method,
                "attribution": projected.projection_provenance.attribution,
                "attribution_is_causal": False,
                "refinement_applied": projected.projection_provenance.refinement_applied,
            },
            "pipeline_audit": dict(frame.pipeline_audit),
            "views": view_audits,
            "six_view_contact_sheet": str(contact_sheet),
            "six_view_contact_sheet_sha256": _sha256_file(contact_sheet),
            "automated_checks": {
                "six_overlay_pngs_written": len(overlay_paths) == 6,
                "six_overlay_jsons_written": len(view_audits) == 6,
                "canonical_camera_order": frame.camera_order == ORION_CAMERA_ORDER,
                "post_augmentation_projection_declared": True,
                "bottom_origin_gt_declared": frame.box_z_origin == "bottom",
                "forty_by_forty_patch_alignment": tuple(projected.support_provenance.patch_hw) == PATCH_HW,
                "object_axis_aligned": (
                    frame.gt_boxes_lidar.shape[0]
                    == frame.gt_classes.shape[0]
                    == frame.gt_actor_ids.shape[0]
                ),
            },
            "human_visual_review": {
                "performed": False,
                "passed": False,
                "g1_projection_overlay_gate_passed": False,
                "reason": "PNG generation is not evidence of visual alignment; human review is pending.",
            },
        }
        frame_json = frame_dir / "frame_audit.json"
        frame_json.write_text(
            json.dumps(frame_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        frame_audit["frame_audit_json"] = str(frame_json)
        frame_audit["frame_audit_json_sha256"] = _sha256_file(frame_json)
        frame_audits.append(frame_audit)

    candidate_audit = next(row for row in frame_audits if row["frame_idx"] == candidate_frame)
    mock_fixture_used = any(
        frame.pipeline_audit.get("mock_fixture") is True for frame in frames
    )
    manifest = {
        "schema_version": OVERLAY_PREFLIGHT_SCHEMA_VERSION,
        "route_key": ROUTE_KEY,
        "folder": ROUTE_FOLDER,
        "selected_frames": sorted(by_index),
        "candidate_frame": candidate_frame,
        "input_mode": (
            "mock_fixture" if mock_fixture_used else "real_dedicated_target_pipeline"
        ),
        "output_dir": str(output_dir.resolve()),
        "frames": frame_audits,
        "automated_preflight": {
            "passed": bool(
                candidate_audit["nonempty_projected_support"]
                and all(
                    all(row["automated_checks"].values()) for row in frame_audits
                )
            ),
            "candidate_has_nonempty_projected_support": candidate_audit[
                "nonempty_projected_support"
            ],
            "all_frames_use_bottom_origin_gt": all(
                row["box_z_origin"] == "bottom" for row in frame_audits
            ),
            "all_frames_use_canonical_camera_order": all(
                row["camera_order"] == list(ORION_CAMERA_ORDER) for row in frame_audits
            ),
        },
        "claim_boundary": {
            "model_loaded": False,
            "gpu_used": False,
            "slurm_job_submitted": False,
            "carla_used": False,
            "training_performed": False,
            "attribution_is_causal": False,
            "human_visual_alignment_review_performed": False,
            "g1_projection_overlay_gate_passed": False,
            "mock_fixture_used": mock_fixture_used,
            "real_route214_data_used": not mock_fixture_used,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = _sha256_file(manifest_path)
    return manifest


def _remove_non_geometry_vqa_transform(pipeline: list[dict[str, Any]]) -> dict[str, Any]:
    removed_transforms = []
    removed_collect_keys = []
    retained = []
    for node in pipeline:
        if node.get("type") == "LoadAnnoatationCriticalVQATest":
            removed_transforms.append(node.get("type"))
            continue
        transforms = node.get("transforms")
        if isinstance(transforms, list):
            nested = _remove_non_geometry_vqa_transform(transforms)
            removed_transforms.extend(nested["removed_transforms"])
            removed_collect_keys.extend(nested["removed_collect_keys"])
        if node.get("type") == "CustomCollect3D":
            keys = node.get("keys")
            if not isinstance(keys, list):
                raise ProjectionOverlayPreflightError("CustomCollect3D keys malformed")
            for key in ("input_ids", "vlm_labels"):
                if key in keys:
                    keys.remove(key)
                    removed_collect_keys.append(key)
        retained.append(node)
    pipeline[:] = retained
    return {
        "removed_transforms": removed_transforms,
        "removed_collect_keys": removed_collect_keys,
    }


def _normalized_tensor_to_rgb(image: torch.Tensor, meta: Mapping[str, Any]) -> torch.Tensor:
    if image.ndim != 5 or image.shape[:2] != (1, 6):
        raise ProjectionOverlayPreflightError("canonical image tensor must be [1,6,C,H,W]")
    norm = meta.get("img_norm_cfg")
    if not isinstance(norm, Mapping):
        raise ProjectionOverlayPreflightError("img_norm_cfg missing from metadata")
    mean = torch.as_tensor(norm.get("mean"), dtype=image.dtype).view(1, 1, 3, 1, 1)
    std = torch.as_tensor(norm.get("std"), dtype=image.dtype).view(1, 1, 3, 1, 1)
    if mean.numel() != 3 or std.numel() != 3 or bool(torch.any(std <= 0)):
        raise ProjectionOverlayPreflightError("invalid image normalization metadata")
    restored = image.detach().cpu() * std + mean
    return restored[0].permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)


def load_route214_frames_from_dedicated_pipeline(
    *,
    repo_root: Path,
    config_path: Path,
    infos_path: Path,
    dataset_root: Path,
    map_file: Path,
    frames: Sequence[int] = DEFAULT_FRAMES,
) -> tuple[list[Route214ProjectionFrameV1], dict[str, Any]]:
    """Run only the dedicated CPU dataset pipeline for selected Route214 frames."""

    repo_root = Path(repo_root).resolve()
    for path, name in (
        (config_path, "config"),
        (infos_path, "infos"),
        (dataset_root, "dataset_root"),
        (map_file, "map_file"),
    ):
        if not Path(path).exists():
            raise ProjectionOverlayPreflightError("%s does not exist: %s" % (name, path))
    from uq_estimator.orion_actual_target_runner import (
        canonicalize_orion_test_batch,
        load_stage3_agent_config,
        mutate_stage3_agent_config_for_actual_targets,
    )

    source, config_lineage = load_stage3_agent_config(Path(config_path))
    mutated, dedicated_audit = mutate_stage3_agent_config_for_actual_targets(source)
    pipeline = mutated["test_pipeline"]
    geometry_prune = _remove_non_geometry_vqa_transform(pipeline)
    mutated["data"]["test"]["pipeline"] = pipeline
    dataset_cfg = mutated["data"]["test"]
    dataset_cfg["data_root"] = str(Path(dataset_root).resolve())
    dataset_cfg["ann_file"] = str(Path(infos_path).resolve())
    dataset_cfg["map_root"] = str(Path(dataset_root).resolve() / "maps")
    dataset_cfg["map_file"] = str(Path(map_file).resolve())
    dataset_cfg["test_mode"] = True

    try:
        from mmcv.datasets import build_dataset
    except Exception as exc:
        raise ProjectionOverlayPreflightError(
            "cannot import project mmcv dataset pipeline: %s: %s"
            % (type(exc).__name__, exc)
        ) from exc
    try:
        dataset = build_dataset(dataset_cfg)
    except Exception as exc:
        raise ProjectionOverlayPreflightError(
            "cannot construct dedicated target dataset: %s: %s"
            % (type(exc).__name__, exc)
        ) from exc

    requested = tuple(int(value) for value in frames)
    indices: dict[int, int] = {}
    for index, info in enumerate(dataset.data_infos):
        if info.get("folder") == ROUTE_FOLDER and int(info.get("frame_idx", -1)) in requested:
            frame_idx = int(info["frame_idx"])
            if frame_idx in indices:
                raise ProjectionOverlayPreflightError("duplicate Route214 frame in dataset")
            indices[frame_idx] = index
    missing = sorted(set(requested) - set(indices))
    if missing:
        raise ProjectionOverlayPreflightError("Route214 frames missing: %s" % missing)

    pipeline_contract = {
        "dedicated_target_pipeline": dedicated_audit.get("passed") is True,
        "with_light_state": dedicated_audit.get("with_light_state_enabled") is True,
        "with_actor_ids": dedicated_audit.get("with_actor_ids_enabled") is True,
        "post_augmentation_geometry": True,
        "processed_rgb_from_normalized_tensor": True,
        "geometry_only_preflight": True,
        "removed_non_geometry_transforms": geometry_prune["removed_transforms"],
        "removed_non_geometry_collect_keys": geometry_prune["removed_collect_keys"],
        "geometry_pipeline_sha256": _sha256_json(pipeline),
        "source_config": config_lineage,
        "model_loaded": False,
        "gpu_used": False,
        "slurm_job_submitted": False,
        "carla_used": False,
    }
    image_transform_id = "route214-dedicated-geometry-pipeline-" + pipeline_contract[
        "geometry_pipeline_sha256"
    ]
    outputs = []
    for frame_idx in requested:
        try:
            sample = dataset[indices[frame_idx]]
            canonical = canonicalize_orion_test_batch(sample)
        except Exception as exc:
            raise ProjectionOverlayPreflightError(
                "dedicated pipeline failed at Route214 frame %d: %s: %s"
                % (frame_idx, type(exc).__name__, exc)
            ) from exc
        if canonical.frame_idx != frame_idx or canonical.scene_token != ROUTE_FOLDER:
            raise ProjectionOverlayPreflightError("pipeline returned wrong route/frame identity")
        boxes = canonical.data["gt_bboxes_3d"]
        if not hasattr(boxes, "tensor"):
            raise ProjectionOverlayPreflightError("gt_bboxes_3d lacks canonical tensor")
        meta = canonical.img_metas[0]
        filenames = meta.get("filename")
        if not isinstance(filenames, (list, tuple)) or len(filenames) != 6:
            raise ProjectionOverlayPreflightError("metadata lacks six source filenames")
        outputs.append(
            Route214ProjectionFrameV1(
                frame_idx=frame_idx,
                processed_rgb=_normalized_tensor_to_rgb(canonical.data["img"], meta),
                post_augmentation_lidar2img=canonical.data["lidar2img"][0]
                .detach()
                .cpu()
                .float(),
                gt_boxes_lidar=boxes.tensor.detach().cpu().float(),
                gt_classes=canonical.data["gt_labels_3d"].detach().cpu().long(),
                gt_actor_ids=canonical.data["gt_actor_ids"].detach().cpu().long(),
                source_image_paths=tuple(str(value) for value in filenames),
                image_transform_id=image_transform_id,
                pipeline_audit=pipeline_contract,
            )
        )
    return outputs, pipeline_contract


def build_mock_route214_projection_frames() -> list[Route214ProjectionFrameV1]:
    """Build deterministic CPU frames for tests; never claims real pipeline IO."""

    matrices = []
    for view_index in range(6):
        depth_sign = 1.0 if view_index == 0 else -1.0
        matrices.append(
            torch.tensor(
                [
                    [160.0, 0.0, 320.0, 0.0],
                    [0.0, 160.0, 160.0, 0.0],
                    [0.0, 0.0, depth_sign, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
    matrix = torch.stack(matrices)
    audit = {
        "dedicated_target_pipeline": True,
        "with_light_state": True,
        "with_actor_ids": True,
        "post_augmentation_geometry": True,
        "processed_rgb_from_normalized_tensor": True,
        "mock_fixture": True,
        "model_loaded": False,
        "gpu_used": False,
        "slurm_job_submitted": False,
        "carla_used": False,
    }
    outputs = []
    for frame_idx in DEFAULT_FRAMES:
        image = torch.zeros(6, 320, 640, 3, dtype=torch.uint8)
        image[..., 1] = 32
        outputs.append(
            Route214ProjectionFrameV1(
                frame_idx=frame_idx,
                processed_rgb=image,
                post_augmentation_lidar2img=matrix,
                gt_boxes_lidar=torch.tensor(
                    [[0.0, 0.0, 8.0, 2.0, 2.0, 2.0, 0.0]]
                ),
                gt_classes=torch.tensor([0]),
                gt_actor_ids=torch.tensor([214000 + frame_idx]),
                source_image_paths=tuple(
                    "mock://route214/frame%d/%s" % (frame_idx, camera)
                    for camera in ORION_CAMERA_ORDER
                ),
                image_transform_id="mock-post-augmentation-route214/v1",
                pipeline_audit=audit,
            )
        )
    return outputs


__all__ = [
    "DEFAULT_CANDIDATE_FRAME",
    "DEFAULT_FRAMES",
    "OVERLAY_PREFLIGHT_SCHEMA_VERSION",
    "ProjectionOverlayPreflightError",
    "ROUTE_FOLDER",
    "ROUTE_KEY",
    "Route214ProjectionFrameV1",
    "build_mock_route214_projection_frames",
    "generate_route214_projection_overlays",
    "load_route214_frames_from_dedicated_pipeline",
]
