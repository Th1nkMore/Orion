#!/usr/bin/env python3
"""Render pose-matched none/medium native-motion-blur visual diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SCHEMA = "orion.native_motion_blur_clean_revalidation_result.v1"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trace(profile_root):
    paths = tuple(Path(profile_root).glob("records_*/capture_trace.jsonl"))
    if len(paths) != 1:
        raise ValueError("expected one capture trace")
    return paths[0], [json.loads(line) for line in paths[0].read_text().splitlines() if line]


def _nearest(source, targets):
    return [min(range(len(source)), key=lambda index: abs(source[index] - target)) for target in targets]


def _label(image, text):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, result.width, 28), fill=(0, 0, 0))
    draw.text((8, 7), text, fill=(255, 255, 255))
    return result


def _row(images, labels):
    values = [_label(image, label) for image, label in zip(images, labels)]
    result = Image.new("RGB", (sum(image.width for image in values), values[0].height))
    left = 0
    for value in values:
        result.paste(value, (left, 0)); left += value.width
    return result


def _gif(path, frames, fps):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(round(1000/fps)), loop=0, optimize=False)


def _edge_variance(image):
    gray = np.asarray(image.convert("L"), dtype=np.float64)
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    return float(laplacian.var())


def render(
    root,
    protocol_path,
    none_audit_path,
    medium_audit_path,
    output,
    none_root=None,
    medium_root=None,
):
    root, protocol_path, none_audit_path, medium_audit_path, output = map(Path, (root, protocol_path, none_audit_path, medium_audit_path, output))
    profile_roots = {
        "none": Path(none_root) if none_root is not None else root / "captures" / "none",
        "medium": Path(medium_root) if medium_root is not None else root / "captures" / "medium",
    }
    if output.exists():
        raise FileExistsError("refusing to overwrite native blur revalidation")
    protocol = _json(protocol_path)
    if protocol.get("schema") != "orion.native_motion_blur_clean_revalidation_protocol.v1":
        raise ValueError("protocol schema differs")
    audits = {"none": _json(none_audit_path), "medium": _json(medium_audit_path)}
    for profile, audit in audits.items():
        if audit.get("status") != "passed_clean_render_artifact_gate" or not audit["gate"]["passed"]:
            raise RuntimeError("%s failed black-block gate" % profile)
    traces, rows = {}, {}
    for profile in ("none", "medium"):
        traces[profile], rows[profile] = _trace(profile_roots[profile])
        if not rows[profile]:
            raise ValueError("empty capture rows")
        if any(row.get("orion_loaded") is not False for row in rows[profile]):
            raise RuntimeError("capture unexpectedly loaded ORION")
        if any(row.get("corruption_family") != "native_motion_blur" for row in rows[profile]):
            raise RuntimeError("capture family differs")
        if any(row.get("profile") != profile for row in rows[profile]):
            raise RuntimeError("capture profile differs")
        expected_status = "observed" if profile == "none" else "verified"
        if any((row.get("camera_postprocess_readback") or {}).get("status") != expected_status for row in rows[profile]):
            raise RuntimeError("camera actor readback status differs")
    none_progress = [float(row["route_progress"]) for row in rows["none"]]
    medium_progress = [float(row["route_progress"]) for row in rows["medium"]]
    matched_medium = _nearest(medium_progress, none_progress)
    none_frames = [Image.open(row["front"]).convert("RGB") for row in rows["none"]]
    medium_frames = [Image.open(rows["medium"][index]["front"]).convert("RGB") for index in matched_medium]
    if any(image.size != (1600, 900) for image in none_frames + medium_frames):
        raise ValueError("front capture is not 1600x900")
    output.mkdir(parents=True)
    fps = int(protocol["review"]["output_fps"])
    _gif(output / "front_none_fullres.gif", none_frames, fps)
    _gif(output / "front_medium_fullres.gif", medium_frames, fps)
    review_size = tuple(protocol["review"]["review_size"])
    comparison = [
        _row(
            [left.resize(review_size, Image.Resampling.LANCZOS), right.resize(review_size, Image.Resampling.LANCZOS)],
            ["none / unmodified", "native motion blur medium"],
        )
        for left, right in zip(none_frames, medium_frames)
    ]
    _gif(output / "front_none_vs_medium.gif", comparison, fps)
    reference = float(protocol["review"]["contact_progress"])
    contact = min(range(len(none_progress)), key=lambda index: abs(none_progress[index] - reference))
    comparison[contact].save(output / "contact_none_vs_medium.png")
    bev = [
        _label(Image.open(row["bev"]).convert("RGB").resize(review_size, Image.Resampling.LANCZOS),
               "none BEV context; BEV postprocess disabled")
        for row in rows["none"]
    ]
    _gif(output / "none_bev_context.gif", bev, fps)
    none_edges = [_edge_variance(image) for image in none_frames]
    medium_edges = [_edge_variance(image) for image in medium_frames]
    progress_errors = [abs(medium_progress[index] - target) for index, target in zip(matched_medium, none_progress)]
    result = {
        "schema": SCHEMA,
        "status": "captures_passed_black_block_and_readback_gates_pending_human_visual_review",
        "orion_loaded": False,
        "frame_count_none": len(none_frames),
        "frame_count_medium": len(rows["medium"]),
        "matched_frame_count": len(medium_frames),
        "source_resolution": [1600, 900],
        "artifact_gates": {
            profile: {
                "passed": True,
                "suspicious_frame_count": audits[profile]["gate"]["suspicious_frame_count"],
                "frame_count": audits[profile]["gate"]["frame_count"],
                "audit_sha256": _sha(none_audit_path if profile == "none" else medium_audit_path),
            } for profile in ("none", "medium")
        },
        "readback": {
            "none": rows["none"][0]["camera_postprocess_readback"],
            "medium": rows["medium"][0]["camera_postprocess_readback"],
        },
        "pose_matching": {
            "maximum_route_progress_error": max(progress_errors),
            "mean_route_progress_error": float(np.mean(progress_errors)),
            "medium_source_indices": matched_medium,
        },
        "edge_variance": {
            "none_mean": float(np.mean(none_edges)),
            "medium_mean": float(np.mean(medium_edges)),
            "medium_to_none_ratio": float(np.mean(medium_edges) / np.mean(none_edges)),
            "interpretation": "diagnostic only; native motion blur can be object/velocity dependent",
        },
        "contact": {"none_index": contact, "medium_index": matched_medium[contact], "route_progress": none_progress[contact]},
        "provenance": {
            "protocol_sha256": _sha(protocol_path),
            "none_trace_sha256": _sha(traces["none"]),
            "medium_trace_sha256": _sha(traces["medium"]),
            "profile_roots": {profile: str(path) for profile, path in profile_roots.items()},
        },
        "locks": protocol["locks"],
        "claim_boundary": protocol["claim_boundary"],
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["artifacts"] = {
        path.name: {"sha256": _sha(path), "bytes": path.stat().st_size}
        for path in sorted(output.iterdir()) if path.is_file() and path != result_path
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--none-audit", type=Path, required=True)
    parser.add_argument("--medium-audit", type=Path, required=True)
    parser.add_argument("--none-root", type=Path)
    parser.add_argument("--medium-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = render(
        args.root,
        args.protocol,
        args.none_audit,
        args.medium_audit,
        args.output,
        none_root=args.none_root,
        medium_root=args.medium_root,
    )
    print(json.dumps({"status": result["status"], "matched_frames": result["matched_frame_count"]}, indent=2))


if __name__ == "__main__":
    main()
