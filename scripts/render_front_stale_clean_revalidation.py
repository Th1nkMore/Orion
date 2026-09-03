#!/usr/bin/env python3
"""Render full-resolution front-stale review from artifact-gated clean frames."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


SCHEMA = "orion.front_stale_visual_revalidation_result.v1"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        result.paste(value, (left, 0))
        left += value.width
    return result


def _gif(path, frames, fps):
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000 / fps)),
        loop=0,
        optimize=False,
    )


def _source_indices(timestamps, delay_ms):
    delay = delay_ms / 1000.0
    return [
        bisect.bisect_right(timestamps, timestamp - delay + 1e-8) - 1
        for timestamp in timestamps
    ]


def render(source_root, protocol_path, audit_path, output):
    source_root, protocol_path, audit_path, output = map(Path, (source_root, protocol_path, audit_path, output))
    if output.exists():
        raise FileExistsError("refusing to overwrite front-stale revalidation")
    protocol, audit = _json(protocol_path), _json(audit_path)
    if protocol.get("schema") != "orion.front_stale_visual_revalidation_protocol.v1":
        raise ValueError("protocol schema differs")
    if audit.get("status") != "passed_clean_render_artifact_gate" or not audit["gate"]["passed"]:
        raise RuntimeError("clean artifact gate did not pass")
    roots = tuple(source_root.glob("records_*"))
    if len(roots) != 1:
        raise ValueError("expected one source records directory")
    root = roots[0]
    trace_path = root / "capture_trace.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    fronts = sorted((root / "rgb_front").glob("*.png"))
    bevs = sorted((root / "bev").glob("*.png"))
    if not rows or len(rows) != len(fronts) or len(rows) != len(bevs):
        raise ValueError("trace/front/BEV counts differ")
    timestamps = [float(row["sim_time_seconds"]) for row in rows]
    if timestamps != sorted(set(timestamps)):
        raise ValueError("source timestamps are not strictly increasing")
    increments = [right - left for left, right in zip(timestamps[:-1], timestamps[1:])]
    expected_increment = 1.0 / float(protocol["source"]["simulation_hz"])
    if max(abs(value - expected_increment) for value in increments) > 1e-5:
        raise ValueError("source timestamps do not match the preregistered cadence")
    delays = [int(value) for value in protocol["intervention"]["delays_ms"]]
    source_by_delay = {delay: _source_indices(timestamps, delay) for delay in delays}
    maximum_delay = max(delays) / 1000.0
    displayed = [
        index for index, timestamp in enumerate(timestamps)
        if timestamp - timestamps[0] >= maximum_delay - 1e-8
    ]
    if not displayed or any(source_by_delay[delay][index] < 0 for delay in delays for index in displayed):
        raise RuntimeError("full-history displayed frame selection failed")
    realized = {
        delay: [1000 * (timestamps[index] - timestamps[source_by_delay[delay][index]]) for index in displayed]
        for delay in delays
    }
    for delay in delays:
        if max(abs(value - delay) for value in realized[delay]) > 1e-3:
            raise RuntimeError("realized stale delay differs from protocol")

    clean = [Image.open(fronts[index]).convert("RGB") for index in displayed]
    if any(image.size != (1600, 900) for image in clean):
        raise ValueError("source front images are not 1600x900")
    stale = {
        delay: [Image.open(fronts[source_by_delay[delay][index]]).convert("RGB") for index in displayed]
        for delay in delays
    }
    output.mkdir(parents=True)
    fps = int(protocol["review"]["output_fps"])
    _gif(output / "front_clean_fullres.gif", clean, fps)
    for delay in delays:
        _gif(output / ("front_stale_%03dms_fullres.gif" % delay), stale[delay], fps)
    review_size = tuple(protocol["review"]["review_size"])
    comparison = []
    for offset in range(len(displayed)):
        images = [clean[offset]] + [stale[delay][offset] for delay in delays]
        images = [image.resize(review_size, Image.Resampling.LANCZOS) for image in images]
        comparison.append(_row(images, ["clean"] + ["stale %d ms" % delay for delay in delays]))
    _gif(output / "front_clean_vs_stale.gif", comparison, fps)
    progress = [float(rows[index]["route_progress"]) for index in displayed]
    reference = float(protocol["route"]["event_progress_reference"])
    contact_offset = min(range(len(displayed)), key=lambda offset: abs(progress[offset] - reference))
    comparison[contact_offset].save(output / "contact_clean_vs_stale.png")
    bev = [
        _label(Image.open(bevs[index]).convert("RGB").resize(review_size, Image.Resampling.LANCZOS),
               "clean BEV context; only CAM_FRONT is stale")
        for index in displayed
    ]
    _gif(output / "clean_bev_context.gif", bev, fps)
    frame_audit = []
    for offset, target_index in enumerate(displayed):
        frame_audit.append({
            "display_offset": offset,
            "target_capture_index": target_index,
            "target_timestamp": timestamps[target_index],
            "route_progress": float(rows[target_index]["route_progress"]),
            "sources": {
                str(delay): {
                    "capture_index": source_by_delay[delay][target_index],
                    "timestamp": timestamps[source_by_delay[delay][target_index]],
                    "realized_delay_ms": realized[delay][offset],
                }
                for delay in delays
            },
        })
    result = {
        "schema": SCHEMA,
        "status": "passed_timestamp_and_clean_visual_preconditions_pending_human_review",
        "orion_loaded": False,
        "source_frame_count": len(rows),
        "warmup_excluded_frame_count": len(rows) - len(displayed),
        "displayed_frame_count": len(displayed),
        "source_resolution": [1600, 900],
        "source_cadence_seconds": expected_increment,
        "clean_artifact_gate": {
            "passed": True,
            "suspicious_frame_count": audit["gate"]["suspicious_frame_count"],
            "audit_sha256": _sha(audit_path),
        },
        "delays": {
            str(delay): {
                "realized_min_ms": min(realized[delay]),
                "realized_max_ms": max(realized[delay]),
                "realized_mean_ms": sum(realized[delay]) / len(realized[delay]),
            }
            for delay in delays
        },
        "contact": {
            "display_offset": contact_offset,
            "capture_index": displayed[contact_offset],
            "route_progress": progress[contact_offset],
        },
        "frame_audit": frame_audit,
        "provenance": {
            "protocol_sha256": _sha(protocol_path),
            "capture_trace_sha256": _sha(trace_path),
            "first_source_front_sha256": _sha(fronts[0]),
            "last_source_front_sha256": _sha(fronts[-1]),
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
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--clean-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = render(args.source_root, args.protocol, args.clean_audit, args.output)
    print(json.dumps({"status": result["status"], "displayed_frames": result["displayed_frame_count"]}, indent=2))


if __name__ == "__main__":
    main()
