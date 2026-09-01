#!/usr/bin/env python3
"""Build a compact, hash-pinned human review bundle from frozen evidence."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


SCHEMA = "orion.corruption_visual_approval_review_bundle.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def reduced_gif(source: Path, output: Path, width: int = 960) -> dict[str, Any]:
    with Image.open(source) as gif:
        total = int(getattr(gif, "n_frames", 1))
        stride = max(1, total // 36)
        frames = []
        durations = []
        for index in range(0, total, stride):
            gif.seek(index)
            frame = gif.convert("RGB")
            height = max(1, round(frame.height * width / frame.width))
            frames.append(frame.resize((width, height), Image.Resampling.LANCZOS))
            durations.append(int(gif.info.get("duration", 100)) * stride)
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            optimize=True,
        )
    return {
        "source_frames": total,
        "review_frames": len(frames),
        "stride": stride,
        "width": width,
    }


def static_sheet(rows: list[dict[str, Any]], output: Path) -> None:
    target_width = 1600
    margin = 24
    title_height = 66
    resized = []
    for row in rows:
        with Image.open(row["contact"]) as image:
            rgb = image.convert("RGB")
            height = max(1, round(rgb.height * target_width / rgb.width))
            resized.append(rgb.resize((target_width, height), Image.Resampling.LANCZOS))
    canvas_height = margin + sum(title_height + image.height + margin for image in resized)
    canvas = Image.new("RGB", (target_width + 2 * margin, canvas_height), "#15171a")
    draw = ImageDraw.Draw(canvas)
    heading = font(30)
    body = font(20)
    y = margin
    for row, image in zip(rows, resized):
        draw.text((margin, y), row["title"], font=heading, fill="#ffffff")
        draw.text((margin, y + 37), row["conditions"], font=body, fill="#aeb7c2")
        y += title_height
        canvas.paste(image, (margin, y))
        y += image.height + margin
    canvas.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite visual review bundle")
    root = args.repository_root.resolve()
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    if gate.get("schema") != "orion.corruption_hardcase_visual_approval_gate.v2":
        raise ValueError("unexpected visual approval gate schema")
    args.output.mkdir(parents=True)

    specs = [
        {
            "family": "front_stale",
            "title": "Front stale-frame — temporal evidence",
            "conditions": "candidate: delay_ms:200 / delay_ms:400",
            "contact": root / gate["families"]["front_stale"]["evidence"]["review_artifacts"][1]["path"],
            "gif": root / gate["families"]["front_stale"]["evidence"]["review_artifacts"][0]["path"],
            "review_note": "Judge temporal lag in the GIF; a still contact sheet cannot establish stale-frame behavior.",
        },
        {
            "family": "lens_waterdrop_paired_template",
            "title": "Paired-template lens waterdrop — spatial evidence",
            "conditions": "candidate: profile:light / profile:medium / profile:heavy",
            "contact": root / gate["families"]["lens_waterdrop_paired_template"]["evidence"]["review_artifacts"][2]["path"],
            "gif": root / gate["families"]["lens_waterdrop_paired_template"]["evidence"]["review_artifacts"][0]["path"],
            "review_note": "Judge whether the fixed lens-space pattern looks physically plausible and remains temporally stable.",
        },
        {
            "family": "native_motion_blur",
            "title": "CARLA native motion blur — renderer evidence",
            "conditions": "candidate: profile:medium",
            "contact": root / gate["families"]["native_motion_blur"]["evidence"]["review_artifacts"][1]["path"],
            "gif": root / gate["families"]["native_motion_blur"]["evidence"]["review_artifacts"][0]["path"],
            "review_note": "Judge moving-object and edge degradation; this profile is intentionally renderer-native and visually subtle.",
        },
    ]
    for spec in specs:
        if not spec["contact"].is_file() or not spec["gif"].is_file():
            raise FileNotFoundError("review evidence missing for %s" % spec["family"])

    sheet = args.output / "visual_approval_decision_sheet.png"
    static_sheet(specs, sheet)
    artifacts = {"visual_approval_decision_sheet.png": sha256(sheet)}
    family_records = []
    for spec in specs:
        reduced_name = "%s_review.gif" % spec["family"]
        reduced_path = args.output / reduced_name
        reduction = reduced_gif(spec["gif"], reduced_path)
        artifacts[reduced_name] = sha256(reduced_path)
        family_records.append({
            "family": spec["family"],
            "candidate_conditions": gate["families"][spec["family"]]["candidate_conditions"],
            "decision_status": gate["families"][spec["family"]]["decision"]["status"],
            "source_contact": {"path": str(spec["contact"]), "sha256": sha256(spec["contact"])},
            "source_gif": {"path": str(spec["gif"]), "sha256": sha256(spec["gif"])},
            "review_gif": reduced_name,
            "reduction": reduction,
            "review_note": spec["review_note"],
        })

    html_rows = []
    for record in family_records:
        conditions = ", ".join(record["candidate_conditions"])
        html_rows.append(
            "<section><h2>%s</h2><p><b>Exact candidates:</b> %s</p>"
            "<p>%s</p><img src=\"%s\"></section>"
            % (
                html.escape(record["family"]),
                html.escape(conditions),
                html.escape(record["review_note"]),
                html.escape(record["review_gif"]),
            )
        )
    html_path = args.output / "index.html"
    html_path.write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>ORION corruption visual approval</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;background:#111;color:#eee}"
        "section{margin:36px 0;padding:20px;background:#1b1d21;border-radius:12px}"
        "img{max-width:100%;height:auto}code{color:#9ad}</style>"
        "<h1>Pending human visual approval</h1>"
        "<p>No ORION/GPU screen is unlocked by this review bundle.</p>"
        "<img src=\"visual_approval_decision_sheet.png\">"
        + "".join(html_rows),
        encoding="utf-8",
    )
    artifacts["index.html"] = sha256(html_path)

    result = {
        "schema": SCHEMA,
        "status": "rendered_pending_explicit_human_visual_decision",
        "gate": {"path": str(args.gate), "sha256": sha256(args.gate)},
        "families": family_records,
        "artifacts": artifacts,
        "orion_loaded": False,
        "gpu_jobs_submitted": 0,
        "claim_boundary": "Review convenience artifact only; it does not approve any condition or support an ORION/safety claim.",
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "families": len(family_records),
        "artifacts": len(artifacts),
        "result_sha256": sha256(result_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
