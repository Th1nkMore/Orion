#!/usr/bin/env python3
"""Build a local, hash-verified HTML gallery for Stage2-L QA geometry review."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping


QUEUE_SCHEMA = "orion.stage2_l.qa_geometry_review_queue.v1"
VARIANT_ORDER = (
    "observed",
    "zero_uq",
    "on_path_uq",
    "off_path_uq",
    "view_shuffled_uq",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_contact_sheet(
    *, event_id: str, reference: Mapping[str, Any], contact_sheet_root: Path
) -> Path:
    remote = Path(str(reference["path"]))
    try:
        event_index = remote.parts.index(event_id)
    except ValueError as error:
        raise ValueError(
            "contact-sheet path does not contain event id %s" % event_id
        ) from error
    local = contact_sheet_root / event_id / Path(*remote.parts[event_index + 1 :])
    if not local.is_file():
        raise FileNotFoundError("local contact sheet is absent: %s" % local)
    if _sha256(local) != reference.get("sha256"):
        raise ValueError("local contact-sheet hash mismatch: %s" % local)
    return local.resolve()


def build_gallery(
    *, queue_path: Path, contact_sheet_root: Path, output_path: Path
) -> Dict[str, int]:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    if queue.get("schema") != QUEUE_SCHEMA:
        raise ValueError("unsupported QA geometry review queue")
    if queue.get("status") != "pending_human_qa_geometry_review":
        raise ValueError("gallery must be built from a pending review queue")

    event_sections = []
    image_count = 0
    frame_count = 0
    for event in queue.get("review_order", []):
        event_id = str(event["event_id"])
        frame_sections = []
        for frame in event.get("visualizations", []):
            frame_count += 1
            cards = []
            variants = frame.get("variants", {})
            if set(variants) != set(VARIANT_ORDER):
                raise ValueError("review frame does not contain exactly five variants")
            for variant in VARIANT_ORDER:
                local = _local_contact_sheet(
                    event_id=event_id,
                    reference=variants[variant]["contact_sheet"],
                    contact_sheet_root=contact_sheet_root,
                )
                image_count += 1
                relative = os.path.relpath(local, output_path.parent.resolve())
                quoted = html.escape(relative, quote=True)
                cards.append(
                    '<figure><figcaption>%s</figcaption>'
                    '<a href="%s"><img loading="lazy" src="%s" alt="%s %s"></a>'
                    "</figure>"
                    % (
                        html.escape(variant),
                        quoted,
                        quoted,
                        html.escape(event_id),
                        html.escape(variant),
                    )
                )
            frame_sections.append(
                '<section class="frame"><h3>saved frame %s</h3><div class="grid">%s</div></section>'
                % (html.escape(str(frame["selected_saved_frame_index"])), "".join(cards))
            )
        checks = "".join(
            "<li>☐ %s</li>" % html.escape(str(check))
            for check in event.get("required_checks", [])
        )
        event_sections.append(
            '<details class="event" open><summary>%s · %s frames</summary>'
            '<ul class="checks">%s</ul>%s</details>'
            % (
                html.escape(event_id),
                html.escape(str(event["keyframe_count"])),
                checks,
                "".join(frame_sections),
            )
        )

    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stage2-L pilot8 QA geometry review</title>
<style>
body{font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;background:#f5f6f8;color:#172033}
h1{margin-bottom:4px}.boundary{max-width:1000px;color:#596274}.event{background:white;border:1px solid #d8dde7;border-radius:10px;margin:18px 0;padding:12px}
summary{cursor:pointer;font-size:20px;font-weight:650}.checks{columns:2;list-style:none;padding:0;color:#445066}.frame{border-top:1px solid #e5e8ee;margin-top:16px;padding-top:8px}
.grid{display:grid;grid-template-columns:repeat(5,minmax(220px,1fr));gap:10px;overflow-x:auto}figure{margin:0;min-width:220px}figcaption{font-weight:600;margin-bottom:5px}
img{width:100%;height:auto;border:1px solid #cbd2df;border-radius:6px;background:#111}a:focus img,a:hover img{outline:3px solid #3b82f6}
@media(max-width:1100px){.grid{grid-template-columns:repeat(2,minmax(280px,1fr))}.checks{columns:1}}
</style></head><body>
<h1>Stage2-L pilot8 QA geometry review</h1>
<p class="boundary">Pending human review. This gallery checks QA construction geometry only; it does not validate Stage1 uncertainty, VLM understanding, planning, or safety.</p>
__EVENT_SECTIONS__
</body></html>
""".replace("__EVENT_SECTIONS__", "\n".join(event_sections))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return {
        "event_count": len(event_sections),
        "frame_count": frame_count,
        "image_count": image_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--contact-sheet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite local review gallery")
    counts = build_gallery(
        queue_path=args.review_queue.resolve(),
        contact_sheet_root=args.contact_sheet_root.resolve(),
        output_path=args.output.resolve(),
    )
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
