#!/usr/bin/env python3
"""Load Qwen-Drive and score selected fields of one frozen v15.2 state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluate_qwen_drive_text_oracle_v1 import (
    FIELD_VOCABULARIES,
    QwenTextScorer,
    TAG_ORDER,
    _read_json,
    field_prompt,
    load_frozen_states,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--orion-report", type=Path, required=True)
    parser.add_argument("--state-index", type=int, default=1)
    parser.add_argument("--tag", choices=(*TAG_ORDER, "ALL"), default="U_VIEW")
    args = parser.parse_args()
    states = load_frozen_states(_read_json(args.orion_report.resolve()))
    keys = sorted(states)
    if not 1 <= args.state_index <= len(keys):
        raise ValueError("state index is outside the frozen state set")
    key = keys[args.state_index - 1]
    state = states[key]
    scorer = QwenTextScorer(args.model.resolve(), "cuda", "bfloat16")
    tags = TAG_ORDER if args.tag == "ALL" else (args.tag,)
    for tag in tags:
        print(json.dumps({"scoring_state": key, "tag": tag}), flush=True)
        candidates = FIELD_VOCABULARIES[tag]
        nlls = scorer.score(field_prompt(state["fields"], tag), candidates)
        predicted = candidates[min(range(len(nlls)), key=nlls.__getitem__)]
        print(
            json.dumps(
                {
                    "status": "qwen_drive_text_oracle_smoke_complete",
                    "state": key,
                    "tag": tag,
                    "expected": state["fields"][tag],
                    "predicted": predicted,
                    "candidate_nlls": dict(zip(candidates, nlls)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
