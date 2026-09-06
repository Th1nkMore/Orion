#!/usr/bin/env python3
"""Build the deterministic V1e row-addressed visibility curriculum."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load local module from %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package = types.ModuleType("_orion_qwen_visibility_row_curriculum_package")
package.__path__ = [str(PROJECT_ROOT / "uq_estimator")]
sys.modules[package.__name__] = package
_load_local_module(
    package.__name__ + ".qwen_visibility_belief",
    PROJECT_ROOT / "uq_estimator" / "qwen_visibility_belief.py",
)
_load_local_module(
    package.__name__ + ".qwen_visibility_grounding",
    PROJECT_ROOT / "uq_estimator" / "qwen_visibility_grounding.py",
)
_curriculum = _load_local_module(
    package.__name__ + ".qwen_visibility_grounding_curriculum",
    PROJECT_ROOT / "uq_estimator" / "qwen_visibility_grounding_curriculum.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps-per-field", type=int, default=90)
    args = parser.parse_args()
    curriculum = _curriculum.build_route151_row_grounding_curriculum(
        base_manifest_path=args.base_manifest,
        output_path=args.output,
        steps_per_field=args.steps_per_field,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "example_count": curriculum["example_count"],
                "field_counts": curriculum["field_counts"],
                "label_counts": curriculum["label_counts"],
                "optimizer_steps": curriculum["optimizer_steps"],
                "reportable_generalization": curriculum[
                    "reportable_generalization"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
