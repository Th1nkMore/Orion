#!/usr/bin/env python3
"""Build the deterministic V1f matched-pair route-readout curriculum."""

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


package = types.ModuleType("_orion_qwen_visibility_route_readout_package")
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
    parser.add_argument("--train-pair-variants", type=int, default=3)
    parser.add_argument("--evaluation-pair-variants", type=int, default=2)
    parser.add_argument("--optimizer-steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    curriculum = _curriculum.build_route151_route_readout_curriculum(
        base_manifest_path=args.base_manifest,
        output_path=args.output,
        train_pair_variants=args.train_pair_variants,
        evaluation_pair_variants=args.evaluation_pair_variants,
        optimizer_steps=args.optimizer_steps,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "example_count": curriculum["example_count"],
                "training_example_count": len(
                    curriculum["training_example_ids"]
                ),
                "evaluation_example_count": len(
                    curriculum["evaluation_example_ids"]
                ),
                "training_label_counts": curriculum[
                    "training_label_counts"
                ],
                "evaluation_label_counts": curriculum[
                    "evaluation_label_counts"
                ],
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
