#!/usr/bin/env python3
"""Build the sparse Route-151 structured-U grounding plumbing manifest."""

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


# Avoid importing the legacy uq_estimator package initializer in the lightweight
# CARLA/data environment.  The grounding module's relative physical-token import
# still resolves through this deliberately minimal package shell.
package = types.ModuleType("_orion_qwen_visibility_grounding_package")
package.__path__ = [str(PROJECT_ROOT / "uq_estimator")]
sys.modules[package.__name__] = package
_load_local_module(
    package.__name__ + ".qwen_visibility_belief",
    PROJECT_ROOT / "uq_estimator" / "qwen_visibility_belief.py",
)
_grounding = _load_local_module(
    package.__name__ + ".qwen_visibility_grounding",
    PROJECT_ROOT / "uq_estimator" / "qwen_visibility_grounding.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutation-seed", type=int, default=20260906)
    args = parser.parse_args()
    manifest = _grounding.build_route151_grounding_manifest(
        token_root=args.token_root,
        audit_root=args.audit_root,
        output_path=args.output,
        permutation_seed=args.permutation_seed,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "record_count": manifest["record_count"],
                "target_counts": manifest["target_counts"],
                "reportable_generalization": manifest[
                    "reportable_generalization"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

