"""Production integration factory for the bounded Route214 target smoke.

This module does not construct ORION or a dataloader.  It binds the concrete
branch-target builder and four named, persistent QA-evidence callbacks for the
launch script.  A missing, stale, lineage-mismatched, or boolean-only evidence
file returns ``False`` and keeps runner preflight fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Mapping, Tuple

from uq_estimator.orion_actual_target_builder import (
    ProductionActualTargetBranchBuilderV1,
    ProductionBranchTargetConfigV1,
)
from uq_estimator.orion_decode_adapter import (
    ORIONDecodeAdapterConfigV1,
)


INTEGRATION_FACTORY_SCHEMA_VERSION = "orion-route214-production-integration/v1"
QA_EVIDENCE_SCHEMA_VERSION = "orion-route214-qa-evidence/v1"
FACTORY_CONTEXT_SCHEMA_VERSION = "orion-route214-production-factory-context/v1"
QA_HOOK_IDS = {
    "decoder_parity_check": "route214-decoder-parity-persistent-evidence/v1",
    "selected_motion_mode_check": "route214-selected-mode-persistent-evidence/v1",
    "projection_overlay_check": "route214-projection-overlay-persistent-evidence/v1",
    "gt_axis_alignment_check": "route214-gt-axis-persistent-evidence/v1",
}


class Route214ProductionIntegrationError(RuntimeError):
    """Raised when production integration metadata would require guessing."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Route214ProductionIntegrationError("%s must be non-empty" % name)
    return value.strip()


def _sha256_text(value: Any, name: str) -> str:
    result = _text(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise Route214ProductionIntegrationError(
            "%s must be a lowercase SHA-256 digest" % name
        )
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if completed.returncode == 0 and len(revision) == 40:
        return revision

    # The compute deployment is an rsynced source tree without `.git`. Record
    # that fact explicitly and bind provenance to the exact production code
    # bundle instead of inventing a commit hash.
    critical_paths = (
        "adzoo/orion/configs/orion_stage3_agent.py",
        "mmcv/datasets/pipelines/formating.py",
        "mmcv/datasets/pipelines/loading.py",
        "mmcv/datasets/pipelines/transforms_3d.py",
        "mmcv/models/dense_heads/orion_head.py",
        "uq_estimator/bev_target_rasterizer.py",
        "uq_estimator/decoded_actual_target_export.py",
        "uq_estimator/orion_actual_target_builder.py",
        "uq_estimator/orion_actual_target_runner.py",
        "uq_estimator/orion_decode_adapter.py",
        "uq_estimator/projected_visible_support.py",
        "uq_estimator/orion_route214_production_integration.py",
        "scripts/run_orion_actual_target_route214.py",
    )
    digest = hashlib.sha256()
    for relative in critical_paths:
        path = repo_root / relative
        if not path.is_file():
            raise Route214ProductionIntegrationError(
                "source-tree provenance file is missing: %s" % relative
            )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return "deployed-source-tree-sha256:%s" % digest.hexdigest()


@dataclass
class PersistentRoute214QAEvidenceV1:
    """Named callback that validates one immutable evidence JSON and artifacts."""

    qa_kind: str
    evidence_path: Path
    expected_plan_id: str
    expected_checkpoint_sha256: str
    expected_config_sha256: str
    expected_route_folder: str
    production_hook_id: str

    def __post_init__(self) -> None:
        if self.qa_kind not in QA_HOOK_IDS:
            raise Route214ProductionIntegrationError("unknown QA kind")
        if self.production_hook_id != QA_HOOK_IDS[self.qa_kind]:
            raise Route214ProductionIntegrationError("QA hook ID disagrees with kind")
        self.evidence_path = Path(self.evidence_path).expanduser().resolve()
        _text(self.expected_plan_id, "expected_plan_id")
        _sha256_text(
            self.expected_checkpoint_sha256, "expected_checkpoint_sha256"
        )
        _sha256_text(self.expected_config_sha256, "expected_config_sha256")
        _text(self.expected_route_folder, "expected_route_folder")
        self.last_audit: Dict[str, Any] = {
            "passed": False,
            "reason": "not_evaluated",
            "evidence_path": str(self.evidence_path),
        }

    def __call__(self) -> bool:
        audit: Dict[str, Any] = {
            "passed": False,
            "evidence_path": str(self.evidence_path),
            "production_hook_id": self.production_hook_id,
            "qa_kind": self.qa_kind,
        }
        if not self.evidence_path.is_file():
            audit["reason"] = "evidence_json_missing"
            self.last_audit = audit
            return False
        try:
            payload = json.loads(self.evidence_path.read_text(encoding="utf-8"))
        except Exception as exc:
            audit["reason"] = "invalid_json:%s" % exc.__class__.__name__
            self.last_audit = audit
            return False
        if not isinstance(payload, Mapping):
            audit["reason"] = "evidence_root_not_object"
            self.last_audit = audit
            return False
        expected = {
            "schema_version": QA_EVIDENCE_SCHEMA_VERSION,
            "qa_kind": self.qa_kind,
            "production_hook_id": self.production_hook_id,
            "plan_id": self.expected_plan_id,
            "checkpoint_sha256": self.expected_checkpoint_sha256,
            "config_sha256": self.expected_config_sha256,
            "route_folder": self.expected_route_folder,
            "passed": True,
        }
        disagreements = [
            key for key, value in expected.items() if payload.get(key) != value
        ]
        checks = payload.get("checks")
        if (
            not isinstance(checks, Mapping)
            or not checks
            or any(value is not True for value in checks.values())
        ):
            disagreements.append("checks")
        generated_by = payload.get("generated_by")
        if not isinstance(generated_by, str) or not generated_by.strip():
            disagreements.append("generated_by")
        artifacts = payload.get("artifacts")
        artifact_audit = []
        if not isinstance(artifacts, list) or not artifacts:
            disagreements.append("artifacts")
        else:
            for index, raw in enumerate(artifacts):
                if not isinstance(raw, Mapping):
                    disagreements.append("artifacts[%d]" % index)
                    continue
                path_value = raw.get("path")
                digest = raw.get("sha256")
                artifact_path = Path(str(path_value)).expanduser()
                if not artifact_path.is_absolute():
                    artifact_path = self.evidence_path.parent / artifact_path
                artifact_path = artifact_path.resolve()
                exists = artifact_path.is_file()
                digest_ok = False
                if exists and isinstance(digest, str):
                    digest_ok = _sha256_file(artifact_path) == digest
                artifact_audit.append(
                    {
                        "path": str(artifact_path),
                        "exists": exists,
                        "sha256_verified": digest_ok,
                    }
                )
                if not exists or not digest_ok:
                    disagreements.append("artifacts[%d].lineage" % index)
        audit.update(
            {
                "evidence_sha256": _sha256_file(self.evidence_path),
                "artifacts": artifact_audit,
                "disagreements": disagreements,
                "reason": "passed" if not disagreements else "evidence_mismatch",
                "passed": not disagreements,
            }
        )
        self.last_audit = audit
        return bool(audit["passed"])


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Route214ProductionIntegrationError("%s must be a mapping" % name)
    return value


def build_route214_production_integration_v1(
    context: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build the concrete branch builder and four persistent QA callbacks."""

    context = _require_mapping(context, "factory context")
    if context.get("schema_version") != FACTORY_CONTEXT_SCHEMA_VERSION:
        raise Route214ProductionIntegrationError("unsupported factory context")
    plan = _require_mapping(context.get("plan"), "plan")
    route = _require_mapping(plan.get("route"), "plan.route")
    corruption = _require_mapping(plan.get("corruption"), "plan.corruption")
    if route.get("canonical_route_key") != "Town04/Route214":
        raise Route214ProductionIntegrationError("factory is Route214-only")
    if route.get("smoke_prefix_frame_range_inclusive") != [0, 63]:
        raise Route214ProductionIntegrationError("factory requires prefix 0..63")
    if corruption.get("family") != "local_occlusion":
        raise Route214ProductionIntegrationError("factory requires local_occlusion")
    if corruption.get("event_window_frames_inclusive") != [0, 63]:
        raise Route214ProductionIntegrationError("factory requires full-prefix window")
    decode_config = context.get("decode_config")
    if not isinstance(decode_config, ORIONDecodeAdapterConfigV1):
        raise Route214ProductionIntegrationError("decode_config has the wrong type")
    config_lineage = _require_mapping(context.get("config_lineage"), "config_lineage")
    checkpoint_sha256 = _sha256_text(
        context.get("checkpoint_sha256"), "checkpoint_sha256"
    )
    config_sha256 = _sha256_text(config_lineage.get("sha256"), "config sha256")
    repo_root = Path(context.get("repo_root", "")).expanduser().resolve()
    if not repo_root.is_dir():
        raise Route214ProductionIntegrationError("repo_root does not exist")
    git_revision = _git_revision(repo_root)
    builder = ProductionActualTargetBranchBuilderV1(
        ProductionBranchTargetConfigV1(
            base_checkpoint_sha256=checkpoint_sha256,
            inference_config_sha256=config_sha256,
            git_revision=git_revision,
            route_id=_text(route.get("folder"), "route folder"),
            town=_text(route.get("town"), "route town"),
            class_mapping_id=decode_config.class_mapping_id,
            decoder_policy_id=decode_config.decoder_policy_id,
            image_transform_id="orion-stage3-agent-post-augmentation-320x640/v1",
            observed_corruption_family="local_occlusion",
            observed_severity=float(corruption["severity"]),
            observed_seed=int(corruption["seed"]),
            event_window_frames=(0, 63),
        )
    )
    evidence_paths = _require_mapping(
        context.get("qa_evidence_paths"), "qa_evidence_paths"
    )
    callbacks: Dict[str, PersistentRoute214QAEvidenceV1] = {}
    for qa_kind, hook_id in QA_HOOK_IDS.items():
        if qa_kind not in evidence_paths:
            raise Route214ProductionIntegrationError(
                "qa_evidence_paths missing %s" % qa_kind
            )
        callbacks[qa_kind] = PersistentRoute214QAEvidenceV1(
            qa_kind=qa_kind,
            evidence_path=Path(evidence_paths[qa_kind]),
            expected_plan_id=_text(plan.get("plan_id"), "plan_id"),
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_config_sha256=config_sha256,
            expected_route_folder=_text(route.get("folder"), "route folder"),
            production_hook_id=hook_id,
        )
    return {
        "schema_version": INTEGRATION_FACTORY_SCHEMA_VERSION,
        "branch_target_builder": builder,
        **callbacks,
    }


__all__ = [
    "FACTORY_CONTEXT_SCHEMA_VERSION",
    "INTEGRATION_FACTORY_SCHEMA_VERSION",
    "PersistentRoute214QAEvidenceV1",
    "QA_EVIDENCE_SCHEMA_VERSION",
    "QA_HOOK_IDS",
    "Route214ProductionIntegrationError",
    "build_route214_production_integration_v1",
]
