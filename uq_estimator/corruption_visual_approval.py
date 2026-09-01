"""Fail-closed lineage and human-approval checks for hard-case corruptions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "orion.corruption_hardcase_visual_approval_gate.v2"


class VisualApprovalError(RuntimeError):
    """Raised when a corruption is not explicitly approved or its lineage changed."""


@dataclass(frozen=True)
class VisualApprovalRecord:
    family: str
    condition: str | None
    decision_status: str
    approved_conditions: tuple[str, ...]
    gate_path: str
    gate_sha256: str
    evidence_sha256: dict[str, str]
    implementation_path: str
    implementation_sha256: str
    human_authorization: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "condition": self.condition,
            "decision_status": self.decision_status,
            "approved_conditions": list(self.approved_conditions),
            "gate_path": self.gate_path,
            "gate_sha256": self.gate_sha256,
            "evidence_sha256": dict(self.evidence_sha256),
            "implementation_path": self.implementation_path,
            "implementation_sha256": self.implementation_sha256,
            "human_authorization": self.human_authorization,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_file(repository_root: Path, relative: str) -> Path:
    root = repository_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise VisualApprovalError("approval path escapes repository root") from error
    if not path.is_file():
        raise VisualApprovalError("approval evidence is missing: %s" % relative)
    return path


def _verified_file(repository_root: Path, record: dict[str, Any]) -> tuple[Path, str]:
    path = _repo_file(repository_root, record["path"])
    actual = _sha256(path)
    if actual != record.get("sha256"):
        raise VisualApprovalError(
            "approval evidence hash differs for %s" % record["path"]
        )
    return path, actual


def _verified_evidence_file(
    repository_root: Path, record: dict[str, Any]
) -> tuple[Path | None, str | None]:
    repository_path = (repository_root.resolve() / record["path"]).resolve()
    candidates = []
    try:
        repository_path.relative_to(repository_root.resolve())
        candidates.append(repository_path)
    except ValueError as error:
        raise VisualApprovalError("approval path escapes repository root") from error
    remote = record.get("remote_path")
    if remote:
        remote_path = Path(remote)
        if not remote_path.is_absolute():
            raise VisualApprovalError("remote approval evidence path is not absolute")
        candidates.append(remote_path)
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        if record.get("required_at_runtime", True) is False:
            return None, None
        raise VisualApprovalError(
            "approval evidence is missing locally and remotely: %s"
            % record["path"]
        )
    actual = _sha256(path)
    if actual != record.get("sha256"):
        raise VisualApprovalError(
            "approval evidence hash differs for %s" % record["path"]
        )
    return path, actual


def verify_visual_approval(
    *,
    gate_path: str | Path,
    repository_root: str | Path,
    family: str,
    condition: str | None = None,
    require_approved: bool = True,
) -> VisualApprovalRecord:
    repository = Path(repository_root).resolve()
    gate_file = Path(gate_path).resolve()
    if not gate_file.is_file():
        raise VisualApprovalError("visual approval gate is missing")
    gate = json.loads(gate_file.read_text(encoding="utf-8"))
    if gate.get("schema") != SCHEMA:
        raise VisualApprovalError("visual approval gate schema differs")
    families = gate.get("families", {})
    if family not in families:
        raise VisualApprovalError("unknown corruption family: %s" % family)
    candidate = families[family]

    implementation_path, implementation_sha = _verified_file(
        repository, candidate["implementation"]
    )
    retired_paths = {
        row.get("path") for row in gate.get("retired_implementations", [])
    }
    if candidate["implementation"]["path"] in retired_paths:
        raise VisualApprovalError("active implementation is also marked retired")

    evidence_hashes: dict[str, str] = {}
    for integration in candidate.get("runtime_integration", []):
        _, integration_sha = _verified_file(repository, integration)
        evidence_hashes[integration["path"]] = integration_sha
    result_record = candidate["evidence"]["result"]
    result_path, result_sha = _verified_evidence_file(repository, result_record)
    if result_path is None or result_sha is None:
        raise VisualApprovalError("visual result cannot be runtime-optional")
    evidence_hashes[result_record["path"]] = result_sha
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema") != result_record.get("schema"):
        raise VisualApprovalError("visual result schema differs")
    if result.get("status") != result_record.get("allowed_status"):
        raise VisualApprovalError("visual result status differs")
    if result.get("orion_loaded") is not False:
        raise VisualApprovalError("visual result unexpectedly loaded ORION")
    for artifact in candidate["evidence"].get("review_artifacts", []):
        _, artifact_sha = _verified_evidence_file(repository, artifact)
        if artifact_sha is not None:
            evidence_hashes[artifact["path"]] = artifact_sha

    decision = candidate.get("decision", {})
    decision_status = decision.get("status")
    approved_conditions = tuple(decision.get("approved_conditions", []))
    candidate_conditions = tuple(candidate.get("candidate_conditions", []))
    if any(item not in candidate_conditions for item in approved_conditions):
        raise VisualApprovalError("approved condition was not a visual candidate")
    human_authorization = decision.get("human_authorization")

    if decision_status == "approved":
        if family not in gate["authority"].get("orion_screen_unlocked_families", []):
            raise VisualApprovalError("approved family is absent from unlock list")
        if not approved_conditions:
            raise VisualApprovalError("approved family has no approved condition")
        if not isinstance(human_authorization, dict):
            raise VisualApprovalError("approved family lacks human authorization")
        if human_authorization.get("source") != "explicit_user_message":
            raise VisualApprovalError("human authorization source is not explicit")
        amendment = human_authorization.get("amendment")
        amendment_sha = human_authorization.get("amendment_sha256")
        if not amendment or not amendment_sha:
            raise VisualApprovalError("human authorization amendment is incomplete")
        _, actual_amendment_sha = _verified_file(
            repository, {"path": amendment, "sha256": amendment_sha}
        )
        evidence_hashes[amendment] = actual_amendment_sha
    elif decision_status != "pending":
        raise VisualApprovalError("decision status must be pending or approved")

    if require_approved:
        if decision_status != "approved":
            raise VisualApprovalError(
                "%s is pending human visual approval" % family
            )
        if condition is None:
            raise VisualApprovalError("an exact approved condition is required")
        if condition not in approved_conditions:
            raise VisualApprovalError(
                "%s is not approved for %s" % (condition, family)
            )

    return VisualApprovalRecord(
        family=family,
        condition=condition,
        decision_status=decision_status,
        approved_conditions=approved_conditions,
        gate_path=str(gate_file),
        gate_sha256=_sha256(gate_file),
        evidence_sha256=evidence_hashes,
        implementation_path=str(implementation_path),
        implementation_sha256=implementation_sha,
        human_authorization=human_authorization,
    )
