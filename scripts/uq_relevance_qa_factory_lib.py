"""CPU-only contracts for Stage2-L spatial-UQ/relevance QA data.

The Stage-1 adapter is frozen and supplies only normalized observation
uncertainty.  A separate dense target describes route/task relevance.  Their
pointwise product is the task-risk map used to make QA text deterministic.
Synthetic corruption identity, severity, TTC and outcomes are deliberately
excluded from every model input.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


FRAME_BUNDLE_SCHEMA = "orion.uq_relevance_frame_bundle.v1"
QA_DATASET_SCHEMA = "orion.uq_relevance_qa_dataset.v1"
QA_RECORD_SCHEMA = "orion.uq_relevance_qa_record.v1"
MAP_SIDECAR_SCHEMA = "orion.uq_relevance_map_sidecar.v1"
ALLOWED_SPLITS = ("train", "dev", "test")
ALLOWED_COUNTERFACTUAL_VARIANTS = (
    "observed",
    "zero_uq",
    "on_path_uq",
    "off_path_uq",
    "view_shuffled_uq",
)
QUESTION_FAMILIES = (
    "observation_semantics",
    "epistemic_limitation",
    "task_relevance",
    "driving_implication",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
QA_FACTORY_CONFIG_SCHEMAS = (
    "orion.uq_relevance_qa_factory_config.v1",
    "orion.uq_relevance_qa_factory_config.v2",
)


class QAFactoryError(ValueError):
    """Raised when Stage2-L data provenance or semantics are ambiguous."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_text(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise QAFactoryError("%s must be non-empty" % name)
    return result


def _require_sha(value: Any, name: str) -> str:
    result = _require_text(value, name)
    if not _SHA256_RE.fullmatch(result):
        raise QAFactoryError("%s must be a lowercase SHA-256 digest" % name)
    return result


def _resolved_file(reference: Mapping[str, Any], base_dir: Path, name: str) -> Path:
    raw = _require_text(reference.get("path"), "%s.path" % name)
    path = Path(raw)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.is_file():
        raise QAFactoryError("%s does not exist: %s" % (name, path))
    expected = _require_sha(reference.get("sha256"), "%s.sha256" % name)
    actual = sha256_file(path)
    if expected != actual:
        raise QAFactoryError("%s SHA-256 mismatch" % name)
    return path


def _load_map(
    reference: Mapping[str, Any],
    *,
    base_dir: Path,
    key: str,
    name: str,
) -> Tuple[np.ndarray, Path]:
    path = _resolved_file(reference, base_dir, name)
    with np.load(path, allow_pickle=False) as payload:
        if key not in payload.files:
            raise QAFactoryError("%s is missing array key %r" % (name, key))
        value = np.asarray(payload[key], dtype=np.float32)
    if not np.all(np.isfinite(value)):
        raise QAFactoryError("%s must contain finite values" % name)
    if np.any(value < 0.0) or np.any(value > 1.0):
        raise QAFactoryError("%s must be normalized to [0, 1]" % name)
    declared_shape = tuple(int(item) for item in reference.get("shape", []))
    if declared_shape != value.shape:
        raise QAFactoryError(
            "%s declared shape %s does not match %s"
            % (name, declared_shape, value.shape)
        )
    return value, path


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def find_forbidden_model_inputs(
    value: Any,
    forbidden_keys: Sequence[str],
    *,
    prefix: str = "model_input",
) -> List[str]:
    forbidden = {_normalized_key(item) for item in forbidden_keys}
    findings: List[str] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized = _normalized_key(key)
                if normalized in forbidden or any(
                    normalized.startswith(token + "_") for token in forbidden
                ):
                    findings.append(path + "." + str(key))
                visit(child, path + "." + str(key))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, "%s[%d]" % (path, index))

    visit(value, prefix)
    return findings


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") not in QA_FACTORY_CONFIG_SCHEMAS:
        raise QAFactoryError("unsupported QA factory config schema")
    camera_order = tuple(config.get("camera_order", []))
    if len(camera_order) != 6 or len(set(camera_order)) != 6:
        raise QAFactoryError("camera_order must contain six unique views")
    if tuple(config.get("question_families", [])) != QUESTION_FAMILIES:
        raise QAFactoryError("question family order is part of the v1 contract")
    medium = float(config["level_thresholds"]["medium"])
    high = float(config["level_thresholds"]["high"])
    caution = float(config["planning_stance_thresholds"]["caution"])
    prepare = float(config["planning_stance_thresholds"]["prepare_to_yield"])
    if not 0.0 < medium < high < 1.0:
        raise QAFactoryError("level thresholds must increase inside (0, 1)")
    if not 0.0 < caution < prepare < 1.0:
        raise QAFactoryError("planning thresholds must increase inside (0, 1)")
    if config.get("rearward_high_risk_stance_cap") != "caution":
        raise QAFactoryError(
            "rearward_high_risk_stance_cap must be caution; Stage2-L must not "
            "teach generic yielding for rearward conflicts"
        )
    if config.get("schema") == QA_FACTORY_CONFIG_SCHEMAS[1]:
        policy = config.get("supervision_policy", {})
        if tuple(policy.get("hard_stance_variants", [])) != (
            "zero_uq", "off_path_uq", "on_path_uq"
        ):
            raise QAFactoryError("v2 hard stance variants violate the frozen contract")
        expected_exclusions = {
            ("observed", "driving_implication"),
            ("view_shuffled_uq", "driving_implication"),
        }
        exclusions = {
            (str(row.get("variant")), str(row.get("question_family")))
            for row in policy.get("hard_language_exclusions", [])
        }
        if exclusions != expected_exclusions:
            raise QAFactoryError("v2 hard language exclusions violate the frozen contract")
        optimizer_group = policy.get("optimizer_group", {})
        if (
            int(optimizer_group.get("records", -1)) != 20
            or int(optimizer_group.get("optimizer_steps_inside_group", -1)) != 0
            or int(optimizer_group.get("optimizer_steps_after_group", -1)) != 1
        ):
            raise QAFactoryError("v2 optimizer-group contract is malformed")


def loss_policy_for_record(
    variant: str, family: str, config: Mapping[str, Any]
) -> Dict[str, Any]:
    """Materialize immutable per-record loss eligibility for QA factory v2."""

    _validate_config(config)
    if variant not in ALLOWED_COUNTERFACTUAL_VARIANTS:
        raise QAFactoryError("unsupported counterfactual variant")
    if family not in QUESTION_FAMILIES:
        raise QAFactoryError("unsupported question family")
    if config.get("schema") == QA_FACTORY_CONFIG_SCHEMAS[0]:
        hard_language = True
        hard_stance = family == "driving_implication"
        contract = "legacy_v1_all_rendered_answers"
    else:
        exclusions = {
            (str(row["variant"]), str(row["question_family"]))
            for row in config["supervision_policy"]["hard_language_exclusions"]
        }
        hard_variants = set(config["supervision_policy"]["hard_stance_variants"])
        hard_language = (variant, family) not in exclusions
        hard_stance = family == "driving_implication" and variant in hard_variants
        contract = "matched_magnitude_cross_family_v2"
    return {
        "contract": contract,
        "hard_language_target": hard_language,
        "hard_stance_target": hard_stance,
        "dense_relevance_target": True,
        "cross_family_preference_anchor": hard_language,
        "optimizer_group_complete_before_step": True,
    }


def validate_frame_bundle(
    bundle: Mapping[str, Any],
    *,
    bundle_path: Path,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    _validate_config(config)
    if bundle.get("schema") != FRAME_BUNDLE_SCHEMA:
        raise QAFactoryError("unsupported frame-bundle schema")
    split = bundle.get("split")
    if split not in ALLOWED_SPLITS:
        raise QAFactoryError("split must be train, dev or test")
    for name in ("event_id", "frame_id", "town", "scenario_family"):
        _require_text(bundle.get(name), name)
    route = bundle.get("route") or {}
    _require_text(route.get("route_id"), "route.route_id")
    counterfactual = bundle.get("counterfactual") or {}
    _require_text(counterfactual.get("group_id"), "counterfactual.group_id")
    variant = counterfactual.get("variant")
    if variant not in ALLOWED_COUNTERFACTUAL_VARIANTS:
        raise QAFactoryError("unsupported counterfactual variant %r" % variant)

    model_input = bundle.get("model_input") or {}
    findings = find_forbidden_model_inputs(
        model_input, config["forbidden_model_input_keys"]
    )
    if findings:
        raise QAFactoryError(
            "forbidden information leaked into model input: %s"
            % ", ".join(findings)
        )
    observation = model_input.get("observation") or {}
    camera_files = observation.get("camera_files") or []
    camera_order = tuple(config["camera_order"])
    if tuple(item.get("view") for item in camera_files) != camera_order:
        raise QAFactoryError("camera files must exactly follow configured camera_order")
    camera_hashes = []
    for index, reference in enumerate(camera_files):
        path = _resolved_file(
            reference,
            bundle_path.parent,
            "camera_files[%d]" % index,
        )
        camera_hashes.append(sha256_file(path))
    declared_observation_sha = _require_sha(
        observation.get("observation_sha256"),
        "model_input.observation.observation_sha256",
    )
    computed_observation_sha = hashlib.sha256(
        "\n".join(camera_hashes).encode("ascii")
    ).hexdigest()
    if declared_observation_sha != computed_observation_sha:
        raise QAFactoryError("observation_sha256 does not match ordered camera files")
    route_context = model_input.get("route_context") or {}
    route_context_sha = _require_sha(
        route_context.get("sha256"), "model_input.route_context.sha256"
    )
    route_payload = route_context.get("payload")
    route_schema = route_context.get("schema")
    if route_schema is not None and route_schema != "orion.route_context.v2":
        raise QAFactoryError("unsupported route_context schema")
    if route_schema == "orion.route_context.v2":
        if not isinstance(route_payload, Mapping):
            raise QAFactoryError("route_context.v2 payload must be an object")
        ego_state = route_payload.get("ego_state")
        if not isinstance(ego_state, Mapping) or set(ego_state) != {
            "speedometer_mps"
        }:
            raise QAFactoryError(
                "route_context.v2 requires only the current speedometer reading"
            )
        speedometer_mps = float(ego_state["speedometer_mps"])
        if not math.isfinite(speedometer_mps):
            raise QAFactoryError(
                "route_context.v2 ego speedometer reading is invalid"
            )
    encoded_route = json.dumps(
        route_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if hashlib.sha256(encoded_route).hexdigest() != route_context_sha:
        raise QAFactoryError("route_context.sha256 does not match its payload")

    uq, uq_path = _load_map(
        model_input.get("stage1_observation_uq") or {},
        base_dir=bundle_path.parent,
        key="uncertainty",
        name="stage1_observation_uq",
    )
    if uq.ndim == 3:
        uq = uq[None, ...]
    if uq.ndim != 4:
        raise QAFactoryError("Stage-1 UQ must have shape [V,H,W] or [T,V,H,W]")
    if uq.shape[1] != len(camera_order):
        raise QAFactoryError("Stage-1 UQ view count does not match camera_order")
    stage1 = model_input["stage1_observation_uq"]
    if stage1.get("source") not in (
        "frozen_stage1_observation_adapter",
        "controlled_stage1_uq_counterfactual",
    ):
        raise QAFactoryError(
            "Stage-1 UQ source must be a frozen adapter output or a controlled UQ counterfactual"
        )
    _require_sha(stage1.get("checkpoint_sha256"), "stage1 checkpoint SHA-256")
    if stage1.get("control_influence") is not False:
        raise QAFactoryError("route-screening Stage-1 UQ must have no control influence")
    components = None
    if stage1.get("component_key") is not None:
        component_reference = dict(stage1)
        component_reference["shape"] = stage1.get("component_shape")
        components, _ = _load_map(
            component_reference,
            base_dir=bundle_path.parent,
            key=stage1["component_key"],
            name="stage1_observation_uq_components",
        )
        if components.ndim != 5 or components.shape[:-1] != uq.shape:
            raise QAFactoryError(
                "Stage-1 component maps must have shape [T,V,H,W,C] matching scalar UQ"
            )
        component_names = stage1.get("component_names") or []
        if len(component_names) != components.shape[-1] or len(set(component_names)) != len(component_names):
            raise QAFactoryError("Stage-1 component names do not match component maps")
        if not np.allclose(uq, components.mean(axis=-1), atol=1e-5):
            raise QAFactoryError("scalar Stage-1 UQ must be the mean of normalized components")

    target = bundle.get("supervision") or {}
    relevance, relevance_path = _load_map(
        target.get("task_relevance") or {},
        base_dir=bundle_path.parent,
        key="relevance",
        name="task_relevance",
    )
    if relevance.ndim != 3 or relevance.shape != uq.shape[1:]:
        raise QAFactoryError("task relevance must match UQ [V,H,W]")
    source = target["task_relevance"].get("source")
    if source not in (
        "projected_actor_route_corridor_geometry_v1",
        "matched_counterfactual_route_geometry_v1",
        "human_audited_geometry_v1",
    ):
        raise QAFactoryError("task relevance supervision source is not allowed")
    if target["task_relevance"].get("uses_corruption_label") is not False:
        raise QAFactoryError("task relevance target must attest no corruption labels")

    return {
        "uq": uq,
        "relevance": relevance,
        "uq_path": uq_path,
        "relevance_path": relevance_path,
        "components": components,
        "observation_sha256": declared_observation_sha,
        "route_context_sha256": route_context_sha,
    }


def _level(value: float, config: Mapping[str, Any]) -> str:
    thresholds = config["level_thresholds"]
    if value >= float(thresholds["high"]):
        return "high"
    if value >= float(thresholds["medium"]):
        return "medium"
    return "low"


def _planning_stance(
    value: float, peak_view: str, config: Mapping[str, Any]
) -> str:
    thresholds = config["planning_stance_thresholds"]
    if value >= float(thresholds["prepare_to_yield"]):
        if peak_view in {"CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"}:
            return str(config["rearward_high_risk_stance_cap"])
        return "prepare_to_yield"
    if value >= float(thresholds["caution"]):
        return "caution"
    return "maintain"


def _region(row: int, column: int, height: int, width: int) -> str:
    row_name = ("upper", "middle", "lower")[min(2, (3 * row) // height)]
    column_name = ("left", "center", "right")[min(2, (3 * column) // width)]
    return row_name + "_" + column_name


def summarize_maps(
    uq: np.ndarray,
    relevance: np.ndarray,
    *,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    latest = uq[-1]
    task_risk = latest * relevance
    uq_index = np.unravel_index(int(np.argmax(latest)), latest.shape)
    risk_index = np.unravel_index(int(np.argmax(task_risk)), task_risk.shape)
    uq_value = float(latest[uq_index])
    relevance_at_uq = float(relevance[uq_index])
    risk_value = float(task_risk[risk_index])
    temporal_peak_region = uq[:, uq_index[0], uq_index[1], uq_index[2]]
    if uq.shape[0] < 2:
        trend = "unknown"
        temporal_delta = None
    else:
        temporal_delta = float(temporal_peak_region[-1] - temporal_peak_region[0])
        trend = (
            "rising"
            if temporal_delta > 0.05
            else "falling"
            if temporal_delta < -0.05
            else "stable"
        )
    views = config["camera_order"]
    return {
        "observation_uncertainty": {
            "level": _level(uq_value, config),
            "peak_score": uq_value,
            "peak_view": views[uq_index[0]],
            "peak_region": _region(uq_index[1], uq_index[2], latest.shape[1], latest.shape[2]),
            "temporal_trend": trend,
            "temporal_peak_region_delta": temporal_delta,
            "temporal_summary_scope": "latest_peak_patch_across_time",
        },
        "relevance_at_most_uncertain_region": {
            "level": _level(relevance_at_uq, config),
            "score": relevance_at_uq,
        },
        "task_risk": {
            "level": _level(risk_value, config),
            "peak_score": risk_value,
            "peak_view": views[risk_index[0]],
            "peak_region": _region(risk_index[1], risk_index[2], latest.shape[1], latest.shape[2]),
        },
        "planning_implication": {
            "stance": _planning_stance(risk_value, views[risk_index[0]], config),
            "risk_bearing": (
                "rearward"
                if views[risk_index[0]] in {
                    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
                }
                else "forward_or_crossing"
            ),
            "is_direct_control_command": False,
        },
        "task_risk_map": task_risk,
    }


def render_answer(family: str, summary: Mapping[str, Any]) -> str:
    observation = summary["observation_uncertainty"]
    relevance = summary["relevance_at_most_uncertain_region"]
    risk = summary["task_risk"]
    stance = summary["planning_implication"]["stance"]
    if family == "observation_semantics":
        return (
            "Observation uncertainty is {level}; its strongest region is "
            "{view}/{region}, and its temporal trend is {trend}."
        ).format(
            level=observation["level"],
            view=observation["peak_view"],
            region=observation["peak_region"],
            trend=observation["temporal_trend"],
        )
    if family == "epistemic_limitation":
        return (
            "The uncertainty map identifies unreliable visual evidence at "
            "{view}/{region}; it does not reveal which hidden object or fact is "
            "missing. Task relevance must be inferred separately."
        ).format(view=observation["peak_view"], region=observation["peak_region"])
    if family == "task_relevance":
        return (
            "<task_relevance_map> The most uncertain region has {level} task "
            "relevance. The strongest combined task risk is {risk_level} at "
            "{view}/{region}."
        ).format(
            level=relevance["level"],
            risk_level=risk["level"],
            view=risk["peak_view"],
            region=risk["peak_region"],
        )
    if family == "driving_implication":
        return (
            "<task_relevance_map> The uncertainty-aware planning stance is "
            "{stance}. This is a planning implication, not a direct brake or "
            "steering command."
        ).format(stance=stance)
    raise QAFactoryError("unsupported question family %r" % family)


def question_for_family(family: str) -> str:
    questions = {
        "observation_semantics": (
            "Where is the current visual observation uncertain, how strong is it, "
            "and how is it changing over time?"
        ),
        "epistemic_limitation": (
            "What does the observation-uncertainty signal tell you, and what can it "
            "not tell you?"
        ),
        "task_relevance": (
            "Given the route context and spatial observation uncertainty, predict "
            "the task-relevance map and summarize which uncertain region matters."
        ),
        "driving_implication": (
            "How should the task-relevant observation uncertainty influence the "
            "planning stance?"
        ),
    }
    try:
        return questions[family]
    except KeyError as error:
        raise QAFactoryError("unsupported question family %r" % family) from error


def build_records_for_bundle(
    bundle: Mapping[str, Any],
    *,
    bundle_path: Path,
    config: Mapping[str, Any],
    sidecar_relative_path: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], np.ndarray]:
    validated = validate_frame_bundle(
        bundle, bundle_path=bundle_path, config=config
    )
    summary = summarize_maps(
        validated["uq"], validated["relevance"], config=config
    )
    task_risk = summary.pop("task_risk_map")
    core_id = "%s/%s/%s" % (
        bundle["event_id"],
        bundle["frame_id"],
        bundle["counterfactual"]["variant"],
    )
    records = []
    for family in QUESTION_FAMILIES:
        answer = render_answer(family, summary)
        records.append({
            "schema": QA_RECORD_SCHEMA,
            "sample_id": core_id + "/" + family,
            "split": bundle["split"],
            "event_id": bundle["event_id"],
            "frame_id": bundle["frame_id"],
            "town": bundle["town"],
            "scenario_family": bundle["scenario_family"],
            "route_id": bundle["route"]["route_id"],
            "counterfactual": dict(bundle["counterfactual"]),
            "question_family": family,
            "loss_policy": loss_policy_for_record(
                str(bundle["counterfactual"]["variant"]), family, config
            ),
            "model_input": bundle["model_input"],
            "conversation": [
                {"from": "human", "value": question_for_family(family)},
                {"from": "gpt", "value": answer},
            ],
            "target": {
                "structured_summary": summary,
                "map_sidecar": {
                    "schema": MAP_SIDECAR_SCHEMA,
                    "path": sidecar_relative_path,
                    "relevance_key": "task_relevance",
                    "task_risk_key": "task_risk",
                    "dense_relevance_is_authoritative": True,
                },
                "rendered_answer": answer,
            },
            "provenance": {
                "frame_bundle_path": str(bundle_path.resolve()),
                "frame_bundle_sha256": sha256_file(bundle_path),
                "relevance_supervision": bundle["supervision"]["task_relevance"],
            },
        })
    sidecar = {
        "schema": MAP_SIDECAR_SCHEMA,
        "shape": list(validated["relevance"].shape),
        "camera_order": list(config["camera_order"]),
        "uncertainty_definition": "normalized frozen Stage-1 observation UQ",
        "relevance_definition": "task relevance independent of UQ magnitude",
        "task_risk_definition": "latest observation UQ multiplied by relevance",
    }
    arrays = np.stack((validated["relevance"], task_risk), axis=0)
    return records, sidecar, arrays


def audit_dataset(
    dataset: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    dataset_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    _validate_config(config)
    records = dataset.get("records", [])
    if dataset.get("schema") != QA_DATASET_SCHEMA:
        raise QAFactoryError("unsupported QA dataset schema")
    if not records:
        raise QAFactoryError("QA dataset is empty")
    sample_ids = [record.get("sample_id") for record in records]
    map_text_mismatches = []
    map_sidecar_mismatches = []
    forbidden = []
    validated_map_samples = set()
    missing_component_maps = []
    loss_policy_mismatches = []
    for record in records:
        if record.get("schema") != QA_RECORD_SCHEMA:
            raise QAFactoryError("dataset contains unsupported record schema")
        family = record.get("question_family")
        variant = str(record.get("counterfactual", {}).get("variant", ""))
        expected_loss_policy = loss_policy_for_record(variant, family, config)
        if record.get("loss_policy") != expected_loss_policy:
            loss_policy_mismatches.append(record.get("sample_id"))
        expected = render_answer(family, record["target"]["structured_summary"])
        actual = record["conversation"][1]["value"]
        if actual != expected or record["target"]["rendered_answer"] != expected:
            map_text_mismatches.append(record["sample_id"])
        paths = find_forbidden_model_inputs(
            record["model_input"], config["forbidden_model_input_keys"]
        )
        forbidden.extend(record["sample_id"] + ":" + path for path in paths)
        if record["model_input"]["stage1_observation_uq"].get("component_key") is None:
            missing_component_maps.append(record["sample_id"])
        if dataset_dir is not None:
            sidecar = record["target"]["map_sidecar"]
            sidecar_id = (sidecar.get("path"), sidecar.get("sha256"))
            if sidecar_id not in validated_map_samples:
                validated_map_samples.add(sidecar_id)
                try:
                    sidecar_path = _resolved_file(
                        sidecar,
                        dataset_dir,
                        "target.map_sidecar",
                    )
                    with np.load(sidecar_path, allow_pickle=False) as payload:
                        relevance = np.asarray(
                            payload[sidecar["relevance_key"]], dtype=np.float32
                        )
                        task_risk = np.asarray(
                            payload[sidecar["task_risk_key"]], dtype=np.float32
                        )
                    bundle_path = Path(record["provenance"]["frame_bundle_path"])
                    uq, _ = _load_map(
                        record["model_input"]["stage1_observation_uq"],
                        base_dir=bundle_path.parent,
                        key="uncertainty",
                        name="stage1_observation_uq",
                    )
                    latest = uq if uq.ndim == 3 else uq[-1]
                    if relevance.shape != latest.shape or task_risk.shape != latest.shape:
                        raise QAFactoryError("map sidecar shapes do not match Stage-1 UQ")
                    if not np.allclose(task_risk, latest * relevance, atol=1e-6):
                        raise QAFactoryError("task-risk sidecar is not K=U_latest*R")
                except (KeyError, OSError, QAFactoryError, ValueError) as error:
                    map_sidecar_mismatches.append(
                        "%s:%s" % (record["sample_id"], error)
                    )

    split_sets: Dict[str, Dict[str, set]] = {
        split: {"routes": set(), "events": set(), "groups": set()}
        for split in ALLOWED_SPLITS
    }
    for record in records:
        split = record["split"]
        split_sets[split]["routes"].add(record["route_id"])
        split_sets[split]["events"].add(record["event_id"])
        split_sets[split]["groups"].add(record["counterfactual"]["group_id"])

    variant_families: Dict[Tuple[str, str, str, str], set] = {}
    counterfactual_groups: Dict[str, Dict[str, Any]] = {}
    for record in records:
        counterfactual = record["counterfactual"]
        variant_key = (
            record["split"],
            record["event_id"],
            record["frame_id"],
            counterfactual["variant"],
        )
        variant_families.setdefault(variant_key, set()).add(
            record["question_family"]
        )
        group = counterfactual_groups.setdefault(
            counterfactual["group_id"],
            {"variants": set(), "observation_hashes": set(), "route_hashes": set(), "uq_hashes": {}},
        )
        group["variants"].add(counterfactual["variant"])
        group["observation_hashes"].add(
            record["model_input"]["observation"]["observation_sha256"]
        )
        group["route_hashes"].add(
            record["model_input"]["route_context"]["sha256"]
        )
        group["uq_hashes"].setdefault(
            counterfactual["variant"],
            record["model_input"]["stage1_observation_uq"]["sha256"],
        )
    incomplete_question_families = [
        "/".join(key)
        for key, families in variant_families.items()
        if families != set(QUESTION_FAMILIES)
    ]
    inconsistent_counterfactual_groups = []
    matched_counterfactual_groups = 0
    for group_id, group in counterfactual_groups.items():
        variants = group["variants"]
        matched = "off_path_uq" in variants and bool(
            {"on_path_uq", "observed"} & variants
        )
        matched_counterfactual_groups += int(matched)
        uq_hashes = [group["uq_hashes"][variant] for variant in variants]
        if (
            len(group["observation_hashes"]) != 1
            or len(group["route_hashes"]) != 1
            or (len(variants) > 1 and len(set(uq_hashes)) != len(variants))
        ):
            inconsistent_counterfactual_groups.append(group_id)

    def overlaps(kind: str) -> List[str]:
        collisions = []
        for left_index, left in enumerate(ALLOWED_SPLITS):
            for right in ALLOWED_SPLITS[left_index + 1:]:
                shared = split_sets[left][kind] & split_sets[right][kind]
                collisions.extend("%s:%s:%s" % (left, right, value) for value in sorted(shared))
        return collisions

    event_rows = {}
    for record in records:
        key = (record["split"], record["event_id"])
        event_rows.setdefault(key, record)
    independent = list(event_rows.values())
    gates = config["formal_training_gates"]
    counts = {
        "records": len(records),
        "independent_events": len({row["event_id"] for row in independent}),
        "towns": len({row["town"] for row in independent}),
        "scenario_families": len({row["scenario_family"] for row in independent}),
        "splits_nonempty": sum(bool(split_sets[split]["events"]) for split in ALLOWED_SPLITS),
        "matched_counterfactual_groups": matched_counterfactual_groups,
    }
    route_overlap = overlaps("routes")
    event_overlap = overlaps("events")
    group_overlap = overlaps("groups")
    checks = {
        "unique_sample_ids": len(sample_ids) == len(set(sample_ids)),
        "map_text_consistent": not map_text_mismatches,
        "map_sidecars_valid": not map_sidecar_mismatches,
        "stage1_component_maps_present": not missing_component_maps,
        "no_forbidden_model_inputs": not forbidden,
        "loss_policy_consistent": not loss_policy_mismatches,
        "route_disjoint_splits": not route_overlap,
        "event_disjoint_splits": not event_overlap,
        "counterfactual_groups_split_intact": not group_overlap,
        "counterfactual_groups_input_matched": not inconsistent_counterfactual_groups,
        "all_question_families_per_variant": not incomplete_question_families,
        "all_splits_nonempty": counts["splits_nonempty"] == len(ALLOWED_SPLITS),
        "minimum_independent_events": counts["independent_events"] >= int(gates["minimum_independent_events"]),
        "minimum_towns": counts["towns"] >= int(gates["minimum_towns"]),
        "minimum_scenario_families": counts["scenario_families"] >= int(gates["minimum_scenario_families"]),
        "minimum_matched_counterfactual_groups": counts["matched_counterfactual_groups"] >= int(gates["minimum_matched_counterfactual_groups"]),
    }
    return {
        "schema": "orion.uq_relevance_qa_audit.v1",
        "counts": counts,
        "checks": checks,
        "formal_training_ready": all(checks.values()),
        "findings": {
            "duplicate_sample_ids": sorted(
                {item for item in sample_ids if sample_ids.count(item) > 1}
            ),
            "map_text_mismatches": map_text_mismatches,
            "map_sidecar_mismatches": map_sidecar_mismatches,
            "missing_stage1_component_maps": missing_component_maps,
            "loss_policy_mismatches": loss_policy_mismatches,
            "forbidden_model_inputs": forbidden,
            "route_split_overlap": route_overlap,
            "event_split_overlap": event_overlap,
            "counterfactual_group_split_overlap": group_overlap,
            "incomplete_question_families": incomplete_question_families,
            "inconsistent_counterfactual_groups": inconsistent_counterfactual_groups,
        },
    }


__all__ = [
    "ALLOWED_COUNTERFACTUAL_VARIANTS",
    "ALLOWED_SPLITS",
    "FRAME_BUNDLE_SCHEMA",
    "MAP_SIDECAR_SCHEMA",
    "QA_DATASET_SCHEMA",
    "QAFactoryError",
    "audit_dataset",
    "build_records_for_bundle",
    "find_forbidden_model_inputs",
    "render_answer",
    "sha256_file",
    "summarize_maps",
    "validate_frame_bundle",
]
