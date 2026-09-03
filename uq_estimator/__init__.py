from uq_estimator.model import UQEstimator, UQOutput
from uq_estimator.density import DensityUQEstimator, compute_view_moments
from uq_estimator.token_projector import UQTokenProjector
from uq_estimator.vision_adapter import UQVisionAdapter
from uq_estimator.grounding import UQGroundingHead, grounding_loss
from uq_estimator.corruptions import (
    BatchCorruptionResultV1,
    CORRUPTION_METADATA_SCHEMA_V1,
    CorruptionMetadataV1,
    CorruptionResultV1,
    corrupt_batch_images,
    corrupt_batch_images_with_metadata,
    corrupt_multiview_images,
    corrupt_multiview_images_with_metadata,
)
from uq_estimator.spatial_uq import (
    EnsembleVarianceDecomposition,
    PathRiskAggregation,
    SpatialPatchUQHead,
    SpatialUQOutput,
    brier_loss,
    cvar_path_risk,
    decompose_ensemble_variance,
    heteroscedastic_gaussian_nll,
    paired_cosine_representation_error,
    paired_error_ranking_loss,
)
from uq_estimator.path_projection import (
    ProjectedPathCorridor,
    project_path_corridor_to_patches,
)
from uq_estimator.trajectory_adapter import (
    PathRiskTrajectoryAdapter,
    TrajectoryAdapterOutput,
    trajectory_adapter_loss,
)
from uq_estimator.spatial_metrics import (
    BinarySpatialMetrics,
    TemporalEventMetrics,
    area_under_risk_coverage,
    binary_spatial_metrics,
    spearman_correlation,
    temporal_event_metrics,
)
from uq_estimator.counterfactual_regions import (
    MatchedCounterfactualRegions,
    select_matched_counterfactual_regions,
)
from uq_estimator.risk_governor import (
    RiskDecision,
    UQRiskGovernor,
    load_score_trace,
)
from uq_estimator.risk_qa import (
    RISK_QA_QUESTION,
    RELIABILITY_QA_QUESTION,
    RiskQAAnswer,
    build_risk_qa_answer,
    mask_to_final_supervised_span,
    parse_risk_qa_answer,
    parse_natural_risk_qa_answer,
    parse_reliability_answer,
    parse_risk_synthesis_answer,
    render_natural_risk_qa_answer,
    render_reliability_answer,
    render_critical_object_context,
    render_risk_synthesis_answer,
    render_risk_qa_answer,
    select_balanced_sample_ids,
    select_critical_objects,
)
from uq_estimator.losses import CombinedUQLoss
from uq_estimator.dataset import UQFeatureDataset
from uq_estimator.bev_uncertainty import (
    compute_patch_quality,
    compute_bev_uncertainty,
    compute_bev_uncertainty_ipm,
    make_b2d_calibration,
    compute_trajectory_cost,
    adjust_mode_scores,
    render_bev_heatmap,
)

__all__ = [
    "UQEstimator", "UQOutput", "DensityUQEstimator", "compute_view_moments",
    "UQTokenProjector",
    "UQVisionAdapter",
    "UQGroundingHead", "grounding_loss",
    "CORRUPTION_METADATA_SCHEMA_V1",
    "CorruptionMetadataV1", "CorruptionResultV1", "BatchCorruptionResultV1",
    "corrupt_batch_images", "corrupt_multiview_images",
    "corrupt_batch_images_with_metadata",
    "corrupt_multiview_images_with_metadata",
    "SpatialUQOutput", "SpatialPatchUQHead",
    "paired_cosine_representation_error",
    "heteroscedastic_gaussian_nll", "brier_loss",
    "paired_error_ranking_loss", "EnsembleVarianceDecomposition",
    "decompose_ensemble_variance", "PathRiskAggregation", "cvar_path_risk",
    "ProjectedPathCorridor", "project_path_corridor_to_patches",
    "TrajectoryAdapterOutput", "PathRiskTrajectoryAdapter",
    "trajectory_adapter_loss",
    "BinarySpatialMetrics", "TemporalEventMetrics",
    "area_under_risk_coverage", "binary_spatial_metrics",
    "spearman_correlation", "temporal_event_metrics",
    "MatchedCounterfactualRegions",
    "select_matched_counterfactual_regions",
    "RiskDecision", "UQRiskGovernor", "load_score_trace",
    "RISK_QA_QUESTION", "RELIABILITY_QA_QUESTION", "RiskQAAnswer",
    "build_risk_qa_answer",
    "mask_to_final_supervised_span",
    "parse_risk_qa_answer", "render_risk_qa_answer",
    "parse_natural_risk_qa_answer", "render_natural_risk_qa_answer",
    "parse_reliability_answer", "render_reliability_answer",
    "parse_risk_synthesis_answer", "render_critical_object_context",
    "render_risk_synthesis_answer",
    "select_balanced_sample_ids",
    "select_critical_objects",
    "CombinedUQLoss", "UQFeatureDataset",
    "compute_patch_quality", "compute_bev_uncertainty",
    "compute_bev_uncertainty_ipm", "make_b2d_calibration",
    "compute_trajectory_cost", "adjust_mode_scores", "render_bev_heatmap",
]
