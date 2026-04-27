"""Phase 1 retrieval modules (spec, see memory/project_phase1_plan.md)."""

from .evaluator import (
    SplitReport,
    build_groundtruth,
    build_split_report,
    coverage_of_pool,
    recall_precision_at_k,
)
from .popularity import (
    compute_popularity,
    recommend_popularity,
    user_seen_from_train,
)
from .random_baseline import recommend_random
from .rule_based import (
    TuningResult,
    build_user_facet_affinity,
    normalized_popularity_prior,
    recommend_rule_based,
    tune_weights,
)

__all__ = [
    "SplitReport",
    "build_groundtruth",
    "build_split_report",
    "coverage_of_pool",
    "recall_precision_at_k",
    "compute_popularity",
    "recommend_popularity",
    "user_seen_from_train",
    "recommend_random",
    "TuningResult",
    "build_user_facet_affinity",
    "normalized_popularity_prior",
    "recommend_rule_based",
    "tune_weights",
]
