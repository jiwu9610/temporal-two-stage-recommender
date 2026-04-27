"""Phase 0 data layer."""

from .canonicalize import canonicalize
from .download import download_category
from .feature_store import build_item_features, build_user_features
from .filtering import iterative_kcore
from .loader import load_metadata, load_sampled, stream_stats
from .preprocessing import (
    compute_bought_together_hit_rate,
    cross_category_users,
    filter_by_counts,
)
from .preprocessing_pipeline import run_category
from .splitting import leave_last_two_split
from .text_alignment import align_text_embeddings

__all__ = [
    "canonicalize",
    "iterative_kcore",
    "leave_last_two_split",
    "build_user_features",
    "build_item_features",
    "align_text_embeddings",
    "run_category",
    "load_sampled",
    "load_metadata",
    "stream_stats",
    "download_category",
    "filter_by_counts",
    "compute_bought_together_hit_rate",
    "cross_category_users",
]
