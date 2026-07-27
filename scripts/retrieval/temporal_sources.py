"""Point-in-time retrieval sources for Phase 3A.

Additional candidate sources computed STRICTLY from snapshot-visible history
(every builder raises if any input row sits at/after the snapshot cutoff):

  * windowed_popularity     positives in the trailing N days before the cutoff
  * decayed_popularity      sum of 0.5^(age_days / half_life) over positives
  * co-visitation           item-to-item co-occurrence within users' visible
                            positive histories; per-user retrieval seeds from
                            their most recent positives

Determinism contract (same as the rest of the temporal stack):
  * item scores tie-break alphabetically by parent_asin;
  * co-visitation neighbor lists and seed selection are fully ordered
    (score desc, item asc / recency desc, item asc);
  * no dependence on catalog row order.

Score-based recommendation reuses scripts.retrieval.popularity's
recommend_popularity, which accepts any per-item score Series.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np
import pandas as pd

MS_PER_DAY = 86_400_000


def _guard_no_future(history_df: pd.DataFrame, as_of_ms: int, time_col: str) -> None:
    if len(history_df) and (history_df[time_col] >= as_of_ms).any():
        raise ValueError(
            "history contains events at/after as_of_ms -- snapshot history "
            "must be strictly before the cutoff (temporal leakage guard)"
        )


def windowed_popularity(
    history_df: pd.DataFrame,
    as_of_ms: int,
    window_days: int,
    item_col: str = "parent_asin",
    time_col: str = "timestamp",
) -> pd.Series:
    """Positive-interaction counts within [as_of - window, as_of)."""
    _guard_no_future(history_df, as_of_ms, time_col)
    start = as_of_ms - window_days * MS_PER_DAY
    recent = history_df[
        (history_df[time_col] >= start) & (history_df["label"] == 1)
    ]
    counts = recent[item_col].value_counts()
    counts.index.name = item_col
    counts.name = f"popularity_last_{window_days}d"
    return counts


def decayed_popularity(
    history_df: pd.DataFrame,
    as_of_ms: int,
    half_life_days: float,
    item_col: str = "parent_asin",
    time_col: str = "timestamp",
) -> pd.Series:
    """Exponentially decayed positive counts: sum 0.5^(age_days/half_life)."""
    _guard_no_future(history_df, as_of_ms, time_col)
    pos = history_df[history_df["label"] == 1]
    age_days = (as_of_ms - pos[time_col].to_numpy(np.float64)) / MS_PER_DAY
    w = np.power(0.5, age_days / float(half_life_days))
    scores = pd.Series(w, index=pos[item_col].to_numpy()).groupby(level=0).sum()
    scores = scores.sort_values(ascending=False)
    scores.index.name = item_col
    scores.name = f"decayed_popularity_hl{half_life_days:g}"
    return scores


# ---------------------------------------------------------------------------
# Co-visitation
# ---------------------------------------------------------------------------

def build_covisitation(
    history_df: pd.DataFrame,
    as_of_ms: int,
    max_basket_items: int = 30,
    min_support: int = 2,
    normalization: str = "cosine",          # "cosine" | "count"
    max_neighbors_store: int = 100,
    user_chunk: int = 20_000,
    item_col: str = "parent_asin",
    user_col: str = "user_id",
    time_col: str = "timestamp",
) -> Dict[str, List[Tuple[str, float]]]:
    """Item -> ordered neighbor list from co-occurrence in visible positive
    histories.

    Baskets = each user's most recent `max_basket_items` POSITIVE items before
    the cutoff (cap bounds the pairwise blow-up on heavy users). Support is
    the raw co-occurrence count across baskets; pairs below `min_support` are
    dropped. `cosine` normalization divides by sqrt(freq_i * freq_j) to damp
    globally popular items; `count` keeps raw supports.

    Neighbor lists are sorted (score desc, item asc) and truncated to
    `max_neighbors_store`. Fully deterministic.
    """
    _guard_no_future(history_df, as_of_ms, time_col)
    if normalization not in ("cosine", "count"):
        raise ValueError(f"unknown normalization {normalization!r}")

    pos = history_df[history_df["label"] == 1][[user_col, item_col, time_col]].copy()
    pos[user_col] = pos[user_col].astype(str)
    pos[item_col] = pos[item_col].astype(str)
    # Most recent max_basket_items per user; deterministic under ts ties.
    pos = pos.sort_values([user_col, time_col, item_col],
                          ascending=[True, False, True], kind="mergesort")
    baskets = pos.groupby(user_col).head(max_basket_items)
    # One row per (user, item): repeated purchases are impossible upstream
    # (dedup keep-earliest) but be defensive.
    baskets = baskets.drop_duplicates(subset=[user_col, item_col])

    item_freq = baskets[item_col].value_counts()

    users = baskets[user_col].unique()
    chunk_counts: List[pd.Series] = []
    for start in range(0, len(users), user_chunk):
        chunk_users = set(users[start:start + user_chunk])
        sub = baskets[baskets[user_col].isin(chunk_users)][[user_col, item_col]]
        pairs = sub.merge(sub, on=user_col, suffixes=("_a", "_b"))
        pairs = pairs[pairs[f"{item_col}_a"] != pairs[f"{item_col}_b"]]
        if len(pairs):
            chunk_counts.append(
                pairs.groupby([f"{item_col}_a", f"{item_col}_b"]).size()
            )
    if not chunk_counts:
        return {}
    support = pd.concat(chunk_counts).groupby(level=[0, 1]).sum()
    support = support[support >= min_support]
    if not len(support):
        return {}

    df = support.rename("support").reset_index()
    df.columns = ["item_a", "item_b", "support"]
    if normalization == "cosine":
        fa = df["item_a"].map(item_freq).to_numpy(np.float64)
        fb = df["item_b"].map(item_freq).to_numpy(np.float64)
        df["score"] = df["support"].to_numpy(np.float64) / np.sqrt(fa * fb)
    else:
        df["score"] = df["support"].astype(np.float64)

    df = df.sort_values(["item_a", "score", "item_b"],
                        ascending=[True, False, True], kind="mergesort")
    df = df.groupby("item_a").head(max_neighbors_store)

    neighbors: Dict[str, List[Tuple[str, float]]] = {}
    for item_a, g in df.groupby("item_a", sort=False):
        neighbors[item_a] = list(zip(g["item_b"], g["score"].astype(float)))
    return neighbors


def recommend_covisitation(
    history_df: pd.DataFrame,
    user_ids: Iterable[str],
    neighbors: Mapping[str, List[Tuple[str, float]]],
    user_seen: Mapping[str, Set[str]],
    candidate_pool: Optional[Set[str]] = None,
    max_seeds: int = 5,
    seed_decay: float = 0.8,
    max_neighbors_per_seed: int = 50,
    k: int = 100,
    item_col: str = "parent_asin",
    user_col: str = "user_id",
    time_col: str = "timestamp",
    return_scores: bool = False,
):
    """Per-user co-visitation top-K.

    Seeds = the user's most recent `max_seeds` POSITIVE history items
    (recency desc, item asc). Seed s at recency rank r contributes
    seed_decay^r * neighbor_score (seed_decay=1.0 disables recency
    weighting). Seen items are removed; ties break (score desc, item asc).
    Zero-history / no-positive users get an empty list -- the popularity
    source of the union is their fallback.
    """
    pos = history_df[history_df["label"] == 1][[user_col, item_col, time_col]].copy()
    pos[user_col] = pos[user_col].astype(str)
    pos[item_col] = pos[item_col].astype(str)
    pos = pos.sort_values([user_col, time_col, item_col],
                          ascending=[True, False, True], kind="mergesort")
    seeds_per_user: Dict[str, List[str]] = {
        u: g[item_col].tolist()[:max_seeds]
        for u, g in pos.groupby(user_col)
    }

    out: Dict[str, List[str]] = {}
    out_scores: Dict[str, List[float]] = {}
    for u in user_ids:
        seeds = seeds_per_user.get(u, [])
        if not seeds:
            out[u] = []
            out_scores[u] = []
            continue
        seen = user_seen.get(u, set())
        scores: Dict[str, float] = {}
        for r, seed in enumerate(seeds):
            w = seed_decay ** r
            for nbr, sc in neighbors.get(seed, [])[:max_neighbors_per_seed]:
                if nbr in seen:
                    continue
                if candidate_pool is not None and nbr not in candidate_pool:
                    continue
                scores[nbr] = scores.get(nbr, 0.0) + w * sc
        ranked = sorted(scores.items(), key=lambda t: (-t[1], t[0]))[:k]
        out[u] = [it for it, _ in ranked]
        out_scores[u] = [s for _, s in ranked]
    if return_scores:
        return out, out_scores
    return out


# ---------------------------------------------------------------------------
# Target-status classification (cold vs low-support vs eligible-not-retrieved)
# ---------------------------------------------------------------------------

def classify_targets(
    gt_df: pd.DataFrame,
    hist_item_counts: pd.Series,
    eligible_items: Set[str],
    retrieved_per_user: Mapping[str, Set[str]],
    item_col: str = "parent_asin",
    user_col: str = "user_id",
) -> pd.Series:
    """Per ground-truth user, classify the target item:

        cold                    never appeared before the snapshot
        low_support             appeared, but below the eligibility threshold
        eligible_not_retrieved  in the eligible pool, absent from the user's
                                candidate union
        retrieved               present in the user's candidate union
    """
    out = {}
    for u, target in zip(gt_df[user_col].astype(str), gt_df[item_col].astype(str)):
        if target in retrieved_per_user.get(u, set()):
            out[u] = "retrieved"
        elif target in eligible_items:
            out[u] = "eligible_not_retrieved"
        elif hist_item_counts.get(target, 0) > 0:
            out[u] = "low_support"
        else:
            out[u] = "cold"
    return pd.Series(out, name="target_status")
