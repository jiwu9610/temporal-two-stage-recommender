"""Global temporal snapshot split (mode: ``global_temporal_snapshots``).

Replaces per-user leave-last-two with calendar cutoffs shared by EVERY user:

    train_end <= val_end <= test_end        (epoch milliseconds, UTC)

    validation snapshot:
        history = events with timestamp <  train_end
        labels  = each user's FIRST positive event in [train_end, val_end)
    test snapshot:
        history = events with timestamp <  val_end     (INCLUDES the val window)
        labels  = each user's FIRST positive event in [val_end, test_end)

Why: under leave-last-two, each user's split boundary is their own last-two
interactions, so train rows of one user can postdate test rows of another --
any statistic pooled across users (popularity, affinity, k-core survival) sees
the future. Global cutoffs give every snapshot a single point-in-time boundary
that mimics serving: "score traffic arriving after T using only what was known
at T".

Eligibility (history-only; replaces full-timeline k-core)
---------------------------------------------------------
The Phase 0 iterative k-core runs on the whole timeline, so item/user survival
depends on events inside the eval windows -- future information. Snapshot mode
therefore starts from ``interactions_clean`` (pre-k-core) and applies a
history-only eligibility filter instead:

    eligible item (candidate pool) : >= min_item_history_interactions events
                                     in the snapshot's history
    warm user                      : >= warm_user_min_history events
                                     in the snapshot's history

Nothing is dropped: label users below the warm threshold (including users with
zero history) are kept, flagged cold, and reported. Label items outside the
pool are kept in the ground truth (they bound achievable recall and are
reported as coverage).

Known upstream bias (documented, not fixed here): canonicalize.py dedups
(user, item) keeping the EARLIEST row, so a repeat interaction with the same
item inside a label window is invisible if its first occurrence predates the
cutoff. Labels are therefore "first positive on a NEW item" -- consistent with
how the seen-item mask is used at serving time, but stricter than raw traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import pandas as pd

MS_PER_DAY = 86_400_000


def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def iso_to_ms(value) -> int:
    """ISO date/datetime string -> epoch ms. Naive inputs are taken as UTC."""
    ts = pd.Timestamp(value)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return int(ts.value // 1_000_000)


@dataclass
class CutoffSpec:
    """Resolved global cutoffs. Source is recorded so the manifest can state
    whether dates were pinned in config or derived from timestamp quantiles."""
    train_end_ms: int
    val_end_ms: int
    test_end_ms: int
    source: str                      # "explicit" | "quantile"
    quantiles: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if not (self.train_end_ms < self.val_end_ms <= self.test_end_ms):
            raise ValueError(
                f"cutoffs must satisfy train_end < val_end <= test_end, got "
                f"{self.train_end_ms}, {self.val_end_ms}, {self.test_end_ms}"
            )

    def to_dict(self) -> dict:
        return {
            "train_end_ms": self.train_end_ms,
            "val_end_ms": self.val_end_ms,
            "test_end_ms": self.test_end_ms,
            "train_end_iso": ms_to_iso(self.train_end_ms),
            "val_end_iso": ms_to_iso(self.val_end_ms),
            "test_end_iso": ms_to_iso(self.test_end_ms),
            "source": self.source,
            "quantiles": self.quantiles,
        }


def resolve_cutoffs(
    clean_df: pd.DataFrame,
    time_col: str = "timestamp",
    explicit: Optional[Dict[str, str]] = None,
    train_end_quantile: float = 0.80,
    val_end_quantile: float = 0.90,
) -> CutoffSpec:
    """Resolve global cutoffs for one category.

    Explicit ISO dates (config `temporal_snapshots.cutoffs.{category}`) win.
    Otherwise cutoffs come from timestamp quantiles of the full clean
    interaction set. Quantile resolution reads the whole timeline -- that is a
    one-time protocol design choice (like picking the dates by hand), recorded
    in the manifest; it never feeds any feature.
    """
    ts = clean_df[time_col]
    max_ms = int(ts.max())
    if explicit and explicit.get("train_end") and explicit.get("val_end"):
        return CutoffSpec(
            train_end_ms=iso_to_ms(explicit["train_end"]),
            val_end_ms=iso_to_ms(explicit["val_end"]),
            test_end_ms=(
                iso_to_ms(explicit["test_end"]) if explicit.get("test_end")
                else max_ms + 1
            ),
            source="explicit",
        )
    return CutoffSpec(
        train_end_ms=int(ts.quantile(train_end_quantile)),
        val_end_ms=int(ts.quantile(val_end_quantile)),
        test_end_ms=max_ms + 1,
        source="quantile",
        quantiles={"train_end": train_end_quantile, "val_end": val_end_quantile},
    )


@dataclass
class Snapshot:
    """One point-in-time evaluation snapshot.

    history        : all clean events strictly before history_end_ms
    labels         : one row per label user -- their first positive event in
                     [history_end_ms, label_end_ms), plus warm/cold and
                     pool-membership flags (see build_snapshot)
    eligible_items : pd.Index of items with >= min_item_history_interactions
                     history events (the default retrieval candidate pool)
    """
    name: str
    history_end_ms: int
    label_end_ms: int
    history: pd.DataFrame
    labels: pd.DataFrame
    eligible_items: pd.Index
    stats: dict


def first_positive_labels(
    window_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "parent_asin",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """One row per user: earliest label==1 event in the window.

    Same deterministic ordering convention as leave_last_two_split: stable
    mergesort on (user, ts, item) so millisecond ties resolve alphabetically.

    This answers "is the user's NEXT positive in our top-K". It is the wrong
    target when the question is coverage of everything they went on to buy --
    see all_positive_labels, and note that under this mode Precision@K is
    exactly Recall@K / K and therefore carries no independent information.
    """
    pos = window_df[window_df["label"] == 1]
    pos = pos.sort_values([user_col, time_col, item_col], kind="mergesort")
    return pos.drop_duplicates(subset=[user_col], keep="first").reset_index(drop=True)


def all_positive_labels(
    window_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "parent_asin",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Every distinct positive a user has in the window -- several rows per user.

    This is the ground truth for "did the recommender cover what the user
    actually bought over the window", as opposed to first_positive_labels'
    next-item framing. Keeping only the first positive discards 5.9% (All_Beauty)
    to 61.3% (Books) of the window's positives, measured on the T2->T3 window.

    Deduplicated on (user, item) keeping the EARLIEST occurrence, so a user who
    reviews one item twice still contributes a single target -- otherwise
    |gt[u]| would double-count and depress per-user recall for no reason.
    Same stable mergesort convention as first_positive_labels, so row order is
    a function of the data alone and never of input order.
    """
    pos = window_df[window_df["label"] == 1]
    pos = pos.sort_values([user_col, time_col, item_col], kind="mergesort")
    return pos.drop_duplicates(
        subset=[user_col, item_col], keep="first"
    ).reset_index(drop=True)


def relabel_snapshot_all_positive(
    snapshot: "Snapshot",
    clean_df: pd.DataFrame,
    warm_user_min_history: int = 3,
    user_col: str = "user_id",
    item_col: str = "parent_asin",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """All-positive label frame for an ALREADY BUILT snapshot.

    Rebuilding the whole snapshot just to swap the label rule costs a second
    full pass over ~1M rows per category and recomputes history, eligibility
    and value_counts that cannot possibly differ: build_snapshot derives
    history purely from `ts < history_end_ms`, which does not depend on
    label_mode. This re-derives ONLY the label frame, annotated from the
    snapshot's own history and eligible pool, so the two modes are identical
    by construction rather than by coincidence.
    """
    ts = clean_df[time_col]
    window = clean_df[(ts >= snapshot.history_end_ms) & (ts < snapshot.label_end_ms)]
    labels = all_positive_labels(window, user_col, item_col, time_col)

    history = snapshot.history
    item_hist_index = pd.Index(history[item_col].unique())
    eligible = set(snapshot.eligible_items)
    user_hist_counts = history.groupby(user_col).size()

    labels = labels.assign(
        n_history_events=labels[user_col].map(user_hist_counts).fillna(0).astype(int),
        item_in_history=labels[item_col].isin(item_hist_index).astype(int),
        item_in_eligible_pool=labels[item_col].isin(eligible).astype(int),
    )
    labels["is_warm_user"] = (
        labels["n_history_events"] >= warm_user_min_history
    ).astype(int)
    return labels


def build_snapshot(
    clean_df: pd.DataFrame,
    name: str,
    history_end_ms: int,
    label_end_ms: int,
    min_item_history_interactions: int = 5,
    warm_user_min_history: int = 3,
    user_col: str = "user_id",
    item_col: str = "parent_asin",
    time_col: str = "timestamp",
    label_mode: str = "first_positive",
) -> Snapshot:
    """Build one snapshot: history strictly before the cutoff, labels in
    [history_end_ms, label_end_ms), history-only eligibility, warm/cold flags.

    label_mode:
      "first_positive"  one row per user, the earliest positive (default —
                        preserves every frozen artifact byte for byte)
      "all_positive"    every distinct positive in the window, several rows
                        per user; the coverage framing
    Only the label frame changes; history, eligibility and cutoffs are
    identical across modes, so the two are directly comparable.
    """
    if label_mode not in ("first_positive", "all_positive"):
        raise ValueError(f"unknown label_mode {label_mode!r}")
    ts = clean_df[time_col]
    history = clean_df[ts < history_end_ms].reset_index(drop=True)
    window = clean_df[(ts >= history_end_ms) & (ts < label_end_ms)]
    label_fn = (first_positive_labels if label_mode == "first_positive"
                else all_positive_labels)
    labels = label_fn(window, user_col, item_col, time_col)

    item_hist_counts = history[item_col].value_counts()
    eligible_items = item_hist_counts[
        item_hist_counts >= min_item_history_interactions
    ].index
    user_hist_counts = history.groupby(user_col).size()

    labels = labels.assign(
        n_history_events=labels[user_col].map(user_hist_counts).fillna(0).astype(int),
        item_in_history=labels[item_col].isin(item_hist_counts.index).astype(int),
        item_in_eligible_pool=labels[item_col].isin(set(eligible_items)).astype(int),
    )
    labels["is_warm_user"] = (
        labels["n_history_events"] >= warm_user_min_history
    ).astype(int)

    # Rows != users once label_mode == "all_positive". Every "…_users" stat
    # below must count DISTINCT users or it silently reports a row count.
    n_labels = len(labels)
    n_label_users = int(labels[user_col].nunique()) if n_labels else 0
    warm_users = (labels.loc[labels["is_warm_user"] == 1, user_col].nunique()
                  if n_labels else 0)
    n_warm = int(warm_users)
    stats = {
        "name": name,
        "history_end_ms": history_end_ms,
        "history_end_iso": ms_to_iso(history_end_ms),
        "label_window": [ms_to_iso(history_end_ms), ms_to_iso(label_end_ms)],
        "history": {
            "n_rows": int(len(history)),
            "n_users": int(history[user_col].nunique()),
            "n_items": int(history[item_col].nunique()),
            "ts_min": int(history[time_col].min()) if len(history) else None,
            "ts_max": int(history[time_col].max()) if len(history) else None,
        },
        "window_n_rows": int(len(window)),
        "labels": {
            "label_mode": label_mode,
            # n_rows == n_users only under first_positive. Both are reported so
            # a consumer can never mistake one for the other.
            "n_rows": n_labels,
            "n_users": n_label_users,
            "positives_per_user": (n_labels / n_label_users) if n_label_users else 0.0,
            "n_items": int(labels[item_col].nunique()),
            "ts_min": int(labels[time_col].min()) if n_labels else None,
            "ts_max": int(labels[time_col].max()) if n_labels else None,
            "n_warm_users": n_warm,
            "n_cold_users": n_label_users - n_warm,
            "warm_user_min_history": warm_user_min_history,
            "n_label_items_in_history": int(labels["item_in_history"].sum()),
            "n_label_items_in_eligible_pool": int(labels["item_in_eligible_pool"].sum()),
            # These two are the coverage ceilings quoted against the headline
            # recall, and the headline recall is MACRO per user on both eval
            # paths. A .mean() over label ROWS is the MICRO rate, which only
            # coincides with the macro one while rows == users. Publish both
            # under explicit names rather than letting the old key quietly
            # change which aggregation it denotes.
            "label_item_in_history_rate_per_positive": (
                float(labels["item_in_history"].mean()) if n_labels else 0.0
            ),
            "label_item_in_eligible_pool_rate_per_positive": (
                float(labels["item_in_eligible_pool"].mean()) if n_labels else 0.0
            ),
            "label_item_in_history_rate": (
                float(labels.groupby(user_col)["item_in_history"].mean().mean())
                if n_labels else 0.0
            ),
            "label_item_in_eligible_pool_rate": (
                float(labels.groupby(user_col)["item_in_eligible_pool"].mean().mean())
                if n_labels else 0.0
            ),
        },
        "eligibility": {
            "min_item_history_interactions": min_item_history_interactions,
            "n_eligible_items": int(len(eligible_items)),
            "n_history_items_below_threshold": int(
                (item_hist_counts < min_item_history_interactions).sum()
            ),
        },
    }
    return Snapshot(
        name=name,
        history_end_ms=history_end_ms,
        label_end_ms=label_end_ms,
        history=history,
        labels=labels,
        eligible_items=eligible_items,
        stats=stats,
    )


def build_temporal_snapshots(
    clean_df: pd.DataFrame,
    cutoffs: CutoffSpec,
    min_item_history_interactions: int = 5,
    warm_user_min_history: int = 3,
    user_col: str = "user_id",
    item_col: str = "parent_asin",
    time_col: str = "timestamp",
) -> Tuple[Snapshot, Snapshot, dict]:
    """Build the validation and test snapshots plus a manifest.

    Guaranteed by construction (and asserted in tests):
      - val history < train_end; test history < val_end
      - test history is a superset of val history and CONTAINS the val window
      - the two snapshots share nothing derived: pools, seen-masks and
        features must be recomputed per snapshot from its own history.
    """
    val = build_snapshot(
        clean_df, "val", cutoffs.train_end_ms, cutoffs.val_end_ms,
        min_item_history_interactions, warm_user_min_history,
        user_col, item_col, time_col,
    )
    test = build_snapshot(
        clean_df, "test", cutoffs.val_end_ms, cutoffs.test_end_ms,
        min_item_history_interactions, warm_user_min_history,
        user_col, item_col, time_col,
    )
    manifest = {
        "strategy": "global_temporal_snapshots",
        "cutoffs": cutoffs.to_dict(),
        "base_universe": {
            "n_rows": int(len(clean_df)),
            "n_users": int(clean_df[user_col].nunique()),
            "n_items": int(clean_df[item_col].nunique()),
            "ts_min": int(clean_df[time_col].min()),
            "ts_max": int(clean_df[time_col].max()),
            "note": (
                "interactions_clean (pre-k-core). Full-timeline k-core is NOT "
                "applied in temporal mode because it uses future events to "
                "decide eligibility; see history-only eligibility in snapshots."
            ),
        },
        "snapshots": {"val": val.stats, "test": test.stats},
    }
    return val, test, manifest
