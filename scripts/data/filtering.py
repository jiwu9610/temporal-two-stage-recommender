"""Iterative bipartite k-core filtering.

Repeatedly remove items with fewer than `min_item` interactions and users with
fewer than `min_user` interactions, alternating until no rows are removed in a
full pass. The fixed point doesn't depend on item-vs-user order, but the
intermediate per-iteration sizes do — we record every round for traceability.

Why iterate
-----------
Removing low-engagement items can drop some users below `min_user`, and removing
those users can drop some items below `min_item`. A single pass leaves rows that
violate the thresholds. The standard fix is to alternate until convergence.

Default order is `[item, user]`: remove items first each round, then users.
This is just a convention -- swap at the config level if you want the symmetric run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd


@dataclass
class FilteringReport:
    min_user_interactions: int
    min_item_interactions: int
    filter_order: List[str]
    max_iterations: int
    converged: bool
    n_iterations: int
    iterations: list = field(default_factory=list)
    initial: dict = field(default_factory=dict)
    final: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "min_user_interactions": self.min_user_interactions,
            "min_item_interactions": self.min_item_interactions,
            "filter_order": self.filter_order,
            "max_iterations": self.max_iterations,
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "initial": self.initial,
            "iterations": self.iterations,
            "final": self.final,
        }


def _snapshot(df: pd.DataFrame, user_col: str, item_col: str) -> dict:
    return {
        "n_users": int(df[user_col].nunique()) if len(df) else 0,
        "n_items": int(df[item_col].nunique()) if len(df) else 0,
        "n_interactions": int(len(df)),
    }


def iterative_kcore(
    df: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
    filter_order: List[str] = ("item", "user"),
    max_iterations: int = 50,
    user_col: str = "user_id",
    item_col: str = "parent_asin",
) -> tuple[pd.DataFrame, FilteringReport]:
    """Iterative k-core: prune until no rows are removed in a full pass.

    Parameters
    ----------
    df : DataFrame with at least `user_col` and `item_col`.
    min_user_interactions, min_item_interactions : k-core thresholds.
    filter_order : sequence of "item" / "user" strings. Applied in order each round.
    max_iterations : safety cap. Convergence usually < 10 rounds.

    Returns
    -------
    (filtered_df, report)
    """
    if not set(filter_order).issubset({"item", "user"}):
        raise ValueError(f"filter_order entries must be 'item' or 'user', got {filter_order}")

    report = FilteringReport(
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
        filter_order=list(filter_order),
        max_iterations=max_iterations,
        converged=False,
        n_iterations=0,
        initial=_snapshot(df, user_col, item_col),
    )

    cur = df
    for it in range(1, max_iterations + 1):
        n_before = len(cur)
        for kind in filter_order:
            if kind == "item":
                counts = cur[item_col].value_counts()
                keep = counts[counts >= min_item_interactions].index
                cur = cur[cur[item_col].isin(keep)]
            elif kind == "user":
                counts = cur[user_col].value_counts()
                keep = counts[counts >= min_user_interactions].index
                cur = cur[cur[user_col].isin(keep)]
        snap = _snapshot(cur, user_col, item_col)
        snap["iter"] = it
        snap["n_removed_this_iter"] = n_before - len(cur)
        report.iterations.append(snap)

        if len(cur) == n_before:
            report.converged = True
            report.n_iterations = it
            break
    else:
        report.n_iterations = max_iterations

    cur = cur.reset_index(drop=True)
    report.final = _snapshot(cur, user_col, item_col)
    return cur, report
