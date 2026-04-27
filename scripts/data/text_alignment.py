"""Verify or rebuild item text-embedding alignment to canonical parent_asin.

The legacy `title_bert.npz` files in `data/processed/{category}/` were built before
the canonicalization pass: their `asins` array contains the raw parent_asin from
metadata. After Phase 0 canonicalizes (title-duplicate collapse), some raw asins
no longer represent independent items -- they map to a canonical parent_asin.

This module re-keys the embedding matrix to canonical parent_asin and re-orders
its rows to match `item_features['parent_asin']` (so downstream code can index
into `embs[i]` for the i-th item in `item_features` without a separate lookup).

Output schema (matches input):
    asins : np.ndarray of canonical parent_asin, ordered as item_features
    embs  : np.ndarray (n_items_in_features, embedding_dim), aligned row-for-row

If the source npz is missing (BERT not yet computed for this category), we still
produce a valid output: an empty `asins` and a (0, 0) `embs`, plus a report
status of "not_present". This lets the orchestrator run end-to-end and
downstream code can branch on `report["status"]`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TextAlignmentReport:
    status: str                                  # "aligned", "rebuilt", "not_present"
    source_path: Optional[str] = None
    output_path: Optional[str] = None
    n_input_embeddings: int = 0
    n_unique_raw_keys: int = 0
    n_canonical_groups: int = 0
    n_groups_collapsed: int = 0                  # canonical groups built from >1 raw embedding
    n_aligned_to_item_features: int = 0
    n_items_in_features: int = 0
    n_items_missing_embedding: int = 0
    embedding_dim: int = 0
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def align_text_embeddings(
    source_npz_path: Optional[str | Path],
    item_features: pd.DataFrame,
    canonical_item_map: pd.DataFrame,
    output_npz_path: str | Path,
    item_col: str = "parent_asin",
) -> TextAlignmentReport:
    """Re-key + re-order BERT embeddings to canonical parent_asin order.

    Parameters
    ----------
    source_npz_path : path to legacy npz with arrays {"asins", "embs"}, or None.
    item_features : DataFrame with column `parent_asin` (canonical, in row order).
    canonical_item_map : DataFrame with [raw_parent_asin, canonical_parent_asin].
    output_npz_path : where to write the aligned npz.

    Returns
    -------
    TextAlignmentReport with coverage + status.
    """
    output_path = Path(output_npz_path)
    item_features_asins = item_features[item_col].astype(str).to_numpy()
    n_items = len(item_features_asins)

    # ---- handle missing source: write an empty output, report not_present -----
    if source_npz_path is None or not Path(source_npz_path).exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            asins=np.array([], dtype=object),
            embs=np.zeros((0, 0), dtype=np.float32),
        )
        return TextAlignmentReport(
            status="not_present",
            source_path=str(source_npz_path) if source_npz_path else None,
            output_path=str(output_path),
            n_items_in_features=n_items,
            n_items_missing_embedding=n_items,
            notes=["Source BERT npz not present; downstream must treat embeddings as unavailable."],
        )

    # ---- load source ----------------------------------------------------------
    src = np.load(source_npz_path, allow_pickle=True)
    if "asins" not in src.files or "embs" not in src.files:
        raise ValueError(
            f"{source_npz_path} missing expected arrays {{'asins','embs'}}; got {src.files}"
        )
    raw_asins = np.asarray(src["asins"]).astype(str)
    embs = np.asarray(src["embs"], dtype=np.float32)
    if embs.ndim != 2 or embs.shape[0] != raw_asins.shape[0]:
        raise ValueError(
            f"embedding matrix shape {embs.shape} inconsistent with asins {raw_asins.shape}"
        )
    embedding_dim = int(embs.shape[1])

    # ---- remap raw -> canonical ----------------------------------------------
    raw_to_canon = dict(
        zip(canonical_item_map["raw_parent_asin"].astype(str),
            canonical_item_map["canonical_parent_asin"].astype(str))
    )
    canonical_keys = np.array([raw_to_canon.get(a, a) for a in raw_asins])

    # Group rows by canonical key; average embeddings for collapsed title-groups.
    # We only need rows for canonical keys that actually appear in item_features.
    target_set = set(item_features_asins)
    keep_mask = np.array([k in target_set for k in canonical_keys])
    canonical_keys_kept = canonical_keys[keep_mask]
    embs_kept = embs[keep_mask]

    # Mean-pool by canonical key.
    df = pd.DataFrame(embs_kept)
    df["_key"] = canonical_keys_kept
    grouped = df.groupby("_key").mean()
    n_groups_collapsed = int(
        (df.groupby("_key").size() > 1).sum()
    )
    # Reorder to match item_features row order; missing items become zero rows.
    aligned = grouped.reindex(item_features_asins)
    n_aligned = int(aligned.notna().any(axis=1).sum())
    aligned_filled = aligned.fillna(0.0).to_numpy(dtype=np.float32)

    # ---- write aligned npz ----------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        asins=item_features_asins.astype(object),
        embs=aligned_filled,
    )

    # status: aligned if no remapping was needed (all raw == canonical), else rebuilt
    needed_remap = bool((np.asarray(canonical_keys) != raw_asins).any())
    status = "rebuilt" if (needed_remap or n_groups_collapsed > 0) else "aligned"

    return TextAlignmentReport(
        status=status,
        source_path=str(source_npz_path),
        output_path=str(output_path),
        n_input_embeddings=int(len(raw_asins)),
        n_unique_raw_keys=int(np.unique(raw_asins).size),
        n_canonical_groups=int(np.unique(canonical_keys).size),
        n_groups_collapsed=n_groups_collapsed,
        n_aligned_to_item_features=n_aligned,
        n_items_in_features=n_items,
        n_items_missing_embedding=int(n_items - n_aligned),
        embedding_dim=embedding_dim,
    )
