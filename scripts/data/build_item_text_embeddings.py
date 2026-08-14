"""P2.4: regenerate item content embeddings on the canon-v2 identity.

The npz on disk was produced ad hoc on the v1 (title-only-merge) identity: two
distinct products that shared a title share one id and therefore one vector,
reproducing the C1 confusion inside the embedding space. This regenerates the
matrix keyed on the canon-v2 canonical ids, with the approved D6 text recipe:

    "title | store | first two category levels"

Books falls back to `details.Author` in place of store when the store field is
predominantly missing there (spec D6; the share is measured, not assumed, and
recorded in the sidecar JSON either way).

Model: sentence-transformers/all-MiniLM-L6-v2 (384-d, normalized) — the same
family as the v1 matrix, so downstream cosine plumbing is unchanged. The model
must already be in the HF cache: compute nodes have no internet (download once
on the login node).

    python -m scripts.data.build_item_text_embeddings --category Books

Output: data/processed/{cat}/item_text_embeddings.npz  {asins, embs}
        + item_text_embeddings.recipe.json (provenance sidecar)
Run AFTER the category's Phase 0 rebuild — it reads the NEW canonical map.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CATEGORIES = ("All_Beauty", "Video_Games", "Books", "Electronics")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
AUTHOR_FALLBACK_THRESHOLD = 0.5   # share of missing/Unknown store that flips Books to author


def _clean(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).strip()
    return "" if s.lower() in ("", "nan", "none", "unknown") else s


def _author_of(details) -> str:
    if isinstance(details, dict):
        return _clean(details.get("Author") or details.get("author"))
    return ""


def _cat_levels(categories, main_category) -> str:
    if isinstance(categories, (list, np.ndarray)) and len(categories):
        return " ".join(_clean(c) for c in list(categories)[:2] if _clean(c))
    return _clean(main_category)


def build_category(category: str, batch_size: int = 1024) -> dict:
    from sentence_transformers import SentenceTransformer
    import torch

    out_dir = PROCESSED_DIR / category
    canon = pd.read_parquet(out_dir / "canonical_item_map.parquet")
    winners = set(canon["canonical_parent_asin"].astype(str))

    # `details` is a struct column with ~830 subfields; loading it for Books'
    # 4.4M rows OOM-killed the 64G embed job. It is only needed when the
    # author fallback actually fires, so read it in a second, projected pass.
    meta = pd.read_parquet(
        RAW_DIR / category / "metadata.parquet",
        columns=["parent_asin", "title", "store", "main_category",
                 "categories", "rating_number"])
    meta["parent_asin"] = meta["parent_asin"].astype(str)
    # Mirror canonicalize's dedup so each winner contributes its winning row.
    meta = (meta.sort_values(["rating_number", "parent_asin"],
                             ascending=[False, True], kind="mergesort")
                .drop_duplicates("parent_asin", keep="first"))
    meta = meta[meta["parent_asin"].isin(winners)].reset_index(drop=True)
    assert len(meta), f"{category}: no canonical winners found in raw metadata"

    store_clean = meta["store"].map(_clean)
    missing_store_share = float((store_clean == "").mean())
    use_author = (category == "Books"
                  and missing_store_share > AUTHOR_FALLBACK_THRESHOLD)
    if use_author:
        det = pd.read_parquet(RAW_DIR / category / "metadata.parquet",
                              columns=["parent_asin", "details"])
        det["parent_asin"] = det["parent_asin"].astype(str)
        det = det.drop_duplicates("parent_asin", keep="first")
        author = det.set_index("parent_asin")["details"].map(_author_of)
        middle = meta["parent_asin"].map(author).fillna("")
        del det, author
    else:
        middle = store_clean

    texts = []
    for title, mid, cats, main_cat in zip(
            meta["title"].map(_clean), middle,
            meta["categories"], meta["main_category"]):
        parts = [p for p in (title, mid, _cat_levels(cats, main_cat)) if p]
        texts.append(" | ".join(parts) if parts else "unknown item")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=True).astype(np.float32)
    assert embs.shape == (len(meta), 384), embs.shape

    asins = meta["parent_asin"].to_numpy(dtype=object)
    np.savez(out_dir / "item_text_embeddings.npz", asins=asins, embs=embs)

    sidecar = {
        "recipe": "title | store | first-2 category levels",
        "middle_field": "details.Author" if use_author else "store",
        "missing_store_share": missing_store_share,
        "author_fallback_threshold": AUTHOR_FALLBACK_THRESHOLD,
        "model": MODEL_NAME,
        "n_items": int(len(meta)),
        "identity": "canon_v2 (norm_title, store)",
        "built_utc": datetime.now(tz=timezone.utc).isoformat(),
        "device": device,
    }
    with open(out_dir / "item_text_embeddings.recipe.json", "w") as f:
        json.dump(sidecar, f, indent=1)
    print(f"[embed] {category}: {len(meta):,} items, middle={sidecar['middle_field']} "
          f"(missing-store share {missing_store_share:.1%}), device={device}", flush=True)
    return sidecar


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--category", action="append", choices=list(CATEGORIES))
    p.add_argument("--batch-size", type=int, default=1024)
    args = p.parse_args(argv)
    for cat in (args.category or list(CATEGORIES)):
        build_category(cat, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
