"""Fetch the OFFICIAL 5-core Video_Games benchmark split .

CLI:
    python -m scripts.anchor.fetch_official_vg

Downloads from HF ``McAuley-Lab/Amazon-Reviews-2023`` (login node has internet;
compute nodes do not — run this at setup time):

    benchmark/5core/rating_only/Video_Games.csv          full 5-core set (stats anchor)
    benchmark/5core/last_out/Video_Games.{train,valid,test}.csv   official LOO split (leg E)

HARD GATE (abort non-zero on failure): the rating_only stats must equal
    94,762 users / 25,612 items / 814,586 interactions
— the Pctx (arXiv:2510.21276 Table 5) / LLaDA-Rec (arXiv:2511.06254 Table 2)
double-anchored triple. The last_out train+valid+test union must reproduce the
same triple, and valid/test must hold exactly one row per user.

Also emits RecBole benchmark-mode atomic files under
``data/anchor/_official/Video_Games/`` (``Video_Games.{train,valid,test}.inter``)
so the P1.2 runner can point RecBole's ``benchmark_filename`` at them without
re-deriving the official split.

ISOLATION: official-benchmark artifacts live under data/anchor/_official/ and
follow the anchor isolation rule — never read by production pipelines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from scripts.anchor.export_anchor_splits import ISOLATION_DECLARATION, _git_state, _now_iso

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = REPO_ROOT / "data" / "anchor" / "_official"

HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"
FILES = [
    "benchmark/5core/rating_only/Video_Games.csv",
    "benchmark/5core/last_out/Video_Games.train.csv",
    "benchmark/5core/last_out/Video_Games.valid.csv",
    "benchmark/5core/last_out/Video_Games.test.csv",
]

# Official 5-core VG triple, double-anchored (Pctx Table 5; LLaDA-Rec Table 2).
EXPECTED_USERS = 94_762
EXPECTED_ITEMS = 25_612
EXPECTED_INTERACTIONS = 814_586

EXPECTED_COLUMNS = ["user_id", "parent_asin", "rating", "timestamp"]


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"user_id": str, "parent_asin": str,
                                  "rating": float, "timestamp": "int64"})
    assert list(df.columns) == EXPECTED_COLUMNS, (
        f"{path.name}: columns {list(df.columns)} != expected {EXPECTED_COLUMNS}"
    )
    return df


def _triple(df: pd.DataFrame) -> tuple[int, int, int]:
    return (int(df["user_id"].nunique()), int(df["parent_asin"].nunique()), int(len(df)))


def _write_inter(df: pd.DataFrame, path: Path) -> None:
    """RecBole atomic file, preserving the official CSV row order exactly."""
    out = pd.DataFrame({
        "user_id:token": df["user_id"],
        "item_id:token": df["parent_asin"],
        "rating:float": df["rating"],
        "timestamp:float": df["timestamp"],
    })
    out.to_csv(path, sep="\t", index=False)


def main() -> int:
    OFFICIAL_ROOT.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    revision = api.repo_info(HF_REPO, repo_type="dataset").sha
    print(f"[official_vg] {HF_REPO} @ {revision}", flush=True)

    local: dict[str, Path] = {}
    for fname in FILES:
        p = hf_hub_download(
            repo_id=HF_REPO, filename=fname, repo_type="dataset",
            revision=revision, local_dir=OFFICIAL_ROOT,
        )
        local[fname] = Path(p)
        print(f"[official_vg] fetched {fname}", flush=True)

    # ---- HARD GATE: rating_only triple ---------------------------------------
    rating_only = _read_csv(local[FILES[0]])
    users, items, inter = _triple(rating_only)
    expected = (EXPECTED_USERS, EXPECTED_ITEMS, EXPECTED_INTERACTIONS)
    print(f"[official_vg] rating_only stats: {users:,} users / {items:,} items / "
          f"{inter:,} interactions (expected {expected})", flush=True)
    if (users, items, inter) != expected:
        print(f"[official_vg] GATE FAIL: stats {(users, items, inter)} != {expected}. "
              f"ABORT — refusing to emit .inter files.", file=sys.stderr, flush=True)
        return 1
    print("[official_vg] GATE PASS: official triple matches", flush=True)

    # ---- cross-check: last_out union reproduces the triple --------------------
    splits = {name: _read_csv(local[f"benchmark/5core/last_out/Video_Games.{name}.csv"])
              for name in ["train", "valid", "test"]}
    union = pd.concat(splits.values(), ignore_index=True)
    u_users, u_items, u_inter = _triple(union)
    assert (u_users, u_items, u_inter) == expected, (
        f"last_out union {(u_users, u_items, u_inter)} != rating_only triple {expected}"
    )
    for name in ["valid", "test"]:
        df = splits[name]
        assert len(df) == EXPECTED_USERS and df["user_id"].is_unique, (
            f"last_out {name}: {len(df)} rows, expected exactly one per "
            f"{EXPECTED_USERS} users"
        )
    print("[official_vg] cross-check PASS: last_out union == triple; "
          "valid/test one row per user", flush=True)

    # ---- RecBole benchmark-mode atomic files ----------------------------------
    inter_dir = OFFICIAL_ROOT / "Video_Games"
    inter_dir.mkdir(exist_ok=True)
    for name, df in splits.items():
        _write_inter(df, inter_dir / f"Video_Games.{name}.inter")
    print(f"[official_vg] wrote RecBole benchmark files -> {inter_dir}", flush=True)

    manifest = {
        "created_utc": _now_iso(),
        "git": _git_state(),
        "isolation_declaration": ISOLATION_DECLARATION,
        "source": {"hf_repo": HF_REPO, "revision": revision, "files": FILES},
        "gate": {
            "expected_users_items_interactions": list(expected),
            "measured_rating_only": [users, items, inter],
            "measured_last_out_union": [u_users, u_items, u_inter],
            "passed": True,
        },
        "split_rows": {name: int(len(df)) for name, df in splits.items()},
        "recbole_benchmark_dir": str(inter_dir),
        "anchor_note": (
            "Official per-user LOO split, downloaded as-is. Row order of the "
            "official CSVs is preserved byte-for-byte into the .inter files; "
            "no re-sorting, no re-splitting."
        ),
    }
    with open(OFFICIAL_ROOT / "official_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[official_vg] manifest -> {OFFICIAL_ROOT / 'official_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
