"""Random retrieval baseline.

For every reported Recall@K we want a floor that comes purely from drawing K
items uniformly at random from the candidate pool (minus user-seen). Without
this, sparse-catalog numbers are easy to misread -- e.g. All_Beauty's pool is
744 items, so Recall@100 already gets ~13% by random chance.

Determinism: each user gets their own permutation derived from
(base_seed XOR crc32(user_id)). Different users do see different orderings,
but a given (seed, user_id) pair always produces the same ranking, so the
report is reproducible.
"""

from __future__ import annotations

import zlib
from typing import Dict, Iterable, List, Mapping, Set

import numpy as np


def recommend_random(
    candidate_pool: Set[str],
    user_ids: Iterable[str],
    user_seen: Mapping[str, Set[str]],
    k: int = 100,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Per-user random top-K from candidate_pool minus user_seen[u]."""
    pool_arr = np.array(sorted(candidate_pool))
    out: Dict[str, List[str]] = {}
    for u in user_ids:
        u_seed = (seed ^ zlib.crc32(u.encode("utf-8"))) & 0xFFFFFFFF
        rng = np.random.RandomState(u_seed)
        perm = rng.permutation(pool_arr)
        seen = user_seen.get(u, set())
        if seen:
            perm = perm[~np.isin(perm, list(seen))]
        out[u] = list(perm[:k])
    return out
