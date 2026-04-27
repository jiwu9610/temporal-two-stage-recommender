"""Build notebooks/08_ranker_cross_features.ipynb programmatically.

Run:  python notebooks/_build_nb08.py
Output: notebooks/08_ranker_cross_features.ipynb
"""
import json
import os

CELLS = []


def md(source, cid=None):
    CELLS.append({
        "cell_type": "markdown",
        "id": cid or f"md-{len(CELLS)}",
        "metadata": {},
        "source": source if isinstance(source, list) else [source],
    })


def code(source, cid=None):
    CELLS.append({
        "cell_type": "code",
        "id": cid or f"code-{len(CELLS)}",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source if isinstance(source, list) else [source],
    })


# -------------------------------------------------------------------- cells
md(
    """# Session 8 — Ranker with user×item cross features (Video_Games)

**What notebook 07 taught us.** Adding `tt_score` as a LightGBM feature lifted
val AUC from 0.533 → 0.736 (+0.204) but still lost to retrieval-only on e2e
Recall@10 (0.030 vs 0.040). The mechanical reason: the `no_tt` variant had
AUC 0.533 — i.e. all existing user/item aggregate features together are
barely above random at distinguishing positives from hard negatives. With
`tt_score` dominating (10× the gain of the next feature), LGB's output is
effectively a noisy re-sort of retrieval's own score. You can't improve a
sort by adding noise to its key.

**What's missing.** Aggregate features are *constant across candidates* for a
given user (or *constant across users* for a given item). They encode nothing
about the specific (user, item) match. The fix is **cross features** — values
that depend on both the user and the item.

**This notebook.** Five cross features computed from each user's interaction
history:

| feature | meaning |
|---|---|
| `u_store_n` | how many times user has touched items from this item's store |
| `u_store_mean_rating` | user's mean rating given to items from this store (NaN if never) |
| `u_cat_n` | same, for this item's main_category |
| `u_cat_mean_rating` | same, for this item's main_category |
| `u_price_log_ratio` | log(item price) − log(user's avg historical price) |

**Ablation — 4 ranker variants**. A 2×2 design isolates the contribution of
each feature family:

| variant | aggregates | tt_score | cross |
|---|:-:|:-:|:-:|
| `base`      | ✓ | ✗ | ✗ |
| `+tt`       | ✓ | ✓ | ✗ |
| `+cross`    | ✓ | ✗ | ✓ |
| `+tt+cross` | ✓ | ✓ | ✓ |

**Success criterion.** `+tt+cross` Recall@10 > retrieval-only Recall@10 on
Video_Games. Secondary: cross features contribute AUC lift independent of
`tt_score` (comparing `+cross` vs `base`).
""",
    cid="h-title",
)

code(
    """import sys, os, gc, time
sys.path.insert(0, os.path.abspath('..'))
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from scripts.retrieval import TwoTowerRetriever
from scripts.features import build_user_profiles, add_cross_features, cross_features as cf_mod
CROSS_COLS = cf_mod.CROSS_COLS

PROCESSED_ROOT = '../data/processed'
RESULTS_DIR = '../results'
os.makedirs(RESULTS_DIR, exist_ok=True)

CATEGORY = 'Video_Games'
NEG_PER_POS = 4
NEG_POOL_K = 500
N_EVAL_USERS = 1000
K_RETRIEVE = 1000
K_RANK = [10, 50, 100]
SEED = 42

TT_EPOCHS = 30
TT_BATCH_SIZE = 8192

rng = np.random.default_rng(SEED)
print(f'Category: {CATEGORY}  device: {"cuda" if torch.cuda.is_available() else "cpu"}')
""",
    cid="h-imports",
)

md("## 1. Load data")

code(
    """d = f'{PROCESSED_ROOT}/{CATEGORY}'
train_raw = pd.read_parquet(f'{d}/train.parquet')
val_raw   = pd.read_parquet(f'{d}/val.parquet')
item_features = pd.read_parquet(f'{d}/item_features.parquet')
user_features = pd.read_parquet(f'{d}/user_features.parquet')

train_pos = train_raw[train_raw['label'] == 1][['user_id', 'parent_asin']].reset_index(drop=True)
val_pos   = val_raw[val_raw['label'] == 1][['user_id', 'parent_asin']].reset_index(drop=True)
user_seen = train_raw.groupby('user_id')['parent_asin'].agg(set).to_dict()

print(f'train rows: {len(train_raw):,}  val rows: {len(val_raw):,}')
print(f'train pos:  {len(train_pos):,}  val pos:  {len(val_pos):,}')
""",
    cid="h-data",
)

md(
    """## 2. Two-tower retriever + user profiles

Fit the two-tower (same config as notebooks 03 and 07 so results are
comparable) and build per-user store/category/price profiles from train
positives only — this is the data source for the five cross features.
""",
    cid="h-tt-prof-md",
)

code(
    """t0 = time.time()
tt = TwoTowerRetriever(epochs=TT_EPOCHS, batch_size=TT_BATCH_SIZE, seed=SEED).fit(train_raw)
print(f'two-tower fit: {time.time()-t0:.1f}s')

t0 = time.time()
profiles = build_user_profiles(train_raw, item_features)
user_store_agg, user_cat_agg, user_price_agg = profiles
print(f'profile build: {time.time()-t0:.1f}s')
print(f'  user_store_agg: {user_store_agg.shape}')
print(f'  user_cat_agg:   {user_cat_agg.shape}')
print(f'  user_price_agg: {user_price_agg.shape}')


def tt_score_pairs(user_ids, item_ids):
    '''Compute u·v for parallel arrays of (user_id, parent_asin).'''
    user_ids = np.asarray(user_ids)
    item_ids = np.asarray(item_ids)
    u_idx = np.array([tt.user2idx_.get(u, -1) for u in user_ids])
    i_idx = np.array([tt.item2idx_.get(i, -1) for i in item_ids])
    out = np.zeros(len(user_ids), dtype=np.float32)
    mask = (u_idx >= 0) & (i_idx >= 0)
    if not mask.any():
        return out
    u_known = u_idx[mask]
    i_known = i_idx[mask]
    tt.user_tower_.eval()
    B = 50000
    u_embs = np.empty((len(u_known), tt.out_dim), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(u_known), B):
            e = min(s + B, len(u_known))
            idx_t = torch.as_tensor(u_known[s:e], dtype=torch.long, device=tt.device)
            u_embs[s:e] = tt.user_tower_(idx_t).cpu().numpy()
    v_embs = tt.item_emb_[i_known]
    out[mask] = np.einsum('nd,nd->n', u_embs, v_embs)
    return out
""",
    cid="h-tt-prof",
)

md("## 3. Hard negatives from two-tower top-K")

code(
    """def sample_hard_negatives_tt(pos_df, neg_per_pos, pool_k, rng, batch_users=5000):
    uniq_users = pos_df['user_id'].unique()
    pos_count = pos_df.groupby('user_id').size().to_dict()
    user_to_neg = {}
    for s in range(0, len(uniq_users), batch_users):
        batch = list(uniq_users[s:s + batch_users])
        recs = tt.recommend_batch(batch, k=pool_k, exclude_seen=True)
        for uid in batch:
            pool_items = recs[uid]
            if len(pool_items) == 0:
                continue
            n = min(pos_count[uid] * neg_per_pos, len(pool_items))
            chosen_idx = rng.choice(len(pool_items), size=n, replace=False)
            user_to_neg[uid] = pool_items[chosen_idx]
    rows = [(u, it) for u, items in user_to_neg.items() for it in items]
    return pd.DataFrame(rows, columns=['user_id', 'parent_asin'])


t0 = time.time()
neg_train = sample_hard_negatives_tt(train_pos, NEG_PER_POS, NEG_POOL_K, rng)
neg_val   = sample_hard_negatives_tt(val_pos,   NEG_PER_POS, NEG_POOL_K, rng)
print(f'hard neg sampling: {time.time()-t0:.1f}s')
print(f'train neg: {len(neg_train):,}  val neg: {len(neg_val):,}')


def label_and_concat(pos_df, neg_df):
    p = pos_df.copy(); p['label'] = 1
    n = neg_df.copy(); n['label'] = 0
    return pd.concat([p, n], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


train_new = label_and_concat(train_pos, neg_train)
val_new   = label_and_concat(val_pos,   neg_val)
print(f'train_new: {train_new.shape}  val_new: {val_new.shape}')
""",
    cid="h-neg",
)

md(
    """## 4. Feature builder

Single function with two flags — flip `add_tt` and `add_cross` to get the
four ablation variants. Always merges user/item aggregates (the baseline).
""",
    cid="h-feat-md",
)

code(
    """USER_FEAT_COLS = ['n_reviews', 'avg_rating', 'std_rating', 'n_unique_items',
                  'active_days', 'verified_rate', 'avg_helpful_vote']
ITEM_FEAT_COLS = ['average_rating', 'rating_number', 'price',
                  'has_bought_together', 'n_reviews_actual',
                  'avg_rating_actual', 'n_unique_reviewers']
CAT_COLS = ['main_category', 'store']


def build_features(df, add_tt=False, add_cross=False):
    df = df.merge(user_features[USER_FEAT_COLS], left_on='user_id', right_index=True, how='left')
    df = df.merge(item_features[ITEM_FEAT_COLS + CAT_COLS], left_on='parent_asin', right_index=True, how='left')
    if add_tt:
        df = df.copy()
        df['tt_score'] = tt_score_pairs(df['user_id'].to_numpy(), df['parent_asin'].to_numpy())
    if add_cross:
        cross = add_cross_features(df[['user_id', 'parent_asin']], item_features, profiles)
        df = pd.concat([df.reset_index(drop=True), cross], axis=1)
    y = df['label'].to_numpy() if 'label' in df.columns else None
    X = df.drop(columns=[c for c in ['user_id', 'parent_asin', 'label'] if c in df.columns])
    for c in CAT_COLS:
        X[c] = X[c].fillna('UNK').astype('category')
    return X, y


t0 = time.time()
X_tr_base,  y_tr = build_features(train_new, add_tt=False, add_cross=False)
X_va_base,  y_va = build_features(val_new,   add_tt=False, add_cross=False)
X_tr_tt,    _    = build_features(train_new, add_tt=True,  add_cross=False)
X_va_tt,    _    = build_features(val_new,   add_tt=True,  add_cross=False)
X_tr_cr,    _    = build_features(train_new, add_tt=False, add_cross=True)
X_va_cr,    _    = build_features(val_new,   add_tt=False, add_cross=True)
X_tr_tc,    _    = build_features(train_new, add_tt=True,  add_cross=True)
X_va_tc,    _    = build_features(val_new,   add_tt=True,  add_cross=True)
print(f'feature build: {time.time()-t0:.1f}s')
print(f'shapes: base={X_tr_base.shape}  +tt={X_tr_tt.shape}  +cross={X_tr_cr.shape}  +tt+cross={X_tr_tc.shape}')

# Sanity check on cross feature values (should have real spread, not all zeros).
print(f'\\ncross-feature summary (train):')
print(X_tr_cr[CROSS_COLS].describe().round(3).to_string())
""",
    cid="h-feat",
)

md(
    """## 5. Train 4 ranker variants

Same LightGBM config for all four so any AUC delta is purely from features.
""",
    cid="h-train-md",
)

code(
    """LGB_PARAMS = dict(
    objective='binary', metric=['auc', 'binary_logloss'],
    learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
    verbose=-1, seed=SEED,
)


def train_lgb(X_tr, y_tr, X_va, y_va, name):
    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CAT_COLS)
    dval   = lgb.Dataset(X_va, label=y_va, categorical_feature=CAT_COLS, reference=dtrain)
    t0 = time.time()
    model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=500, valid_sets=[dval],
                      valid_names=['val'],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    p = model.predict(X_va, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_va, p)
    print(f'[{name}] train: {time.time()-t0:.1f}s  best_iter: {model.best_iteration}  val AUC: {auc:.4f}')
    return model, auc


models = {}
aucs = {}
for name, Xt, Xv in [
    ('base',      X_tr_base, X_va_base),
    ('+tt',       X_tr_tt,   X_va_tt),
    ('+cross',    X_tr_cr,   X_va_cr),
    ('+tt+cross', X_tr_tc,   X_va_tc),
]:
    models[name], aucs[name] = train_lgb(Xt, y_tr, Xv, y_va, name)

print(f'\\nAUC summary:')
for k, v in aucs.items():
    print(f'  {k:12s}  {v:.4f}')
print(f'\\nAUC deltas:')
print(f'  +tt       - base : {aucs["+tt"] - aucs["base"]:+.4f}')
print(f'  +cross    - base : {aucs["+cross"] - aucs["base"]:+.4f}')
print(f'  +tt+cross - base : {aucs["+tt+cross"] - aucs["base"]:+.4f}')
print(f'  +tt+cross - +tt  : {aucs["+tt+cross"] - aucs["+tt"]:+.4f}  (marginal value of cross)')
print(f'  +tt+cross - +cross: {aucs["+tt+cross"] - aucs["+cross"]:+.4f}  (marginal value of tt)')

# Feature importance for the full model.
imp = pd.DataFrame({
    'feature': models['+tt+cross'].feature_name(),
    'gain':    models['+tt+cross'].feature_importance(importance_type='gain'),
    'split':   models['+tt+cross'].feature_importance(importance_type='split'),
}).sort_values('gain', ascending=False)
print(f'\\nTop features by gain (+tt+cross variant):')
print(imp.head(12).to_string(index=False))
""",
    cid="h-train",
)

md("## 6. End-to-end evaluation: two-tower retrieval → four reranker variants")

code(
    """gt = val_raw[val_raw['label'] == 1].groupby('user_id')['parent_asin'].agg(set).to_dict()
train_users = set(train_raw['user_id'].unique())
eval_users = [u for u in gt if u in train_users]
sample_users = list(rng.choice(eval_users, size=min(N_EVAL_USERS, len(eval_users)), replace=False))
print(f'eval users total: {len(eval_users):,}  sampled: {len(sample_users):,}')

t0 = time.time()
candidates = tt.recommend_batch(sample_users, k=K_RETRIEVE, exclude_seen=True)
print(f'two-tower retrieval: {time.time()-t0:.1f}s')

# Long (user, candidate) form for vectorized scoring.
users_long, items_long, offsets = [], [], {}
cur = 0
for uid in sample_users:
    c = candidates[uid]
    users_long.append(np.full(len(c), uid, dtype=object))
    items_long.append(c)
    offsets[uid] = (cur, cur + len(c))
    cur += len(c)
users_long = np.concatenate(users_long)
items_long = np.concatenate(items_long)
score_df = pd.DataFrame({'user_id': users_long, 'parent_asin': items_long})
print(f'candidate rows to score: {len(score_df):,}')

# Build once with BOTH extensions enabled, then slice columns as needed.
t0 = time.time()
X_score_full, _ = build_features(score_df, add_tt=True, add_cross=True)
print(f'featurize full: {time.time()-t0:.1f}s')

# Column sets per variant
cols_base = [c for c in X_score_full.columns if c != 'tt_score' and c not in CROSS_COLS]
cols_tt   = cols_base + ['tt_score']
cols_cr   = cols_base + CROSS_COLS
cols_tc   = cols_base + ['tt_score'] + CROSS_COLS

score_variants = {
    'base':      X_score_full[cols_base],
    '+tt':       X_score_full[cols_tt],
    '+cross':    X_score_full[cols_cr],
    '+tt+cross': X_score_full[cols_tc],
}


def rerank(users, scores, items_long, offsets):
    out = {}
    for uid in users:
        s, e = offsets[uid]
        order = np.argsort(-scores[s:e])
        out[uid] = items_long[s:e][order]
    return out


reranked = {}
for name, Xs in score_variants.items():
    t0 = time.time()
    s = models[name].predict(Xs, num_iteration=models[name].best_iteration)
    reranked[name] = rerank(sample_users, s, items_long, offsets)
    print(f'  scored {name}: {time.time()-t0:.1f}s')
""",
    cid="h-e2e",
)

code(
    """def recall_mrr_at_k(recs, gt, k):
    recalls, mrrs = [], []
    for uid, items in recs.items():
        truth = gt.get(uid, set())
        if not truth: continue
        top = items[:k]
        recalls.append(len(set(top) & truth) / len(truth))
        mrr = 0.0
        for rank, it in enumerate(top, 1):
            if it in truth:
                mrr = 1.0 / rank
                break
        mrrs.append(mrr)
    return float(np.mean(recalls)), float(np.mean(mrrs))


rows = []
for k in K_RANK:
    r, m = recall_mrr_at_k(candidates, gt, k)
    rows.append({'k': k, 'variant': 'retrieval_only', 'recall': r, 'mrr': m})
    for name, recs in reranked.items():
        r, m = recall_mrr_at_k(recs, gt, k)
        rows.append({'k': k, 'variant': f'rerank_{name}', 'recall': r, 'mrr': m})

e2e_df = pd.DataFrame(rows)
out_path = f'{RESULTS_DIR}/e2e_cross_features_{CATEGORY}.csv'
e2e_df.to_csv(out_path, index=False)
print(f'Saved: {out_path}\\n')

pivot_recall = e2e_df.pivot(index='k', columns='variant', values='recall').round(4)
print('Recall @ K:')
print(pivot_recall)

# Lift vs retrieval_only
lift = pivot_recall.subtract(pivot_recall['retrieval_only'], axis=0).drop(columns='retrieval_only').round(4)
print(f'\\nRecall lift vs retrieval_only (positive = reranker helped):')
print(lift)

pivot_mrr = e2e_df.pivot(index='k', columns='variant', values='mrr').round(4)
print(f'\\nMRR @ K:')
print(pivot_mrr)
""",
    cid="h-metrics",
)

md(
    """## 7. Interpretation

**Three questions the ablation table answers directly.**

1. *Do cross features alone help?* Look at `+cross` vs `base` AUC. If
   `+cross` > `base` by a meaningful margin, cross features carry orthogonal
   signal the existing aggregates couldn't capture.

2. *Do cross features complement `tt_score`?* Look at `+tt+cross` vs `+tt`
   AUC. If `+tt+cross` > `+tt` by a real margin, the two families are
   complementary (cross adds signal beyond what retrieval scores provide).

3. *Do we finally beat retrieval-only on e2e?* The Recall@10 row is the
   verdict: `rerank_+tt+cross` > `retrieval_only` closes the project story.

**Interpreting feature importance.** In the `+tt+cross` model:
- `tt_score` should still rank high but no longer dominate by 10× — if cross
  features together exceed half of `tt_score`'s gain, the ranker is using
  both signals, not just retrieval.
- Among cross features, `u_store_n` / `u_store_mean_rating` typically rank
  highest on Amazon data (store affinity is strong).

**Decision tree for next step**:

- ✅ `rerank_+tt+cross` beats `retrieval_only` on Recall@10 → lock this
  design, extend to the other 3 categories (P1b).
- ⚠️ Only on Recall@10, not Recall@50/100 → that's actually fine; production
  reranking cares about top-10. Note in the report that the ranker re-sorts
  within the top-K set retrieval delivers.
- ❌ Still losing on Recall@10 → time to switch to LambdaRank (P2). Binary
  classification loss doesn't directly optimize ranking; a listwise objective
  might close the final gap.
""",
    cid="h-interp",
)


# -------------------------------------------------------------------- write
nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(os.path.dirname(__file__), "08_ranker_cross_features.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"wrote {out}  ({len(CELLS)} cells)")
