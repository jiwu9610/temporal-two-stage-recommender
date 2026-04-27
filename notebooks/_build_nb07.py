"""Build notebooks/07_ranker_tt_features.ipynb programmatically.

Run:  python notebooks/_build_nb07.py
Output: notebooks/07_ranker_tt_features.ipynb
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
    """# Session 7 — Ranker with two-tower features (Video_Games)

**Story so far.** Notebook 06 showed that even with hard-negative training, the
LightGBM ranker *still* hurts Recall@K over retrieval alone on All_Beauty
(Recall@100: 0.006 vs 0.056 retrieval-only). The missing ingredient is the
*retrieval signal itself* — the ranker is being asked to rerank two-tower
candidates but has no idea what two-tower thinks of each (user, item) pair.

**Fix (Phase 1).** Add `tt_score = u·v` (the two-tower cosine similarity) as a
LightGBM feature. This is the standard industry pattern (YouTube, Pinterest):
stage-2 reranker consumes stage-1 retrieval scores as features so it learns to
*refine* the retrieval ordering, not replace it.

**Category choice.** We start on `Video_Games` because that's the one category
where two-tower already beats popularity at Recall@100 (0.1336 vs 0.1278) —
i.e. the two-tower embeddings actually carry personalization signal worth
exposing to the ranker. If the method doesn't work here, it won't work anywhere.

**Success criterion.** `two-tower retrieval + LGB(+tt_score)` >
`two-tower retrieval alone` on Recall@{10, 50, 100}.

**What's new vs notebook 06.**
- Retriever at both training and eval time is **two-tower** (not popularity).
- Hard negatives are sampled from each user's personalized two-tower top-K,
  matching the serving-time candidate distribution.
- Ranker gets the `tt_score` feature in addition to user/item features.

**Ablation.** We run three ranker variants to isolate the contribution:
1. `no_tt` — same hardneg LGB as notebook 06 (no tt_score).
2. `tt_score` — adds `tt_score` feature.
3. `retrieval_only` — baseline (two-tower alone).
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

PROCESSED_ROOT = '../data/processed'
RESULTS_DIR = '../results'
os.makedirs(RESULTS_DIR, exist_ok=True)

CATEGORY = 'Video_Games'
NEG_PER_POS = 4
NEG_POOL_K = 500           # per-user two-tower top-K pool for hard negatives
N_EVAL_USERS = 1000
K_RETRIEVE = 1000
K_RANK = [10, 50, 100]
SEED = 42

# Match notebook 03's two-tower training config so the embeddings we produce
# here are the same quality as those that got 0.1336 Recall@100.
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

# Seen set from the full train (includes low-rating interactions), used to
# keep hard negatives away from anything the user has actually touched.
user_seen = train_raw.groupby('user_id')['parent_asin'].agg(set).to_dict()

print(f'train rows: {len(train_raw):,}  val rows: {len(val_raw):,}')
print(f'train pos:  {len(train_pos):,}  val pos:  {len(val_pos):,}')
print(f'users in seen map: {len(user_seen):,}  items: {item_features.shape[0]:,}')
""",
    cid="h-data",
)

md(
    """## 2. Fit two-tower retriever

Fit once on the full training data. Pass `eval_user_ids=None` so `user_seen_`
is populated for every training user — we need it for both (a) hard-negative
sampling (exclude seen items) and (b) e2e retrieval (exclude seen at serve
time).
""",
    cid="h-tt-md",
)

code(
    """t0 = time.time()
tt = TwoTowerRetriever(epochs=TT_EPOCHS, batch_size=TT_BATCH_SIZE, seed=SEED).fit(train_raw)
print(f'two-tower fit: {time.time()-t0:.1f}s')
print(f'n_users in tower: {len(tt.user_ids_):,}  n_items: {len(tt.item_ids_):,}  out_dim: {tt.out_dim}')


def tt_score_pairs(user_ids, item_ids):
    '''Compute u·v for parallel arrays of (user_id, parent_asin). Unknown
    users or items get score=0 (both sides default to zero vectors).'''
    user_ids = np.asarray(user_ids)
    item_ids = np.asarray(item_ids)
    u_idx = np.array([tt.user2idx_.get(u, -1) for u in user_ids])
    i_idx = np.array([tt.item2idx_.get(i, -1) for i in item_ids])
    out = np.zeros(len(user_ids), dtype=np.float32)

    # Item embeddings: precomputed, direct lookup.
    valid_item = i_idx >= 0
    # User embeddings: need one forward pass for the unique known users.
    valid_user = u_idx >= 0
    mask = valid_item & valid_user
    if not mask.any():
        return out

    u_known = u_idx[mask]
    i_known = i_idx[mask]

    # Batch the user-tower forward pass to avoid GPU OOM on huge arrays.
    tt.user_tower_.eval()
    B = 50000
    u_embs = np.empty((len(u_known), tt.out_dim), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(u_known), B):
            e = min(s + B, len(u_known))
            idx_t = torch.as_tensor(u_known[s:e], dtype=torch.long, device=tt.device)
            u_embs[s:e] = tt.user_tower_(idx_t).cpu().numpy()
    v_embs = tt.item_emb_[i_known]        # (N, d)
    out[mask] = np.einsum('nd,nd->n', u_embs, v_embs)
    return out


# Smoke test the scoring helper.
sample_u = train_pos['user_id'].head(3).tolist()
sample_i = train_pos['parent_asin'].head(3).tolist()
print('tt_score smoke test:', tt_score_pairs(sample_u, sample_i))
""",
    cid="h-tt",
)

md(
    """## 3. Hard negatives from two-tower top-K

For each positive (user, item), sample `NEG_PER_POS` items from the user's
personal two-tower top-500 that they haven't interacted with. This matches the
serving distribution: at eval time the LGB reranker sees exactly these
candidates.
""",
    cid="h-neg-md",
)

code(
    """def sample_hard_negatives_tt(pos_df, neg_per_pos, pool_k, rng, batch_users=5000):
    '''For each positive row, sample neg_per_pos items from that user's two-tower
    top-pool_k (minus seen). Done in batches of users to amortize the user-tower
    forward pass.'''
    uniq_users = pos_df['user_id'].unique()
    pos_count = pos_df.groupby('user_id').size().to_dict()
    user_to_neg = {}

    for s in range(0, len(uniq_users), batch_users):
        batch = list(uniq_users[s:s + batch_users])
        # recommend_batch already masks seen items (exclude_seen=True).
        recs = tt.recommend_batch(batch, k=pool_k, exclude_seen=True)
        for uid in batch:
            pool_items = recs[uid]
            if len(pool_items) == 0:
                continue
            n = pos_count[uid] * neg_per_pos
            n = min(n, len(pool_items))
            chosen_idx = rng.choice(len(pool_items), size=n, replace=False)
            user_to_neg[uid] = pool_items[chosen_idx]

    rows = []
    for uid, items in user_to_neg.items():
        rows.extend((uid, it) for it in items)
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
print(f'train_new: {train_new.shape}  pos_rate: {train_new["label"].mean():.3f}')
print(f'val_new:   {val_new.shape}    pos_rate: {val_new["label"].mean():.3f}')
""",
    cid="h-neg",
)

md(
    """## 4. Build features (with optional tt_score)

Same feature set as notebook 06 plus an opt-in `tt_score` column so we can
train both the `no_tt` and `tt_score` variants with a single feature builder.
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


def build_features(df, add_tt=False):
    df = df.merge(user_features[USER_FEAT_COLS], left_on='user_id', right_index=True, how='left')
    df = df.merge(item_features[ITEM_FEAT_COLS + CAT_COLS], left_on='parent_asin', right_index=True, how='left')
    if add_tt:
        df = df.copy()
        df['tt_score'] = tt_score_pairs(df['user_id'].to_numpy(), df['parent_asin'].to_numpy())
    y = df['label'].to_numpy() if 'label' in df.columns else None
    X = df.drop(columns=[c for c in ['user_id', 'parent_asin', 'label'] if c in df.columns])
    for c in CAT_COLS:
        X[c] = X[c].fillna('UNK').astype('category')
    return X, y


t0 = time.time()
X_tr_notts, y_tr = build_features(train_new, add_tt=False)
X_va_notts, y_va = build_features(val_new,   add_tt=False)
X_tr_tts,  _     = build_features(train_new, add_tt=True)
X_va_tts,  _     = build_features(val_new,   add_tt=True)
print(f'feature build: {time.time()-t0:.1f}s')
print(f'no_tt  shape: {X_tr_notts.shape}')
print(f'tt     shape: {X_tr_tts.shape}  tt_score stats: mean={X_tr_tts["tt_score"].mean():.3f}  std={X_tr_tts["tt_score"].std():.3f}')
print(f'tt_score by label (train):\\n{pd.Series(X_tr_tts["tt_score"]).groupby(y_tr).describe()}')
""",
    cid="h-feat",
)

md(
    """## 5. Train LightGBM (both variants)

If the hypothesis is right, `tt_score` lights up in feature importance and val
AUC improves. AUC alone isn't the success criterion — downstream e2e Recall is
— but AUC gain is a sanity check that the feature carries signal.
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
    dt = time.time() - t0
    p = model.predict(X_va, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_va, p)
    print(f'[{name}] train: {dt:.1f}s  best_iter: {model.best_iteration}  val AUC: {auc:.4f}')
    return model, auc


ranker_no_tt, auc_no_tt = train_lgb(X_tr_notts, y_tr, X_va_notts, y_va, 'no_tt')
ranker_tt,    auc_tt    = train_lgb(X_tr_tts,   y_tr, X_va_tts,   y_va, 'tt_score')
print(f'\\nval AUC delta (tt - no_tt): {auc_tt - auc_no_tt:+.4f}')

# Feature importance with tt_score exposed — if tt_score is doing real work it
# should land near the top of the split/gain ranking.
imp = pd.DataFrame({
    'feature': ranker_tt.feature_name(),
    'gain':    ranker_tt.feature_importance(importance_type='gain'),
    'split':   ranker_tt.feature_importance(importance_type='split'),
}).sort_values('gain', ascending=False)
print('\\nTop features by gain (tt_score variant):')
print(imp.head(10).to_string(index=False))
""",
    cid="h-train",
)

md(
    """## 6. End-to-end: two-tower retrieval → LGB rerank

Sample 1000 eval users; retrieve top-1000 with two-tower; rerank with both
ranker variants; compare to the retrieval-only baseline.
""",
    cid="h-e2e-md",
)

code(
    """gt = val_raw[val_raw['label'] == 1].groupby('user_id')['parent_asin'].agg(set).to_dict()
train_users = set(train_raw['user_id'].unique())
eval_users = [u for u in gt if u in train_users]
sample_users = list(rng.choice(eval_users, size=min(N_EVAL_USERS, len(eval_users)), replace=False))
print(f'eval users total: {len(eval_users):,}  sampled: {len(sample_users):,}')

t0 = time.time()
candidates = tt.recommend_batch(sample_users, k=K_RETRIEVE, exclude_seen=True)
print(f'two-tower retrieval: {time.time()-t0:.1f}s')

# Long-format (user, candidate) dataframe so we can featurize in one pass.
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

# Featurize once WITH tt_score so we can slice the column out for the no_tt
# ranker (which wasn't trained on that column).
X_score_tt, _   = build_features(score_df, add_tt=True)
X_score_notts   = X_score_tt.drop(columns=['tt_score'])

t0 = time.time()
s_no_tt = ranker_no_tt.predict(X_score_notts, num_iteration=ranker_no_tt.best_iteration)
s_tt    = ranker_tt.predict(X_score_tt,       num_iteration=ranker_tt.best_iteration)
print(f'scoring: {time.time()-t0:.1f}s')


def rerank(users, scores, items_long, offsets):
    out = {}
    for uid in users:
        s, e = offsets[uid]
        order = np.argsort(-scores[s:e])
        out[uid] = items_long[s:e][order]
    return out


reranked_no_tt = rerank(sample_users, s_no_tt, items_long, offsets)
reranked_tt    = rerank(sample_users, s_tt,    items_long, offsets)
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
    r_retr, m_retr = recall_mrr_at_k(candidates,        gt, k)
    r_nott, m_nott = recall_mrr_at_k(reranked_no_tt,    gt, k)
    r_tt,   m_tt   = recall_mrr_at_k(reranked_tt,       gt, k)
    rows.append({'k': k, 'variant': 'retrieval_only',              'recall': r_retr, 'mrr': m_retr})
    rows.append({'k': k, 'variant': 'retrieval+ranker_hardneg',    'recall': r_nott, 'mrr': m_nott})
    rows.append({'k': k, 'variant': 'retrieval+ranker_hardneg_tt', 'recall': r_tt,   'mrr': m_tt})

e2e_df = pd.DataFrame(rows)
out_path = f'{RESULTS_DIR}/e2e_tt_features_{CATEGORY}.csv'
e2e_df.to_csv(out_path, index=False)
print(f'\\nSaved: {out_path}')

print(f'\\nEnd-to-end on {CATEGORY} (two-tower retriever):')
pivot = e2e_df.pivot(index='k', columns='variant', values='recall').round(4)
print(pivot)

# Lift vs retrieval_only (the success criterion).
lift = pivot.copy()
for col in lift.columns:
    if col != 'retrieval_only':
        lift[col] = (pivot[col] - pivot['retrieval_only']).round(4)
lift = lift.drop(columns=['retrieval_only'])
print(f'\\nRecall lift vs retrieval_only:')
print(lift)
""",
    cid="h-metrics",
)

md(
    """## 7. Interpretation

**Reading the numbers.**
- `retrieval_only` = two-tower top-K, no rerank. This is the bar to clear.
- `retrieval+ranker_hardneg` = LGB reranker without `tt_score`. Same recipe as
  notebook 06 but with two-tower as the retriever. Tests whether hard-negative
  training *alone* is enough.
- `retrieval+ranker_hardneg_tt` = LGB with `tt_score` feature added. This is
  the Phase-1 intervention.

**What we expect.**
- If `tt_score` is in the top-3 LightGBM features, the ranker is using
  retrieval signal as intended.
- `retrieval+ranker_hardneg_tt` > `retrieval_only` on Recall@10 would close
  the loop for the resume story.
- Recall@100 is harder to beat: the retrieval-only top-100 already contains
  most of the recall ceiling, so the ranker can only reorder within that set.
  Recall@10 is the real test.

**Next steps conditional on these numbers.**
- ✅ If `retrieval+ranker_hardneg_tt` > `retrieval_only` on Recall@10:
  promote the approach to the other 3 categories, productionize (save
  two-tower embeddings, see Phase 1b in the roadmap).
- ⚠️ If `tt_score` helps AUC but still loses Recall:
  the retriever-rescored top-K is still too narrow — try widening
  `K_RETRIEVE`, add listwise training, or switch to lambdarank objective.
- ❌ If `tt_score` is flat in feature importance:
  two-tower scores don't carry enough signal — the Video_Games two-tower
  embeddings themselves need more training (more epochs, larger batch) before
  this architecture can work.
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

out = os.path.join(os.path.dirname(__file__), "07_ranker_tt_features.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"wrote {out}  ({len(CELLS)} cells)")
