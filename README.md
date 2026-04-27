# Recommendation System on Amazon Reviews 2023

End-to-end two-stage recommender (retrieval → ranker) on the Amazon Reviews
2023 dataset across four categories: All_Beauty, Video_Games, Books, Electronics.

- **Retrieval**: popularity, rule-based (`store + category + popularity` affinity),
  and a two-tower model (pointwise BCE, explicit positive / soft-negative pairs).
- **Ranker**: MLP and Deep & Cross, both reranking the same per-category
  candidate union of popularity ⊕ rule ⊕ two-tower top-100. Shared training
  loop and eval contract so any A/B is purely architectural.

## Results

4-category test Recall@K. All numbers are held-out-positive Recall on
canonical leave-last-two splits; ceiling = candidate-union coverage upper
bound on R@100.

| Category | popularity | rule_based | two_tower | **MLP** | **Deep+Cross** | ceiling |
|---|---:|---:|---:|---:|---:|---:|
| All_Beauty  R@10  | 0.006 | **0.024** | 0.005 | 0.022 | 0.023 | — |
| All_Beauty  R@50  | 0.042 | 0.044 | 0.054 | **0.094** | **0.094** | — |
| All_Beauty  R@100 | 0.083 | 0.072 | 0.093 | 0.145 | **0.149** | 0.296 |
| Video_Games R@10  | 0.029 | 0.035 | 0.033 | **0.050** | 0.049 | — |
| Video_Games R@50  | 0.084 | 0.085 | 0.094 | 0.125 | **0.126** | — |
| Video_Games R@100 | 0.128 | 0.115 | 0.139 | 0.175 | **0.178** | 0.241 |
| Books R@10        | 0.047 | 0.063 | 0.043 | **0.077** | 0.075 | — |
| Books R@50        | 0.084 | 0.117 | 0.090 | **0.136** | 0.135 | — |
| Books R@100       | 0.105 | 0.143 | 0.120 | **0.168** | **0.168** | 0.199 |
| Electronics R@10  | 0.028 | 0.018 | 0.028 | **0.041** | 0.039 | — |
| Electronics R@50  | 0.065 | 0.044 | 0.072 | **0.102** | 0.100 | — |
| Electronics R@100 | 0.104 | 0.063 | 0.105 | **0.128** | 0.127 | 0.148 |

Reranker reach toward the candidate-union ceiling at R@100:
**AB 50%, VG 74%, Books 84%, Electronics 86%**.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/data/download.py        # raw data, first time only
```

## Run

```bash
# Phase 0 — data layer (per category, two profiles available)
python -m scripts.data.preprocessing_pipeline --category Video_Games --profile default

# Phase 1 — retrieval
python -m scripts.retrieval.run_retrieval --category Video_Games            # popularity + rule + random
CAT=Video_Games VARIANT=metadata_and_ids EPOCHS=100 DIM=64 N_NEG=4 \
    sbatch run_two_tower.sbatch                                              # two-tower (GPU)

# Phase 2 — ranking (candidate_builder → MLP → Deep & Cross)
CAT=Video_Games sbatch run_phase2_ranker.sbatch
CAT=Video_Games SKIP_BUILDER=1 sbatch run_phase2_ranker.sbatch              # skip rebuild on re-run
```

## Repository layout

```
R-project/
├── configs/preprocessing.yaml    # default + strict profiles
├── scripts/
│   ├── data/                     # canonicalize, k-core filter, leave-last-two split, feature store, text alignment
│   ├── retrieval/                # popularity, rule_based, two_tower (+ candidate diagnostics)
│   ├── ranker/                   # candidate_builder, MLP ranker, Deep & Cross (shared train_runner)
│   └── evaluation/               # held-out-positive Recall@K / Precision@K
├── notebooks/                    # 01 EDA, 02 text→rating baselines (kept from earlier work)
├── tests/                        # data-layer invariant tests + retrieval tests
├── results/{phase0, phase1, phase2}/
├── run_two_tower.sbatch
├── run_phase2_ranker.sbatch
└── requirements.txt
```

`data/`, `logs/`, `models/`, `outputs/`, and per-user retrieval prediction
parquets are gitignored — reproducible from the scripts above.

## Project spec

[Recommendation_system_project_0425.pdf](Recommendation_system_project_0425.pdf)
