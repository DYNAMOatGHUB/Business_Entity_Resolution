# Business Entity Resolution: Amazon ML Challenge 2026

Match every Source 1 business record to all Source 2 and Source 3 records for the same real-world business. A Source 1 entity can match zero, one or many records. The leaderboard metric is macro F0.5, which weights precision twice as much as recall.

**Team:** Dhyanesh, Prabhu

**Pipeline:** normalize → block (5 blockers) → pair features → LightGBM + cross-encoder → decision layer → `matching_results.tsv`

---

## Repository structure

```
business_entity_resolution/
├── README.md                       # this file
├── requirements.txt                # pinned dependencies
├── .gitignore                      # ignores dataset/, artifacts/, output/*.tsv, model weights
│
├── configs/
│   └── default.yaml                # K per blocker, thresholds, model names, seeds, paths
│
├── dataset/                        # NOT committed. Copy from student_resource/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── utils/
│   └── validate_submission.py      # official validator (provided, do not edit)
│
├── src/
│   ├── __init__.py
│   ├── run.py                      # entrypoint: --split train|test --stage all|0|1|2|3|4
│   ├── config.py                   # loads configs/default.yaml, sets seeds
│   ├── io_utils.py                 # TSV read/write, ID-list parsing and formatting
│   │
│   ├── normalize.py                # Stage 0: unicode fold, legal suffixes, address abbreviations, digit/postal extraction
│   │
│   ├── blocking/                   # Stage 1: candidate generation
│   │   ├── __init__.py
│   │   ├── tfidf_knn.py            # B1 name char n-gram kNN, B2 address char n-gram kNN
│   │   ├── token_index.py          # B3 rare-token inverted index
│   │   ├── numeric_keys.py         # B4 shared postal / digit tokens
│   │   ├── embed_ann.py            # B5 multilingual MiniLM + FAISS
│   │   └── union.py                # merge blockers, blend rank, cap K per S1
│   │
│   ├── features/                   # Stage 2: pair features
│   │   ├── __init__.py
│   │   ├── build.py                # joins all feature groups on (s1_id, cand_id)
│   │   ├── string_feats.py         # f_str_*: rapidfuzz sims, token overlap, postal/digit checks
│   │   ├── vector_feats.py         # f_vec_*: TF-IDF and embedding cosines
│   │   └── context_feats.py        # f_ctx_*: blocker hits/ranks, per-S1 relative, reverse context
│   │
│   ├── models/                     # Stage 3: scoring
│   │   ├── __init__.py
│   │   ├── lgbm.py                 # GroupKFold OOF training, seed ensemble, test inference
│   │   └── cross_encoder.py        # multilingual pair classifier fine-tune (OOF)
│   │
│   ├── decide.py                   # Stage 4: no-match gate, pair threshold, relative cut, one-owner, unseen-country margin
│   ├── write_outputs.py            # writes both TSVs, runs the validator
│   │
│   └── eval/
│       ├── __init__.py
│       ├── score.py                # exact macro F0.5 (per S1 entity, singletons included)
│       ├── threshold_search.py     # grid search on OOF scores
│       └── reports.py              # blocking recall@K, error dumps
│
├── tests/
│   ├── test_score.py               # spec example must return 0.714
│   ├── test_normalize.py
│   └── test_outputs.py             # format rules: one row per S1, no dups, matches ⊆ candidates
│
├── notebooks/
│   └── eda.ipynb                   # dataset stats, singleton %, noise patterns
│
├── scripts/
│   ├── setup_models.py             # one-time download of model weights to models/
│   └── make_zip.sh                 # builds <team_name>_submission.zip
│
├── models/                         # NOT committed. Cached pretrained weights
├── artifacts/                      # NOT committed. Stage outputs
│   ├── records_norm.parquet        # Stage 0
│   ├── candidates.parquet          # Stage 1
│   ├── features.parquet            # Stage 2
│   ├── scores.parquet              # Stage 3 (OOF for train, predictions for test)
│   ├── scores_loco.parquet         # leave-one-country-out predictions
│   └── emb_<split>.npy             # cached embeddings
│
├── output/
│   ├── matching_results.tsv        # final matches (leaderboard upload)
│   └── candidate_pairs.tsv         # candidate set scored by the model
│
├── reports/
│   ├── experiments.csv             # one row per experiment / upload
│   ├── threshold_grid.csv
│   ├── blocking_report.md
│   ├── errors_<date>.tsv
│   └── uploads/lb-<n>/             # exact TSVs of every leaderboard upload
│
└── docs/
    └── Documentation_template.md   # methodology write-up for the final zip
```

---

## Setup

Python 3.11 is recommended.

```bash
git clone <repo-url> business_entity_resolution
cd business_entity_resolution
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# GPU machine (RTX 50-series needs CUDA 12.8 wheels)
pip install torch --index-url https://download.pytorch.org/whl/cu128
# CPU-only machine
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
python scripts/setup_models.py       # downloads weights once into models/
```

Copy the challenge data into `dataset/` using the layout shown in the tree above, and copy `utils/validate_submission.py` from `student_resource/`.

Check the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Running the pipeline

The same code runs on train and test. The only difference is whether labels exist.

```bash
# Full test run: data → blocking → features → scoring → outputs
python -m src.run --split test --stage all

# Train with out-of-fold predictions (for validation and threshold tuning)
python -m src.run --split train --stage all

# Run a single stage (reads the previous stage's parquet from artifacts/)
python -m src.run --split test --stage 1     # 0 normalize | 1 block | 2 features | 3 score | 4 decide
```

Retune thresholds on the OOF scores without retraining:

```bash
python -m src.eval.threshold_search --scores artifacts/scores.parquet
```

Validate before every upload:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Upload only when it prints `PASS`.

Run the tests:

```bash
pytest tests/
```

---

## Pipeline

| Stage | Module | Output | What it does |
|---|---|---|---|
| 0. Normalize | `src/normalize.py` | `records_norm.parquet` | Accent folding, lowercasing, `&`→and, legal suffixes split into their own field, address abbreviations expanded, digits and postal codes extracted. The same rules apply to every country |
| 1. Block | `src/blocking/` | `candidates.parquet`, `candidate_pairs.tsv` | Union of 5 blockers, searched within the same country string, capped at K per S1 |
| 2. Features | `src/features/` | `features.parquet` | ~50 pair features: string sims, structure checks, vector cosines, blocking context |
| 3. Score | `src/models/` | `scores.parquet` | LightGBM (5-fold GroupKFold by S1), optionally stacked with a cross-encoder |
| 4. Decide | `src/decide.py` | `matching_results.tsv` | Thresholds tuned directly on macro F0.5. Leans precise |

### Blockers

| # | Blocker | Method | Catches |
|---|---|---|---|
| B1 | Name n-gram | TF-IDF char 2-4 grams, cosine kNN | typos, abbreviations |
| B2 | Address n-gram | TF-IDF char 2-4 grams on address | DBA / trade names at the same address |
| B3 | Rare tokens | inverted index on high-IDF name tokens | reordered or partial names |
| B4 | Numeric keys | shared postal code or number tokens | address worded completely differently |
| B5 | Embeddings | multilingual MiniLM + FAISS | transliteration, French text |

`candidate_pairs.tsv` holds exactly the pairs the model scores. Nothing is filtered between blocking and scoring.

### Decision layer (per S1 entity)

1. **No-match gate:** if the best score is below `tau_gate`, output an empty row. A correct singleton earns a full 1.0.
2. **Pair threshold:** keep candidates scoring at or above `tau_pair`.
3. **Relative cut:** drop candidates scoring below `alpha × best score`.
4. **One-owner:** each S2/S3 record goes only to its highest-scoring S1.
5. **Unseen-country margin:** countries not present in train (such as France) get stricter thresholds, sized by a leave-one-country-out run.

---

## Data contracts

Stages communicate only through these files. Don't rename columns without telling the other person.

| File | Columns |
|---|---|
| `records_norm.parquet` | entity_id, source, country, name_raw, addr_raw, name_core, name_legal, name_sorted, name_acronym, addr_norm, digits, postal, addr_first_num, landmark_flag |
| `candidates.parquet` | s1_id, cand_id, cand_source, hit_b1..hit_b5, rank_b1..rank_b5, sim_b1, sim_b5, blend_rank |
| `features.parquet` | s1_id, cand_id, label (train only), `f_str_*`, `f_vec_*`, `f_ctx_*` |
| `scores.parquet` | s1_id, cand_id, fold (train only), p_lgbm, p_ce, p_final |

---

## Validation

- **Scorer:** `src/eval/score.py` replicates the official metric. F0.5 is computed per S1 entity and averaged, with singletons included. An empty prediction on an empty truth scores 1.0. Any prediction on an empty truth scores 0.0.
- **OOF:** 5-fold GroupKFold grouped by S1 id. Used for model selection and threshold tuning.
- **Leave-one-country-out:** train on US and score India, then the reverse. This is our stand-in for the unseen France test data.
- **Blocking gate:** 97%+ recall of true pairs at the chosen K.

Every experiment and upload is logged in `reports/experiments.csv` with its git hash, OOF F0.5, LOCO F0.5 and public LB score. Uploaded commits are tagged `lb-<n>`.

---

## Models and licenses

The competition rules allow only MIT or Apache 2.0 models with 8B parameters or fewer. The pipeline makes no external lookups, API calls or geocoding.

| Model | Use | License | Params |
|---|---|---|---|
| LightGBM | pair classifier | MIT | n/a |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | B5 blocking, vector features | Apache 2.0 | 118M |
| `intfloat/multilingual-e5-small` or `xlm-roberta-base` | cross-encoder | MIT | 118M / 278M |

Weights are downloaded once by `scripts/setup_models.py`. Pipeline runs never touch the network.

---

## Team ownership

| Area | Owner |
|---|---|
| Scorer, EDA, `normalize.py`, `string_feats.py`, `decide.py`, `threshold_search.py`, `write_outputs.py`, uploads, docs | Prabhu |
| Repo skeleton, `config.py`, `run.py`, `blocking/*`, `vector_feats.py`, `context_feats.py`, `models/*`, full-data runs, `make_zip.sh` | Dhyanesh |

### Git workflow

- `main` always runs end to end.
- Name branches by stage, e.g. `stage1/b5-embeddings` or `stage4/relative-cut`.
- Keep PRs small. Run `pytest tests/` before merging.
- Never commit `dataset/`, `artifacts/`, `models/` or large TSVs.

---

## Final submission package

`scripts/make_zip.sh` builds:

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── configs/
│       ├── scripts/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

Before submitting:

- [ ] The validator prints `PASS` on both TSVs
- [ ] The uploaded `matching_results.tsv` is identical to the one in the zip
- [ ] A fresh clone plus a clean venv plus `python -m src.run --split test --stage all` reproduces both outputs
- [ ] `requirements.txt` is pinned
- [ ] No network calls anywhere in `src/`