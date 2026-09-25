"""LightGBM pair scoring: GroupKFold OOF training, seed ensemble, test inference.

PRD §D4 parameters:
- Binary objective, 500 trees, lr 0.05, num_leaves 63, feature_fraction 0.8,
  bagging 0.8, early stopping 50 on log loss.
- GroupKFold(5) grouped by s1_id.
- TF-IDF / IDF weights fit inside each fold for honest OOF.
- OOF probability written to scores.parquet with a `fold` column.

F0.5 note: LightGBM optimises log-loss, not F0.5.  The threshold sweep in
eval/reports.py then finds the cut-off that maximises macro F0.5 on OOF preds.
num_threads is set to all physical cores automatically.
"""

from __future__ import annotations

import os
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold


# ── default hyperparams (PRD §D4) ──────────────────────────────────────────
_DEFAULT_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_estimators": 500,
    "verbose": -1,
    "num_threads": os.cpu_count() or 4,    # all physical cores
}

_EARLY_STOP = 50
_NON_FEATURE_COLS = {"s1_id", "cand_id", "label", "fold", "low_confidence"}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


# ---------------------------------------------------------------------------
# OOF training (D4)
# ---------------------------------------------------------------------------

def train_lgbm_oof(
    features_df: pd.DataFrame,
    n_folds: int = 5,
    seeds: list[int] | None = None,
    params: dict | None = None,
    model_dir: str | Path = "models",
) -> tuple[pd.DataFrame, list]:
    """Train LightGBM with GroupKFold(s1_id) and return OOF predictions + models.

    Parameters
    ----------
    features_df : must contain s1_id, cand_id, label, plus feature columns.
    seeds       : one model trained per seed per fold; predictions averaged.
    params      : override default LightGBM hyperparams.
    model_dir   : directory to persist trained fold models.

    Returns
    -------
    oof_df : pd.DataFrame with s1_id, cand_id, fold, p_lgbm
    models : flat list of trained LGBMClassifier objects (n_folds × n_seeds)
    """
    if seeds is None:
        seeds = [42]
    hp = {**_DEFAULT_PARAMS, **(params or {})}

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    feat_cols = _feature_cols(features_df)
    gkf = GroupKFold(n_splits=n_folds)
    groups = features_df["s1_id"].values
    X = features_df[feat_cols].values.astype(np.float32)
    y = features_df["label"].values.astype(np.int8)

    oof_preds = np.zeros(len(features_df), dtype=np.float64)
    fold_col = np.full(len(features_df), -1, dtype=np.int8)
    all_models: list = []

    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        fold_col[val_idx] = fold_idx
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        fold_preds = np.zeros(len(val_idx), dtype=np.float64)

        for seed in seeds:
            seed_hp = {**hp, "seed": seed, "data_random_seed": seed}
            clf = lgb.LGBMClassifier(**seed_hp)
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(_EARLY_STOP, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )
            fold_preds += clf.predict_proba(X_val)[:, 1]
            all_models.append(clf)

            # Persist to disk
            model_path = Path(model_dir) / f"lgbm_fold{fold_idx}_seed{seed}.pkl"
            joblib.dump(clf, model_path)

        oof_preds[val_idx] = fold_preds / len(seeds)

        pos_rate = y[val_idx].mean()
        mean_pred = oof_preds[val_idx].mean()
        print(
            f"  Fold {fold_idx}: val_size={len(val_idx)}  "
            f"pos_rate={pos_rate:.3f}  mean_pred={mean_pred:.3f}"
        )

    oof_df = features_df[["s1_id", "cand_id"]].copy()
    oof_df["fold"] = fold_col
    oof_df["p_lgbm"] = oof_preds

    return oof_df, all_models


# ---------------------------------------------------------------------------
# Test inference (D4 / D9)
# ---------------------------------------------------------------------------

def predict_lgbm(
    features_df: pd.DataFrame,
    models: list,
) -> pd.DataFrame:
    """Average predictions from all fold models for test inference."""
    feat_cols = _feature_cols(features_df)
    X = features_df[feat_cols].values.astype(np.float32)

    preds = np.zeros(len(features_df), dtype=np.float64)
    for clf in models:
        preds += clf.predict_proba(X)[:, 1]
    preds /= max(len(models), 1)

    out = features_df[["s1_id", "cand_id"]].copy()
    out["p_lgbm"] = preds
    return out


# ---------------------------------------------------------------------------
# LOCO training helper (D8)
# ---------------------------------------------------------------------------

def train_lgbm_loco(
    features_df: pd.DataFrame,
    records_df: pd.DataFrame,
    held_out_country: str,
    seeds: list[int] | None = None,
    params: dict | None = None,
) -> pd.DataFrame:
    """Leave-one-country-out: train on all countries except held_out_country,
    predict on held_out_country.  Returns a DataFrame like oof_df."""
    if seeds is None:
        seeds = [42]
    hp = {**_DEFAULT_PARAMS, **(params or {})}

    feat_cols = _feature_cols(features_df)

    # Join country onto features via s1_id
    s1_country = records_df[records_df["source"] == "S1"][["entity_id", "country"]].rename(
        columns={"entity_id": "s1_id"}
    )
    df = features_df.merge(s1_country, on="s1_id", how="left")

    train_mask = df["country"] != held_out_country
    test_mask = df["country"] == held_out_country

    X_tr = df.loc[train_mask, feat_cols].values.astype(np.float32)
    y_tr = df.loc[train_mask, "label"].values.astype(np.int8)
    X_te = df.loc[test_mask, feat_cols].values.astype(np.float32)

    preds = np.zeros(test_mask.sum(), dtype=np.float64)
    for seed in seeds:
        seed_hp = {**hp, "seed": seed, "data_random_seed": seed}
        clf = lgb.LGBMClassifier(**seed_hp)
        clf.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(period=-1)])
        preds += clf.predict_proba(X_te)[:, 1]
    preds /= max(len(seeds), 1)

    out = df.loc[test_mask, ["s1_id", "cand_id"]].copy()
    out["p_lgbm"] = preds
    out["loco_held_out"] = held_out_country
    return out.reset_index(drop=True)
