"""Synthetic OOF train scores from string features, to smoke-test decide / threshold search.

p_final = mean(f_str_name_tset, f_str_addr_tset), with NaN -> 0 for this synthetic score only
(features keep NaN for missing information). Each S1 gets one random fold (seed from config),
mimicking GroupKFold by S1. Output: ``<artifacts_dir>/scores_synth_train.parquet``, named so it
never collides with the real ``scores_train.parquet``; pass it with ``--scores`` to
``src.decide`` / ``src.eval.threshold_search``.

Run: python -m scripts.synthetic_scores [--sample] [--config PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.contracts import check_artifact, load_artifact

FEATURES = ["f_str_name_tset", "f_str_addr_tset"]
OUT_NAME = "scores_synth_train.parquet"


def synthetic_scores(features: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    """Build a scores-contract frame (s1_id, cand_id, fold, p_lgbm, p_final) from string features."""
    missing = [c for c in FEATURES if c not in features.columns]
    if missing:
        raise ValueError(f"features_str_train lacks {missing}; synthetic scores need {FEATURES}")
    vals = features[FEATURES].astype(float)
    lo, hi = np.nanmin(vals.to_numpy()), np.nanmax(vals.to_numpy())
    if lo < 0 or hi > 1:
        raise ValueError(f"{FEATURES} must be in [0, 1] (got min {lo}, max {hi})")
    p = vals.fillna(0.0).mean(axis=1).to_numpy()

    s1_codes, s1_uniques = pd.factorize(features["s1_id"])
    fold_per_s1 = np.random.default_rng(seed).integers(0, n_folds, size=len(s1_uniques))
    out = features[["s1_id", "cand_id"]].copy()  # keep Arrow-backed ids: no per-row Python objects
    out["fold"] = fold_per_s1[s1_codes].astype(np.int16)
    out["p_lgbm"] = p
    out["p_final"] = p
    return out.reset_index(drop=True)


def main() -> None:
    """CLI: read features_str_train, write scores_synth_train.parquet."""
    parser = argparse.ArgumentParser(description="Synthetic OOF scores from string features")
    parser.add_argument("--sample", action="store_true", help="keep sample.frac of S1 ids (seed sample.seed)")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    feats = load_artifact("features_str", "train", sample=args.sample, cfg=cfg, columns=["s1_id", "cand_id", *FEATURES])
    out = synthetic_scores(feats, int(cfg["models"]["n_folds"]), int(cfg["seed"]))
    check_artifact(out, "scores", "train")
    path = Path(resolve_path(cfg["paths"]["artifacts_dir"])) / OUT_NAME
    out.to_parquet(path, index=False)
    print(f"[synthetic_scores] {len(out):,} pairs, {out['s1_id'].nunique():,} S1, p_final mean {out['p_final'].mean():.3f} -> {path}")


if __name__ == "__main__":
    main()
