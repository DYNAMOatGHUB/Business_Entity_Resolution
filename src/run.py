"""Main pipeline entrypoint — Dhyanesh's stages (1, 2-vec, 3, partial 4).

Usage:
    python -m src.run --split train|test --stage all|1|2|3

Stage ownership:
    Stage 0 (normalize)          → Prabhu  [src/normalize.py]
    Stage 1 (blocking)           → Dhyanesh ← THIS FILE
    Stage 2-vec (vector feats)   → Dhyanesh ← THIS FILE
    Stage 2-str (string feats)   → Prabhu  [src/features/string_feats.py]
    Stage 3 (LightGBM / CE)      → Dhyanesh ← THIS FILE
    Stage 4 (decide + write TSV) → Prabhu  [src/decide.py / write_outputs.py]

Contract:
    Reads  : artifacts/records_norm.parquet  (from Prabhu Stage 0)
    Writes : artifacts/candidates.parquet
             artifacts/features.parquet   (f_vec_* and f_ctx_* columns only)
             artifacts/scores.parquet
             artifacts/scores_loco.parquet (D8)
             output/candidate_pairs.tsv   (validated before write)
             reports/blocking_report.md
             reports/missed_pairs.tsv
             reports/threshold_sweep.md
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import load_config, set_seed
from src.blocking.tfidf_knn import block_name_tfidf_knn, block_addr_tfidf_knn
from src.blocking.numeric_keys import block_numeric_keys
from src.blocking.union import union_blockers
from src.eval.reports import generate_blocking_report, sweep_threshold, dump_errors
from src.models.lgbm import train_lgbm_oof, predict_lgbm, train_lgbm_loco


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_records(config: dict, split: str, test_slice: bool = False) -> pd.DataFrame:
    """Load records_norm.parquet. Falls back to reading raw TSVs if not present."""
    art_path = Path(config["paths"]["artifacts_dir"]) / "records_norm.parquet"
    if art_path.exists() and not test_slice:
        return pd.read_parquet(art_path)
    
    if test_slice:
        print("[WARN] test_slice=True, ignoring cached records_norm.parquet.")

    print(f"[WARN] {art_path} not found — reading raw TSVs directly (Stage 0 not run yet).")
    data_dir = Path(config["paths"]["dataset_dir"]) / split
    frames = []
    for src_label, fname in [
        ("S1", f"{split}_source1.tsv"),
        ("S2", f"{split}_source2.tsv"),
        ("S3", f"{split}_source3.tsv"),
    ]:
        fp = data_dir / fname
        if not fp.exists():
            print(f"  [SKIP] {fp} not found.")
            continue
        df = pd.read_csv(fp, sep="\t", dtype=str)
        df["source"] = src_label
        # Rename raw columns to expected contract names
        df = df.rename(columns={
            "entity_id": "entity_id",
            "business_name": "name_raw",
            "business_address": "addr_raw",
        })
        # Fallback normalised columns (Prabhu will overwrite with proper ones)
        if "name_core" not in df.columns:
            df["name_core"] = df.get("name_raw", "")
        if "addr_norm" not in df.columns:
            df["addr_norm"] = df.get("addr_raw", "")
        if "postal" not in df.columns:
            df["postal"] = ""
        if "digits" not in df.columns:
            df["digits"] = ""
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    if test_slice:
        s1_mask = df_all["source"] == "S1"
        s1_df = df_all[s1_mask].head(500)
        cands_df = df_all[~s1_mask]
        
        # Sample candidates to prevent OOM/slow fits, ensuring country grouping isn't broken
        # Keep at least 500 records per country in candidates (if available)
        cands_sampled = cands_df.groupby("country", group_keys=False).apply(lambda x: x.head(2000))
        
        df_all = pd.concat([s1_df, cands_sampled], ignore_index=True)
        print(f"[WARN] test_slice=True: S1 sampled to {len(s1_df)}, Candidates sampled to {len(cands_sampled)}")
    return df_all


def _load_ground_truth(config: dict, split: str) -> pd.DataFrame | None:
    """Load train_ground_truth.tsv if it exists."""
    gt_path = Path(config["paths"]["dataset_dir"]) / split / f"{split}_ground_truth.tsv"
    if not gt_path.exists():
        return None
    return pd.read_csv(gt_path, sep="\t", dtype=str)


def _write_candidate_pairs_tsv(candidates: pd.DataFrame, output_dir: Path) -> Path:
    """Write output/candidate_pairs.tsv in the required validator format.

    Format: source1_entity_id <TAB> candidate_entity_ids (comma-separated)
    Every test S1 must appear exactly once (even with empty candidates).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "candidate_pairs.tsv"

    grouped = (
        candidates
        .groupby("s1_id")["cand_id"]
        .apply(lambda ids: ",".join(dict.fromkeys(ids)))  # dedup, preserve order
        .reset_index()
        .rename(columns={"s1_id": "source1_entity_id", "cand_id": "candidate_entity_ids"})
    )
    grouped.to_csv(out_path, sep="\t", index=False)
    print(f"[TSV] candidate_pairs.tsv written → {out_path}  ({len(grouped)} rows)")
    return out_path


def _validate_candidate_tsv(tsv_path: Path, test_dir: Path, validator_path: Path) -> bool:
    """Run the official validator on candidate_pairs.tsv; return True on PASS."""
    if not validator_path.exists():
        print(f"[WARN] Validator not found at {validator_path}. Skipping.")
        return True
    cmd = [
        sys.executable, str(validator_path),
        "--candidate", str(tsv_path),
        "--test-dir", str(test_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("[FAIL] Validator rejected candidate_pairs.tsv:")
        print(result.stderr)
        return False
    print("[PASS] Validator accepted candidate_pairs.tsv")
    return True


# ---------------------------------------------------------------------------
# Stage 1: Blocking
# ---------------------------------------------------------------------------

def stage_blocking(config: dict, split: str, records: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Stage 1: Blocking ===")
    s1_df = records[records["source"] == "S1"].copy()
    cand_df = records[records["source"].isin(["S2", "S3"])].copy()

    b_cfg = config["blocking"]

    # B1 — Name TF-IDF kNN
    print(f"B1: Name char-kNN  K={b_cfg['b1_name_ngram_k']}")
    b1 = block_name_tfidf_knn(s1_df, cand_df, k=b_cfg["b1_name_ngram_k"])

    # B4 — Numeric keys
    print(f"B4: Numeric keys   K={b_cfg['b4_numeric_k']}")
    b4 = block_numeric_keys(s1_df, cand_df, k=b_cfg["b4_numeric_k"])

    # B2 — Address TF-IDF kNN (D5, may not be ready yet — soft import)
    b2 = pd.DataFrame()
    try:
        print(f"B2: Addr char-kNN  K={b_cfg['b2_addr_ngram_k']}")
        b2 = block_addr_tfidf_knn(s1_df, cand_df, k=b_cfg["b2_addr_ngram_k"])
    except Exception as e:
        print(f"[SKIP] B2 failed: {e}")

    # B3 — Rare token index (D5)
    b3 = pd.DataFrame()
    try:
        from src.blocking.token_index import block_rare_token_index
        print(f"B3: Rare-token     K={b_cfg['b3_rare_tokens_k']}")
        b3 = block_rare_token_index(s1_df, cand_df, k=b_cfg["b3_rare_tokens_k"])
    except Exception as e:
        print(f"[SKIP] B3 not ready: {e}")

    # B5 — Dense embedding ANN (D5)
    b5 = pd.DataFrame()
    try:
        from src.blocking.embed_ann import block_embed_ann
        print(f"B5: Dense embed    K={b_cfg['b5_embed_ann_k']}")
        b5 = block_embed_ann(s1_df, cand_df, k=b_cfg["b5_embed_ann_k"])
    except Exception as e:
        print(f"[SKIP] B5 not ready: {e}")

    cap_k = b_cfg["cap_k_per_s1"]
    print(f"Union + cap K={cap_k}")
    candidates = union_blockers(b1, b2, b3, b4, b5, cand_df=cand_df, cap_k=cap_k)

    art_dir = Path(config["paths"]["artifacts_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)
    cands_path = art_dir / "candidates.parquet"
    candidates.to_parquet(cands_path, index=False)
    print(f"candidates.parquet → {cands_path}  ({len(candidates)} pairs, {candidates['s1_id'].nunique()} S1s)")

    # Write candidate_pairs.tsv and validate
    out_dir = Path(config["paths"]["output_dir"])
    tsv_path = _write_candidate_pairs_tsv(candidates, out_dir)
    # Validator requires test_source{1,2,3}.tsv regardless of split being processed
    test_dir = Path(config["paths"]["dataset_dir"]) / "test"
    validator = Path("../student_resource/utils/validate_submission.py")
    _validate_candidate_tsv(tsv_path, test_dir, validator)

    # Blocking report (train only)
    if split == "train":
        gt = _load_ground_truth(config, split)
        if gt is not None:
            report = generate_blocking_report(
                candidates, gt, out_path="reports/blocking_report.md"
            )
            recall_40 = report["recall_at_k"].get(40, 0)
            if recall_40 < 0.97:
                print(f"[GATE FAIL] Recall@40 = {recall_40:.4f} < 97%. "
                      "Increase K or add more blockers before proceeding.")

    return candidates


# ---------------------------------------------------------------------------
# Stage 2 (vector features) — Dhyanesh's f_vec_* and f_ctx_* columns
# ---------------------------------------------------------------------------

def stage_vector_features(
    config: dict, split: str, candidates: pd.DataFrame, records: pd.DataFrame
) -> pd.DataFrame:
    print("\n=== Stage 2: Vector & Context Features ===")
    try:
        from src.features.vector_feats import compute_vector_features
        feats = compute_vector_features(candidates, records)
    except Exception as e:
        print(f"[SKIP] vector_feats not ready: {e}")
        feats = candidates[["s1_id", "cand_id"]].copy()

    try:
        from src.features.context_feats import compute_context_features
        ctx = compute_context_features(candidates)
        feats = feats.merge(ctx, on=["s1_id", "cand_id"], how="left")
    except Exception as e:
        print(f"[SKIP] context_feats not ready: {e}")

    art_dir = Path(config["paths"]["artifacts_dir"])
    # Merge Prabhu's string features if available
    str_feats_path = art_dir / "string_features.parquet"
    if str_feats_path.exists():
        str_feats = pd.read_parquet(str_feats_path)
        feats = feats.merge(str_feats, on=["s1_id", "cand_id"], how="left")

    feats_path = art_dir / "features.parquet"
    feats.to_parquet(feats_path, index=False)
    print(f"features.parquet → {feats_path}  ({feats.shape[1]} cols)")
    return feats


# ---------------------------------------------------------------------------
# Stage 3: Score
# ---------------------------------------------------------------------------

def stage_score(
    config: dict, split: str, features: pd.DataFrame, records: pd.DataFrame
) -> pd.DataFrame:
    print("\n=== Stage 3: LightGBM Scoring ===")
    art_dir = Path(config["paths"]["artifacts_dir"])
    m_cfg = config["models"]
    seeds = m_cfg.get("lgbm_seeds", [42])
    n_folds = m_cfg.get("n_folds", 5)

    if split == "train":
        if "label" not in features.columns:
            print("[WARN] No 'label' column in features.parquet. Skipping training.")
            return features

        print(f"Training LightGBM OOF: {n_folds} folds × {len(seeds)} seeds")
        oof_df, models = train_lgbm_oof(
            features, n_folds=n_folds, seeds=seeds, model_dir=config["paths"]["models_dir"]
        )

        # Merge p_lgbm into a scores frame
        scores = oof_df.copy()
        scores["p_ce"] = np.nan
        scores["p_final"] = scores["p_lgbm"]
        scores_path = art_dir / "scores.parquet"
        scores.to_parquet(scores_path, index=False)
        print(f"scores.parquet → {scores_path}")

        # D8: LOCO run
        train_countries = features.merge(
            records[records["source"] == "S1"][["entity_id", "country"]].rename(
                columns={"entity_id": "s1_id"}
            ),
            on="s1_id", how="left"
        )["country"].dropna().unique().tolist()

        if len(train_countries) >= 2:
            loco_frames = []
            for hoc in train_countries:
                print(f"LOCO: held-out country = {hoc}")
                try:
                    loco_df = train_lgbm_loco(
                        features, records, held_out_country=hoc, seeds=seeds
                    )
                    loco_frames.append(loco_df)
                except Exception as e:
                    print(f"  [WARN] LOCO for {hoc} failed: {e}")
            if loco_frames:
                loco_all = pd.concat(loco_frames, ignore_index=True)
                loco_path = art_dir / "scores_loco.parquet"
                loco_all.to_parquet(loco_path, index=False)
                print(f"scores_loco.parquet → {loco_path}")

        # Threshold sweep — give Prabhu the optimal cut
        gt = _load_ground_truth(config, split)
        if gt is not None:
            sweep_threshold(scores, gt, out_path="reports/threshold_sweep.md")

        return scores

    else:
        # Test: load persisted models
        models_dir = Path(config["paths"]["models_dir"])
        import joblib
        model_files = sorted(models_dir.glob("lgbm_fold*.pkl"))
        if not model_files:
            print("[WARN] No saved models found. Run train split first.")
            return features[["s1_id", "cand_id"]].assign(p_lgbm=np.nan)

        models = [joblib.load(p) for p in model_files]
        print(f"Loaded {len(models)} models from {models_dir}")
        preds = predict_lgbm(features, models)
        preds["p_ce"] = np.nan
        preds["p_final"] = preds["p_lgbm"]
        scores_path = art_dir / "scores.parquet"
        preds.to_parquet(scores_path, index=False)
        print(f"scores.parquet → {scores_path}")
        return preds


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run_pipeline(
    split: str = "test",
    stage: str = "all",
    config_path: str = "configs/default.yaml",
    test_slice: bool = False,
):
    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    print(f"\n{'='*60}")
    print(f"  Business Entity Resolution Pipeline")
    print(f"  split={split}   stage={stage}")
    if test_slice:
        print("  MODE: TEST SLICE (500 S1 records max)")
    print(f"{'='*60}\n")

    records = _load_records(config, split, test_slice=test_slice)
    print(f"Records loaded: {len(records)} rows  "
          f"(S1={len(records[records['source']=='S1'])}, "
          f"S2={len(records[records['source']=='S2'])}, "
          f"S3={len(records[records['source']=='S3'])})")

    run_stage = lambda s: stage in ("all", s)

    candidates = None
    features = None

    if run_stage("1"):
        candidates = stage_blocking(config, split, records)

    if run_stage("2"):
        if candidates is None:
            cands_path = Path(config["paths"]["artifacts_dir"]) / "candidates.parquet"
            candidates = pd.read_parquet(cands_path)
        features = stage_vector_features(config, split, candidates, records)

    if run_stage("3"):
        if features is None:
            feats_path = Path(config["paths"]["artifacts_dir"]) / "features.parquet"
            features = pd.read_parquet(feats_path)
        stage_score(config, split, features, records)

    print("\nDone. Hand off artifacts/ to Prabhu for Stage 4 (decide + write_outputs).")


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution — Dhyanesh's stages")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--stage", choices=["all", "1", "2", "3"], default="all")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--test-slice", action="store_true", help="Run on a small subset for testing")
    args = parser.parse_args()
    run_pipeline(split=args.split, stage=args.stage, config_path=args.config, test_slice=args.test_slice)



if __name__ == "__main__":
    main()
