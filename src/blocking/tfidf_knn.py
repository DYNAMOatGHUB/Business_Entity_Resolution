"""Blocker B1 & B2: TF-IDF char n-gram kNN on name and address.

Design notes (PRD D2):
- Fit vectorizer on ALL records (S1 + S2 + S3) of the split, unsupervised.
- Run kNN within the same country; fall back to global if group < 50 records.
- Chunk S1 matrix in 2 000-row slices so sparse dot products fit in RAM.
- min_sim threshold prunes zero-signal candidates early.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
from tqdm import tqdm

# Drop pairs whose cosine similarity is below this — they add noise but not recall.
_MIN_SIM = 1e-4
_CHUNK = 2_000   # S1 rows per sparse-dot chunk


def _sparse_top_k(
    s1_vecs: sp.csr_matrix,
    cand_vecs: sp.csr_matrix,
    s1_ids: np.ndarray,
    cand_ids: np.ndarray,
    k: int,
    prefix: str,
) -> list[tuple]:
    """Chunk-wise sparse cosine top-K without sklearn overhead.

    Returns a list of (s1_id, cand_id, sim, rank) tuples.
    """
    from sklearn.utils.extmath import safe_sparse_dot
    results: list[tuple] = []
    n_s1 = s1_vecs.shape[0]

    # Dynamically pick chunk size so dense result is ~2GB max.
    # 2GB / (n_cand * 4 bytes)
    n_cand = cand_vecs.shape[0]
    safe_chunk = max(1, int(2e9 / (n_cand * 4 + 1)))
    _chunk_size = min(500, safe_chunk)
    
    cand_vecs_t = cand_vecs.T

    for start in range(0, n_s1, _chunk_size):
        chunk = s1_vecs[start : start + _chunk_size]       # (chunk_size, vocab)
        
        # Dense output avoids sparse index memory blowup (the OOM cause)
        scores = safe_sparse_dot(chunk, cand_vecs_t, dense_output=True)

        chunk_size = scores.shape[0]
        k_eff = min(k, scores.shape[1])

        # Partial sort: argpartition then sort the top slice
        top_idx = np.argpartition(scores, -k_eff, axis=1)[:, -k_eff:]
        for local_i in range(chunk_size):
            row_scores = scores[local_i, top_idx[local_i]]
            order = np.argsort(-row_scores)            # descending sim
            s1_id = s1_ids[start + local_i]
            for rank, oi in enumerate(order, 1):
                sim = float(row_scores[oi])
                if sim < _MIN_SIM:
                    break
                cand_id = cand_ids[top_idx[local_i][oi]]
                results.append((s1_id, cand_id, sim, rank))

    return results

def _tfidf_knn_base(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    text_col: str,
    k: int,
    prefix: str,
) -> pd.DataFrame:
    """Fit TF-IDF on all records, then run per-country chunked kNN."""

    s1_id_arr = s1_df["entity_id"].values
    cand_id_arr = cand_df["entity_id"].values

    countries = pd.unique(pd.concat([s1_df["country"], cand_df["country"]]))

    all_results: list[tuple] = []
    small_country_s1_indices = []

    import gc
    for country in tqdm(countries, desc=f"kNN({prefix})"):
        s1_mask = s1_df["country"].values == country
        cand_mask = cand_df["country"].values == country

        n_group = s1_mask.sum() + cand_mask.sum()
        if n_group < 50:
            # Accumulate tiny-country S1 entities to fall back to global cand pool later
            small_country_s1_indices.extend(np.where(s1_mask)[0].tolist())
            continue
            
        s1_rows = np.where(s1_mask)[0]
        cand_rows = np.where(cand_mask)[0]
        if len(s1_rows) == 0 or len(cand_rows) == 0:
            continue
            
        # Localize DataFrames
        local_s1_df = s1_df.iloc[s1_rows]
        local_cand_df = cand_df.iloc[cand_rows]
        
        # Localize Vectorization
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), dtype=np.float32)
        fit_texts = pd.concat([local_s1_df[text_col], local_cand_df[text_col]]).fillna("").tolist()
        vec.fit(fit_texts)
        
        s1_mat_local = vec.transform(local_s1_df[text_col].fillna("")).tocsr()
        cand_mat_local = vec.transform(local_cand_df[text_col].fillna("")).tocsr()

        all_results.extend(
            _sparse_top_k(
                s1_mat_local,
                cand_mat_local,
                s1_id_arr[s1_rows],
                cand_id_arr[cand_rows],
                k,
                prefix,
            )
        )
        
        # Flush RAM
        del vec, s1_mat_local, cand_mat_local, local_s1_df, local_cand_df, fit_texts
        gc.collect()

    # Process all tiny countries against the global candidate pool in ONE pass (no slicing of cand_mat_global)
    if small_country_s1_indices:
        local_s1_df = s1_df.iloc[small_country_s1_indices]
        
        all_texts_df = pd.concat([local_s1_df[text_col], cand_df[text_col]]).fillna("")
        if len(all_texts_df) > 2_000_000:
            fit_texts = all_texts_df.sample(n=2_000_000, random_state=42).tolist()
        else:
            fit_texts = all_texts_df.tolist()
            
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), dtype=np.float32)
        vec.fit(fit_texts)
        
        s1_mat_global = vec.transform(local_s1_df[text_col].fillna("")).tocsr()
        cand_mat_global = vec.transform(cand_df[text_col].fillna("")).tocsr()
        
        all_results.extend(
            _sparse_top_k(
                s1_mat_global,
                cand_mat_global,
                local_s1_df["entity_id"].values,
                cand_id_arr,
                k,
                prefix,
            )
        )
        del vec, s1_mat_global, cand_mat_global, all_texts_df, fit_texts, local_s1_df
        gc.collect()

    if not all_results:
        return pd.DataFrame(
            columns=["s1_id", "cand_id", f"sim_{prefix}", f"rank_{prefix}"]
        )

    df = pd.DataFrame(all_results, columns=["s1_id", "cand_id", "sim", "rank"])

    # Keep best sim per (s1_id, cand_id) in case country fall-back caused duplicates
    df = (
        df.sort_values("sim", ascending=False)
        .drop_duplicates(subset=["s1_id", "cand_id"])
        .rename(columns={"sim": f"sim_{prefix}", "rank": f"rank_{prefix}"})
        .reset_index(drop=True)
    )
    return df


def block_name_tfidf_knn(
    s1_df: pd.DataFrame, cand_df: pd.DataFrame, k: int = 20
) -> pd.DataFrame:
    """B1: Name char n-gram kNN (PRD §D2)."""
    col = "name_core" if "name_core" in s1_df.columns else "business_name"
    # Make copies only if we need to add a column
    if col not in s1_df.columns:
        s1_df = s1_df.copy()
        cand_df = cand_df.copy()
        s1_df[col] = ""
        cand_df[col] = ""
    return _tfidf_knn_base(s1_df, cand_df, col, k, "b1")


def block_addr_tfidf_knn(
    s1_df: pd.DataFrame, cand_df: pd.DataFrame, k: int = 15
) -> pd.DataFrame:
    """B2: Address char n-gram kNN (PRD §D5)."""
    col = "addr_norm" if "addr_norm" in s1_df.columns else "business_address"
    if col not in s1_df.columns:
        s1_df = s1_df.copy()
        cand_df = cand_df.copy()
        s1_df[col] = ""
        cand_df[col] = ""
    return _tfidf_knn_base(s1_df, cand_df, col, k, "b2")
