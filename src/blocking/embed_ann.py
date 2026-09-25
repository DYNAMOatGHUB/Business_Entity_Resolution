"""Blocker B5: Multilingual MiniLM + FAISS approximate nearest neighbors."""

import pandas as pd


def block_embed_ann(s1_df: pd.DataFrame, cand_df: pd.DataFrame, k: int = 25) -> pd.DataFrame:
    """B5: Multilingual MiniLM + FAISS candidate generation."""
    return pd.DataFrame(columns=["s1_id", "cand_id", "sim_b5", "rank_b5"])
