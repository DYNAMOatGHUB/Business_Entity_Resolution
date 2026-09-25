"""Blocker B1 & B2: TF-IDF char n-gram kNN on name and address."""

import pandas as pd


def block_name_tfidf_knn(s1_df: pd.DataFrame, cand_df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """B1: Name char n-gram kNN."""
    return pd.DataFrame(columns=["s1_id", "cand_id", "sim_b1", "rank_b1"])


def block_addr_tfidf_knn(s1_df: pd.DataFrame, cand_df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """B2: Address char n-gram kNN."""
    return pd.DataFrame(columns=["s1_id", "cand_id", "sim_b2", "rank_b2"])
