"""Blocker B3: Rare-token inverted index."""

import pandas as pd


def block_rare_tokens(s1_df: pd.DataFrame, cand_df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """B3: Inverted index on high-IDF name tokens."""
    return pd.DataFrame(columns=["s1_id", "cand_id", "rank_b3"])
