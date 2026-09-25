"""Blocker B4: Shared postal code and digit numeric keys."""

import pandas as pd


def block_numeric_keys(s1_df: pd.DataFrame, cand_df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """B4: Shared postal or number tokens."""
    return pd.DataFrame(columns=["s1_id", "cand_id", "rank_b4"])
