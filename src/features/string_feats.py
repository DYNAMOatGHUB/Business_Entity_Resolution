"""String similarity and structure check features (f_str_*).
rapidfuzz similarities, token overlap, postal/digit checks.
"""

import pandas as pd


def compute_string_features(pairs_df: pd.DataFrame, records_df: pd.DataFrame) -> pd.DataFrame:
    """Compute rapidfuzz sims, token overlap, and postal/digit match indicators."""
    return pairs_df
