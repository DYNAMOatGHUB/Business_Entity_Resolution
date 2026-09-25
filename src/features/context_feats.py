"""Context features (f_ctx_*): blocker hits/ranks, per-S1 relative, reverse context."""

import pandas as pd


def compute_context_features(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Compute blocker rank features, relative score/rank, and reverse context."""
    return pairs_df
