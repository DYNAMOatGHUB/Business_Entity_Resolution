"""Vector similarity features (f_vec_*): TF-IDF and embedding cosines."""

import pandas as pd


def compute_vector_features(pairs_df: pd.DataFrame, records_df: pd.DataFrame) -> pd.DataFrame:
    """Compute vector cosine similarities for TF-IDF and embeddings."""
    return pairs_df
