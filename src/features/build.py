"""Feature builder: joins string, vector, and context features on (s1_id, cand_id)."""

import pandas as pd


def build_pair_features(
    candidates_df: pd.DataFrame,
    records_df: pd.DataFrame,
    is_train: bool = False,
    ground_truth_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Build full pair feature set.
    Output columns: s1_id, cand_id, [label], f_str_*, f_vec_*, f_ctx_*
    """
    return candidates_df
