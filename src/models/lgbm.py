"""LightGBM pair scoring: GroupKFold OOF training, seed ensemble, and inference."""

import pandas as pd


def train_lgbm_oof(features_df: pd.DataFrame, n_folds: int = 5, seeds: list = None) -> pd.DataFrame:
    """Train LightGBM with GroupKFold grouped by s1_id and return OOF predictions."""
    return features_df


def predict_lgbm(features_df: pd.DataFrame, models: list) -> pd.DataFrame:
    """Predict match probabilities on test features."""
    return features_df
