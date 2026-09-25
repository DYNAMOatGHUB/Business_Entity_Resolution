"""Multilingual pair classifier cross-encoder fine-tuning and inference."""

import pandas as pd


def train_cross_encoder(pairs_df: pd.DataFrame, records_df: pd.DataFrame, model_name: str) -> None:
    """Fine-tune cross-encoder for entity resolution pair classification."""
    pass


def predict_cross_encoder(pairs_df: pd.DataFrame, records_df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """Predict cross-encoder scores for candidate pairs."""
    return pairs_df
