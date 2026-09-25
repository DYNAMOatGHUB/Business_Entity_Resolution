"""Tests for Stage 0 normalization."""

import pytest
import pandas as pd
from src.normalize import normalize_records


def test_normalize_empty_df():
    df = pd.DataFrame(columns=["entity_id", "source", "country", "name", "address"])
    norm = normalize_records(df)
    assert isinstance(norm, pd.DataFrame)
