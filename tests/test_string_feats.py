"""Tests for string feature extraction."""

import pytest
import pandas as pd
from src.features.string_feats import compute_string_features


def test_compute_string_features():
    pairs = pd.DataFrame([{"s1_id": "s1_1", "cand_id": "s2_1"}])
    records = pd.DataFrame()
    feats = compute_string_features(pairs, records)
    assert len(feats) == 1
