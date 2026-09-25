"""Tests for decision rules layer."""

import pytest
import pandas as pd
from src.decide import apply_decision_rules


def test_decision_rules_empty():
    scores = pd.DataFrame(columns=["s1_id", "cand_id", "p_final"])
    matches = apply_decision_rules(scores)
    assert "s1_id" in matches.columns
