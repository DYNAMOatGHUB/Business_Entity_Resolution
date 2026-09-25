"""Tests for the macro F0.5 scorer."""

import pandas as pd
import pytest

from src.eval.score import breakdown, f05_entity, load_pred_tsv, load_truth, macro_f05


def test_spec_example():
    """Problem-statement example: P = 2/3, R = 1.0 -> 0.714."""
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    truth = {"S2-00047", "S3-00812"}
    assert f05_entity(pred, truth) == pytest.approx(0.7142857, abs=1e-6)


def test_empty_truth_empty_pred():
    assert f05_entity(set(), set()) == 1.0


def test_empty_truth_with_pred():
    assert f05_entity({"S2-1"}, set()) == 0.0


def test_truth_with_empty_pred():
    assert f05_entity(set(), {"S2-1"}) == 0.0


def test_perfect_match():
    ids = {"S2-1", "S3-2", "S3-3"}
    assert f05_entity(set(ids), set(ids)) == 1.0


def test_no_overlap():
    assert f05_entity({"S2-9"}, {"S2-1"}) == 0.0


def test_missing_s1_counts_as_empty():
    truth = {"S1-a": {"S2-1"}, "S1-b": set()}
    # S1-a missing -> empty pred on non-empty truth (0.0); S1-b missing -> empty/empty (1.0)
    assert macro_f05({}, truth) == pytest.approx(0.5)


def test_macro_average_three_entities():
    truth = {
        "S1-a": {"S2-00047", "S3-00812"},
        "S1-b": set(),
        "S1-c": {"S2-5"},
    }
    pred = {
        "S1-a": {"S2-00047", "S2-00193", "S3-00812"},  # 0.714
        "S1-b": set(),                                   # 1.0
        "S1-c": {"S3-7"},                                # 0.0
        "S1-zzz": {"S2-1"},                              # not in truth -> ignored
    }
    assert macro_f05(pred, truth) == pytest.approx((5 / 7 + 1.0 + 0.0) / 3)


def test_breakdown_groups():
    truth = {"S1-a": {"S2-1", "S3-1"}, "S1-b": set(), "S1-c": {"S3-2"}}
    pred = {"S1-a": {"S2-1", "S3-1"}, "S1-b": {"S2-9"}}
    meta = pd.DataFrame({"entity_id": ["S1-a", "S1-b", "S1-c"], "country": ["X", "X", "Y"]})
    bd = breakdown(pred, truth, meta).set_index(["group", "value"])
    assert bd.loc[("country", "X"), "count"] == 2
    assert bd.loc[("country", "X"), "f05"] == pytest.approx(0.5)
    assert bd.loc[("is_singleton", "True"), "f05"] == 0.0
    assert bd.loc[("true_count_bucket", "2"), "f05"] == 1.0
    assert bd.loc[("truth_source_mix", "both"), "count"] == 1
    assert bd.loc[("truth_source_mix", "S3 only"), "f05"] == 0.0
    # S1-c has an empty prediction -> precision undefined, excluded from mean
    assert pd.isna(bd.loc[("country", "Y"), "precision"])


def test_load_truth_and_pred(tmp_path):
    p = tmp_path / "gt.tsv"
    p.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1, S3-2 \nS1-2\t\n")
    assert load_truth(str(p)) == {"S1-1": {"S2-1", "S3-2"}, "S1-2": set()}
    q = tmp_path / "pred.tsv"
    q.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-2\t\n")
    assert load_pred_tsv(str(q)) == {"S1-1": {"S2-1"}, "S1-2": set()}
