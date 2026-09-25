"""Tests for blocking and error reports (hand-computed toy example)."""

import json

import pandas as pd
import pytest

from src.eval.reports import blocking_stats, dump_errors, generate_blocking_report, md_table, truth_pairs


@pytest.fixture
def records():
    return pd.DataFrame(
        {
            "entity_id": ["S1-a", "S1-b", "S1-c", "S2-1", "S2-2", "S3-1", "S3-2"],
            "source": ["S1", "S1", "S1", "S2", "S2", "S3", "S3"],
            "country": ["X", "X", "Y", "X", "X", "X", "Y"],
            "name_raw": ["A", "B", "C", "A2", "B2", "A3", "Z3"],
            "addr_raw": ["a", "b", "c", "a2", "b2", "a3", "z3"],
        }
    )


@pytest.fixture
def truth():
    return pd.DataFrame({"source1_entity_id": ["S1-a", "S1-b", "S1-c"], "matched_entity_ids": ["S2-1,S3-1", "S2-2", ""]})


@pytest.fixture
def cands():
    # a: S2-1 rank 1 (true), S3-2 rank 2, S3-1 rank 3 (true); b: S3-2 only; c: nothing
    return pd.DataFrame(
        {
            "s1_id": ["S1-a", "S1-a", "S1-a", "S1-b"],
            "cand_id": ["S3-1", "S2-1", "S3-2", "S3-2"],
            "blend_rank": [0.1, 0.9, 0.8, 0.5],
        }
    )


def test_truth_pairs_drops_singletons_and_blanks():
    df = pd.DataFrame({"source1_entity_id": ["S1-a", "S1-b", "S1-c"], "matched_entity_ids": ["S2-1, S3-1,", "", None]})
    assert truth_pairs(df).values.tolist() == [["S1-a", "S2-1"], ["S1-a", "S3-1"]]


def test_blocking_stats(cands, truth, records):
    s = blocking_stats(cands, truth, records, ks=(1, 2, 3))
    assert s["n_s1"] == 3 and s["n_cand_pool"] == 4 and s["pairs"] == 4
    assert s["mean_cands_per_s1"] == pytest.approx(4 / 3)
    assert s["median_cands_per_s1"] == 1
    assert s["pct_s1_zero_cands"] == pytest.approx(100 / 3)
    assert s["reduction_ratio"] == pytest.approx(1 - 4 / 12)
    assert s["n_true_pairs"] == 3
    assert s["recall_at_k"] == pytest.approx({1: 1 / 3, 2: 1 / 3, 3: 2 / 3})
    assert s["recall_all"] == pytest.approx(2 / 3)
    assert s["pct_s1_fully_covered"] == pytest.approx(50.0)
    assert s["pct_candidates_true"] == pytest.approx(50.0)
    assert s["missed"][["s1_id", "cand_id"]].values.tolist() == [["S1-b", "S2-2"]]

    t = s["recall_table"].set_index(["group", "value"])
    assert t.loc[("cand_source", "S2"), "n_true"] == 2
    assert t.loc[("cand_source", "S2"), "R@all"] == pytest.approx(0.5)
    assert t.loc[("cand_source", "S3"), "R@1"] == 0
    assert t.loc[("cand_source", "S3"), "R@all"] == 1
    assert t.loc[("country", "X"), "n_true"] == 3

    by_c = s["candidates_by_country"].set_index("country")
    assert by_c.loc["Y", "pct_zero"] == 100 and by_c.loc["X", "pairs"] == 4


def test_blocking_stats_without_truth_or_records(cands):
    s = blocking_stats(cands)
    assert "recall_at_k" not in s
    assert s["n_s1"] == 2 and s["pct_s1_zero_cands"] == 0
    assert s["reduction_ratio"] != s["reduction_ratio"]  # NaN: pool unknown


def test_generate_blocking_report_writes_files(tmp_path, cands, truth, records):
    out = tmp_path / "blocking_report.md"
    s = generate_blocking_report(cands, truth, out, records, ks=(1, 40))
    assert s["recall_at_k"][40] == pytest.approx(2 / 3)
    md = out.read_text()
    assert "| group | value | n_true | R@1 | R@40 | R@all |" in md
    j = json.loads(out.with_suffix(".json").read_text())
    assert j["pairs"] == 4 and j["recall_at_k"]["40"] == pytest.approx(2 / 3)
    missed = pd.read_csv(tmp_path / "blocking_report_missed.tsv", sep="\t")
    assert missed["cand_id"].tolist() == ["S2-2"]


def test_dump_errors(tmp_path, truth, records):
    pred = pd.DataFrame({"s1_id": ["S1-a", "S1-a", "S1-c"], "cand_id": ["S2-1", "S3-2", "S2-2"]})
    scores = pd.DataFrame(
        {"s1_id": ["S1-a", "S1-a", "S1-a", "S1-c"], "cand_id": ["S2-1", "S3-2", "S3-1", "S2-2"], "p_final": [0.9, 0.8, 0.1, 0.6]}
    )
    err = dump_errors(pred, truth, records, scores, tmp_path / "errors.tsv")
    got = err[["error_type", "s1_id", "cand_id"]].values.tolist()
    assert got == [
        ["FP", "S1-a", "S3-2"],
        ["FP", "S1-c", "S2-2"],
        ["FN", "S1-a", "S3-1"],
        ["FN", "S1-b", "S2-2"],
    ]
    assert err["in_candidates"].tolist() == [True, True, True, False]
    assert err.loc[0, "s1_name"] == "A" and err.loc[0, "cand_name"] == "Z3"
    assert (tmp_path / "errors.tsv").exists()


def test_md_table_escapes_and_formats():
    df = pd.DataFrame({"a": ["x|y"], "b": [1234], "c": [0.5], "d": [float("nan")]})
    assert md_table(df).splitlines()[2] == "| x\\|y | 1,234 | 0.5000 |  |"
