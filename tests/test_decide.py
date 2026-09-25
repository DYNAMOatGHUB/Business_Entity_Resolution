"""Tests for the vectorized decision layer."""

import numpy as np
import pandas as pd
import pytest

from src.decide import DecideParams, PairArrays, decide


def _cfg(**kw) -> dict:
    base = dict(tau_gate=0.5, tau_pair=0.5, tau_s2=None, tau_s3=None, alpha=0.0, one_owner=False, delta_unseen=0.0, min_p=0.0)
    return {"decide": base | kw}


def _pairs(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["s1_id", "cand_id", "p_final"])


def _table(ids, country="X") -> pd.DataFrame:
    return pd.DataFrame({"s1_id": ids, "country": country})


def _kept(df) -> set:
    return set(map(tuple, df[["s1_id", "cand_id"]].values.tolist()))


def test_gate_drops_whole_s1():
    scores = _pairs([("S1-a", "S2-1", 0.45), ("S1-a", "S3-1", 0.40), ("S1-b", "S2-2", 0.9), ("S1-b", "S3-2", 0.35)])
    out = decide(scores, _table(["S1-a", "S1-b"]), _cfg(tau_gate=0.5, tau_pair=0.3), ["X"])
    assert _kept(out) == {("S1-b", "S2-2"), ("S1-b", "S3-2")}


def test_pair_threshold_by_source():
    scores = _pairs([("S1-a", "S2-1", 0.65), ("S1-a", "S3-1", 0.55), ("S1-a", "S2-2", 0.75)])
    out = decide(scores, _table(["S1-a"]), _cfg(tau_gate=0.5, tau_pair=0.5, tau_s2=0.7), ["X"])
    assert _kept(out) == {("S1-a", "S3-1"), ("S1-a", "S2-2")}


def test_relative_cut():
    scores = _pairs([("S1-a", "S2-1", 0.9), ("S1-a", "S2-2", 0.75), ("S1-a", "S3-1", 0.7)])
    out = decide(scores, _table(["S1-a"]), _cfg(alpha=0.8), ["X"])  # cut = 0.72
    assert _kept(out) == {("S1-a", "S2-1"), ("S1-a", "S2-2")}


def test_one_owner_keeps_best_s1_per_candidate():
    scores = _pairs([("S1-a", "S2-1", 0.8), ("S1-b", "S2-1", 0.9), ("S1-a", "S2-2", 0.7), ("S1-c", "S2-3", 0.6), ("S1-b", "S2-3", 0.6)])
    table = _table(["S1-a", "S1-b", "S1-c"])
    assert _kept(decide(scores, table, _cfg(one_owner=True), ["X"])) == {
        ("S1-b", "S2-1"), ("S1-a", "S2-2"), ("S1-b", "S2-3")  # tie 0.6/0.6 -> earlier S1 in s1_table order
    }
    assert len(decide(scores, table, _cfg(one_owner=False), ["X"])) == 5


def test_one_owner_applies_after_thresholds():
    # S1-b's higher score on S2-1 is below its gate, so S1-a keeps S2-1
    scores = _pairs([("S1-a", "S2-1", 0.6), ("S1-b", "S2-1", 0.45)])
    out = decide(scores, _table(["S1-a", "S1-b"]), _cfg(tau_gate=0.5, tau_pair=0.4, one_owner=True), ["X"])
    assert _kept(out) == {("S1-a", "S2-1")}


def test_unseen_country_offset():
    scores = _pairs([("S1-a", "S2-1", 0.55), ("S1-b", "S2-2", 0.55)])
    table = pd.DataFrame({"s1_id": ["S1-a", "S1-b"], "country": ["Seen", "New"]})
    out = decide(scores, table, _cfg(delta_unseen=0.1), ["Seen"])
    assert _kept(out) == {("S1-a", "S2-1")}
    out_neg = decide(_pairs([("S1-b", "S2-2", 0.45)]), table, _cfg(delta_unseen=-0.1, min_p=0.5), ["Seen"])
    assert _kept(out_neg) == {("S1-b", "S2-2")}  # negative offset lowers the prune point too


def test_pruning_never_changes_result():
    rng = np.random.default_rng(0)
    scores = _random_scores(rng, n_s1=200)
    table = _table(sorted(scores["s1_id"].unique()))
    for min_p in (0.0, 0.2, 0.45, 0.9):
        cfg = _cfg(tau_gate=0.55, tau_pair=0.45, alpha=0.6, one_owner=True, min_p=min_p)
        assert _kept(decide(scores, table, cfg, ["X"])) == _kept(decide(scores, table, _cfg(**cfg["decide"] | {"min_p": 0.0}), ["X"]))


def test_unknown_s1_rows_dropped_and_empty_input():
    arr = PairArrays.build(_pairs([("S1-zz", "S2-1", 0.9)]), _table(["S1-a"]), ["X"])
    assert len(arr) == 0 and arr.n_dropped_unknown_s1 == 1
    out = decide(_pairs([]), _table(["S1-a"]), _cfg(one_owner=True, tau_s2=0.6), ["X"])
    assert list(out.columns) == ["s1_id", "cand_id"] and out.empty


def _random_scores(rng, n_s1: int) -> pd.DataFrame:
    rows = []
    for i in range(n_s1):
        for _ in range(rng.integers(0, 8)):
            src = "S2" if rng.random() < 0.5 else "S3"
            rows.append((f"S1-{i}", f"{src}-{rng.integers(0, n_s1 * 2)}", float(np.round(rng.random(), 2))))
    return _pairs(rows).drop_duplicates(["s1_id", "cand_id"]).reset_index(drop=True)


def _reference(scores: pd.DataFrame, table: pd.DataFrame, p: DecideParams, train_countries) -> set:
    """Deliberately naive pandas implementation of rules a-e."""
    df = scores.merge(table, on="s1_id")
    df["off"] = np.where(df["country"].isin(train_countries), 0.0, p.delta_unseen)
    df["smax"] = df.groupby("s1_id")["p_final"].transform("max")
    src = df["cand_id"].str[:2]
    thr = np.where(src == "S2", p.tau_s2 if p.tau_s2 is not None else p.tau_pair,
                   np.where(src == "S3", p.tau_s3 if p.tau_s3 is not None else p.tau_pair, p.tau_pair))
    keep = (df["smax"] >= p.tau_gate + df["off"]) & (df["p_final"] >= thr + df["off"])
    if p.alpha > 0:
        keep &= df["p_final"] >= p.alpha * df["smax"]
    df = df[keep]
    if p.one_owner:
        order = {s: i for i, s in enumerate(table["s1_id"])}
        df = df.assign(o=df["s1_id"].map(order)).sort_values(["cand_id", "p_final", "o"], ascending=[True, False, True])
        df = df.drop_duplicates("cand_id")
    return _kept(df)


@pytest.mark.parametrize("seed", range(6))
def test_matches_naive_reference(seed):
    rng = np.random.default_rng(seed)
    scores = _random_scores(rng, n_s1=150)
    ids = sorted(scores["s1_id"].unique())
    table = pd.DataFrame({"s1_id": ids, "country": rng.choice(["Seen", "New"], len(ids))})
    p = DecideParams(
        tau_gate=float(rng.choice([0.3, 0.5, 0.7])), tau_pair=float(rng.choice([0.3, 0.5])),
        tau_s2=[None, 0.4][seed % 2], tau_s3=[None, 0.6][(seed // 2) % 2], alpha=float(rng.choice([0.0, 0.8])),
        one_owner=bool(seed % 3), delta_unseen=float(rng.choice([0.0, 0.1])),
    )
    cfg = {"decide": {**p.__dict__, "min_p": 0.2}}
    assert _kept(decide(scores, table, cfg, ["Seen"])) == _reference(scores, table, p, ["Seen"])
