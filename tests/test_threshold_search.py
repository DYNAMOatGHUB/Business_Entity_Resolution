"""Tests for the vectorized threshold search: the fast scorer must equal src.eval.score.macro_f05."""

import numpy as np
import pandas as pd
import pytest
import yaml

from src.decide import DecideParams, decide
from src.eval.score import f05_entity, macro_f05
from src.eval.threshold_search import (
    build_search_data,
    config_updates,
    evaluate,
    per_s1_f05,
    stage_a,
    update_config_text,
)


def _synthetic(seed: int = 42, n_s1: int = 500):
    """500 S1s: ~15% singletons, ~10% zero-candidate S1s, multi-match, shared candidates, blocking misses."""
    rng = np.random.default_rng(seed)
    s1_ids = [f"S1-{i}" for i in range(n_s1)]
    table = pd.DataFrame({"s1_id": s1_ids, "country": rng.choice(["A", "B"], n_s1)})
    score_rows, truth = [], {}
    for i, s1 in enumerate(s1_ids):
        singleton = rng.random() < 0.15
        n_true = 0 if singleton else int(rng.integers(1, 6))
        true_ids = [f"S{rng.choice([2, 3])}-{i}-{k}" for k in range(n_true)]
        truth[s1] = set(true_ids)
        if rng.random() < 0.10:
            continue  # zero-candidate S1
        for t in true_ids:
            if rng.random() < 0.9:  # 10% blocking misses
                score_rows.append((s1, t, float(np.clip(rng.normal(0.75, 0.2), 0, 1))))
        for _ in range(int(rng.integers(0, 10))):
            cand = f"S{rng.choice([2, 3])}-shared-{rng.integers(0, 300)}"  # shared -> one-owner matters
            score_rows.append((s1, cand, float(np.clip(rng.normal(0.35, 0.2), 0, 1))))
    scores = pd.DataFrame(score_rows, columns=["s1_id", "cand_id", "p_final"]).drop_duplicates(["s1_id", "cand_id"])
    truth_df = pd.DataFrame({"source1_entity_id": s1_ids, "matched_entity_ids": [",".join(sorted(truth[s])) for s in s1_ids]})
    return scores.reset_index(drop=True), truth_df, table, truth


def _reference_f05(scores, table, truth, params: DecideParams) -> float:
    kept = decide(scores, table, {"decide": {**params.__dict__, "min_p": 0.0}}, ["A", "B"])
    pred = kept.groupby("s1_id")["cand_id"].agg(set).to_dict()
    return macro_f05(pred, truth)


PARAMS = [
    DecideParams(tau_gate=0.5, tau_pair=0.5, one_owner=False),
    DecideParams(tau_gate=0.7, tau_pair=0.4, one_owner=False),
    DecideParams(tau_gate=0.6, tau_pair=0.5, alpha=0.8, one_owner=True),
    DecideParams(tau_gate=0.55, tau_pair=0.45, tau_s2=0.6, tau_s3=0.4, alpha=0.5, one_owner=True),
    DecideParams(tau_gate=0.95, tau_pair=0.95, one_owner=True),
    DecideParams(tau_gate=0.3, tau_pair=0.3, one_owner=False),
]


@pytest.mark.parametrize("params", PARAMS)
def test_fast_scorer_equals_reference(params):
    scores, truth_df, table, truth = _synthetic()
    data = build_search_data(scores, truth_df, table, ["A", "B"], min_p=0.2)
    assert data.n_s1 == 500
    assert evaluate(data, params)["f05_macro"] == pytest.approx(_reference_f05(scores, table, truth, params), abs=1e-12)


def test_synthetic_set_has_all_cases():
    scores, _, table, truth = _synthetic()
    with_cands = set(scores["s1_id"])
    assert sum(1 for s in table["s1_id"] if not truth[s]) > 30                      # singletons
    assert sum(1 for s in table["s1_id"] if s not in with_cands and truth[s]) > 10  # truth but zero candidates
    assert sum(1 for s in table["s1_id"] if s not in with_cands and not truth[s]) > 0
    assert sum(1 for s in truth.values() if len(s) >= 3) > 100                      # multi-match


def test_pruning_does_not_change_score():
    scores, truth_df, table, _ = _synthetic(seed=7)
    full = build_search_data(scores, truth_df, table, ["A", "B"], min_p=0.0)
    pruned = build_search_data(scores, truth_df, table, ["A", "B"], min_p=0.3)
    assert len(pruned.arrays) < len(full.arrays)
    assert (pruned.n_true == full.n_true).all()
    for params in PARAMS:
        assert evaluate(pruned, params)["f05_macro"] == pytest.approx(evaluate(full, params)["f05_macro"], abs=1e-12)


def test_threshold_below_min_p_rejected():
    scores, truth_df, table, _ = _synthetic()
    data = build_search_data(scores, truth_df, table, ["A", "B"], min_p=0.4)
    with pytest.raises(ValueError, match="min_p"):
        evaluate(data, DecideParams(tau_gate=0.5, tau_pair=0.35))


def test_zero_candidate_s1s_count():
    table = pd.DataFrame({"s1_id": ["S1-a", "S1-b", "S1-c"], "country": "A"})
    truth_df = pd.DataFrame({"source1_entity_id": ["S1-a", "S1-b", "S1-c"], "matched_entity_ids": ["", "S2-9", "S2-1"]})
    scores = pd.DataFrame({"s1_id": ["S1-c"], "cand_id": ["S2-1"], "p_final": [0.9]})
    data = build_search_data(scores, truth_df, table, ["A"], min_p=0.2)
    # a: singleton + empty = 1; b: truth + no candidates = 0; c: perfect = 1
    assert evaluate(data, DecideParams(one_owner=False))["f05_macro"] == pytest.approx(2 / 3)


def test_per_s1_f05_matches_entity_formula():
    rng = np.random.default_rng(3)
    n_true = rng.integers(0, 6, 2000)
    n_pred = rng.integers(0, 8, 2000)
    tp = np.minimum(rng.integers(0, 6, 2000), np.minimum(n_true, n_pred))
    ref = [f05_entity(set(range(p)), set(range(t)) if t == 0 else set(range(k)) | {f"x{j}" for j in range(t - k)})
           for t, p, k in zip(n_true, n_pred, tp)]
    assert np.allclose(per_s1_f05(tp, n_pred, n_true), ref)


def test_stage_a_finds_perfect_separation():
    table = pd.DataFrame({"s1_id": ["S1-a", "S1-b"], "country": "A"})
    truth_df = pd.DataFrame({"source1_entity_id": ["S1-a", "S1-b"], "matched_entity_ids": ["S2-1", ""]})
    scores = pd.DataFrame({"s1_id": ["S1-a", "S1-a", "S1-b"], "cand_id": ["S2-1", "S3-1", "S2-2"], "p_final": [0.9, 0.6, 0.55]})
    grid = stage_a(build_search_data(scores, truth_df, table, ["A"], min_p=0.2))
    assert len(grid) == 27 * 27
    assert grid["f05_macro"].max() == 1.0


def test_update_config_text_only_touches_decide_keys():
    text = (
        'seed: 42\npaths:\n  dataset_dir: "dataset"\ntrain_countries: []\n\ndecide:\n  tau_gate: 0.50\n'
        "  tau_pair: 0.40\n  one_owner: true\n  tau_s2: null\n  tau_s3: null\n  alpha: 0.0\n  delta_unseen: 0.0\n"
    )
    best = DecideParams(tau_gate=0.625, tau_pair=0.45, tau_s2=0.45, tau_s3=0.5, alpha=0.8, one_owner=False)
    new = update_config_text(text, config_updates(best, 0.2), ["India", "US"])
    assert '  dataset_dir: "dataset"' in new and new.startswith("seed: 42\n")
    cfg = yaml.safe_load(new)
    assert cfg["train_countries"] == ["India", "US"]
    assert cfg["decide"] == {
        "tau_gate": 0.625, "tau_pair": 0.45, "one_owner": False, "tau_s2": None, "tau_s3": 0.5,
        "alpha": 0.8, "delta_unseen": 0.0, "min_p": 0.2,
    }
