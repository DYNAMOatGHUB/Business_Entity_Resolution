"""Vectorized threshold search for the decision layer on OOF train scores.

Scale trick: pairs with ``p_final`` < ``decide.min_p`` can never be predicted (every grid
threshold is >= min_p), so they are dropped at parquet-read time. Recall denominators come
from the ground truth (``n_true`` per S1), never from the candidate set, and the S1 table
covers ALL train S1s (zero-candidate S1s included: singleton + empty = 1.0, truth + empty = 0.0),
so pruning does not bias the score.

Everything is precomputed once (integer S1 / cand codes, label per pair, S1 max score);
each config evaluation is ``PairArrays.mask`` + two ``bincount`` calls. With
F0.5 = 1.25 PR / (0.25 P + R), P = tp / n_pred and R = tp / n_true, per-S1 F0.5 reduces to
1.25 tp / (0.25 n_true + n_pred) when n_true > 0.

Stage A: tau_gate x tau_pair grid (alpha 0, no one-owner, no per-source thresholds).
Stage B (around the best A): alpha x one_owner x tau_s2 / tau_s3 at -0.05 / 0 / +0.05.

Run: python -m src.eval.threshold_search [--sample] [--scores PATH] [--write-config [--yes]]
Writes reports/threshold_grid.csv and reports/last_oof.json.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.decide import DEFAULT_MIN_P, DecideParams, PairArrays, codes_in
from src.eval.reports import truth_pairs

GRID_TAUS = np.round(np.arange(0.30, 0.95 + 1e-9, 0.025), 3)
STAGE_B_ALPHAS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)
STAGE_B_OWNER = (False, True)
STAGE_B_SOURCE_DELTAS = (-0.05, 0.0, 0.05)
BASELINE = DecideParams(tau_gate=0.5, tau_pair=0.5, alpha=0.0, one_owner=False)
PARAM_COLS = ["tau_gate", "tau_pair", "tau_s2", "tau_s3", "alpha", "one_owner", "delta_unseen"]


@dataclass
class SearchData:
    """Precomputed arrays: pairs (pruned) + per-S1 truth counts, country and fold."""

    arrays: PairArrays
    label: np.ndarray
    n_true: np.ndarray
    country: np.ndarray
    fold: np.ndarray
    min_p: float

    @property
    def n_s1(self) -> int:
        return self.arrays.n_s1


def build_search_data(
    scores_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    s1_table: pd.DataFrame,
    train_countries,
    min_p: float = DEFAULT_MIN_P,
) -> SearchData:
    """Encode scores against the full S1 table and join labels once via integer pair keys.

    ``s1_table``: s1_id, country (every S1 to average over). ``truth_df``: raw GT or long form.
    """
    arrays = PairArrays.build(scores_df, s1_table, train_countries, min_p)
    n_s1, n_cand = arrays.n_s1, len(arrays.cand_ids)

    tp = truth_pairs(truth_df)
    t_s1 = codes_in(tp["s1_id"], pd.Index(arrays.s1_ids))
    in_table = t_s1 >= 0
    n_true = np.bincount(t_s1[in_table], minlength=n_s1).astype(np.int64)

    t_cand = codes_in(tp.loc[in_table, "cand_id"], pd.Index(arrays.cand_ids))
    found = t_cand >= 0
    truth_keys = t_s1[in_table][found].astype(np.int64) * max(n_cand, 1) + t_cand[found]
    row_keys = arrays.s1_code.astype(np.int64) * max(n_cand, 1) + arrays.cand_code
    label = np.isin(row_keys, truth_keys)

    fold = np.full(n_s1, -1, dtype=np.int16)
    if "fold" in scores_df.columns:
        fold[arrays.s1_code] = scores_df["fold"].to_numpy()[arrays.source_rows].astype(np.int16)
    return SearchData(arrays, label, n_true, s1_table["country"].to_numpy(), fold, min_p)


def per_s1_f05(tp: np.ndarray, n_pred: np.ndarray, n_true: np.ndarray) -> np.ndarray:
    """Per-S1 F0.5 with the metric's empty-set rules, vectorized."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n_true == 0, (n_pred == 0).astype(np.float64), 1.25 * tp / (0.25 * n_true + n_pred))


def score_mask(data: SearchData, keep: np.ndarray, per_s1: bool = False) -> dict:
    """Macro F0.5 / precision / recall over ALL S1s in ``data`` for a kept-row mask."""
    code = data.arrays.s1_code[keep]
    n_pred = np.bincount(code, minlength=data.n_s1)
    tp = np.bincount(code, weights=data.label[keep], minlength=data.n_s1)
    f = per_s1_f05(tp, n_pred, data.n_true)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(n_pred > 0, tp / n_pred, np.nan)
        rec = np.where(data.n_true > 0, tp / data.n_true, np.nan)
    out = {
        "f05_macro": float(f.mean()) if len(f) else float("nan"),
        "precision": float(np.nanmean(prec)) if (n_pred > 0).any() else float("nan"),
        "recall": float(np.nanmean(rec)) if (data.n_true > 0).any() else float("nan"),
        "n_pred_pairs": int(keep.sum()),
        "n_pred_s1": int((n_pred > 0).sum()),
    }
    if per_s1:
        out["per_s1"] = f
    return out


def evaluate(data: SearchData, params: DecideParams) -> dict:
    """Score one decision config (params + metrics + wall time)."""
    _check_prunable(params, data.min_p)
    t = time.perf_counter()
    res = score_mask(data, data.arrays.mask(params))
    return {**asdict(params), **res, "secs": time.perf_counter() - t}


def _check_prunable(params: DecideParams, min_p: float) -> None:
    """Pruning at ``min_p`` is only exact if no threshold can fall below it."""
    lowest = min(params.min_threshold(), params.tau_gate + min(0.0, params.delta_unseen))
    if lowest < min_p - 1e-12:
        raise ValueError(f"threshold {lowest} < min_p {min_p}: pruned pairs could be predicted; lower decide.min_p")


def stage_a(data: SearchData, taus=GRID_TAUS, base: DecideParams = BASELINE) -> pd.DataFrame:
    """tau_gate x tau_pair grid with the other rules neutral."""
    rows = [evaluate(data, replace(base, tau_gate=float(g), tau_pair=float(p), tau_s2=None, tau_s3=None))
            for g in taus for p in taus]
    return pd.DataFrame(rows).assign(stage="A")


def stage_b(
    data: SearchData,
    best: DecideParams,
    alphas=STAGE_B_ALPHAS,
    owners=STAGE_B_OWNER,
    deltas=STAGE_B_SOURCE_DELTAS,
) -> pd.DataFrame:
    """alpha x one_owner x per-source thresholds around the best stage-A config."""
    rows = []
    for a in alphas:
        for o in owners:
            for d2 in deltas:
                for d3 in deltas:
                    params = replace(
                        best, alpha=float(a), one_owner=bool(o),
                        tau_s2=round(best.tau_pair + d2, 3), tau_s3=round(best.tau_pair + d3, 3),
                    )
                    rows.append(evaluate(data, params))
    return pd.DataFrame(rows).assign(stage="B")


def params_from_row(row: pd.Series) -> DecideParams:
    """DecideParams from a grid row (NaN per-source thresholds -> None)."""
    opt = lambda v: None if v is None or pd.isna(v) else float(v)  # noqa: E731
    return DecideParams(
        tau_gate=float(row["tau_gate"]), tau_pair=float(row["tau_pair"]), tau_s2=opt(row["tau_s2"]),
        tau_s3=opt(row["tau_s3"]), alpha=float(row["alpha"]), one_owner=bool(row["one_owner"]),
        delta_unseen=float(row["delta_unseen"]),
    )


def fold_scores(data: SearchData, per_s1: np.ndarray) -> dict[str, float]:
    """Macro F0.5 per OOF fold (S1s without any pair >= min_p have no fold and are excluded)."""
    return {str(int(k)): float(per_s1[data.fold == k].mean()) for k in np.unique(data.fold) if k >= 0}


# ----------------------------------------------------------------------------- config writing


def _yaml_scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(round(v, 4))
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(json.dumps(x) for x in v) + "]"
    return json.dumps(v)


def update_config_text(text: str, decide_updates: dict, train_countries: list[str]) -> str:
    """Rewrite only the given ``decide:`` keys and ``train_countries`` line, keeping all other lines."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if re.match(r"^decide:\s*$", l)), None)
    if start is None:
        raise ValueError("config has no top-level 'decide:' block")
    end = start + 1
    while end < len(lines) and (lines[end].startswith((" ", "\t")) or not lines[end].strip()):
        end += 1
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    indent = re.match(r"^(\s+)", lines[start + 1]).group(1) if end > start + 1 else "  "
    for key, val in decide_updates.items():
        pat = re.compile(rf"^(\s+){re.escape(key)}:.*$")
        idx = next((i for i in range(start + 1, end) if pat.match(lines[i])), None)
        new_line = f"{indent}{key}: {_yaml_scalar(val)}"
        if idx is None:
            lines.insert(end, new_line)
            end += 1
        else:
            lines[idx] = new_line
    tc = f"train_countries: {_yaml_scalar(list(train_countries))}"
    idx = next((i for i, l in enumerate(lines) if re.match(r"^train_countries:", l)), None)
    if idx is None:
        lines.append(tc)
    else:
        lines[idx] = tc
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def config_updates(best: DecideParams, min_p: float) -> dict:
    """Decide keys to write; per-source thresholds equal to tau_pair are written as null."""
    same = lambda t: None if t is None or abs(t - best.tau_pair) < 1e-9 else t  # noqa: E731
    return {
        "tau_gate": best.tau_gate,
        "tau_pair": best.tau_pair,
        "tau_s2": same(best.tau_s2),
        "tau_s3": same(best.tau_s3),
        "alpha": best.alpha,
        "one_owner": best.one_owner,
        "min_p": min_p,
    }


# ----------------------------------------------------------------------------- CLI


def _fmt_top(df: pd.DataFrame, n: int) -> str:
    cols = ["stage", *PARAM_COLS[:-1], "f05_macro", "precision", "recall", "n_pred_pairs"]
    with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
        return df[cols].head(n).to_string(index=False)


def main() -> None:
    """CLI: two-stage grid search on OOF train scores."""
    from src.config import load_config, resolve_path
    from src.contracts import load_artifact, load_s1_meta
    from src.decide import resolve_train_countries
    from src.eval.score import breakdown, macro_f05
    from src.io_utils import read_truth

    parser = argparse.ArgumentParser(description="Vectorized decision-threshold search on OOF scores")
    parser.add_argument("--sample", action="store_true", help="keep sample.frac of S1 ids (seed sample.seed)")
    parser.add_argument("--scores", default=None, help="override scores parquet path (contract still checked)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--write-config", action="store_true", help="print the config diff for the best params")
    parser.add_argument("--yes", action="store_true", help="with --write-config: actually write the config")
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = load_config(args.config)
    min_p = float(cfg["decide"].get("min_p", DEFAULT_MIN_P))
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])

    scores = load_artifact(
        "scores", "train", sample=args.sample, cfg=cfg, path=args.scores,
        columns=["s1_id", "cand_id", "p_final", "fold"], filters=[("p_final", ">=", min_p)],
    )
    s1_table = load_s1_meta("train", sample=args.sample, cfg=cfg).rename(columns={"entity_id": "s1_id"})
    truth = read_truth("train", resolve_path(cfg["paths"]["dataset_dir"]))
    train_countries = resolve_train_countries(cfg)
    t_load = time.perf_counter()

    data = build_search_data(scores, truth, s1_table, train_countries, min_p)
    n_loaded = len(scores)
    del scores
    t_build = time.perf_counter()
    print(
        f"[search] S1={data.n_s1:,} (zero pairs >= {min_p}: {int((np.bincount(data.arrays.s1_code, minlength=data.n_s1) == 0).sum()):,}) "
        f"pairs>={min_p}: {len(data.arrays):,} (loaded {n_loaded:,}) true-in-pairs: {int(data.label.sum()):,} / {int(data.n_true.sum()):,} "
        f"| load {t_load - t0:.1f}s build {t_build - t_load:.1f}s"
    )

    base = evaluate(data, BASELINE)
    grid_a = stage_a(data)
    best_a = params_from_row(grid_a.sort_values("f05_macro", ascending=False, kind="stable").iloc[0])
    grid_b = stage_b(data, best_a)
    grid = pd.concat([grid_a, grid_b], ignore_index=True).sort_values("f05_macro", ascending=False, kind="stable")
    best = params_from_row(grid.iloc[0])
    t_grid = time.perf_counter()

    reports_dir.mkdir(parents=True, exist_ok=True)
    grid[["stage", *PARAM_COLS, "f05_macro", "precision", "recall", "n_pred_pairs", "n_pred_s1", "secs"]].to_csv(
        reports_dir / "threshold_grid.csv", index=False
    )
    print(f"\n[search] {len(grid)} configs in {t_grid - t_build:.1f}s ({grid['secs'].mean() * 1000:.0f} ms/config mean, "
          f"{grid['secs'].max() * 1000:.0f} ms max)")
    print("\nTop 10:\n" + _fmt_top(grid, 10))
    print(f"\nBaseline 0.5/0.5: f05={base['f05_macro']:.4f}  P={base['precision']:.4f}  R={base['recall']:.4f}")
    print(f"Best: {best}  f05={grid.iloc[0]['f05_macro']:.4f}  (+{grid.iloc[0]['f05_macro'] - base['f05_macro']:.4f} vs baseline)")

    keep = data.arrays.mask(best)
    best_res = score_mask(data, keep, per_s1=True)
    folds = fold_scores(data, best_res["per_s1"])

    # Cross-check with the reference scorer and print its breakdown (one-off, not in the grid loop).
    kept = data.arrays.to_frame(keep)
    pred = kept.groupby("s1_id")["cand_id"].agg(set).to_dict()
    tp_long = truth_pairs(truth)
    tp_long = tp_long[tp_long["s1_id"].isin(set(data.arrays.s1_ids))]
    truth_dict = dict.fromkeys(data.arrays.s1_ids.tolist(), frozenset()) | tp_long.groupby("s1_id")["cand_id"].agg(set).to_dict()
    ref = macro_f05(pred, truth_dict)
    print(f"\nReference scorer check: score.macro_f05={ref:.6f}  fast={best_res['f05_macro']:.6f}  "
          f"{'OK' if abs(ref - best_res['f05_macro']) < 1e-9 else 'MISMATCH'}")
    bd = breakdown(pred, truth_dict, s1_table.rename(columns={"s1_id": "entity_id"}))
    with pd.option_context("display.width", 120, "display.float_format", "{:.4f}".format):
        print("\nBest config breakdown:\n" + bd[bd["group"].isin(["country", "is_singleton"])].to_string(index=False))
    print("Fold scores: " + ", ".join(f"{k}={v:.4f}" for k, v in folds.items()))

    summary = {
        "scores_path": str(args.scores or "contract scores_train"),
        "sample": args.sample,
        "n_s1": data.n_s1,
        "n_pairs_ge_min_p": len(data.arrays),
        "min_p": min_p,
        "best": asdict(best),
        "f05_macro": best_res["f05_macro"],
        "precision": best_res["precision"],
        "recall": best_res["recall"],
        "baseline": {k: base[k] for k in ("f05_macro", "precision", "recall")},
        "fold_scores": folds,
        "reference_f05": ref,
        "train_countries": train_countries,
        "n_configs": len(grid),
        "ms_per_config_mean": float(grid["secs"].mean() * 1000),
        "total_secs": time.perf_counter() - t0,
    }
    (reports_dir / "last_oof.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[search] wrote {reports_dir / 'threshold_grid.csv'} and {reports_dir / 'last_oof.json'} "
          f"(total {summary['total_secs']:.1f}s)")

    if args.write_config:
        cfg_path = resolve_path(args.config)
        old = cfg_path.read_text(encoding="utf-8")
        new = update_config_text(old, config_updates(best, min_p), train_countries)
        diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), str(cfg_path), str(cfg_path)))
        print("\nConfig diff:\n" + (diff or "(no changes)"))
        if args.yes and diff:
            cfg_path.write_text(new, encoding="utf-8")
            print(f"[search] wrote {cfg_path}")
        elif diff:
            print("[search] dry run: re-run with --write-config --yes to apply")


if __name__ == "__main__":
    main()
