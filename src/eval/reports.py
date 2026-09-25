"""Blocking and error reports.

- Blocking report: candidate volume, reduction ratio, recall@K overall / per country / per
  candidate source, and the list of missed true pairs. Writes ``blocking_report.md`` and
  ``blocking_report.json`` (plus ``blocking_report_missed.tsv``) under ``paths.reports_dir``.
- Error report: false positives and false negatives of predicted matches, side by side.

Threshold search lives in ``src/eval/threshold_search.py``; the metric in ``src/eval/score.py``.
All computations are vectorized pandas (no Python loops over S1s); markdown tables are
rendered by ``md_table`` (no tabulate dependency).

Run:
    python -m src.eval.reports blocking --split train [--sample]
    python -m src.eval.reports errors --split train [--sample] [--n 50]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_KS: tuple[int, ...] = (1, 5, 10, 20, 30, 40, 50)
UNKNOWN_COUNTRY = "<unknown>"
PAIR_KEY = ["s1_id", "cand_id"]


# ----------------------------------------------------------------------------- helpers


def truth_pairs(truth_df: pd.DataFrame) -> pd.DataFrame:
    """Ground truth as unique long-form (s1_id, cand_id) pairs; singletons contribute no rows.

    Accepts the raw TSV format (``source1_entity_id, matched_entity_ids``) or long form.
    """
    if set(PAIR_KEY) <= set(truth_df.columns):
        return truth_df[PAIR_KEY].drop_duplicates().reset_index(drop=True)
    long = pd.DataFrame(
        {
            "s1_id": truth_df["source1_entity_id"].astype(str).str.strip(),
            "cand_id": truth_df["matched_entity_ids"].fillna("").astype(str).str.split(","),
        }
    ).explode("cand_id")
    long["cand_id"] = long["cand_id"].str.strip()
    long = long[long["cand_id"].notna() & (long["cand_id"] != "")]
    return long.drop_duplicates().reset_index(drop=True)


def id_source(ids: pd.Series) -> pd.Series:
    """Source label from the ID prefix (``S2-123`` -> ``S2``)."""
    return ids.astype(str).str.split("-", n=1).str[0]


def rank_within_s1(cands: pd.DataFrame, rank_col: str = "blend_rank") -> pd.Series:
    """1-based rank of each candidate within its S1 (higher ``rank_col`` = better).

    Ties keep input order. Without a usable ``rank_col`` the input order is the rank.
    """
    if rank_col in cands.columns and cands[rank_col].notna().any():
        key = pd.to_numeric(cands[rank_col], errors="coerce").fillna(-np.inf)
        return key.groupby(cands["s1_id"]).rank(method="first", ascending=False).astype(int)
    return cands.groupby("s1_id").cumcount() + 1


def md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a DataFrame as a GitHub markdown table (pipes escaped, NaN shown as blank)."""
    def fmt(v) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        if isinstance(v, (float, np.floating)):
            return format(float(v), floatfmt)
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        return str(v).replace("|", "\\|").replace("\n", " ")

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _jsonable(obj):
    """Recursively convert numpy scalars / NaN to plain JSON values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if math.isnan(obj) else float(obj)
    if obj is pd.NA:
        return None
    return obj


def _s1_universe(records_df: pd.DataFrame | None, cands: pd.DataFrame, truth: pd.DataFrame | None) -> tuple[pd.Series, int | None]:
    """(S1 id -> country, size of the S2+S3 pool). Without records: ids from candidates + truth, pool unknown."""
    if records_df is not None:
        s1 = records_df.loc[records_df["source"] == "S1", ["entity_id", "country"]].drop_duplicates("entity_id")
        return s1.set_index("entity_id")["country"], int((records_df["source"] != "S1").sum())
    ids = pd.Index(cands["s1_id"].unique())
    if truth is not None:
        ids = ids.union(pd.Index(truth["source1_entity_id"].astype(str).str.strip().unique()))
    return pd.Series(UNKNOWN_COUNTRY, index=ids), None


# ----------------------------------------------------------------------------- blocking


def recall_table(true_ranked: pd.DataFrame, ks: Sequence[int]) -> pd.DataFrame:
    """Recall@K for all true pairs, per S1 country and per candidate source.

    ``true_ranked`` has one row per true pair with ``rank`` (NaN = not a candidate),
    ``country`` and ``cand_source``.
    """
    hit_all = true_ranked["rank"].notna()
    parts = []
    for dim in (None, "country", "cand_source"):
        grp = pd.Series("all", index=true_ranked.index) if dim is None else true_ranked[dim].astype(str)
        t = pd.DataFrame({"n_true": true_ranked.groupby(grp).size()})
        for k in ks:
            t[f"R@{k}"] = (true_ranked["rank"] <= k).groupby(grp).mean()
        t["R@all"] = hit_all.groupby(grp).mean()
        t.insert(0, "group", dim or "all")
        parts.append(t.rename_axis("value").reset_index())
    cols = ["group", "value", "n_true", *[f"R@{k}" for k in ks], "R@all"]
    return pd.concat(parts, ignore_index=True)[cols]


def blocking_stats(
    candidates_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame | None = None,
    records_df: pd.DataFrame | None = None,
    ks: Sequence[int] = DEFAULT_KS,
    rank_col: str = "blend_rank",
) -> dict:
    """Candidate volume + recall statistics.

    ``records_df`` (entity_id, source, country) defines the S1 universe, S1 countries and the
    S2+S3 pool size; without it countries are ``<unknown>`` and the reduction ratio is NaN.
    ``ground_truth_df`` is optional (test split): recall fields are then omitted.
    """
    keep = [c for c in (*PAIR_KEY, rank_col) if c in candidates_df.columns]
    cands = candidates_df[keep].drop_duplicates(PAIR_KEY)
    s1_country, n_pool = _s1_universe(records_df, cands, ground_truth_df)
    cands = cands[cands["s1_id"].isin(s1_country.index)].copy()
    cands["rank"] = rank_within_s1(cands, rank_col)

    per_s1 = cands.groupby("s1_id").size().reindex(s1_country.index, fill_value=0)
    n_s1, n_pairs = len(per_s1), int(per_s1.sum())
    by_country = (
        pd.DataFrame({"country": s1_country.fillna(UNKNOWN_COUNTRY).astype(str).values, "n": per_s1.values})
        .groupby("country")["n"]
        .agg(n_s1="size", pairs="sum", mean_per_s1="mean", median_per_s1="median", pct_zero=lambda x: (x == 0).mean() * 100)
        .reset_index()
    )
    out: dict = {
        "n_s1": n_s1,
        "n_cand_pool": n_pool,
        "pairs": n_pairs,
        "mean_cands_per_s1": float(per_s1.mean()) if n_s1 else float("nan"),
        "median_cands_per_s1": float(per_s1.median()) if n_s1 else float("nan"),
        "max_cands_per_s1": int(per_s1.max()) if n_s1 else 0,
        "pct_s1_zero_cands": float((per_s1 == 0).mean() * 100) if n_s1 else float("nan"),
        "reduction_ratio": 1 - n_pairs / (n_s1 * n_pool) if n_pool and n_s1 else float("nan"),
        "candidates_by_country": by_country,
        "ks": list(ks),
    }
    if ground_truth_df is None:
        return out

    tp = truth_pairs(ground_truth_df)
    tp = tp[tp["s1_id"].isin(s1_country.index)].merge(cands[[*PAIR_KEY, "rank"]], on=PAIR_KEY, how="left")
    tp["country"] = tp["s1_id"].map(s1_country).fillna(UNKNOWN_COUNTRY)
    tp["cand_source"] = id_source(tp["cand_id"])
    table = recall_table(tp, ks)
    overall = table.iloc[0]
    hit = tp["rank"].notna()
    out.update(
        {
            "n_true_pairs": len(tp),
            "recall_at_k": {k: float(overall[f"R@{k}"]) for k in ks},
            "recall_all": float(overall["R@all"]),
            "pct_s1_fully_covered": float(hit.groupby(tp["s1_id"]).all().mean() * 100) if len(tp) else float("nan"),
            "pct_candidates_true": len(tp[hit]) / n_pairs * 100 if n_pairs else float("nan"),
            "recall_table": table,
            "missed": tp.loc[~hit, ["s1_id", "cand_id", "country", "cand_source"]].reset_index(drop=True),
        }
    )
    return out


def render_blocking_md(stats: dict, title: str = "Blocking report") -> str:
    """Markdown rendering of ``blocking_stats`` output."""
    s = stats
    lines = [
        f"# {title}",
        "",
        f"- S1 entities: {s['n_s1']:,}; S2+S3 pool: {s['n_cand_pool']:,}" if s["n_cand_pool"] else f"- S1 entities: {s['n_s1']:,}; S2+S3 pool: unknown",
        f"- Candidate pairs: {s['pairs']:,}",
        f"- Candidates per S1: mean {s['mean_cands_per_s1']:.2f}, median {s['median_cands_per_s1']:.1f}, max {s['max_cands_per_s1']:,}",
        f"- S1 with zero candidates: {s['pct_s1_zero_cands']:.2f}%",
        f"- Reduction ratio 1 - pairs / (|S1| x |S2 u S3|): {s['reduction_ratio']:.6f}" if not math.isnan(s["reduction_ratio"]) else "- Reduction ratio: n/a (no records)",
    ]
    if "recall_at_k" in s:
        ks = s["ks"]
        lines += [
            f"- True pairs: {s['n_true_pairs']:,}; recall (all candidates): **{s['recall_all']:.4f}**; "
            f"missed: {len(s['missed']):,}",
            f"- Non-singleton S1 with every true match in candidates: {s['pct_s1_fully_covered']:.2f}%",
            f"- Share of candidate pairs that are true: {s['pct_candidates_true']:.2f}%",
            "",
            "## Recall@K",
            "Rank = position by `blend_rank` within the S1 (ties keep input order).",
            "",
            md_table(s["recall_table"]),
            "",
            f"Recall@K overall: " + ", ".join(f"@{k}={s['recall_at_k'][k]:.4f}" for k in ks),
        ]
    lines += ["", "## Candidates per country", "", md_table(s["candidates_by_country"], floatfmt=".2f"), ""]
    if "missed" in s and len(s["missed"]):
        lines += [
            "## Missed true pairs",
            "Full list in the `*_missed.tsv` next to this report. First 20:",
            "",
            md_table(s["missed"].head(20)),
            "",
        ]
    return "\n".join(lines)


def generate_blocking_report(
    candidates_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame | None = None,
    out_path: str | Path = "reports/blocking_report.md",
    records_df: pd.DataFrame | None = None,
    ks: Sequence[int] = DEFAULT_KS,
    title: str = "Blocking report",
) -> dict:
    """Compute ``blocking_stats`` and write ``<out>.md``, ``<out>.json`` and ``<out>_missed.tsv``.

    Returns the stats dict (``recall_at_k`` is keyed by int K).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = blocking_stats(candidates_df, ground_truth_df, records_df, ks)
    out_path.write_text(render_blocking_md(stats, title), encoding="utf-8")
    out_path.with_suffix(".json").write_text(
        json.dumps(_jsonable({k: v for k, v in stats.items() if k != "missed"}), indent=2), encoding="utf-8"
    )
    if "missed" in stats:
        stats["missed"].to_csv(out_path.parent / f"{out_path.stem}_missed.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE)
    msg = f"[blocking] pairs={stats['pairs']:,} mean/S1={stats['mean_cands_per_s1']:.1f} zero={stats['pct_s1_zero_cands']:.2f}%"
    if "recall_all" in stats:
        msg += f" recall_all={stats['recall_all']:.4f}"
    print(f"{msg} -> {out_path}")
    return stats


# ----------------------------------------------------------------------------- errors


def dump_errors(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    records_df: pd.DataFrame | None = None,
    scores_df: pd.DataFrame | None = None,
    out_path: str | Path = "reports/errors.tsv",
    n: int = 50,
    score_col: str = "p_final",
) -> pd.DataFrame:
    """Write the top-``n`` false positives and false negatives with both sides' text.

    ``predictions_df``: predicted matched pairs (s1_id, cand_id). ``scores_df`` (s1_id, cand_id,
    ``score_col``) attaches scores to both error types; FNs absent from it are flagged
    ``in_candidates=False`` (a blocking miss). FPs are sorted by score (most confident first),
    FNs by score (nearest misses first). With ``records_df`` the S1 universe is its S1 ids and
    text comes from ``name_raw``/``addr_raw`` (or ``business_name``/``business_address``).
    """
    pred = predictions_df[PAIR_KEY].drop_duplicates()
    truth = truth_pairs(ground_truth_df)
    if records_df is not None:
        s1_ids = records_df.loc[records_df["source"] == "S1", "entity_id"]
        pred, truth = pred[pred["s1_id"].isin(s1_ids)], truth[truth["s1_id"].isin(s1_ids)]

    m = pred.merge(truth, on=PAIR_KEY, how="outer", indicator=True)
    err = m[m["_merge"] != "both"].copy()
    err["error_type"] = np.where(err["_merge"] == "left_only", "FP", "FN")
    err = err.drop(columns="_merge")
    n_fp, n_fn = int((err["error_type"] == "FP").sum()), int((err["error_type"] == "FN").sum())

    score_src = scores_df if scores_df is not None else (predictions_df if score_col in predictions_df.columns else None)
    if score_src is not None:
        sc = score_src[[*PAIR_KEY, score_col]].drop_duplicates(PAIR_KEY)
        err = err.merge(sc.rename(columns={score_col: "score"}), on=PAIR_KEY, how="left")
    else:
        err["score"] = np.nan
    err["in_candidates"] = err["score"].notna() if scores_df is not None else pd.NA

    err = (
        err.sort_values(["error_type", "score"], ascending=[False, False], na_position="last")
        .groupby("error_type", sort=False)
        .head(n)
    )

    if records_df is not None:
        name_col = "name_raw" if "name_raw" in records_df.columns else "business_name"
        addr_col = "addr_raw" if "addr_raw" in records_df.columns else "business_address"
        rec = records_df.drop_duplicates("entity_id").set_index("entity_id")
        for side, id_col in (("s1", "s1_id"), ("cand", "cand_id")):
            err[f"{side}_name"] = err[id_col].map(rec[name_col]) if name_col in rec else pd.NA
            err[f"{side}_addr"] = err[id_col].map(rec[addr_col]) if addr_col in rec else pd.NA
        err["country"] = err["s1_id"].map(rec["country"]) if "country" in rec else pd.NA

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err.to_csv(out_path, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
    print(f"[errors] FP={n_fp:,} FN={n_fn:,} (top {n} of each) -> {out_path}")
    return err.reset_index(drop=True)


# ----------------------------------------------------------------------------- CLI


def _records_meta(split: str, sample: bool, cfg: dict) -> pd.DataFrame:
    """entity_id/source/country (+ text) from records_norm, falling back to raw TSV ids if Stage 0 has not run."""
    from src.config import resolve_path
    from src.contracts import load_artifact, sample_frame, sampled_s1_ids

    try:
        return load_artifact("records_norm", split, sample=sample, cfg=cfg)
    except FileNotFoundError as e:
        print(f"[reports] {e}\n[reports] falling back to raw TSV ids/countries")
    base = resolve_path(cfg["paths"]["dataset_dir"]) / split
    frames = [
        pd.read_csv(
            base / f"{split}_source{n}.tsv", sep="\t", dtype=str, keep_default_na=False,
            quoting=csv.QUOTE_NONE, usecols=["entity_id", "country"],
        ).assign(source=f"S{n}")
        for n in (1, 2, 3)
    ]
    df = pd.concat(frames, ignore_index=True)
    return sample_frame(df, "records_norm", sampled_s1_ids(split, cfg)) if sample else df


def main() -> None:
    """CLI: ``blocking`` or ``errors`` report for a split."""
    from src.config import load_config, resolve_path
    from src.contracts import load_artifact
    from src.io_utils import read_truth

    parser = argparse.ArgumentParser(description="Blocking and error reports")
    parser.add_argument("report", choices=["blocking", "errors"])
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--sample", action="store_true", help="keep sample.frac of S1 ids (seed sample.seed)")
    parser.add_argument("--n", type=int, default=50, help="errors: rows per error type")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    truth = read_truth(args.split, resolve_path(cfg["paths"]["dataset_dir"])) if args.split == "train" else None
    records = _records_meta(args.split, args.sample, cfg)
    tag = "" if args.split == "train" else f"_{args.split}"

    if args.report == "blocking":
        cap = int(cfg["blocking"]["cap_k_per_s1"])
        ks = sorted({k for k in DEFAULT_KS if k < cap} | {cap})
        cands = load_artifact("candidates", args.split, sample=args.sample, cfg=cfg)
        title = f"Blocking report ({args.split}{', sample' if args.sample else ''})"
        generate_blocking_report(cands, truth, reports_dir / f"blocking_report{tag}.md", records, ks, title)
    else:
        if truth is None:
            parser.error("errors report needs ground truth (train split)")
        dec = load_artifact("decisions", args.split, sample=args.sample, cfg=cfg)
        scores = load_artifact("scores", args.split, sample=args.sample, cfg=cfg)
        dump_errors(dec[dec["is_match"].astype(bool)], truth, records, scores, reports_dir / "errors.tsv", n=args.n)


if __name__ == "__main__":
    main()
