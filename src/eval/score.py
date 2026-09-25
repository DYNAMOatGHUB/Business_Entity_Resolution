"""Macro F0.5 scorer matching the official competition metric.

F0.5 = 1.25*P*R / (0.25*P + R), computed per S1 entity and macro-averaged over ALL S1 entities
in the ground truth (singletons included):
- truth empty + pred empty     -> 1.0
- truth empty + pred non-empty -> 0.0
- truth non-empty + pred empty -> 0.0
An S1 present in the truth but missing from the predictions counts as an empty prediction.

Usage:
    python -m src.eval.score --pred output/matching_results.tsv \
        --truth dataset/train/train_ground_truth.tsv [--by-country]
"""

import argparse

import numpy as np
import pandas as pd

from src.io_utils import parse_id_list, read_tsv

BETA_SQ = 0.25
COUNT_BUCKETS = ["0", "1", "2", "3+"]
MISSING_COUNTRY = "<missing>"


def f05_entity(pred: set[str], truth: set[str]) -> float:
    """F0.5 for a single S1 entity, including both empty-set rules."""
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(truth)
    return (1 + BETA_SQ) * p * r / (BETA_SQ * p + r)


def macro_f05(pred: dict[str, set[str]], truth: dict[str, set[str]]) -> float:
    """Macro F0.5 over the keys of ``truth``; S1s missing from ``pred`` count as empty predictions."""
    if not truth:
        return float("nan")
    return sum(f05_entity(pred.get(s1, set()), t) for s1, t in truth.items()) / len(truth)


def _source_mix(ids: set[str]) -> str:
    """Label a truth set by which sources it contains: none / S2 only / S3 only / both."""
    has_s2 = any(i.startswith("S2-") for i in ids)
    has_s3 = any(i.startswith("S3-") for i in ids)
    if has_s2 and has_s3:
        return "both"
    if has_s2:
        return "S2 only"
    if has_s3:
        return "S3 only"
    return "none"


def per_entity(pred: dict[str, set[str]], truth: dict[str, set[str]]) -> pd.DataFrame:
    """One row per truth S1: f05, precision, recall, tp, n_pred, n_true, bucket and source mix.

    Precision is NaN when the prediction is empty and recall is NaN when the truth is empty
    (undefined, so group means skip them); ``f05`` always follows the metric rules.
    """
    rows = []
    for s1, t in truth.items():
        p = pred.get(s1, set())
        tp = len(p & t)
        rows.append(
            (
                s1,
                f05_entity(p, t),
                tp / len(p) if p else np.nan,
                tp / len(t) if t else np.nan,
                tp,
                len(p),
                len(t),
                _source_mix(t),
            )
        )
    df = pd.DataFrame(
        rows,
        columns=["s1_id", "f05", "precision", "recall", "tp", "n_pred", "n_true", "truth_source_mix"],
    )
    df["is_singleton"] = df["n_true"] == 0
    df["true_count_bucket"] = pd.Categorical(
        np.where(df["n_true"] >= 3, "3+", df["n_true"].astype(str)), categories=COUNT_BUCKETS, ordered=True
    )
    return df


def breakdown(pred: dict[str, set[str]], truth: dict[str, set[str]], s1_meta: pd.DataFrame) -> pd.DataFrame:
    """Mean F0.5 / precision / recall and count, grouped separately by each slice dimension.

    Dimensions: country, is_singleton, true_count_bucket (0/1/2/3+), truth_source_mix.
    ``s1_meta`` needs ``entity_id`` and ``country``; S1s absent from it get country ``<missing>``.
    Returns a long DataFrame with columns ``group, value, count, f05, precision, recall``.
    """
    ent = per_entity(pred, truth)
    country = s1_meta.drop_duplicates("entity_id").set_index("entity_id")["country"]
    ent["country"] = ent["s1_id"].map(country).fillna(MISSING_COUNTRY)

    parts = []
    for dim in ["country", "is_singleton", "true_count_bucket", "truth_source_mix"]:
        g = (
            ent.groupby(dim, observed=True)
            .agg(count=("f05", "size"), f05=("f05", "mean"), precision=("precision", "mean"), recall=("recall", "mean"))
            .reset_index()
            .rename(columns={dim: "value"})
        )
        g["value"] = g["value"].astype(str)
        g.insert(0, "group", dim)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _id_map(df: pd.DataFrame) -> dict[str, set[str]]:
    """Turn a two-column (s1_id, id-list) frame into ``{s1_id: set(ids)}``."""
    key, val = df.columns[0], df.columns[1]
    return {k.strip(): set(parse_id_list(v)) for k, v in zip(df[key], df[val])}


def load_truth(path: str) -> dict[str, set[str]]:
    """Load the ground-truth TSV into ``{s1_id: set(matched ids)}`` (empty string -> empty set)."""
    return _id_map(read_tsv(path))


def load_pred_tsv(path: str) -> dict[str, set[str]]:
    """Load ``matching_results.tsv`` (``source1_entity_id, matched_entity_ids``) into ``{s1_id: set(ids)}``."""
    return _id_map(read_tsv(path))


def main() -> None:
    """CLI: print macro F0.5 of a prediction TSV against a ground-truth TSV."""
    from src.config import load_config, resolve_path

    cfg = load_config()
    default_dir = resolve_path(cfg["paths"]["dataset_dir"]) / "train"
    parser = argparse.ArgumentParser(description="Macro F0.5 scorer")
    parser.add_argument("--pred", required=True, help="matching_results.tsv to score")
    parser.add_argument("--truth", default=str(default_dir / "train_ground_truth.tsv"))
    parser.add_argument("--s1", default=str(default_dir / "train_source1.tsv"), help="S1 file for country lookup")
    parser.add_argument("--by-country", action="store_true", help="also print per-country breakdown")
    parser.add_argument("--breakdown", action="store_true", help="print the full slice breakdown")
    args = parser.parse_args()

    truth = load_truth(args.truth)
    pred = load_pred_tsv(args.pred)
    extra = len(set(pred) - set(truth))
    missing = len(set(truth) - set(pred))
    print(f"truth S1: {len(truth):,}  pred S1: {len(pred):,}  missing from pred: {missing:,}  not in truth (ignored): {extra:,}")
    print(f"macro F0.5: {macro_f05(pred, truth):.6f}")

    if args.by_country or args.breakdown:
        s1_meta = read_tsv(args.s1)[["entity_id", "country"]]
        bd = breakdown(pred, truth, s1_meta)
        if not args.breakdown:
            bd = bd[bd["group"] == "country"]
        with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
            print(bd.to_string(index=False))


if __name__ == "__main__":
    main()
