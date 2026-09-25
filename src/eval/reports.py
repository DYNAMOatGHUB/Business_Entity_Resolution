"""Reporting and error analysis utilities (PRD §D3, §D4).

Dhyanesh's responsibilities:
- Blocking recall@K report (K = 10, 20, 40, 60), per-country and per-source breakdown.
- List of missed true pairs for Prabhu's normalisation fixes.
- F0.5 threshold sweep on OOF scores to find optimal cut-off.
- Error dumps: 50 worst FPs and 50 worst FNs side-by-side.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Blocking recall@K report (D3)
# ---------------------------------------------------------------------------

def generate_blocking_report(
    candidates_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    out_path: str = "reports/blocking_report.md",
) -> dict:
    """Compute recall@K, per-country recall, per-source recall; write markdown report.

    The ground_truth_df must have columns: source1_entity_id, matched_entity_ids
    (the raw TSV format from the dataset) OR (s1_id, cand_id) long form.

    Returns a dict with recall values for programmatic use.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Normalise ground truth to long form (s1_id, cand_id)
    if "source1_entity_id" in ground_truth_df.columns:
        rows = []
        for _, row in ground_truth_df.iterrows():
            s1 = row["source1_entity_id"]
            ids_str = str(row.get("matched_entity_ids", "")).strip()
            if ids_str and ids_str != "nan":
                for cid in ids_str.split(","):
                    cid = cid.strip()
                    if cid:
                        rows.append((s1, cid))
        gt_long = pd.DataFrame(rows, columns=["s1_id", "cand_id"])
    else:
        gt_long = ground_truth_df[["s1_id", "cand_id"]].copy()

    gt_pairs = set(zip(gt_long["s1_id"], gt_long["cand_id"]))
    total_true = len(gt_pairs)

    # Rank candidates within each S1 by blend_rank descending
    cands = candidates_df.copy()
    cands["_rank"] = (
        cands.groupby("s1_id")["blend_rank"]
        .rank(method="first", ascending=False)
    )

    ks = [10, 20, 40, 60]
    recall_at_k: dict[int, float] = {}
    for k in ks:
        top_k = cands[cands["_rank"] <= k]
        hit = len(set(zip(top_k["s1_id"], top_k["cand_id"])).intersection(gt_pairs))
        recall_at_k[k] = hit / total_true if total_true > 0 else 0.0

    avg_cands = len(cands) / cands["s1_id"].nunique() if not cands.empty else 0.0

    # Missed pairs
    all_cand_pairs = set(zip(cands["s1_id"], cands["cand_id"]))
    missed = gt_pairs - all_cand_pairs

    # Per-source recall (S2 vs S3) using cand_id prefix
    def _recall_by_source(source_prefix: str) -> float:
        src_pairs = {(s, c) for s, c in gt_pairs if c.startswith(source_prefix)}
        if not src_pairs:
            return float("nan")
        hit = len(all_cand_pairs.intersection(src_pairs))
        return hit / len(src_pairs)

    recall_s2 = _recall_by_source("S2-")
    recall_s3 = _recall_by_source("S3-")

    # Per-country recall (join via s1_id back to a records frame if available)
    # We emit per-source only here; per-country requires the records frame which
    # the caller can pass in via a wrapper if needed.

    # Write report
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Blocking Report\n\n")
        f.write(f"- **Total true pairs**: {total_true}\n")
        f.write(f"- **S1 entities in candidates**: {cands['s1_id'].nunique()}\n")
        f.write(f"- **Avg candidates per S1**: {avg_cands:.2f}\n\n")

        f.write("## Recall@K\n\n")
        f.write("| K | Recall |\n|---|--------|\n")
        for k in ks:
            gate = "✅" if recall_at_k[k] >= 0.97 else "❌"
            f.write(f"| {k} | {recall_at_k[k]:.4f} {gate} |\n")

        f.write(f"\n**Gate (97%+)**: smallest K that clears it = ")
        passed = [k for k in ks if recall_at_k[k] >= 0.97]
        f.write(f"{min(passed) if passed else 'NONE — increase K'}\n\n")

        f.write("## Per-Source Recall\n\n")
        f.write(f"- S2: {recall_s2:.4f}\n")
        f.write(f"- S3: {recall_s3:.4f}\n\n")

        f.write(f"## Missed Pairs ({len(missed)} total)\n\n")
        for pair in list(sorted(missed))[:50]:
            f.write(f"- {pair[0]}  →  {pair[1]}\n")
        if len(missed) > 50:
            f.write(f"…and {len(missed) - 50} more.  See missed_pairs.tsv.\n")

    # Also write missed pairs as TSV for Prabhu
    missed_path = Path(out_path).parent / "missed_pairs.tsv"
    pd.DataFrame(sorted(missed), columns=["s1_id", "cand_id"]).to_csv(
        missed_path, sep="\t", index=False
    )

    print(f"[Blocking report] {out_path}")
    print(f"  Recall@10={recall_at_k[10]:.4f}  @20={recall_at_k[20]:.4f}"
          f"  @40={recall_at_k[40]:.4f}  @60={recall_at_k[60]:.4f}")
    print(f"  Avg cands/S1={avg_cands:.1f}   Missed={len(missed)}")

    return {
        "recall_at_k": recall_at_k,
        "avg_cands": avg_cands,
        "missed": missed,
        "recall_s2": recall_s2,
        "recall_s3": recall_s3,
    }


# ---------------------------------------------------------------------------
# F0.5 threshold sweep (D4 / D6) — Dhyanesh uses this on OOF scores
# ---------------------------------------------------------------------------

def _f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


def sweep_threshold(
    oof_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    score_col: str = "p_lgbm",
    thresholds: list[float] | None = None,
    out_path: str = "reports/threshold_sweep.md",
) -> dict:
    """Sweep decision thresholds on OOF scores; maximise macro F0.5.

    oof_df must have columns: s1_id, cand_id, <score_col>
    ground_truth_df: source1_entity_id, matched_entity_ids  (raw TSV format)

    Returns dict with best_threshold and best_f05.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.30, 0.96, 0.02)]

    # Normalise GT to long
    if "source1_entity_id" in ground_truth_df.columns:
        rows = []
        for _, row in ground_truth_df.iterrows():
            s1 = row["source1_entity_id"]
            ids_str = str(row.get("matched_entity_ids", "")).strip()
            if ids_str and ids_str != "nan":
                for cid in ids_str.split(","):
                    cid = cid.strip()
                    if cid:
                        rows.append((s1, cid))
        gt_long = pd.DataFrame(rows, columns=["s1_id", "cand_id"])
    else:
        gt_long = ground_truth_df[["s1_id", "cand_id"]].copy()

    gt_by_s1: dict[str, set] = {}
    for s1, cid in zip(gt_long["s1_id"], gt_long["cand_id"]):
        gt_by_s1.setdefault(s1, set()).add(cid)

    all_s1 = list(set(oof_df["s1_id"].unique()) | set(gt_long["s1_id"].unique()))

    best_f05 = -1.0
    best_thresh = 0.5
    records = []

    for tau in thresholds:
        # Predicted matches per S1 at this threshold
        above = oof_df[oof_df[score_col] >= tau]
        pred_by_s1: dict[str, set] = {}
        for s1, cid in zip(above["s1_id"], above["cand_id"]):
            pred_by_s1.setdefault(s1, set()).add(cid)

        # Macro F0.5
        f05_list = []
        for s1 in all_s1:
            pred = pred_by_s1.get(s1, set())
            truth = gt_by_s1.get(s1, set())

            if not truth and not pred:
                f05_list.append(1.0)
            elif not truth:
                f05_list.append(0.0)
            elif not pred:
                f05_list.append(0.0)
            else:
                tp = len(pred & truth)
                prec = tp / len(pred) if pred else 0.0
                rec = tp / len(truth) if truth else 0.0
                f05_list.append(_f_beta(prec, rec))

        macro_f05 = float(np.mean(f05_list))
        records.append({"threshold": tau, "macro_f05": macro_f05})

        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_thresh = tau

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sweep_df = pd.DataFrame(records)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Threshold Sweep (F0.5)\n\n")
        f.write(f"Best threshold: **{best_thresh}**  →  macro F0.5 = **{best_f05:.4f}**\n\n")
        f.write(sweep_df.to_markdown(index=False))

    print(f"[Threshold sweep] best_threshold={best_thresh}  best_f0.5={best_f05:.4f}")
    return {"best_threshold": best_thresh, "best_f05": best_f05, "sweep": sweep_df}


# ---------------------------------------------------------------------------
# Error dump (D4 / D6)
# ---------------------------------------------------------------------------

def dump_errors(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    records_df: pd.DataFrame | None = None,
    out_path: str = "reports/errors.tsv",
    n: int = 50,
) -> None:
    """Dump top-n FPs and top-n FNs for manual error analysis.

    predictions_df must have: s1_id, cand_id, p_final (or p_lgbm).
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    score_col = "p_final" if "p_final" in predictions_df.columns else "p_lgbm"

    if "source1_entity_id" in ground_truth_df.columns:
        rows = []
        for _, row in ground_truth_df.iterrows():
            s1 = row["source1_entity_id"]
            ids_str = str(row.get("matched_entity_ids", "")).strip()
            if ids_str and ids_str != "nan":
                for cid in ids_str.split(","):
                    cid = cid.strip()
                    if cid:
                        rows.append((s1, cid))
        gt_long = pd.DataFrame(rows, columns=["s1_id", "cand_id"])
    else:
        gt_long = ground_truth_df[["s1_id", "cand_id"]].copy()

    gt_set = set(zip(gt_long["s1_id"], gt_long["cand_id"]))

    preds = predictions_df.copy()
    preds["is_tp"] = [
        (s, c) in gt_set for s, c in zip(preds["s1_id"], preds["cand_id"])
    ]

    # FPs: predicted as positive but not in GT
    threshold = 0.5
    positives = preds[preds[score_col] >= threshold]
    fps = positives[~positives["is_tp"]].nlargest(n, score_col)

    # FNs: in GT but predicted below threshold (or absent)
    pred_set = set(zip(positives["s1_id"], positives["cand_id"]))
    fn_pairs = gt_set - pred_set
    fns = gt_long[
        [
            (s, c) in fn_pairs
            for s, c in zip(gt_long["s1_id"], gt_long["cand_id"])
        ]
    ].head(n)

    fps["error_type"] = "FP"
    fns = fns.copy()
    fns["error_type"] = "FN"
    fns[score_col] = None

    errors = pd.concat([fps, fns], ignore_index=True)
    errors.to_csv(out_path, sep="\t", index=False)
    print(f"[Error dump] FPs={len(fps)}  FNs={len(fns)}  → {out_path}")
