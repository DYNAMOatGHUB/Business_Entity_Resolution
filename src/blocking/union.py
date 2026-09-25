"""Union and blend blocker candidates, capping at K per S1 (PRD §D3).

Blend rank formula (PRD):
    blend_rank = max(sim_b1, sim_b5) + 0.1 * <number of blockers that hit>

Singleton safety: S1 entities that received ONLY very-low-similarity candidates
(all sim_b1 < MIN_SIM_KEEP and no B4/B3/B2 hit) keep their candidates in the
union but are flagged via `low_confidence=True`. The decision layer (Prabhu)
can use this flag; alternatively the pipeline can prune them before scoring
to avoid F0.5 penalty from false merges on singletons.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# Candidates whose best sim across all blockers falls below this are very likely
# singletons. Flagged but NOT dropped here — Prabhu's decision layer decides.
_LOW_CONF_SIM_THRESHOLD = 0.05


def union_blockers(
    *blocker_dfs: pd.DataFrame,
    cand_df: pd.DataFrame | None = None,
    cap_k: int = 40,
) -> pd.DataFrame:
    """Merge candidates across B1-B5, compute blend_rank, cap at cap_k per S1.

    Parameters
    ----------
    *blocker_dfs:
        Any subset of B1-B5 DataFrames. Each must contain at least
        ``s1_id`` and ``cand_id`` columns.
    cand_df:
        Full candidate records DataFrame (with ``entity_id`` and ``source``).
        Used to populate ``cand_source`` column.
    cap_k:
        Maximum candidates to keep per S1 entity after ranking by blend_rank.

    Returns
    -------
    pd.DataFrame with the contract columns:
        s1_id, cand_id, cand_source,
        hit_b1..hit_b5, rank_b1..rank_b5,
        sim_b1, sim_b5, blend_rank, low_confidence
    """
    active = [df for df in blocker_dfs if df is not None and not df.empty]
    if not active:
        return pd.DataFrame()

    merged = active[0]
    for df in active[1:]:
        merged = pd.merge(merged, df, on=["s1_id", "cand_id"], how="outer")

    # Materialise hit / rank / sim columns for every blocker slot
    for i in range(1, 6):
        rank_col = f"rank_b{i}"
        sim_col = f"sim_b{i}"
        hit_col = f"hit_b{i}"

        if rank_col not in merged.columns:
            merged[rank_col] = np.nan
        if sim_col not in merged.columns:
            merged[sim_col] = 0.0

        merged[hit_col] = merged[rank_col].notna().astype(np.int8)
        merged[sim_col] = merged[sim_col].fillna(0.0)

    # Blend rank
    merged["sim_b1"] = merged["sim_b1"].fillna(0.0)
    merged["sim_b5"] = merged["sim_b5"].fillna(0.0)
    hit_sum = merged[[f"hit_b{i}" for i in range(1, 6)]].sum(axis=1)
    merged["blend_rank"] = merged[["sim_b1", "sim_b5"]].max(axis=1) + 0.1 * hit_sum

    # Singleton safety flag — very low signal pairs
    max_sim_any = merged[[f"sim_b{i}" for i in range(1, 6)]].max(axis=1)
    merged["low_confidence"] = (max_sim_any < _LOW_CONF_SIM_THRESHOLD) & (hit_sum <= 1)

    # Sort and cap
    merged = (
        merged
        .sort_values(["s1_id", "blend_rank"], ascending=[True, False])
        .groupby("s1_id", sort=False)
        .head(cap_k)
        .reset_index(drop=True)
    )

    # Populate cand_source
    if cand_df is not None and not cand_df.empty:
        src_col = "source" if "source" in cand_df.columns else None
        if src_col:
            cand_map = cand_df.set_index("entity_id")[src_col].to_dict()
            merged["cand_source"] = merged["cand_id"].map(cand_map).fillna("")
        else:
            merged["cand_source"] = ""
    else:
        merged["cand_source"] = ""

    out_cols = [
        "s1_id", "cand_id", "cand_source",
        "hit_b1", "hit_b2", "hit_b3", "hit_b4", "hit_b5",
        "rank_b1", "rank_b2", "rank_b3", "rank_b4", "rank_b5",
        "sim_b1", "sim_b5", "blend_rank", "low_confidence",
    ]
    for col in out_cols:
        if col not in merged.columns:
            merged[col] = None

    return merged[out_cols]
