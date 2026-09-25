"""Union and blend blocker candidates, capping at K per S1."""

import pandas as pd


def union_blockers(*blocker_dfs: pd.DataFrame, cap_k: int = 50) -> pd.DataFrame:
    """Merge candidates across B1-B5, compute blend_rank, and cap at cap_k per S1.
    Expected output columns:
    s1_id, cand_id, cand_source, hit_b1..hit_b5, rank_b1..rank_b5, sim_b1, sim_b5, blend_rank
    """
    return pd.DataFrame(columns=[
        "s1_id", "cand_id", "cand_source",
        "hit_b1", "hit_b2", "hit_b3", "hit_b4", "hit_b5",
        "rank_b1", "rank_b2", "rank_b3", "rank_b4", "rank_b5",
        "sim_b1", "sim_b5", "blend_rank"
    ])
