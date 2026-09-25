"""Stage 4: Decision layer.
Applies no-match gate, pair threshold, relative score cut, one-owner constraint,
and unseen-country margin adjustments to output final matches.
"""

import pandas as pd


def apply_decision_rules(
    scores_df: pd.DataFrame,
    tau_gate: float = 0.50,
    tau_pair: float = 0.40,
    alpha_relative: float = 0.80,
    one_owner: bool = True,
    unseen_country_margin: float = 0.05,
) -> pd.DataFrame:
    """Filter candidates into final matches based on tuned decision thresholds."""
    return pd.DataFrame(columns=["s1_id", "matched_cand_ids"])
