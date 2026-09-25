"""Blocker B4: Shared postal code and digit numeric keys (PRD §D2).

Design notes:
- Inverted index on postal (longest 5-or-6 digit token) and on digit tokens of length ≥ 3.
- Skip any key that is shared by > 200 candidate records (noise; PRD rule).
- Run within the same country; fall back to global if country group < 50.
- Returns (s1_id, cand_id, rank_b4).  No sim score — numeric hits are binary.
"""

from __future__ import annotations

import pandas as pd
from collections import defaultdict
from tqdm import tqdm


_NOISE_CAP = 200   # posting-list size above which key is discarded (PRD)


def _extract_keys(row_entity_id, postal_val, digits_val) -> set[str]:
    keys: set[str] = set()

    postal = str(postal_val) if postal_val and str(postal_val) not in ("nan", "None", "") else ""
    if postal:
        keys.add(f"p_{postal}")

    digits = str(digits_val) if digits_val and str(digits_val) not in ("nan", "None", "") else ""
    for d in digits.split():
        if len(d) >= 3:
            keys.add(f"d_{d}")

    return keys


def _build_index(df: pd.DataFrame) -> dict[str, list[str]]:
    """Build inverted key → [entity_id, …] index from a DataFrame."""
    raw_index: dict[str, list[str]] = defaultdict(list)
    key_counts: dict[str, int] = defaultdict(int)

    has_postal = "postal" in df.columns
    has_digits = "digits" in df.columns

    for row in df.itertuples(index=False):
        postal_val = getattr(row, "postal", "") if has_postal else ""
        digits_val = getattr(row, "digits", "") if has_digits else ""
        eid = row.entity_id
        for key in _extract_keys(eid, postal_val, digits_val):
            raw_index[key].append(eid)
            key_counts[key] += 1

    # Filter noise keys
    return {k: v for k, v in raw_index.items() if key_counts[k] <= _NOISE_CAP}


def block_numeric_keys(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    """B4: Shared postal or digit-token inverted index (PRD §D2)."""

    has_postal = "postal" in cand_df.columns
    has_digits = "digits" in cand_df.columns

    # Unique countries
    countries = pd.unique(pd.concat([s1_df["country"], cand_df["country"]]))

    results: list[tuple] = []

    for country in tqdm(countries, desc="B4 numeric-key"):
        s1_c = s1_df[s1_df["country"] == country]
        cand_c = cand_df[cand_df["country"] == country]

        n_group = len(s1_c) + len(cand_c)

        # Fall back to global candidate pool for tiny country groups
        cand_pool = cand_c if n_group >= 50 else cand_df

        if len(s1_c) == 0:
            continue

        # Build index on the candidate pool for this country
        cand_index = _build_index(cand_pool)

        for row in s1_c.itertuples(index=False):
            postal_val = getattr(row, "postal", "") if has_postal else ""
            digits_val = getattr(row, "digits", "") if has_digits else ""
            s1_id = row.entity_id

            matched: set[str] = set()
            for key in _extract_keys(s1_id, postal_val, digits_val):
                if key in cand_index:
                    matched.update(cand_index[key])

            # Cap at k; assign rank in insertion order (all ranks are equally uncertain)
            for rank, cand_id in enumerate(list(matched)[:k], 1):
                results.append((s1_id, cand_id, rank))

    if not results:
        return pd.DataFrame(columns=["s1_id", "cand_id", "rank_b4"])

    df = pd.DataFrame(results, columns=["s1_id", "cand_id", "rank_b4"])
    # Drop duplicates that can appear when country fall-back fires
    df = df.drop_duplicates(subset=["s1_id", "cand_id"]).reset_index(drop=True)
    return df
