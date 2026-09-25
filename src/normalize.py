"""Stage 0: Normalization module.
Handles unicode folding, legal suffixes extraction, address abbreviations,
and digit/postal code extraction.
"""

import pandas as pd


def normalize_records(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize input records DataFrame.
    Expected output columns:
    entity_id, source, country, name_raw, addr_raw, name_core,
    name_legal, name_sorted, name_acronym, addr_norm, digits,
    postal, addr_first_num, landmark_flag
    """
    # Placeholder implementation to be extended
    return df
