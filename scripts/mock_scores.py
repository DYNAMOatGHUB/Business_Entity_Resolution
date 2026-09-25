"""Generate mock candidate pairs and scores for testing pipeline downstream stages."""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_mock_scores(out_path: str = "artifacts/scores_train.parquet", n_records: int = 100):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(1, n_records + 1):
        s1_id = f"s1_{i}"
        for j in range(1, 4):
            cand_id = f"s2_{i*10 + j}"
            p_lgbm = float(np.random.uniform(0.1, 0.95))
            rows.append({
                "s1_id": s1_id,
                "cand_id": cand_id,
                "fold": i % 5,
                "p_lgbm": p_lgbm,
                "p_ce": p_lgbm,
                "p_final": p_lgbm,
            })
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"Generated mock scores to {out_path}")


if __name__ == "__main__":
    generate_mock_scores()
