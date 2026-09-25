"""Output generation and submission validation.
Writes matching_results.tsv and candidate_pairs.tsv, then executes official validator.
"""

from pathlib import Path
import subprocess
import pandas as pd


def write_submission_files(
    candidates_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    output_dir: str = "output",
    validate: bool = True,
    test_dir: str = "dataset/test",
) -> None:
    """Write output TSVs and optionally run validation script."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cand_file = out_path / "candidate_pairs.tsv"
    match_file = out_path / "matching_results.tsv"

    candidates_df.to_csv(cand_file, sep="\t", index=False)
    matches_df.to_csv(match_file, sep="\t", index=False)

    if validate:
        validator_script = Path("utils/validate_submission.py")
        if validator_script.exists():
            subprocess.run([
                "python3", str(validator_script),
                "--matching", str(match_file),
                "--candidate", str(cand_file),
                "--test-dir", str(test_dir),
            ], check=True)
