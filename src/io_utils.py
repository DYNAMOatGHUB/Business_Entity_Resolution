"""I/O utilities: reading source/truth TSVs, ID-list parsing/formatting, sampling, ID-only TSV writing.

All TSVs are read with ``sep="\\t", dtype=str, keep_default_na=False`` so empty fields stay ``""``.
"""

import argparse
import csv
from pathlib import Path

import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def _dataset_dir(dataset_dir: str | Path | None) -> Path:
    """Resolve the dataset directory, defaulting to ``paths.dataset_dir`` from the config."""
    if dataset_dir is not None:
        return Path(dataset_dir)
    from src.config import load_config, resolve_path

    return resolve_path(load_config()["paths"]["dataset_dir"])


def read_tsv(path: str | Path) -> pd.DataFrame:
    """Read any challenge TSV as all-string columns with empty strings kept as ``""``."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


def read_source(split: str, n: int, dataset_dir: str | Path | None = None) -> pd.DataFrame:
    """Read ``{split}_source{n}.tsv`` and add a ``source`` column (``S1``/``S2``/``S3``).

    Raises ValueError if the file's columns differ from the frozen schema.
    """
    path = _dataset_dir(dataset_dir) / split / f"{split}_source{n}.tsv"
    df = read_tsv(path)
    if list(df.columns) != SOURCE_COLUMNS:
        raise ValueError(f"{path}: expected columns {SOURCE_COLUMNS}, got {list(df.columns)}")
    df["source"] = f"S{n}"
    return df


def read_all_sources(split: str, dataset_dir: str | Path | None = None) -> pd.DataFrame:
    """Read and concatenate all three sources of a split (with ``source`` column)."""
    return pd.concat([read_source(split, n, dataset_dir) for n in (1, 2, 3)], ignore_index=True)


def read_truth(split: str = "train", dataset_dir: str | Path | None = None) -> pd.DataFrame:
    """Read ``{split}_ground_truth.tsv`` as a DataFrame (columns ``source1_entity_id, matched_entity_ids``)."""
    path = _dataset_dir(dataset_dir) / split / f"{split}_ground_truth.tsv"
    df = read_tsv(path)
    if list(df.columns) != TRUTH_COLUMNS:
        raise ValueError(f"{path}: expected columns {TRUTH_COLUMNS}, got {list(df.columns)}")
    return df


def parse_id_list(s: str | None) -> list[str]:
    """Parse a comma-separated ID string into a list; whitespace is stripped, empties dropped.

    ``""``, ``None`` and NaN give ``[]``.
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def format_id_list(ids) -> str:
    """Join IDs with commas and no spaces, dropping duplicates while keeping first-seen order."""
    return ",".join(dict.fromkeys(str(i).strip() for i in ids if str(i).strip()))


def sample_s1_ids(s1_ids, frac: float, seed: int = 42) -> list[str]:
    """Deterministically sample a fraction of S1 ids (independent of input order), returned sorted."""
    unique = pd.Series(sorted(set(s1_ids)), dtype=str)
    return sorted(unique.sample(frac=frac, random_state=seed).tolist())


def write_id_tsv(df: pd.DataFrame, path: str | Path, header: list[str]) -> None:
    """Write a two-column ID-only TSV with the given header, no quoting and ``\\n`` line endings."""
    if len(header) != df.shape[1]:
        raise ValueError(f"header has {len(header)} names but df has {df.shape[1]} columns")
    out = df.copy()
    out.columns = header
    out = out.fillna("").astype(str)
    for col in header:
        if out[col].str.contains(r"[\t\n\r\"]", regex=True).any():
            raise ValueError(f"column {col!r} contains tab/newline/quote characters")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE, lineterminator="\n")


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Save dataframe to Parquet with parent directory creation."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: str) -> pd.DataFrame:
    """Read parquet file into dataframe."""
    return pd.read_parquet(path)


def main() -> None:
    """Print row counts per source and country value counts per file, for a quick load check."""
    parser = argparse.ArgumentParser(description="Sanity-check that source files load correctly")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    args = parser.parse_args()
    for split in args.splits:
        for n in (1, 2, 3):
            df = read_source(split, n)
            print(f"\n{split}_source{n}.tsv: {len(df):,} rows")
            print(df["country"].value_counts().to_string())


if __name__ == "__main__":
    main()
