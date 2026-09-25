"""Artifact data contracts: file names, required columns, validation and loading.

Every stage reads and writes artifacts through this module so file names and columns stay
frozen (see CLAUDE.md "Data contracts"). All artifacts carry a split suffix (``_train`` /
``_test``) except ``scores_loco``, which only exists for train.

Sampling (``--sample``) keeps a fraction of the split's S1 ids, drawn from the raw
``{split}_source1.tsv`` id list with ``io_utils.sample_s1_ids`` (``sample.frac``, ``seed``),
so every artifact and every module sees the same S1 subset.

Run: ``python -m src.contracts --split train [--sample]`` to check every artifact on disk.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.config import load_config, resolve_path
from src.io_utils import sample_s1_ids

SPLITS = ("train", "test")

ARTIFACTS: dict[str, str] = {
    "records_norm": "records_norm_{split}.parquet",
    "candidates": "candidates_{split}.parquet",
    "features_str": "features_str_{split}.parquet",
    "features": "features_{split}.parquet",
    "scores": "scores_{split}.parquet",
    "scores_loco": "scores_loco_train.parquet",
    "decisions": "decisions_{split}.parquet",
}

TRAIN_ONLY_ARTIFACTS = {"scores_loco"}

PRODUCED_BY: dict[str, str] = {
    "records_norm": "Stage 0 normalize (python -m src.normalize, Prabhu)",
    "candidates": "Stage 1 blocking (python -m src.run --stage 1, Dhyanesh)",
    "features_str": "Stage 2 string features (python -m src.features.string_feats, Prabhu)",
    "features": "Stage 2 feature build (python -m src.run --stage 2, Dhyanesh)",
    "scores": "Stage 3 scoring (python -m src.run --stage 3, Dhyanesh)",
    "scores_loco": "Stage 3 leave-one-country-out scoring (python -m src.run --stage 3, Dhyanesh)",
    "decisions": "Stage 4 decide (python -m src.decide, Prabhu)",
}

_BLOCKERS = range(1, 6)

REQUIRED_COLUMNS: dict[str, list[str]] = {
    "records_norm": [
        "entity_id", "source", "country", "name_raw", "addr_raw", "name_core", "name_legal",
        "name_sorted", "name_acronym", "addr_norm", "digits", "postal", "addr_first_num", "landmark_flag",
    ],
    "candidates": [
        "s1_id", "cand_id", "cand_source",
        *[f"hit_b{i}" for i in _BLOCKERS], *[f"rank_b{i}" for i in _BLOCKERS],
        "sim_b1", "sim_b5", "blend_rank",
    ],
    "features_str": ["s1_id", "cand_id"],
    "features": ["s1_id", "cand_id"],
    "scores": ["s1_id", "cand_id", "p_lgbm", "p_final"],
    "scores_loco": ["s1_id", "cand_id", "p_lgbm", "p_final", "holdout_country"],
    "decisions": ["s1_id", "cand_id", "p_final", "is_match"],
}

TRAIN_ONLY_COLUMNS: dict[str, list[str]] = {
    "features_str": ["label"],
    "features": ["label"],
    "scores": ["fold"],
}

OPTIONAL_COLUMNS: dict[str, list[str]] = {
    "scores": ["p_ce"],
    "scores_loco": ["p_ce", "fold"],
}

ALLOWED_PREFIXES: dict[str, tuple[str, ...]] = {
    "features_str": ("f_str_",),
    "features": ("f_str_", "f_vec_", "f_ctx_"),
}

KEY_COLUMNS: dict[str, list[str]] = {name: ["s1_id", "cand_id"] for name in ARTIFACTS} | {"records_norm": ["entity_id"]}


def _check_name_split(name: str, split: str) -> None:
    """Raise ValueError on an unknown artifact, unknown split, or a train-only artifact on test."""
    if name not in ARTIFACTS:
        raise ValueError(f"unknown artifact {name!r}; expected one of {sorted(ARTIFACTS)}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if name in TRAIN_ONLY_ARTIFACTS and split != "train":
        raise ValueError(f"artifact {name!r} only exists for split 'train', got {split!r}")


def required_columns(name: str, split: str) -> list[str]:
    """Required columns of artifact ``name`` for ``split`` (train adds label / fold where applicable)."""
    _check_name_split(name, split)
    extra = TRAIN_ONLY_COLUMNS.get(name, []) if split == "train" else []
    return REQUIRED_COLUMNS[name] + extra


def artifact_path(name: str, split: str, cfg: dict | None = None) -> Path:
    """Path of artifact ``name`` for ``split`` under ``paths.artifacts_dir``."""
    _check_name_split(name, split)
    cfg = cfg if cfg is not None else load_config()
    return resolve_path(cfg["paths"]["artifacts_dir"]) / ARTIFACTS[name].format(split=split)


def _column_problems(cols: list[str], name: str, split: str) -> list[str]:
    """Missing required columns and extra columns (not required, optional or feature-prefixed)."""
    required = required_columns(name, split)
    allowed = set(required) | set(OPTIONAL_COLUMNS.get(name, []))
    prefixes = ALLOWED_PREFIXES.get(name, ())
    problems = []
    missing = [c for c in required if c not in cols]
    if missing:
        problems.append(f"missing columns {missing}")
    extra = [c for c in cols if c not in allowed and not (prefixes and c.startswith(prefixes))]
    if extra:
        hint = f" (allowed prefixes: {list(prefixes)})" if prefixes else ""
        problems.append(f"unexpected columns {extra}{hint}")
    return problems


def _duplicate_problems(df: pd.DataFrame, name: str) -> list[str]:
    """Duplicate key rows (skipped when the key columns are not all present)."""
    key = KEY_COLUMNS[name]
    if not all(k in df.columns for k in key):
        return []
    dup = df.duplicated(key)
    n_dup = int(dup.sum())
    if not n_dup:
        return []
    examples = df.loc[df.duplicated(key, keep=False), key].drop_duplicates().head(3).values.tolist()
    return [f"{n_dup} duplicate ({', '.join(key)}) rows, e.g. {examples}"]


def _raise(problems: list[str], name: str, split: str) -> None:
    if problems:
        raise ValueError(f"{name}_{split} contract violated: " + "; ".join(problems))


def check_artifact(df: pd.DataFrame, name: str, split: str) -> None:
    """Validate ``df`` against the contract of ``name``/``split``.

    Raises ValueError listing every problem at once: missing required columns, extra columns
    (not required, optional or carrying an allowed feature prefix) and duplicate key rows.
    """
    _raise(_column_problems(list(df.columns), name, split) + _duplicate_problems(df, name), name, split)


@lru_cache(maxsize=8)
def _sampled_ids_cached(s1_path: str, frac: float, seed: int) -> frozenset[str]:
    ids = pd.read_csv(
        s1_path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, usecols=["entity_id"]
    )["entity_id"]
    return frozenset(sample_s1_ids(ids, frac, seed))


def sampled_s1_ids(split: str, cfg: dict | None = None) -> frozenset[str]:
    """The ``--sample`` S1 id subset of ``split``: ``sample.frac`` of raw S1 ids, seed ``sample.seed``."""
    cfg = cfg if cfg is not None else load_config()
    s1_path = resolve_path(cfg["paths"]["dataset_dir"]) / split / f"{split}_source1.tsv"
    return _sampled_ids_cached(str(s1_path), float(cfg["sample"]["frac"]), int(cfg["sample"]["seed"]))


def sample_frame(df: pd.DataFrame, name: str, keep_s1: frozenset[str] | set[str]) -> pd.DataFrame:
    """Restrict an artifact frame to the kept S1 ids.

    Pair artifacts keep every row of a kept S1 (whole groups). ``records_norm`` keeps the kept
    S1 records and all S2/S3 records, since candidates are not known at that stage.
    """
    if name == "records_norm":
        mask = (df["source"] != "S1") | df["entity_id"].isin(keep_s1)
    else:
        mask = df["s1_id"].isin(keep_s1)
    return df.loc[mask].reset_index(drop=True)


def load_artifact(
    name: str,
    split: str,
    sample: bool = False,
    cfg: dict | None = None,
    path: str | Path | None = None,
    columns: list[str] | None = None,
    filters: list | None = None,
) -> pd.DataFrame:
    """Read an artifact, validate it against its contract and optionally apply the S1 sample.

    ``path`` overrides the contract file name (e.g. a synthetic scores file) but the contract of
    ``name`` still applies. Columns are validated from the parquet schema, so ``columns`` /
    ``filters`` (pyarrow row filters, e.g. ``[("p_final", ">=", 0.2)]``) can load a slice of a
    large file; duplicate keys are checked on the rows actually loaded.
    Raises FileNotFoundError naming the expected path and the stage that produces it.
    """
    cfg = cfg if cfg is not None else load_config()
    path = Path(path) if path is not None else artifact_path(name, split, cfg)
    _check_name_split(name, split)
    if not path.exists():
        raise FileNotFoundError(f"{name} artifact not found at {path}; it is produced by {PRODUCED_BY[name]}")
    schema_cols = [c for c in pq.read_schema(path).names if not c.startswith("__index_level_")]
    _raise(_column_problems(schema_cols, name, split), name, split)
    df = pd.read_parquet(path, columns=columns, filters=filters)
    _raise(_duplicate_problems(df, name), name, split)
    if sample:
        df = sample_frame(df, name, sampled_s1_ids(split, cfg))
    return df


def load_s1_meta(split: str, sample: bool = False, cfg: dict | None = None) -> pd.DataFrame:
    """S1 ``entity_id, country`` of a split: from records_norm, or the raw S1 TSV if Stage 0 has not run."""
    cfg = cfg if cfg is not None else load_config()
    try:
        rec = load_artifact("records_norm", split, cfg=cfg, columns=["entity_id", "source", "country"])
        meta = rec.loc[rec["source"] == "S1", ["entity_id", "country"]].reset_index(drop=True)
    except FileNotFoundError as e:
        print(f"[contracts] {e}\n[contracts] using raw {split}_source1.tsv for S1 countries")
        s1_path = resolve_path(cfg["paths"]["dataset_dir"]) / split / f"{split}_source1.tsv"
        meta = pd.read_csv(
            s1_path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, usecols=["entity_id", "country"]
        )
    if sample:
        meta = meta[meta["entity_id"].isin(sampled_s1_ids(split, cfg))].reset_index(drop=True)
    return meta


def main() -> None:
    """CLI: validate every artifact present on disk for a split."""
    parser = argparse.ArgumentParser(description="Check artifacts against the data contracts")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--sample", action="store_true", help="also report row counts after the S1 sample")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    failed = False
    for name in ARTIFACTS:
        if name in TRAIN_ONLY_ARTIFACTS and args.split != "train":
            continue
        try:
            df = load_artifact(name, args.split, sample=args.sample, cfg=cfg)
            print(f"[OK]      {name:13s} {len(df):>12,} rows")
        except FileNotFoundError as e:
            print(f"[MISSING] {e}")
        except ValueError as e:
            failed = True
            print(f"[FAIL]    {e}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
