"""Stage 4: decision layer: turn pair scores into kept matches, fully vectorized.

Rules, in order (all numpy on integer codes, no per-S1 Python loops):
  a. offset: ``delta_unseen`` for S1s whose country is not in ``train_countries``, else 0
  b. gate: drop every pair of an S1 whose max ``p_final`` < ``tau_gate`` + offset
  c. pair threshold: ``p_final`` >= (``tau_s2`` / ``tau_s3`` by candidate source if set, else
     ``tau_pair``) + offset
  d. relative cut: if ``alpha`` > 0, ``p_final`` >= ``alpha`` * S1 max
  e. one-owner: each ``cand_id`` keeps only its highest-scoring surviving row

Pairs scoring below the lowest threshold that can ever apply are pruned first; that never
changes the result (every kept pair must clear the pair threshold). ``PairArrays`` is shared
with ``src.eval.threshold_search`` so the search and the pipeline apply identical logic.

Run: ``python -m src.decide --split train|test [--sample] [--scores PATH]``
Writes ``decisions_<split>.parquet`` (s1_id, cand_id, p_final, is_match) for every pair that
survived pruning.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_MIN_P = 0.2


def codes_in(values: pd.Series, index: pd.Index) -> np.ndarray:
    """Position of each value in ``index`` (-1 if absent) without materializing Python strings.

    Factorizes ``values`` first (cheap on Arrow strings) and looks up only the uniques.
    """
    codes, uniques = pd.factorize(values)
    lookup = index.get_indexer(uniques)
    return np.where(codes >= 0, lookup[codes], -1).astype(np.int32)


def ids_to_arrow(codes: np.ndarray, uniques: np.ndarray) -> pa.Array:
    """Arrow string array ``uniques[codes]`` built via a dictionary (no per-row Python objects)."""
    return pa.DictionaryArray.from_arrays(pa.array(codes, pa.int32()), pa.array(list(uniques), pa.string())).cast(pa.string())


@dataclass(frozen=True)
class DecideParams:
    """Decision thresholds (``decide`` section of the config)."""

    tau_gate: float = 0.5
    tau_pair: float = 0.5
    tau_s2: float | None = None
    tau_s3: float | None = None
    alpha: float = 0.0
    one_owner: bool = True
    delta_unseen: float = 0.0

    @classmethod
    def from_cfg(cls, cfg: dict) -> "DecideParams":
        """Build from a full config or its ``decide`` section (unknown keys ignored)."""
        d = cfg.get("decide", cfg)
        opt = lambda k: None if d.get(k) is None else float(d[k])  # noqa: E731
        return cls(
            tau_gate=float(d.get("tau_gate", cls.tau_gate)),
            tau_pair=float(d.get("tau_pair", cls.tau_pair)),
            tau_s2=opt("tau_s2"),
            tau_s3=opt("tau_s3"),
            alpha=float(d.get("alpha", cls.alpha) or 0.0),
            one_owner=bool(d.get("one_owner", cls.one_owner)),
            delta_unseen=float(d.get("delta_unseen", cls.delta_unseen) or 0.0),
        )

    def source_thresholds(self) -> dict[str, float]:
        """Pair threshold per candidate source label; sources not listed use ``tau_pair``."""
        out = {}
        if self.tau_s2 is not None:
            out["S2"] = self.tau_s2
        if self.tau_s3 is not None:
            out["S3"] = self.tau_s3
        return out

    def min_threshold(self) -> float:
        """Lowest pair threshold any row can face (after a possibly negative offset)."""
        return min([self.tau_pair, *self.source_thresholds().values()]) + min(0.0, self.delta_unseen)


@dataclass
class PairArrays:
    """Scored pairs as integer-coded numpy arrays for fast repeated decisions.

    ``s1_ids`` are the S1 universe (``s1_table`` order); ``s1_code`` indexes it per row.
    ``source_rows`` maps each row back to its position in the input frame (to align extra columns).
    """

    s1_ids: np.ndarray
    s1_code: np.ndarray
    cand_ids: np.ndarray
    cand_code: np.ndarray
    src_labels: np.ndarray
    src_code: np.ndarray
    p: np.ndarray
    s1_max: np.ndarray
    unseen: np.ndarray
    source_rows: np.ndarray
    n_dropped_unknown_s1: int = 0
    _owner_order: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_s1(self) -> int:
        return len(self.s1_ids)

    def __len__(self) -> int:
        return len(self.p)

    @classmethod
    def build(
        cls,
        scores_df: pd.DataFrame,
        s1_table: pd.DataFrame,
        train_countries,
        min_p: float = 0.0,
        score_col: str = "p_final",
    ) -> "PairArrays":
        """Encode ``scores_df`` (s1_id, cand_id, ``score_col``) against ``s1_table`` (s1_id, country).

        Rows with ``score_col`` < ``min_p`` or with an S1 absent from ``s1_table`` are dropped.
        """
        s1_ids = s1_table["s1_id"].to_numpy(dtype=object)
        p_all = scores_df[score_col].to_numpy(dtype=np.float64)
        s1_code = codes_in(scores_df["s1_id"], pd.Index(s1_ids))
        keep = (p_all >= min_p) & (s1_code >= 0)
        n_unknown = int(((s1_code < 0) & (p_all >= min_p)).sum())
        rows = np.flatnonzero(keep)

        s1_code = s1_code[rows]
        p = p_all[rows]
        cand_code_all, cand_ids = pd.factorize(scores_df["cand_id"])
        cand_code = cand_code_all[rows].astype(np.int32)
        cand_ids = np.asarray(cand_ids, dtype=object)
        src_code, src_labels = pd.factorize(pd.Series(cand_ids, dtype=object).str.split("-", n=1).str[0])
        src_code = src_code.astype(np.int16)[cand_code] if len(cand_ids) else np.zeros(0, np.int16)

        s1_max = pd.Series(p).groupby(s1_code).transform("max").to_numpy() if len(p) else p.copy()
        unseen_s1 = ~s1_table["country"].isin(list(train_countries)).to_numpy()
        return cls(
            s1_ids=s1_ids,
            s1_code=s1_code,
            cand_ids=cand_ids,
            cand_code=cand_code,
            src_labels=np.asarray(src_labels, dtype=object),
            src_code=src_code,
            p=p,
            s1_max=s1_max,
            unseen=unseen_s1[s1_code],
            source_rows=rows,
            n_dropped_unknown_s1=n_unknown,
        )

    def _order_for_owner(self) -> np.ndarray:
        """Row order by cand, then score desc, then S1 code (deterministic tie-break); cached."""
        if self._owner_order is None:
            self._owner_order = np.lexsort((self.s1_code, -self.p, self.cand_code))
        return self._owner_order

    def one_owner(self, keep: np.ndarray) -> np.ndarray:
        """Restrict ``keep`` so each cand keeps only its highest-scoring kept row."""
        order = self._order_for_owner()
        kept_rows = order[keep[order]]
        c = self.cand_code[kept_rows]
        first = np.ones(len(c), dtype=bool)
        first[1:] = c[1:] != c[:-1]
        out = np.zeros(len(keep), dtype=bool)
        out[kept_rows[first]] = True
        return out

    def mask(self, params: DecideParams) -> np.ndarray:
        """Boolean mask of kept rows under ``params`` (rules a-e)."""
        offset = np.where(self.unseen, params.delta_unseen, 0.0) if params.delta_unseen else 0.0
        thr_by_src = params.source_thresholds()
        if thr_by_src:
            src_thr = np.array([thr_by_src.get(s, params.tau_pair) for s in self.src_labels], dtype=np.float64)
            pair_thr = src_thr[self.src_code] if len(self.src_labels) else np.full(len(self), params.tau_pair)
        else:
            pair_thr = params.tau_pair
        keep = (self.s1_max >= params.tau_gate + offset) & (self.p >= pair_thr + offset)
        if params.alpha > 0:
            keep &= self.p >= params.alpha * self.s1_max
        if params.one_owner:
            keep = self.one_owner(keep)
        return keep

    def to_frame(self, keep: np.ndarray) -> pd.DataFrame:
        """Kept rows as (s1_id, cand_id)."""
        return pd.DataFrame({"s1_id": self.s1_ids[self.s1_code[keep]], "cand_id": self.cand_ids[self.cand_code[keep]]})


def decide(scores_df: pd.DataFrame, s1_table: pd.DataFrame, cfg: dict, train_countries) -> pd.DataFrame:
    """Apply rules a-e to ``scores_df`` (s1_id, cand_id, p_final); return kept (s1_id, cand_id).

    ``s1_table`` needs s1_id and country; ``cfg`` is the full config or its ``decide`` section.
    Pruning uses min(``decide.min_p``, lowest reachable threshold), so it is always exact.
    """
    params = DecideParams.from_cfg(cfg)
    min_p = float(cfg.get("decide", cfg).get("min_p", DEFAULT_MIN_P))
    arrays = PairArrays.build(scores_df, s1_table, train_countries, min(min_p, params.min_threshold()))
    return arrays.to_frame(arrays.mask(params))


def resolve_train_countries(cfg: dict) -> list[str]:
    """``train_countries`` from the config, or (if empty) every country present in train S1."""
    from src.contracts import load_s1_meta

    listed = cfg.get("train_countries") or []
    return sorted(listed) if listed else sorted(load_s1_meta("train", cfg=cfg)["country"].unique())


def main() -> None:
    """CLI: decide on a split's scores and write ``decisions_<split>.parquet``."""
    from src.config import load_config
    from src.contracts import artifact_path, check_artifact, load_artifact, load_s1_meta

    parser = argparse.ArgumentParser(description="Stage 4 decision layer")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--sample", action="store_true", help="keep sample.frac of S1 ids (seed sample.seed)")
    parser.add_argument("--scores", default=None, help="override scores parquet path (contract still checked)")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    params = DecideParams.from_cfg(cfg)
    min_p = min(float(cfg["decide"].get("min_p", DEFAULT_MIN_P)), params.min_threshold())
    scores = load_artifact(
        "scores", args.split, sample=args.sample, cfg=cfg, path=args.scores,
        columns=["s1_id", "cand_id", "p_final"], filters=[("p_final", ">=", min_p)],
    )
    meta = load_s1_meta(args.split, sample=args.sample, cfg=cfg).rename(columns={"entity_id": "s1_id"})
    train_countries = resolve_train_countries(cfg)
    unseen = sorted(set(meta["country"]) - set(train_countries))
    print(f"[decide] {params}  min_p={min_p}  train_countries={train_countries}  unseen={unseen}")

    arrays = PairArrays.build(scores, meta, train_countries, min_p)
    keep = arrays.mask(params)
    out = pa.table(
        {
            "s1_id": ids_to_arrow(arrays.s1_code, arrays.s1_ids),
            "cand_id": ids_to_arrow(arrays.cand_code, arrays.cand_ids),
            "p_final": arrays.p,
            "is_match": keep,
        }
    )
    check_artifact(out.slice(0, 0).to_pandas(), "decisions", args.split)
    path = artifact_path("decisions", args.split, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, path)
    n_matched_s1 = int(np.unique(arrays.s1_code[keep]).size)
    print(
        f"[decide] {len(out):,} pairs >= {min_p}, {int(keep.sum()):,} kept; "
        f"{n_matched_s1:,}/{arrays.n_s1:,} S1 with >=1 match; dropped (S1 not in table): {arrays.n_dropped_unknown_s1:,} -> {path}"
    )


if __name__ == "__main__":
    main()
