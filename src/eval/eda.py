"""Exploratory data analysis: dataset stats, ground-truth structure, noise patterns.

Writes ``reports/eda_summary.md`` (numbers + short bullets) and ``reports/eda_examples.md``
(random matched pairs, singleton near-misses, records from test-only countries).

Heavy parts are sampled: pair text stats use a fraction of S1 ids (``sample.frac``), token
tops and nearest-name lookups use a deterministic hash sample of rows. Everything else
(counts, ground-truth structure, lengths) runs on the full data with lazy polars scans.

Run: ``python -m src.eval.eda [--sample-frac 0.2]``
"""

from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz, process, utils

from src.config import load_config, resolve_path
from src.io_utils import SOURCE_COLUMNS, TRUTH_COLUMNS, sample_s1_ids

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)
TOKEN_SPLIT_RE = r"[^\p{L}\p{M}\p{N}]+"
POSTAL_RE = r"\b\d{5,6}\b"
DECISIONS_START = "<!-- DECISIONS:START (hand-curated, preserved on rerun) -->"
DECISIONS_END = "<!-- DECISIONS:END -->"


# ----------------------------------------------------------------------------- loading


def _scan_tsv(path: Path, expected: list[str]) -> pl.LazyFrame:
    """Lazily scan a challenge TSV as all-string columns, empty fields kept as ``""``."""
    lf = pl.scan_csv(path, separator="\t", quote_char=None, infer_schema=False, empty_string_is_null=False)
    cols = lf.collect_schema().names()
    if cols != expected:
        raise ValueError(f"{path}: expected columns {expected}, got {cols}")
    return lf


def scan_source(dataset_dir: Path, split: str, n: int) -> pl.LazyFrame:
    """Lazy scan of ``{split}_source{n}.tsv`` with an added ``source`` column (``S1``/``S2``/``S3``)."""
    path = dataset_dir / split / f"{split}_source{n}.tsv"
    return _scan_tsv(path, SOURCE_COLUMNS).with_columns(source=pl.lit(f"S{n}"))


def scan_all(dataset_dir: Path, split: str, sources=SOURCES) -> pl.LazyFrame:
    """Concatenated lazy scan of the given sources of one split (adds ``split`` column)."""
    return pl.concat([scan_source(dataset_dir, split, n) for n in sources]).with_columns(split=pl.lit(split))


def load_truth_pairs(dataset_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (truth, pairs): truth with ``n_matches`` per S1; pairs exploded to (s1_id, cand_id, cand_source)."""
    gt = _scan_tsv(dataset_dir / "train" / "train_ground_truth.tsv", TRUTH_COLUMNS).collect()
    ids = pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().str.strip_chars()).list.eval(
        pl.element().filter(pl.element() != "")
    )
    gt = gt.with_columns(ids.alias("ids")).with_columns(n_matches=pl.col("ids").list.len())
    pairs = (
        gt.select(s1_id=pl.col("source1_entity_id"), cand_id=pl.col("ids"))
        .explode("cand_id", empty_as_null=True)
        .drop_nulls("cand_id")
        .with_columns(cand_source=pl.col("cand_id").str.slice(0, 2))
    )
    return gt.drop("matched_entity_ids"), pairs


def hash_sample(lf: pl.LazyFrame, frac: float, seed: int) -> pl.LazyFrame:
    """Deterministic row sample by hashing ``entity_id`` (keeps ~``frac`` of rows)."""
    buckets = 10_000
    return lf.filter(pl.col("entity_id").hash(seed=seed) % buckets < int(frac * buckets))


# ----------------------------------------------------------------------------- item 1


def row_counts(dataset_dir: Path) -> pl.DataFrame:
    """Rows per split x source x country (full data)."""
    frames = [
        scan_all(dataset_dir, split).group_by("split", "source", "country").agg(n=pl.len()).collect()
        for split in SPLITS
    ]
    return pl.concat(frames).sort("split", "source", "country")


def id_checks(dataset_dir: Path) -> pl.DataFrame:
    """Duplicate and prefix checks on entity_id per split x source."""
    rows = []
    for split in SPLITS:
        for n in SOURCES:
            df = scan_source(dataset_dir, split, n).select("entity_id").collect()
            rows.append(
                {
                    "split": split,
                    "source": f"S{n}",
                    "n": df.height,
                    "dup_ids": df.height - df["entity_id"].n_unique(),
                    "bad_prefix": df.filter(~pl.col("entity_id").str.starts_with(f"S{n}-")).height,
                }
            )
    return pl.DataFrame(rows)


def test_only_countries(counts: pl.DataFrame) -> list[str]:
    """Countries present in test but absent from train (any source)."""
    train = set(counts.filter(pl.col("split") == "train")["country"])
    test = set(counts.filter(pl.col("split") == "test")["country"])
    return sorted(test - train)


# ----------------------------------------------------------------------------- items 2-5


def match_stats(gt: pl.DataFrame, pairs: pl.DataFrame) -> dict:
    """Singleton rate, matches-per-S1 buckets, S2/S3 share and per-S1 source composition."""
    n = gt.height
    bucket = pl.when(pl.col("n_matches") >= 4).then(pl.lit("4+")).otherwise(pl.col("n_matches").cast(pl.String))
    dist = gt.group_by(bucket.alias("matches")).agg(n=pl.len()).with_columns(pct=pl.col("n") / n * 100).sort("matches")
    per_s1 = pairs.group_by("s1_id").agg(
        n_s2=(pl.col("cand_source") == "S2").sum(), n_s3=(pl.col("cand_source") == "S3").sum()
    )
    n_matched = per_s1.height
    return {
        "n_s1": n,
        "n_singleton": int((gt["n_matches"] == 0).sum()),
        "singleton_rate": float((gt["n_matches"] == 0).mean()),
        "dist": dist,
        "n_pairs": pairs.height,
        "share_s2": float((pairs["cand_source"] == "S2").mean()),
        "share_s3": float((pairs["cand_source"] == "S3").mean()),
        "other_prefix": int((~pairs["cand_source"].is_in(["S2", "S3"])).sum()),
        "dup_pairs": pairs.height - pairs.unique(["s1_id", "cand_id"]).height,
        "mean_matches_nonsingleton": pairs.height / max(n_matched, 1),
        "max_matches": int(gt["n_matches"].max()),
        "s2_only": int(((per_s1["n_s2"] > 0) & (per_s1["n_s3"] == 0)).sum()) / max(n_matched, 1),
        "s3_only": int(((per_s1["n_s3"] > 0) & (per_s1["n_s2"] == 0)).sum()) / max(n_matched, 1),
        "both": int(((per_s1["n_s2"] > 0) & (per_s1["n_s3"] > 0)).sum()) / max(n_matched, 1),
        "multi_s2": int((per_s1["n_s2"] >= 2).sum()) / max(n_matched, 1),
        "multi_s3": int((per_s1["n_s3"] >= 2).sum()) / max(n_matched, 1),
    }


def truth_coverage(dataset_dir: Path, gt: pl.DataFrame, pairs: pl.DataFrame) -> dict:
    """Check GT S1 ids == train S1 ids and every matched id exists in train S2/S3."""
    s1 = scan_source(dataset_dir, "train", 1).select("entity_id").collect()["entity_id"]
    s23 = scan_all(dataset_dir, "train", (2, 3)).select("entity_id").collect()["entity_id"]
    gt_s1 = gt["source1_entity_id"]
    return {
        "s1_missing_from_gt": int((~s1.is_in(gt_s1.implode())).sum()),
        "gt_s1_not_in_s1": int((~gt_s1.is_in(s1.implode())).sum()),
        "gt_cands_not_in_s23": int((~pairs["cand_id"].is_in(s23.implode())).sum()),
    }


def one_owner(pairs: pl.DataFrame, dataset_dir: Path, n_examples: int = 5) -> tuple[int, pl.DataFrame]:
    """Count S2/S3 ids matched under 2+ distinct S1s; return (count, examples with names)."""
    owners = pairs.group_by("cand_id").agg(s1_ids=pl.col("s1_id").unique().sort(), n_owners=pl.col("s1_id").n_unique())
    multi = owners.filter(pl.col("n_owners") >= 2).sort("n_owners", "cand_id", descending=[True, False])
    if multi.height == 0:
        return 0, multi
    ex = multi.head(n_examples).explode("s1_ids").rename({"s1_ids": "s1_id"})
    names = pl.concat([scan_source(dataset_dir, "train", n) for n in SOURCES]).filter(
        pl.col("entity_id").is_in(pl.concat([ex["cand_id"], ex["s1_id"]]).implode())
    ).select("entity_id", "business_name", "business_address").collect()
    ex = (
        ex.join(names.rename({"entity_id": "cand_id", "business_name": "cand_name", "business_address": "cand_addr"}), on="cand_id", how="left")
        .join(names.rename({"entity_id": "s1_id", "business_name": "s1_name", "business_address": "s1_addr"}), on="s1_id", how="left")
    )
    return multi.height, ex.select("cand_id", "cand_name", "cand_addr", "n_owners", "s1_id", "s1_name", "s1_addr")


def unmatched_s23(dataset_dir: Path, pairs: pl.DataFrame) -> pl.DataFrame:
    """Train S2/S3 records matched to no S1, per source x country."""
    matched = pairs.select(entity_id="cand_id").unique().with_columns(matched=pl.lit(True))
    s23 = scan_all(dataset_dir, "train", (2, 3)).select("entity_id", "source", "country").collect()
    return (
        s23.join(matched, on="entity_id", how="left")
        .group_by("source", "country")
        .agg(n=pl.len(), unmatched=pl.col("matched").is_null().sum())
        .with_columns(pct_unmatched=pl.col("unmatched") / pl.col("n") * 100)
        .sort("source", "country")
    )


# ----------------------------------------------------------------------------- item 6


def sampled_pairs_text(dataset_dir: Path, pairs: pl.DataFrame, s1_ids: list[str]) -> pl.DataFrame:
    """Matched pairs for sampled S1 ids joined with both sides' raw name/address/country."""
    sp = pairs.filter(pl.col("s1_id").is_in(s1_ids))
    cols = ["entity_id", "business_name", "business_address", "country"]
    s1_keys = sp.select(entity_id=pl.col("s1_id").unique()).lazy()
    c_keys = sp.select(entity_id=pl.col("cand_id").unique()).lazy()
    s1 = scan_source(dataset_dir, "train", 1).select(cols).join(s1_keys, on="entity_id", how="semi").collect(engine="streaming")
    s23 = scan_all(dataset_dir, "train", (2, 3)).select(cols).join(c_keys, on="entity_id", how="semi").collect(engine="streaming")
    return sp.join(s1.rename({c: f"s1_{c}" for c in cols[1:]} | {"entity_id": "s1_id"}), on="s1_id").join(
        s23.rename({c: f"c_{c}" for c in cols[1:]} | {"entity_id": "cand_id"}), on="cand_id"
    )


def _norm_expr(col: str) -> pl.Expr:
    return pl.col(col).str.to_lowercase().str.strip_chars()


def pair_text_features(pt: pl.DataFrame) -> pl.DataFrame:
    """Add per-pair flags: exact lower name, token_set_ratio, shared 5-6 digit token, empty address, cross-country."""
    tsr = process.cpdist(
        pt["s1_business_name"].to_list(),
        pt["c_business_name"].to_list(),
        scorer=fuzz.token_set_ratio,
        processor=utils.default_process,
        workers=-1,
    )
    s1_post = pl.col("s1_business_address").str.extract_all(POSTAL_RE)
    c_post = pl.col("c_business_address").str.extract_all(POSTAL_RE)
    return pt.with_columns(
        name_exact=_norm_expr("s1_business_name") == _norm_expr("c_business_name"),
        name_tsr=pl.Series(tsr),
        postal_shared=s1_post.list.set_intersection(c_post).list.len() > 0,
        postal_both_have=(s1_post.list.len() > 0) & (c_post.list.len() > 0),
        addr_empty_s1=pl.col("s1_business_address").str.strip_chars() == "",
        addr_empty_c=pl.col("c_business_address").str.strip_chars() == "",
        cross_country=pl.col("s1_country") != pl.col("c_country"),
    ).with_columns(
        name_tsr_lt50=pl.col("name_tsr") < 50,
        addr_empty_any=pl.col("addr_empty_s1") | pl.col("addr_empty_c"),
    )


def pair_text_summary(pf: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Percent of pairs with each flag, grouped by ``by`` (empty list = overall)."""
    aggs = [
        pl.len().alias("pairs"),
        (pl.col("name_exact").mean() * 100).alias("%name_exact_lc"),
        (pl.col("name_tsr_lt50").mean() * 100).alias("%name_tsr<50"),
        pl.col("name_tsr").median().alias("tsr_median"),
        (pl.col("postal_shared").mean() * 100).alias("%postal_shared"),
        (pl.col("postal_shared").sum() / pl.col("postal_both_have").sum() * 100).alias("%postal_shared|both_have"),
        (pl.col("addr_empty_any").mean() * 100).alias("%addr_empty_either"),
        (pl.col("addr_empty_s1").mean() * 100).alias("%addr_empty_s1"),
        (pl.col("addr_empty_c").mean() * 100).alias("%addr_empty_cand"),
        (pl.col("cross_country").mean() * 100).alias("%cross_country"),
    ]
    if not by:
        return pf.select(aggs)
    return pf.group_by(by).agg(aggs).sort(by)


# ----------------------------------------------------------------------------- item 7


def length_stats(dataset_dir: Path) -> pl.DataFrame:
    """Char-length quantiles of name/address per split x source x country (full data)."""
    qs = (0.05, 0.5, 0.95)
    frames = []
    for split in SPLITS:
        lf = scan_all(dataset_dir, split).with_columns(
            nl=pl.col("business_name").str.len_chars(), al=pl.col("business_address").str.len_chars()
        )
        aggs = [pl.len().alias("n")]
        for c, lab in (("nl", "name"), ("al", "addr")):
            aggs += [pl.col(c).quantile(q, "nearest").alias(f"{lab}_p{int(q * 100)}") for q in qs]
            aggs.append(pl.col(c).max().alias(f"{lab}_max"))
        aggs += [
            ((pl.col("nl") == 0).mean() * 100).alias("%name_empty"),
            ((pl.col("al") == 0).mean() * 100).alias("%addr_empty"),
            (pl.col("business_name").str.contains(r"[^\x00-\x7F]").mean() * 100).alias("%name_nonascii"),
        ]
        frames.append(lf.group_by("split", "source", "country").agg(aggs).collect(engine="streaming"))
    return pl.concat(frames).sort("split", "country", "source")


NOISE_PATTERNS: dict[str, tuple[str, str]] = {
    "%addr_null_literal": ("business_address", r"(?i)(^|[^\p{L}])<?null>?([^\p{L}]|$)"),
    "%name_domain_like": ("business_name", r"(?i)^[\p{L}\p{N}-]+\.[a-z]{2,4}$"),
    "%name_leading_symbol": ("business_name", r"^[^\p{L}\p{N}]"),
    "%name_non_latin": ("business_name", r"[^\p{Latin}\p{Common}\p{Inherited}]"),
    "%name_all_upper": ("business_name", r"^[^\p{Ll}]*\p{Lu}[^\p{Ll}]*$"),
    "%addr_all_upper": ("business_address", r"^[^\p{Ll}]*\p{Lu}[^\p{Ll}]*$"),
    "%name_has_brackets": ("business_name", r"[\(\[]"),
    "%name_dotted_abbrev": ("business_name", r"\b\p{L}\.\p{L}\."),
}


def noise_patterns(dataset_dir: Path) -> pl.DataFrame:
    """Percent of records matching each generic noise regex, per split x source x country (full data)."""
    frames = []
    for split in SPLITS:
        aggs = [pl.len().alias("n")] + [
            (pl.col(col).str.contains(rx).mean() * 100).alias(name) for name, (col, rx) in NOISE_PATTERNS.items()
        ]
        frames.append(scan_all(dataset_dir, split).group_by("split", "source", "country").agg(aggs).collect(engine="streaming"))
    return pl.concat(frames).sort("split", "country", "source")


# ----------------------------------------------------------------------------- item 8


def _tokens(col: str) -> pl.Expr:
    return (
        pl.col(col).str.to_lowercase().str.replace_all(TOKEN_SPLIT_RE, " ").str.strip_chars().str.split(" ")
        .list.eval(pl.element().filter(pl.element() != ""))
    )


def top_tokens(dataset_dir: Path, frac: float, seed: int, k: int = 40) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Top-k last name tokens and top-k non-numeric address tokens per country (hash-sampled rows, both splits).

    Counts are records containing the token; ``pct`` is relative to sampled records of that country.
    Aggregated per file with the streaming engine so only counts are materialized.
    """
    name_parts, addr_parts, row_parts = [], [], []
    for split in SPLITS:
        for n in SOURCES:
            lf = hash_sample(scan_source(dataset_dir, split, n), frac, seed)
            row_parts.append(lf.group_by("country").agg(n_rows=pl.len()).collect(engine="streaming"))
            name_parts.append(
                lf.select("country", token=_tokens("business_name").list.last()).drop_nulls("token")
                .group_by("country", "token").agg(n=pl.len()).collect(engine="streaming")
            )
            addr_parts.append(
                lf.select("country", token=_tokens("business_address").list.unique())
                .explode("token", empty_as_null=True).drop_nulls("token")
                .filter(~pl.col("token").str.contains(r"^\d+$"))
                .group_by("country", "token").agg(n=pl.len()).collect(engine="streaming")
            )
    n_country = pl.concat(row_parts).group_by("country").agg(pl.col("n_rows").sum())

    def _top(parts: list[pl.DataFrame]) -> pl.DataFrame:
        return (
            pl.concat(parts).group_by("country", "token").agg(pl.col("n").sum())
            .join(n_country, on="country")
            .with_columns(pct=pl.col("n") / pl.col("n_rows") * 100)
            .sort(["country", "n", "token"], descending=[False, True, False])
            .group_by("country", maintain_order=True).head(k)
        )

    return _top(name_parts), _top(addr_parts)


# ----------------------------------------------------------------------------- items 9-10


def random_matched_pairs(pt: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    """``n`` random matched pairs (raw text of both sides)."""
    return pt.sample(n=min(n, pt.height), seed=seed).select(
        "s1_id", "s1_business_name", "s1_business_address", "cand_id", "c_business_name", "c_business_address", "s1_country"
    )


def singleton_near_misses(dataset_dir: Path, gt: pl.DataFrame, s1_ids: list[str], frac: float, seed: int, n: int) -> pl.DataFrame:
    """For ``n`` random sampled singleton S1s: closest S2/S3 name (token_sort_ratio) within the same country.

    Choices are a hash sample (``frac``) of train S2 + S3, grouped by country dynamically.
    """
    singles = gt.filter((pl.col("n_matches") == 0) & pl.col("source1_entity_id").is_in(s1_ids))
    pick = singles.sample(n=min(n, singles.height), seed=seed)["source1_entity_id"]
    s1 = scan_source(dataset_dir, "train", 1).filter(pl.col("entity_id").is_in(pick.implode())).collect()
    choices = hash_sample(scan_all(dataset_dir, "train", (2, 3)), frac, seed).filter(
        pl.col("country").is_in(s1["country"].unique().implode())
    ).select("entity_id", "business_name", "business_address", "country").collect()
    by_country = {c[0]: g for c, g in choices.group_by("country")}
    rows = []
    for r in s1.iter_rows(named=True):
        g = by_country.get(r["country"])
        best = None
        if g is not None:
            best = process.extractOne(r["business_name"], g["business_name"].to_list(), scorer=fuzz.token_sort_ratio, processor=utils.default_process)
        rows.append(
            {
                "s1_id": r["entity_id"], "s1_name": r["business_name"], "s1_addr": r["business_address"], "country": r["country"],
                "best_id": g["entity_id"][best[2]] if best else None,
                "best_name": best[0] if best else None,
                "best_addr": g["business_address"][best[2]] if best else None,
                "score": round(best[1], 1) if best else None,
            }
        )
    return pl.DataFrame(rows)


def new_country_examples(dataset_dir: Path, countries: list[str], n: int, seed: int) -> pl.DataFrame:
    """``n`` random test records per source for each test-only country."""
    if not countries:
        return pl.DataFrame()
    df = scan_all(dataset_dir, "test").filter(pl.col("country").is_in(countries)).collect()
    return pl.concat(
        [g.sample(n=min(n, g.height), seed=seed) for _, g in df.group_by("country", "source", maintain_order=True)]
    ).sort("country", "source").select("country", "source", "entity_id", "business_name", "business_address")


# ----------------------------------------------------------------------------- rendering


def md_table(df: pl.DataFrame, floats: int = 2) -> str:
    """Render a polars frame as a GitHub markdown table (pipes escaped)."""
    def fmt(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.{floats}f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v).replace("|", "\\|").replace("\n", " ")
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.iter_rows()]
    return "\n".join([head, sep, *body])


def _pivot_tokens(tops: pl.DataFrame, k: int) -> pl.DataFrame:
    """Side-by-side ``token (pct%)`` columns per country, one row per rank."""
    cols = {}
    for (country,), g in tops.group_by("country", maintain_order=True):
        vals = [f"{t} ({p:.1f}%)" for t, p in zip(g["token"], g["pct"])]
        cols[country] = vals + [""] * (k - len(vals))
    return pl.DataFrame({"rank": list(range(1, k + 1)), **cols})


def _read_decisions(path: Path) -> str | None:
    """Return the hand-curated decisions block from an existing summary, if present."""
    if not path.exists():
        return None
    m = re.search(re.escape(DECISIONS_START) + r"(.*?)" + re.escape(DECISIONS_END), path.read_text(encoding="utf-8"), re.S)
    return m.group(1).strip("\n") if m else None


def render_summary(r: dict, decisions: str | None) -> str:
    """Build the markdown summary from the results dict."""
    ms = r["match_stats"]
    ov = r["pair_overall"].row(0, named=True)
    lines = [
        "# EDA summary",
        "",
        f"_Generated by `python -m src.eval.eda` in {r['runtime_s']:.0f}s. Sampled parts use frac={r['frac']} seed={r['seed']}._",
        "",
        "## Headline",
        f"- **Singleton rate (train S1 with empty match list): {ms['singleton_rate'] * 100:.2f}%** "
        f"({ms['n_singleton']:,} / {ms['n_s1']:,})",
        f"- One-owner violations (S2/S3 id under 2+ S1s): **{r['n_multi_owner']:,}**",
        f"- Test-only countries: **{', '.join(r['new_countries']) or 'none'}**",
        "",
        "## 1. Row counts",
        "Per split x source x country (full data):",
        "",
        md_table(r["counts"].pivot(on="source", index=["split", "country"], values="n").sort("split", "country")),
        "",
        "ID checks (duplicates / wrong prefix):",
        "",
        md_table(r["id_checks"]),
        "",
        f"- Train countries: {', '.join(r['train_countries'])}; test countries: {', '.join(r['test_countries'])}.",
        f"- Only in test: **{', '.join(r['new_countries']) or 'none'}**.",
        "",
        "## 2-3. Ground-truth structure (full train)",
        f"- S1 rows: {ms['n_s1']:,}; matched pairs: {ms['n_pairs']:,}; mean matches per non-singleton S1: {ms['mean_matches_nonsingleton']:.2f}; max: {ms['max_matches']}.",
        f"- Coverage: S1 missing from GT {r['coverage']['s1_missing_from_gt']}, GT S1 not in S1 file {r['coverage']['gt_s1_not_in_s1']}, "
        f"GT match ids not in train S2/S3 {r['coverage']['gt_cands_not_in_s23']}, duplicate pairs {ms['dup_pairs']}, non-S2/S3 prefixes {ms['other_prefix']}.",
        "",
        md_table(ms["dist"]),
        "",
        f"- Share of matched ids: **S2 {ms['share_s2'] * 100:.1f}%**, **S3 {ms['share_s3'] * 100:.1f}%**.",
        f"- Non-singleton S1s: S2-only {ms['s2_only'] * 100:.1f}%, S3-only {ms['s3_only'] * 100:.1f}%, both {ms['both'] * 100:.1f}%; "
        f"2+ matches within S2 {ms['multi_s2'] * 100:.1f}%, within S3 {ms['multi_s3'] * 100:.1f}%.",
        "",
        "Per country (train S1):",
        "",
        md_table(r["match_by_country"]),
        "",
        "## 4. One-owner check",
        f"- S2/S3 ids appearing under 2+ distinct S1s: **{r['n_multi_owner']:,}** (of {r['n_distinct_cands']:,} distinct matched ids).",
    ]
    if r["n_multi_owner"]:
        lines += ["", md_table(r["multi_owner_examples"]), ""]
    lines += [
        "",
        "## 5. Unmatched train S2/S3 (matched to no S1)",
        "",
        md_table(r["unmatched"]),
        "",
        f"- Total: {r['unmatched']['unmatched'].sum():,} of {r['unmatched']['n'].sum():,} "
        f"({r['unmatched']['unmatched'].sum() / r['unmatched']['n'].sum() * 100:.1f}%).",
        "",
        f"## 6. Matched-pair text stats ({r['n_sample_s1']:,} sampled S1s, {ov['pairs']:,} pairs)",
        "`tsr` = rapidfuzz token_set_ratio on names (default_process). Postal = any shared `\\b\\d{5,6}\\b` token.",
        "",
        md_table(r["pair_overall"]),
        "",
        "By S1 country x candidate source:",
        "",
        md_table(r["pair_by"]),
        "",
        "## 7. Name / address char lengths (full data)",
        "",
        md_table(r["lengths"], floats=1),
        "",
        "### Noise-pattern rates (full data, % of records)",
        "",
        md_table(r["noise"], floats=1),
        "",
        f"## 8. Top {r['k']} tokens per country (hash sample {r['frac']:.0%} of rows, all sources, both splits)",
        "`pct` = % of sampled records of that country containing the token.",
        "",
        "### Last token of business_name",
        "",
        md_table(_pivot_tokens(r["top_name"], r["k"])),
        "",
        "### Address tokens (pure digits excluded)",
        "",
        md_table(_pivot_tokens(r["top_addr"], r["k"])),
        "",
        "## 9-10. Examples",
        "See `reports/eda_examples.md` (15 random matched pairs, 15 singleton near-misses, 20 test-only-country records per source).",
        "",
        "## Decisions",
        f"- **Singleton rate: {ms['singleton_rate'] * 100:.2f}%**.",
        f"- **One-owner rule:** {r['n_multi_owner']:,} S2/S3 ids have 2+ owners in train GT "
        f"({r['n_multi_owner'] / max(r['n_distinct_cands'], 1) * 100:.3f}% of matched ids).",
        "",
        DECISIONS_START,
        decisions or "_TODO: curate legal suffix list, address abbreviation map, one-owner verdict._",
        DECISIONS_END,
        "",
    ]
    return "\n".join(lines)


def render_examples(r: dict) -> str:
    """Markdown with the example dumps (items 9-10)."""
    lines = [
        "# EDA examples",
        "",
        f"_Generated by `python -m src.eval.eda`, seed={r['seed']}._",
        "",
        "## 15 random matched pairs",
        "",
        md_table(r["ex_pairs"]),
        "",
        "## 15 random singleton S1s and their closest S2/S3 name",
        f"Closest by token_sort_ratio within the same country, over a {r['frac']:.0%} hash sample of train S2+S3.",
        "",
        md_table(r["ex_singletons"], floats=1),
        "",
    ]
    for (country,), g in r["ex_new"].group_by("country", maintain_order=True) if r["ex_new"].height else []:
        lines += [f"## Test-only country {country}: 20 random records per source", ""]
        for (src,), gs in g.group_by("source", maintain_order=True):
            lines += [f"### {src}", "", md_table(gs.select("entity_id", "business_name", "business_address")), ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- driver


def run_eda(frac: float, seed: int, dataset_dir: Path, reports_dir: Path, k: int = 40, n_examples: int = 15, n_new: int = 20, log=print) -> dict:
    """Compute every EDA item, write both reports, and return the results dict."""
    t0 = time.time()
    random.seed(seed)
    r: dict = {"frac": frac, "seed": seed, "k": k}

    def step(msg: str) -> None:
        log(f"[{time.time() - t0:6.1f}s] {msg}")

    step("row counts")
    r["counts"] = row_counts(dataset_dir)
    r["id_checks"] = id_checks(dataset_dir)
    r["train_countries"] = sorted(set(r["counts"].filter(pl.col("split") == "train")["country"]))
    r["test_countries"] = sorted(set(r["counts"].filter(pl.col("split") == "test")["country"]))
    r["new_countries"] = test_only_countries(r["counts"])

    step("ground truth")
    gt, pairs = load_truth_pairs(dataset_dir)
    r["match_stats"] = match_stats(gt, pairs)
    r["coverage"] = truth_coverage(dataset_dir, gt, pairs)
    s1c = scan_source(dataset_dir, "train", 1).select(source1_entity_id="entity_id", country="country").collect()
    gtc = gt.join(s1c, on="source1_entity_id", how="left")
    r["match_by_country"] = gtc.group_by("country").agg(
        n_s1=pl.len(),
        pct_singleton=(pl.col("n_matches") == 0).mean() * 100,
        mean_matches=pl.col("n_matches").mean(),
        pct_4plus=(pl.col("n_matches") >= 4).mean() * 100,
    ).sort("country")

    step("one-owner / unmatched")
    r["n_distinct_cands"] = pairs["cand_id"].n_unique()
    r["n_multi_owner"], r["multi_owner_examples"] = one_owner(pairs, dataset_dir)
    r["unmatched"] = unmatched_s23(dataset_dir, pairs)

    step("pair text stats")
    s1_ids = sample_s1_ids(gt["source1_entity_id"].to_list(), frac, seed)
    r["n_sample_s1"] = len(s1_ids)
    pf = pair_text_features(sampled_pairs_text(dataset_dir, pairs, s1_ids))
    r["pair_overall"] = pair_text_summary(pf, [])
    r["pair_by"] = pair_text_summary(pf.rename({"s1_country": "country"}), ["country", "cand_source"])
    r["ex_pairs"] = random_matched_pairs(pf, n_examples, seed)
    del pf

    step("lengths")
    r["lengths"] = length_stats(dataset_dir)

    step("noise patterns")
    r["noise"] = noise_patterns(dataset_dir)

    step("top tokens")
    r["top_name"], r["top_addr"] = top_tokens(dataset_dir, frac, seed, k)

    step("examples")
    r["ex_singletons"] = singleton_near_misses(dataset_dir, gt, s1_ids, frac, seed, n_examples)
    r["ex_new"] = new_country_examples(dataset_dir, r["new_countries"], n_new, seed)

    r["runtime_s"] = time.time() - t0
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "eda_summary.md"
    decisions = _read_decisions(summary_path)
    summary_path.write_text(render_summary(r, decisions), encoding="utf-8")
    (reports_dir / "eda_examples.md").write_text(render_examples(r), encoding="utf-8")
    step(f"wrote {summary_path} and {reports_dir / 'eda_examples.md'}")
    return r


def main() -> None:
    """CLI entry point."""
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Exploratory data analysis -> reports/eda_summary.md")
    parser.add_argument("--sample-frac", type=float, default=cfg["sample"]["frac"], help="fraction for sampled parts")
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    parser.add_argument("--k", type=int, default=40, help="top-k tokens per country")
    args = parser.parse_args()
    run_eda(
        frac=args.sample_frac,
        seed=args.seed,
        dataset_dir=resolve_path(cfg["paths"]["dataset_dir"]),
        reports_dir=resolve_path(cfg["paths"]["reports_dir"]),
        k=args.k,
    )


if __name__ == "__main__":
    main()
