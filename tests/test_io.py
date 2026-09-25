"""Tests for I/O utilities."""

import pytest

from src.config import resolve_path
from src.io_utils import (
    format_id_list,
    parse_id_list,
    read_source,
    sample_s1_ids,
    write_id_tsv,
)
import pandas as pd


def test_parse_format_round_trip():
    ids = ["S2-101", "S3-205", "S2-300"]
    s = format_id_list(ids)
    assert s == "S2-101,S3-205,S2-300"
    assert parse_id_list(s) == ids


def test_parse_strips_whitespace():
    assert parse_id_list(" S2-1 , S3-2,") == ["S2-1", "S3-2"]


def test_parse_empty():
    assert parse_id_list("") == []
    assert parse_id_list(None) == []
    assert format_id_list([]) == ""


def test_format_dedupes_keeping_order():
    assert format_id_list(["S3-2", "S2-1", "S3-2", "S2-1"]) == "S3-2,S2-1"


def test_sample_s1_ids_deterministic_and_order_independent():
    ids = [f"S1-{i}" for i in range(100)]
    a = sample_s1_ids(ids, 0.2, 42)
    b = sample_s1_ids(list(reversed(ids)), 0.2, 42)
    assert a == b and len(a) == 20


def test_write_id_tsv_no_quoting(tmp_path):
    df = pd.DataFrame({"a": ["S1-1", "S1-2"], "b": ["S2-1,S3-2", ""]})
    path = tmp_path / "out.tsv"
    write_id_tsv(df, path, ["source1_entity_id", "matched_entity_ids"])
    assert path.read_text() == "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-2\nS1-2\t\n"


@pytest.mark.skipif(
    not (resolve_path("dataset") / "train" / "train_source1.tsv").exists(), reason="train data not present"
)
def test_read_source_real_train():
    df = read_source("train", 1)
    assert list(df.columns) == ["entity_id", "business_name", "business_address", "country", "source"]
    assert (df["source"] == "S1").all()
    assert df["entity_id"].str.startswith("S1-").all()
    assert len(df) > 0
