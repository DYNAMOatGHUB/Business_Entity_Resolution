"""Tests for artifact data contracts."""

import pandas as pd
import pytest

from src.contracts import (
    REQUIRED_COLUMNS,
    artifact_path,
    check_artifact,
    load_artifact,
    required_columns,
    sampled_s1_ids,
)
from src.io_utils import sample_s1_ids


def _candidates(n_s1: int = 3, per_s1: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n_s1):
        for j in range(per_s1):
            row = {c: 0 for c in REQUIRED_COLUMNS["candidates"]}
            row.update(s1_id=f"S1-{i}", cand_id=f"S2-{i}{j}", cand_source="S2")
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def cfg(tmp_path):
    """Config pointing at a tiny dataset (50 train S1s) and an empty artifacts dir."""
    data = tmp_path / "dataset" / "train"
    data.mkdir(parents=True)
    s1 = pd.DataFrame(
        {"entity_id": [f"S1-{i}" for i in range(50)], "business_name": "x", "business_address": "y", "country": "C"}
    )
    s1.to_csv(data / "train_source1.tsv", sep="\t", index=False)
    (tmp_path / "artifacts").mkdir()
    return {
        "paths": {"artifacts_dir": str(tmp_path / "artifacts"), "dataset_dir": str(tmp_path / "dataset")},
        "sample": {"frac": 0.2, "seed": 42},
    }


def test_artifact_paths(cfg, tmp_path):
    assert artifact_path("records_norm", "train", cfg) == tmp_path / "artifacts" / "records_norm_train.parquet"
    assert artifact_path("candidates", "test", cfg).name == "candidates_test.parquet"
    assert artifact_path("scores_loco", "train", cfg).name == "scores_loco_train.parquet"
    with pytest.raises(ValueError, match="only exists for split 'train'"):
        artifact_path("scores_loco", "test", cfg)
    with pytest.raises(ValueError, match="unknown artifact"):
        artifact_path("nope", "train", cfg)


def test_valid_frame_passes():
    check_artifact(_candidates(), "candidates", "train")


def test_missing_column_raises():
    df = _candidates().drop(columns=["blend_rank", "hit_b3"])
    with pytest.raises(ValueError, match=r"missing columns \['hit_b3', 'blend_rank'\]"):
        check_artifact(df, "candidates", "train")


def test_extra_column_raises():
    df = _candidates().assign(low_confidence=False)
    with pytest.raises(ValueError, match="unexpected columns \\['low_confidence'\\]"):
        check_artifact(df, "candidates", "train")


def test_feature_prefixes_and_label():
    base = pd.DataFrame({"s1_id": ["S1-1"], "cand_id": ["S2-1"], "f_str_jw": [0.9]})
    check_artifact(base.assign(label=1), "features_str", "train")
    check_artifact(base, "features_str", "test")
    with pytest.raises(ValueError, match="missing columns \\['label'\\]"):
        check_artifact(base, "features_str", "train")
    with pytest.raises(ValueError, match="unexpected columns \\['label'\\]"):
        check_artifact(base.assign(label=1), "features_str", "test")
    with pytest.raises(ValueError, match="f_vec_cos"):
        check_artifact(base.assign(f_vec_cos=0.5), "features_str", "test")
    check_artifact(base.assign(f_vec_cos=0.5, f_ctx_rank=1), "features", "test")


def test_scores_fold_train_only():
    assert "fold" in required_columns("scores", "train")
    assert "fold" not in required_columns("scores", "test")


def test_duplicate_key_raises():
    df = pd.concat([_candidates(), _candidates().head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="1 duplicate \\(s1_id, cand_id\\) rows"):
        check_artifact(df, "candidates", "train")


def test_duplicate_entity_id_in_records_raises():
    cols = REQUIRED_COLUMNS["records_norm"]
    df = pd.DataFrame([dict.fromkeys(cols, "x") | {"entity_id": "S1-1"}] * 2)
    with pytest.raises(ValueError, match="duplicate \\(entity_id\\)"):
        check_artifact(df, "records_norm", "train")


def test_problems_reported_together():
    df = pd.concat([_candidates(), _candidates().head(1)], ignore_index=True).drop(columns=["sim_b5"]).assign(junk=1)
    with pytest.raises(ValueError) as e:
        check_artifact(df, "candidates", "train")
    msg = str(e.value)
    assert "missing columns" in msg and "unexpected columns" in msg and "duplicate" in msg


def test_missing_file_names_path_and_stage(cfg):
    with pytest.raises(FileNotFoundError) as e:
        load_artifact("candidates", "train", cfg=cfg)
    assert "candidates_train.parquet" in str(e.value)
    assert "Stage 1 blocking" in str(e.value)


def test_load_validates(cfg):
    _candidates().assign(junk=1).to_parquet(artifact_path("candidates", "train", cfg))
    with pytest.raises(ValueError, match="junk"):
        load_artifact("candidates", "train", cfg=cfg)


def test_sample_keeps_whole_s1_groups(cfg):
    df = _candidates(n_s1=50, per_s1=3)
    df.to_parquet(artifact_path("candidates", "train", cfg))

    full = load_artifact("candidates", "train", cfg=cfg)
    sampled = load_artifact("candidates", "train", sample=True, cfg=cfg)

    expected = set(sample_s1_ids([f"S1-{i}" for i in range(50)], 0.2, 42))
    assert set(sampled["s1_id"]) == expected == set(sampled_s1_ids("train", cfg))
    assert len(expected) == 10
    assert (sampled.groupby("s1_id").size() == 3).all()
    assert len(full) == 150 and len(sampled) == 30


def test_sample_records_keeps_all_candidates(cfg):
    cols = REQUIRED_COLUMNS["records_norm"]
    ids = [f"S1-{i}" for i in range(50)] + ["S2-1", "S3-1"]
    df = pd.DataFrame([dict.fromkeys(cols, "x") | {"entity_id": e, "source": e[:2]} for e in ids])
    df.to_parquet(artifact_path("records_norm", "train", cfg))
    sampled = load_artifact("records_norm", "train", sample=True, cfg=cfg)
    assert (sampled["source"] == "S1").sum() == 10
    assert {"S2-1", "S3-1"} <= set(sampled["entity_id"])
