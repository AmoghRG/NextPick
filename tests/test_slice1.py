"""Slice 1 verification tests: temporal split, causal history, metrics, and popularity baseline."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.baselines.classical import (
    fit_item_cooccurrence,
    recommend_item_knn,
    recommend_repurchase,
)
from src.baselines.popularity import evaluate_popularity_baseline
from src.data.temporal_split import SplitConfig, build_temporal_split
from src.eval.metrics import (
    catalog_coverage_at_k,
    evaluate_recommendations,
    mean_popularity_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def _make_synthetic_hm(tmp_path: Path) -> tuple[Path, Path]:
    start = date(2020, 7, 1)
    # 10 weeks of data; split should retain only the last 8 weeks
    rows = []
    for day_offset in range(70):
        d = (start + timedelta(days=day_offset)).isoformat()
        for u in range(1, 6):
            item = f"art_{(day_offset + u) % 8:03d}"
            rows.append(
                {
                    "t_dat": d,
                    "customer_id": f"user_{u}",
                    "article_id": item,
                    "price": 0.02 + 0.005 * ((day_offset + u) % 4),
                    "sales_channel_id": 2,
                }
            )
    tx_path = tmp_path / "transactions.csv"
    pd.DataFrame(rows).to_csv(tx_path, index=False)

    articles = [
        {
            "article_id": f"art_{i:03d}",
            "prod_name": f"Item {i}",
            "product_type_name": "T-shirt" if i % 2 == 0 else "Trousers",
            "product_group_name": "Garment Upper body" if i % 2 == 0 else "Garment Lower body",
            "colour_group_name": "Black",
            "department_name": f"Dept_{i % 3}",
            "detail_desc": f"Synthetic fashion item {i}",
        }
        for i in range(8)
    ]
    art_path = tmp_path / "articles.csv"
    pd.DataFrame(articles).to_csv(art_path, index=False)
    return tx_path, art_path


def test_metrics_exact_values() -> None:
    ranked = ["a", "b", "c", "d"]
    gt = {"b", "d"}
    assert recall_at_k(ranked, gt, k=2) == pytest.approx(0.5)
    assert recall_at_k(ranked, gt, k=4) == pytest.approx(1.0)
    assert mrr_at_k(ranked, gt, k=4) == pytest.approx(0.5)
    # Hits at positions 2 (idx=1) and 4 (idx=3)
    expected_dcg = (1.0 / 1.584962500721156) + (1.0 / 2.321928094887362)
    expected_idcg = 1.0 + (1.0 / 1.584962500721156)
    assert ndcg_at_k(ranked, gt, k=4) == pytest.approx(expected_dcg / expected_idcg, rel=1e-6)


def test_temporal_split_and_zero_leakage(tmp_path: Path) -> None:
    tx_path, art_path = _make_synthetic_hm(tmp_path)
    out_dir = tmp_path / "processed"
    counts = build_temporal_split(tx_path, art_path, out_dir, SplitConfig())

    assert counts["train"] == 42 * 5
    assert counts["val"] == 7 * 5
    assert counts["test"] == 7 * 5
    assert counts["catalog"] == 8

    con = duckdb.connect()
    max_train = con.execute(f"SELECT MAX(t_dat) FROM read_parquet('{out_dir / 'train.parquet'}')").fetchone()[0]
    min_val = con.execute(f"SELECT MIN(t_dat) FROM read_parquet('{out_dir / 'val.parquet'}')").fetchone()[0]
    max_val = con.execute(f"SELECT MAX(t_dat) FROM read_parquet('{out_dir / 'val.parquet'}')").fetchone()[0]
    min_test = con.execute(f"SELECT MIN(t_dat) FROM read_parquet('{out_dir / 'test.parquet'}')").fetchone()[0]
    assert max_train < min_val <= max_val < min_test

    # First event per user in the 8-week window must have empty history
    first_hist = con.execute(
        f"""
        SELECT hist_article_ids
        FROM read_parquet('{out_dir / 'train.parquet'}')
        WHERE customer_id = 'user_1'
        ORDER BY t_dat, row_id
        LIMIT 1
        """
    ).fetchone()[0]
    assert len(first_hist) == 0

    # History length is bounded by 20
    max_hist_len = con.execute(
        f"SELECT MAX(len(hist_article_ids)) FROM read_parquet('{out_dir / 'test.parquet'}')"
    ).fetchone()[0]
    assert max_hist_len == 20
    con.close()


def test_popularity_baseline_runs(tmp_path: Path) -> None:
    tx_path, art_path = _make_synthetic_hm(tmp_path)
    out_dir = tmp_path / "processed"
    build_temporal_split(tx_path, art_path, out_dir)

    metrics = evaluate_popularity_baseline(
        out_dir / "train.parquet",
        out_dir / "test.parquet",
        recall_k=5,
        ndcg_k=5,
        mrr_k=5,
    )
    assert metrics["num_users"] == 5
    assert 0.0 <= metrics["recall@5"] <= 1.0
    assert 0.0 <= metrics["ndcg@5"] <= 1.0
    assert 0.0 <= metrics["mrr@5"] <= 1.0


def test_beyond_accuracy_metrics_and_classical_baselines() -> None:
    gt = {"u1": {"a", "b"}, "u2": {"c"}}
    preds = {"u1": ["a", "d"], "u2": ["c", "d"]}
    item_counts = {"a": 10, "b": 20, "c": 30, "d": 100}

    assert catalog_coverage_at_k(preds, catalog_size=4, k=2) == pytest.approx(0.75)
    assert mean_popularity_at_k(preds, item_counts, k=2) == pytest.approx((10 + 100 + 30 + 100) / 4)

    summary = evaluate_recommendations(
        gt, preds, recall_k=2, ndcg_k=2, mrr_k=2, catalog_size=4, item_counts=item_counts
    )
    assert summary["coverage@2"] == pytest.approx(0.75)
    assert summary["mean_popularity@2"] == pytest.approx(60.0)

    # Repurchase ranks most-recent/frequent history first and backfills unseen popular items
    rep = recommend_repurchase(["art_2", "art_1", "art_2"], ["art_9", "art_2", "art_8"], k=4)
    assert rep == ["art_2", "art_1", "art_9", "art_8"]

    # ItemKNN co-visitation on same-day baskets
    train_df = pd.DataFrame(
        [
            {"customer_id": "c1", "t_dat": "2020-09-01", "article_id": "A"},
            {"customer_id": "c1", "t_dat": "2020-09-01", "article_id": "B"},
            {"customer_id": "c2", "t_dat": "2020-09-02", "article_id": "A"},
            {"customer_id": "c2", "t_dat": "2020-09-02", "article_id": "B"},
            {"customer_id": "c3", "t_dat": "2020-09-03", "article_id": "A"},
            {"customer_id": "c3", "t_dat": "2020-09-03", "article_id": "C"},
        ]
    )
    neighbors = fit_item_cooccurrence(train_df, max_neighbors=5, min_cooccurrence=2)
    assert "A" in neighbors
    assert neighbors["A"][0][0] == "B"
    recs = recommend_item_knn(["A"], neighbors, ["C", "D"], k=3)
    assert recs == ["B", "C", "D"]
