"""Popularity baseline computed strictly from the training split (zero leakage)."""

from __future__ import annotations

from pathlib import Path

import duckdb

from src.eval.metrics import evaluate_recommendations


def fit_popularity_top_k(train_parquet: Path, k: int = 100) -> list[str]:
    """Return top-k article_ids ranked by training transaction frequency."""
    con = duckdb.connect(database=":memory:")
    rows = con.execute(
        f"""
        SELECT article_id
        FROM read_parquet('{train_parquet}')
        GROUP BY article_id
        ORDER BY COUNT(*) DESC, article_id ASC
        LIMIT {k}
        """
    ).fetchall()
    con.close()
    return [r[0] for r in rows]


def load_ground_truth_by_user(eval_parquet: Path) -> dict[str, set[str]]:
    """Aggregate ground-truth purchased article_ids per customer from an eval split."""
    con = duckdb.connect(database=":memory:")
    rows = con.execute(
        f"""
        SELECT customer_id, LIST(DISTINCT article_id) AS items
        FROM read_parquet('{eval_parquet}')
        GROUP BY customer_id
        """
    ).fetchall()
    con.close()
    return {user: set(items) for user, items in rows}


def evaluate_popularity_baseline(
    train_parquet: Path,
    eval_parquet: Path,
    recall_k: int = 50,
    ndcg_k: int = 10,
    mrr_k: int = 10,
) -> dict[str, float]:
    """Evaluate global training popularity against an evaluation split."""
    top_items = fit_popularity_top_k(train_parquet, k=max(recall_k, ndcg_k, mrr_k))
    gt = load_ground_truth_by_user(eval_parquet)
    preds = {user: top_items for user in gt}
    return evaluate_recommendations(
        gt,
        preds,
        recall_k=recall_k,
        ndcg_k=ndcg_k,
        mrr_k=mrr_k,
    )
