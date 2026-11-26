"""Offline ranking, retrieval, and beyond-accuracy evaluation metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def recall_at_k(ranked: Sequence[str], ground_truth: set[str], k: int = 50) -> float:
    """Fraction of relevant ground-truth items retrieved in the top-k predictions."""
    if not ground_truth:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in ground_truth)
    return hits / len(ground_truth)


def ndcg_at_k(ranked: Sequence[str], ground_truth: set[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at rank k for binary relevance."""
    if not ground_truth:
        return 0.0
    dcg = 0.0
    for idx, item in enumerate(ranked[:k]):
        if item in ground_truth:
            dcg += 1.0 / math.log2(idx + 2)
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg > 0.0 else 0.0


def mrr_at_k(ranked: Sequence[str], ground_truth: set[str], k: int = 10) -> float:
    """Mean Reciprocal Rank at cutoff k."""
    for idx, item in enumerate(ranked[:k]):
        if item in ground_truth:
            return 1.0 / (idx + 1)
    return 0.0


def catalog_coverage_at_k(
    predictions_by_user: Mapping[str, Sequence[str]],
    catalog_size: int,
    k: int = 10,
) -> float:
    """Fraction of active catalog items recommended at least once in users' top-k lists."""
    if catalog_size <= 0 or not predictions_by_user:
        return 0.0
    recommended: set[str] = set()
    for preds in predictions_by_user.values():
        recommended.update(preds[:k])
    return len(recommended) / catalog_size


def mean_popularity_at_k(
    predictions_by_user: Mapping[str, Sequence[str]],
    item_counts: Mapping[str, int | float],
    k: int = 10,
) -> float:
    """Average training interaction count of recommended top-k items (lower = less popularity bias)."""
    total_pop = 0.0
    total_recs = 0
    for preds in predictions_by_user.values():
        for item in preds[:k]:
            total_pop += float(item_counts.get(item, 0))
            total_recs += 1
    return total_pop / total_recs if total_recs > 0 else 0.0


def evaluate_recommendations(
    ground_truth_by_user: Mapping[str, set[str]],
    predictions_by_user: Mapping[str, Sequence[str]],
    recall_k: int = 50,
    ndcg_k: int = 10,
    mrr_k: int = 10,
    catalog_size: int | None = None,
    item_counts: Mapping[str, int | float] | None = None,
) -> dict[str, float]:
    """Compute mean Recall@K, NDCG@K, MRR@K, and optional coverage/popularity-bias metrics."""
    users = [u for u, gt in ground_truth_by_user.items() if gt]
    if not users:
        out: dict[str, float] = {
            f"recall@{recall_k}": 0.0,
            f"ndcg@{ndcg_k}": 0.0,
            f"mrr@{mrr_k}": 0.0,
            "num_users": 0,
        }
        if catalog_size is not None:
            out[f"coverage@{ndcg_k}"] = 0.0
        if item_counts is not None:
            out[f"mean_popularity@{ndcg_k}"] = 0.0
        return out

    total_recall = 0.0
    total_ndcg = 0.0
    total_mrr = 0.0
    eval_preds: dict[str, Sequence[str]] = {}
    for user in users:
        preds = predictions_by_user.get(user, ())
        eval_preds[user] = preds
        gt = ground_truth_by_user[user]
        total_recall += recall_at_k(preds, gt, k=recall_k)
        total_ndcg += ndcg_at_k(preds, gt, k=ndcg_k)
        total_mrr += mrr_at_k(preds, gt, k=mrr_k)

    n = len(users)
    out = {
        f"recall@{recall_k}": total_recall / n,
        f"ndcg@{ndcg_k}": total_ndcg / n,
        f"mrr@{mrr_k}": total_mrr / n,
        "num_users": n,
    }
    if catalog_size is not None:
        out[f"coverage@{ndcg_k}"] = catalog_coverage_at_k(eval_preds, catalog_size, k=ndcg_k)
    if item_counts is not None:
        out[f"mean_popularity@{ndcg_k}"] = mean_popularity_at_k(eval_preds, item_counts, k=ndcg_k)
    return out
