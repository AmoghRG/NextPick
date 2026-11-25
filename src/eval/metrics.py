"""Offline ranking and retrieval evaluation metrics: Recall@K, NDCG@K, MRR@K."""

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


def evaluate_recommendations(
    ground_truth_by_user: Mapping[str, set[str]],
    predictions_by_user: Mapping[str, Sequence[str]],
    recall_k: int = 50,
    ndcg_k: int = 10,
    mrr_k: int = 10,
) -> dict[str, float]:
    """Compute mean Recall@K, NDCG@K, and MRR@K across all evaluation users."""
    users = [u for u, gt in ground_truth_by_user.items() if gt]
    if not users:
        return {
            f"recall@{recall_k}": 0.0,
            f"ndcg@{ndcg_k}": 0.0,
            f"mrr@{mrr_k}": 0.0,
            "num_users": 0,
        }

    total_recall = 0.0
    total_ndcg = 0.0
    total_mrr = 0.0
    for user in users:
        preds = predictions_by_user.get(user, ())
        gt = ground_truth_by_user[user]
        total_recall += recall_at_k(preds, gt, k=recall_k)
        total_ndcg += ndcg_at_k(preds, gt, k=ndcg_k)
        total_mrr += mrr_at_k(preds, gt, k=mrr_k)

    n = len(users)
    return {
        f"recall@{recall_k}": total_recall / n,
        f"ndcg@{ndcg_k}": total_ndcg / n,
        f"mrr@{mrr_k}": total_mrr / n,
        "num_users": n,
    }
