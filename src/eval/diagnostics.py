"""Statistical and candidate-pool diagnostics: bootstrap CIs, paired tests, and channel coverage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Compute mean and percentile bootstrap confidence interval."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    n = arr.size
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = arr[idx].mean()
    lo = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return {"mean": float(arr.mean()), "ci_low": lo, "ci_high": hi}


def paired_comparison(
    model_scores: Sequence[float],
    baseline_scores: Sequence[float],
) -> dict[str, float]:
    """Compute per-user paired win/tie/loss counts and mean delta."""
    a = np.asarray(model_scores, dtype=np.float64)
    b = np.asarray(baseline_scores, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        return {"mean_diff": 0.0, "wins": 0, "ties": 0, "losses": 0, "win_rate_nonzero": 0.0}
    diff = a - b
    wins = int(np.sum(diff > 1e-12))
    losses = int(np.sum(diff < -1e-12))
    ties = int(a.size - wins - losses)
    nonzero = wins + losses
    return {
        "mean_diff": float(diff.mean()),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate_nonzero": float(wins / nonzero) if nonzero > 0 else 0.0,
    }


def evaluate_candidate_pool(
    gt_by_user: Mapping[str, set[str]],
    ann_by_user: Mapping[str, Sequence[str]],
    rep_by_user: Mapping[str, Sequence[str]],
    knn_by_user: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    """Measure Stage-1 multi-channel candidate recall, hit rate, pool size, and unique channel hits."""
    users = [u for u, gt in gt_by_user.items() if gt]
    if not users:
        return {
            "candidate_recall": 0.0,
            "candidate_hit_rate": 0.0,
            "avg_pool_size": 0.0,
            "ann_only_hit_share": 0.0,
            "rep_only_hit_share": 0.0,
            "knn_only_hit_share": 0.0,
        }

    recalls = []
    hits_any = 0
    pool_sizes = []
    total_hits = 0
    ann_only = 0
    rep_only = 0
    knn_only = 0

    for u in users:
        gt = gt_by_user[u]
        s_ann = set(ann_by_user.get(u, ()))
        s_rep = set(rep_by_user.get(u, ()))
        s_knn = set(knn_by_user.get(u, ()))
        union = s_ann | s_rep | s_knn

        pool_sizes.append(len(union))
        matched = gt & union
        recalls.append(len(matched) / len(gt))
        if matched:
            hits_any += 1
        total_hits += len(matched)
        for item in matched:
            in_a = item in s_ann
            in_r = item in s_rep
            in_k = item in s_knn
            if in_a and not in_r and not in_k:
                ann_only += 1
            elif in_r and not in_a and not in_k:
                rep_only += 1
            elif in_k and not in_a and not in_r:
                knn_only += 1

    return {
        "candidate_recall": float(np.mean(recalls)),
        "candidate_hit_rate": float(hits_any / len(users)),
        "avg_pool_size": float(np.mean(pool_sizes)),
        "ann_only_hit_share": float(ann_only / total_hits) if total_hits > 0 else 0.0,
        "rep_only_hit_share": float(rep_only / total_hits) if total_hits > 0 else 0.0,
        "knn_only_hit_share": float(knn_only / total_hits) if total_hits > 0 else 0.0,
    }
