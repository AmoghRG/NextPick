"""Shared zero-skew 14-feature builder for Stage-2 LightGBM training and FastAPI serving."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

STAGE2_FEATURE_NAMES: tuple[str, ...] = (
    "stage1_cosine_score",
    "item_price",
    "user_avg_price",
    "price_abs_diff",
    "dept_affinity_count",
    "group_affinity_count",
    "steps_since_same_dept",
    "item_log_popularity",
    "sales_channel_id",
    "price_zscore",
    "steps_since_same_group",
    "is_repurchase",
    "hist_repurchase_count",
    "item_knn_score",
)


def compute_stage2_features_for_candidates(
    hist_prices: Sequence[float],
    hist_departments: Sequence[str],
    hist_groups: Sequence[str],
    sales_channel_id: int,
    candidates: Sequence[dict],
    hist_article_ids: Sequence[str] | None = None,
) -> np.ndarray:
    """Compute the exact 14 Stage-2 features for a list of candidate dicts.

    Contract: hist_prices, hist_departments, hist_groups (and hist_article_ids if provided)
    MUST have equal lengths and MUST be ordered most-recent-first (index 0 = latest click)
    for accurate recency.
    """
    if not (len(hist_prices) == len(hist_departments) == len(hist_groups)):
        raise ValueError(
            f"History length mismatch: prices={len(hist_prices)}, "
            f"departments={len(hist_departments)}, groups={len(hist_groups)}"
        )
    if hist_article_ids is not None and len(hist_article_ids) != len(hist_prices):
        raise ValueError(
            f"History length mismatch: article_ids={len(hist_article_ids)}, prices={len(hist_prices)}"
        )

    n = len(candidates)
    if n == 0:
        return np.zeros((0, len(STAGE2_FEATURE_NAMES)), dtype=np.float32)

    user_avg_price = float(np.mean(hist_prices)) if hist_prices else 0.0
    user_std_price = float(np.std(hist_prices)) if len(hist_prices) > 1 else 0.0
    dept_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    first_dept_step: dict[str, int] = {}
    first_group_step: dict[str, int] = {}
    article_counts: dict[str, int] = {}

    for step, (dept, grp) in enumerate(zip(hist_departments, hist_groups)):
        dept_counts[dept] = dept_counts.get(dept, 0) + 1
        group_counts[grp] = group_counts.get(grp, 0) + 1
        if dept not in first_dept_step:
            first_dept_step[dept] = step
        if grp not in first_group_step:
            first_group_step[grp] = step

    if hist_article_ids is not None:
        for aid in hist_article_ids:
            key = str(aid)
            article_counts[key] = article_counts.get(key, 0) + 1

    feats = np.zeros((n, len(STAGE2_FEATURE_NAMES)), dtype=np.float32)
    for i, c in enumerate(candidates):
        item_price = float(c.get("price", 0.0))
        dept = str(c.get("department_name", "unknown"))
        grp = str(c.get("product_group_name", "unknown"))
        tx_count = int(c.get("train_tx_count", 0))
        aid = str(c.get("article_id", ""))
        repurchase_cnt = article_counts.get(aid, 0)

        feats[i, 0] = float(c.get("stage1_cosine_score", 0.0))
        feats[i, 1] = item_price
        feats[i, 2] = user_avg_price
        feats[i, 3] = abs(item_price - user_avg_price) if hist_prices else 0.0
        feats[i, 4] = float(dept_counts.get(dept, 0))
        feats[i, 5] = float(group_counts.get(grp, 0))
        feats[i, 6] = float(first_dept_step.get(dept, 99))
        feats[i, 7] = math.log1p(max(tx_count, 0))
        feats[i, 8] = float(sales_channel_id)
        feats[i, 9] = (
            (item_price - user_avg_price) / (user_std_price + 1e-6) if hist_prices else 0.0
        )
        feats[i, 10] = float(first_group_step.get(grp, 99))
        feats[i, 11] = 1.0 if repurchase_cnt > 0 else 0.0
        feats[i, 12] = float(repurchase_cnt)
        feats[i, 13] = float(c.get("item_knn_score", 0.0))

    return feats
