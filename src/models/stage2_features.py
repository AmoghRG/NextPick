"""Shared zero-skew 9-feature builder for Stage-2 LightGBM training and FastAPI serving."""

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
)


def compute_stage2_features_for_candidates(
    hist_prices: Sequence[float],
    hist_departments: Sequence[str],
    hist_groups: Sequence[str],
    sales_channel_id: int,
    candidates: Sequence[dict],
) -> np.ndarray:
    """Compute the exact 9 Stage-2 features for a list of candidate dicts.

    Contract: hist_prices, hist_departments, and hist_groups MUST have equal lengths
    and MUST be ordered most-recent-first (index 0 = latest click) for accurate recency.
    """
    if not (len(hist_prices) == len(hist_departments) == len(hist_groups)):
        raise ValueError(
            f"History length mismatch: prices={len(hist_prices)}, "
            f"departments={len(hist_departments)}, groups={len(hist_groups)}"
        )

    n = len(candidates)
    if n == 0:
        return np.zeros((0, len(STAGE2_FEATURE_NAMES)), dtype=np.float32)

    user_avg_price = float(np.mean(hist_prices)) if hist_prices else 0.0
    dept_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    first_dept_step: dict[str, int] = {}

    for step, (dept, grp) in enumerate(zip(hist_departments, hist_groups)):
        dept_counts[dept] = dept_counts.get(dept, 0) + 1
        group_counts[grp] = group_counts.get(grp, 0) + 1
        if dept not in first_dept_step:
            first_dept_step[dept] = step

    feats = np.zeros((n, len(STAGE2_FEATURE_NAMES)), dtype=np.float32)
    for i, c in enumerate(candidates):
        item_price = float(c.get("price", 0.0))
        dept = str(c.get("department_name", "unknown"))
        grp = str(c.get("product_group_name", "unknown"))
        tx_count = int(c.get("train_tx_count", 0))

        feats[i, 0] = float(c.get("stage1_cosine_score", 0.0))
        feats[i, 1] = item_price
        feats[i, 2] = user_avg_price
        feats[i, 3] = abs(item_price - user_avg_price) if hist_prices else 0.0
        feats[i, 4] = float(dept_counts.get(dept, 0))
        feats[i, 5] = float(group_counts.get(grp, 0))
        feats[i, 6] = float(first_dept_step.get(dept, 99))
        feats[i, 7] = math.log1p(max(tx_count, 0))
        feats[i, 8] = float(sales_channel_id)

    return feats
