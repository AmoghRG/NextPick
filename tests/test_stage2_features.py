"""Unit tests for the shared 9-feature Stage-2 builder."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.models.stage2_features import (
    STAGE2_FEATURE_NAMES,
    compute_stage2_features_for_candidates,
)


def test_stage2_features_exact_values() -> None:
    # Most recent click first: DeptA (idx 0), DeptB (idx 1), DeptA (idx 2)
    hist_prices = [10.0, 20.0, 30.0]
    hist_depts = ["DeptA", "DeptB", "DeptA"]
    hist_groups = ["Grp1", "Grp2", "Grp1"]

    cands = [
        {
            "stage1_cosine_score": 0.85,
            "price": 25.0,
            "department_name": "DeptA",
            "product_group_name": "Grp1",
            "train_tx_count": 99,
        },
        {
            "stage1_cosine_score": 0.50,
            "price": 15.0,
            "department_name": "DeptZ",
            "product_group_name": "GrpZ",
            "train_tx_count": 0,
        },
    ]

    feats = compute_stage2_features_for_candidates(hist_prices, hist_depts, hist_groups, 2, cands)
    assert feats.shape == (2, len(STAGE2_FEATURE_NAMES))

    # Candidate 0 checks
    assert feats[0, 0] == pytest.approx(0.85)
    assert feats[0, 1] == pytest.approx(25.0)
    assert feats[0, 2] == pytest.approx(20.0)  # avg(10, 20, 30)
    assert feats[0, 3] == pytest.approx(5.0)   # |25 - 20|
    assert feats[0, 4] == pytest.approx(2.0)   # 2 DeptA clicks
    assert feats[0, 5] == pytest.approx(2.0)   # 2 Grp1 clicks
    assert feats[0, 6] == pytest.approx(0.0)   # latest click was DeptA
    assert feats[0, 7] == pytest.approx(math.log1p(99))
    assert feats[0, 8] == pytest.approx(2.0)

    # Candidate 1 checks (unseen dept -> recency sentinel 99)
    assert feats[1, 4] == pytest.approx(0.0)
    assert feats[1, 6] == pytest.approx(99.0)


def test_stage2_features_empty_history() -> None:
    feats = compute_stage2_features_for_candidates(
        [],
        [],
        [],
        1,
        [{"stage1_cosine_score": 0.2, "price": 50.0, "department_name": "X", "product_group_name": "Y", "train_tx_count": 3}],
    )
    assert feats[0, 2] == pytest.approx(0.0)
    assert feats[0, 3] == pytest.approx(0.0)
    assert feats[0, 6] == pytest.approx(99.0)


def test_stage2_features_unequal_lengths_raises() -> None:
    with pytest.raises(ValueError, match="History length mismatch"):
        compute_stage2_features_for_candidates([10.0, 20.0], ["DeptA"], ["Grp1"], 2, [])
