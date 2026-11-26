"""Unit tests for the shared 13-feature Stage-2 builder."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.models.stage2_features import (
    STAGE2_FEATURE_NAMES,
    compute_stage2_features_for_candidates,
)


def test_stage2_features_exact_values() -> None:
    # Most recent click first: art_1/DeptA/Grp1 (idx 0), art_2/DeptB/Grp2 (idx 1), art_1/DeptA/Grp1 (idx 2)
    hist_article_ids = ["art_1", "art_2", "art_1"]
    hist_prices = [10.0, 20.0, 30.0]
    hist_depts = ["DeptA", "DeptB", "DeptA"]
    hist_groups = ["Grp1", "Grp2", "Grp1"]

    cands = [
        {
            "article_id": "art_1",
            "stage1_cosine_score": 0.85,
            "price": 25.0,
            "department_name": "DeptA",
            "product_group_name": "Grp1",
            "train_tx_count": 99,
        },
        {
            "article_id": "art_999",
            "stage1_cosine_score": 0.50,
            "price": 15.0,
            "department_name": "DeptZ",
            "product_group_name": "GrpZ",
            "train_tx_count": 0,
        },
    ]

    feats = compute_stage2_features_for_candidates(
        hist_prices,
        hist_depts,
        hist_groups,
        2,
        cands,
        hist_article_ids=hist_article_ids,
    )
    assert feats.shape == (2, len(STAGE2_FEATURE_NAMES))
    assert len(STAGE2_FEATURE_NAMES) == 13

    user_std = float(np.std(hist_prices))
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
    assert feats[0, 9] == pytest.approx((25.0 - 20.0) / (user_std + 1e-6), rel=1e-5)
    assert feats[0, 10] == pytest.approx(0.0)  # latest click was Grp1
    assert feats[0, 11] == pytest.approx(1.0)  # art_1 in history
    assert feats[0, 12] == pytest.approx(2.0)  # art_1 clicked twice

    # Candidate 1 checks (unseen dept/group/article -> recency sentinel 99, repurchase 0)
    assert feats[1, 4] == pytest.approx(0.0)
    assert feats[1, 6] == pytest.approx(99.0)
    assert feats[1, 10] == pytest.approx(99.0)
    assert feats[1, 11] == pytest.approx(0.0)
    assert feats[1, 12] == pytest.approx(0.0)


def test_stage2_features_empty_history() -> None:
    feats = compute_stage2_features_for_candidates(
        [],
        [],
        [],
        1,
        [{"article_id": "art_x", "stage1_cosine_score": 0.2, "price": 50.0, "department_name": "X", "product_group_name": "Y", "train_tx_count": 3}],
        hist_article_ids=[],
    )
    assert feats[0, 2] == pytest.approx(0.0)
    assert feats[0, 3] == pytest.approx(0.0)
    assert feats[0, 6] == pytest.approx(99.0)
    assert feats[0, 9] == pytest.approx(0.0)
    assert feats[0, 10] == pytest.approx(99.0)
    assert feats[0, 11] == pytest.approx(0.0)
    assert feats[0, 12] == pytest.approx(0.0)


def test_stage2_features_unequal_lengths_raises() -> None:
    with pytest.raises(ValueError, match="History length mismatch"):
        compute_stage2_features_for_candidates([10.0, 20.0], ["DeptA"], ["Grp1"], 2, [])
    with pytest.raises(ValueError, match="History length mismatch"):
        compute_stage2_features_for_candidates(
            [10.0], ["DeptA"], ["Grp1"], 2, [], hist_article_ids=["art_1", "art_2"]
        )
