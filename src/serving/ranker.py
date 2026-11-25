"""Stage-2 LightGBM lambdarank scorer using the shared 9 zero-skew features."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.models.stage2_features import compute_stage2_features_for_candidates
from src.serving.session_store import SessionEvent


class Stage2Ranker:
    """Re-ranks Stage-1 Top-100 candidates down to Top-K with rank-delta tracking."""

    def __init__(self, model_path: Path | None = None) -> None:
        self._booster = None
        if model_path and model_path.exists():
            import lightgbm as lgb

            self._booster = lgb.Booster(model_file=str(model_path))

    def rank(
        self,
        history: Sequence[SessionEvent],
        candidates: Sequence[dict],
        sales_channel_id: int = 2,
        top_k: int = 10,
    ) -> list[dict]:
        if not candidates:
            return []

        hist_prices = [e.price for e in history]
        hist_depts = [e.department_name for e in history]
        hist_groups = [e.product_group_name for e in history]

        feats = compute_stage2_features_for_candidates(
            hist_prices=hist_prices,
            hist_departments=hist_depts,
            hist_groups=hist_groups,
            sales_channel_id=sales_channel_id,
            candidates=candidates,
        )

        if self._booster is not None:
            scores = self._booster.predict(feats)
        else:
            # Deterministic heuristic fallback when model artifact isn't present in tests:
            # stage1_cosine + dept_affinity * 0.15 - price_abs_diff * 0.5
            scores = feats[:, 0] + 0.15 * feats[:, 4] - 0.5 * feats[:, 3]

        order = np.argsort(-scores)[:top_k]
        ranked: list[dict] = []
        for stage2_rank, idx in enumerate(order, start=1):
            item = dict(candidates[int(idx)])
            s1_rank = int(item.get("stage1_rank", int(idx) + 1))
            item["stage2_rank"] = stage2_rank
            item["rank_delta"] = s1_rank - stage2_rank
            item["stage2_logit"] = float(scores[int(idx)])
            ranked.append(item)
        return ranked
