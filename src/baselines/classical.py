"""Classical RecSys baselines: User Repurchase and ItemKNN (cosine co-visitation)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import duckdb
import pandas as pd


def _backfill_popular(
    ranked: list[str],
    fallback_popular: Sequence[str],
    k: int,
) -> list[str]:
    """Pad ranked list up to k using unseen popular items."""
    if len(ranked) >= k:
        return ranked[:k]
    seen = set(ranked)
    for item in fallback_popular:
        if item not in seen:
            ranked.append(item)
            seen.add(item)
            if len(ranked) >= k:
                break
    return ranked


def recommend_repurchase(
    hist_article_ids: Sequence[str],
    fallback_popular: Sequence[str],
    k: int = 50,
) -> list[str]:
    """Rank user's own history by recency-weighted frequency, backfilling with popular items.

    Expects `hist_article_ids` ordered most-recent-first (index 0 = latest interaction).
    """
    scores: dict[str, float] = defaultdict(float)
    for step, aid in enumerate(hist_article_ids):
        scores[str(aid)] += 1.0 / (step + 1)
    ranked = [aid for aid, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]
    return _backfill_popular(ranked, fallback_popular, k)


def fit_item_cooccurrence(
    train_df: pd.DataFrame,
    max_neighbors: int = 50,
    min_cooccurrence: int = 2,
) -> dict[str, list[tuple[str, float]]]:
    """Compute cosine-normalized ItemKNN co-visitation neighbors from training baskets.

    Baskets are `(customer_id, t_dat)` pairs in `train_df` (strictly zero-leakage).
    """
    if train_df.empty:
        return {}

    con = duckdb.connect(database=":memory:")
    con.register("train_df", train_df[["customer_id", "t_dat", "article_id"]])
    rows = con.execute(
        f"""
        WITH baskets AS (
            SELECT DISTINCT CAST(customer_id AS VARCHAR) AS customer_id,
                            CAST(t_dat AS VARCHAR) AS t_dat,
                            CAST(article_id AS VARCHAR) AS article_id
            FROM train_df
        ),
        item_cnt AS (
            SELECT article_id, COUNT(*) AS cnt
            FROM baskets
            GROUP BY article_id
        ),
        pairs AS (
            SELECT a.article_id AS item_a,
                   b.article_id AS item_b,
                   COUNT(*) AS co_cnt
            FROM baskets a
            JOIN baskets b
              ON a.customer_id = b.customer_id
             AND a.t_dat = b.t_dat
             AND a.article_id != b.article_id
            GROUP BY 1, 2
            HAVING COUNT(*) >= {int(min_cooccurrence)}
        ),
        scored AS (
            SELECT p.item_a,
                   p.item_b,
                   p.co_cnt / SQRT(ca.cnt * cb.cnt) AS sim,
                   ROW_NUMBER() OVER (
                       PARTITION BY p.item_a
                       ORDER BY p.co_cnt / SQRT(ca.cnt * cb.cnt) DESC, p.item_b ASC
                   ) AS rn
            FROM pairs p
            JOIN item_cnt ca ON p.item_a = ca.article_id
            JOIN item_cnt cb ON p.item_b = cb.article_id
        )
        SELECT item_a, item_b, sim
        FROM scored
        WHERE rn <= {int(max_neighbors)}
        """
    ).fetchall()
    con.close()

    neighbors: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for item_a, item_b, sim in rows:
        neighbors[str(item_a)].append((str(item_b), float(sim)))
    return dict(neighbors)


def recommend_item_knn(
    hist_article_ids: Sequence[str],
    neighbors: Mapping[str, Sequence[tuple[str, float]]],
    fallback_popular: Sequence[str],
    k: int = 50,
) -> list[str]:
    """Score candidate items by recency-weighted sum of ItemKNN cosine similarities."""
    scores: dict[str, float] = defaultdict(float)
    for step, aid in enumerate(hist_article_ids):
        decay = 1.0 / (step + 1)
        for nbr, sim in neighbors.get(str(aid), ()):
            scores[nbr] += decay * sim
    ranked = [aid for aid, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]
    return _backfill_popular(ranked, fallback_popular, k)
