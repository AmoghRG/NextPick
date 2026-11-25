"""NextPick Kaggle Training Pipeline (Weeks 1-3: Data Split -> Two-Tower -> ONNX -> LightGBM).

Run on Kaggle T4 GPU with H&M Personalized Fashion dataset attached at:
  /kaggle/input/h-and-m-personalized-fashion-recommendations/

Produces lightweight serving artifacts in ./artifacts/:
  - user_tower.onnx
  - item_embeddings.parquet
  - lgbm_ranker.txt
  - catalog.parquet
  - ablation_metrics.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path("/kaggle/input/h-and-m-personalized-fashion-recommendations")
WORK_DIR = Path("./data/processed")
ART_DIR = Path("./artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 20
EMBED_DIM = 128


def step1_temporal_split() -> dict[str, int]:
    """Slice H&M to last 8 weeks (6w train / 1w val / 1w test) + DuckDB causal histories."""
    con = duckdb.connect(database=":memory:")
    tx_path = DATA_DIR / "transactions_train.csv"
    art_path = DATA_DIR / "articles.csv"

    con.execute(f"CREATE TABLE raw_tx AS SELECT * FROM read_csv_auto('{tx_path}', header=True)")
    con.execute(f"CREATE TABLE raw_art AS SELECT * FROM read_csv_auto('{art_path}', header=True)")
    max_date = con.execute("SELECT MAX(CAST(t_dat AS DATE)) FROM raw_tx").fetchone()[0]

    con.execute(
        f"""
        CREATE TABLE windowed_tx AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY CAST(t.t_dat AS DATE), t.customer_id, t.article_id) AS row_id,
            CAST(t.t_dat AS DATE) AS t_dat,
            CAST(t.customer_id AS VARCHAR) AS customer_id,
            CAST(t.article_id AS VARCHAR) AS article_id,
            CAST(t.price AS DOUBLE) AS price,
            COALESCE(CAST(t.sales_channel_id AS INTEGER), 1) AS sales_channel_id,
            COALESCE(CAST(a.product_type_name AS VARCHAR), 'unknown') AS product_type_name,
            COALESCE(CAST(a.product_group_name AS VARCHAR), 'unknown') AS product_group_name,
            COALESCE(CAST(a.department_name AS VARCHAR), 'unknown') AS department_name
        FROM raw_tx t
        LEFT JOIN raw_art a ON CAST(t.article_id AS VARCHAR) = CAST(a.article_id AS VARCHAR)
        WHERE CAST(t.t_dat AS DATE) >= DATE '{max_date}' - INTERVAL 55 DAY
        """
    )

    con.execute(
        """
        CREATE TABLE enriched_tx AS
        SELECT
            *,
            COALESCE(LIST(article_id) OVER w, []) AS hist_article_ids,
            COALESCE(LIST(price) OVER w, []) AS hist_prices,
            COALESCE(LIST(product_group_name) OVER w, []) AS hist_groups,
            COALESCE(LIST(department_name) OVER w, []) AS hist_departments
        FROM windowed_tx
        WINDOW w AS (
            PARTITION BY customer_id
            ORDER BY t_dat, row_id
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
        )
        """
    )

    train_end = f"DATE '{max_date}' - INTERVAL 13 DAY"
    val_end = f"DATE '{max_date}' - INTERVAL 6 DAY"

    counts = {}
    for name, pred in [
        ("train", f"t_dat < {train_end}"),
        ("val", f"t_dat >= {train_end} AND t_dat < {val_end}"),
        ("test", f"t_dat >= {val_end}"),
    ]:
        out = WORK_DIR / f"{name}.parquet"
        con.execute(f"COPY (SELECT * FROM enriched_tx WHERE {pred}) TO '{out}' (FORMAT PARQUET)")
        counts[name] = con.execute(f"SELECT COUNT(*) FROM enriched_tx WHERE {pred}").fetchone()[0]

    # Active SKUs in the 8-week window
    cat_out = ART_DIR / "catalog.parquet"
    con.execute(
        f"""
        COPY (
            SELECT
                CAST(a.article_id AS VARCHAR) AS article_id,
                COALESCE(CAST(a.prod_name AS VARCHAR), '') AS prod_name,
                COALESCE(CAST(a.product_type_name AS VARCHAR), 'unknown') AS product_type_name,
                COALESCE(CAST(a.product_group_name AS VARCHAR), 'unknown') AS product_group_name,
                COALESCE(CAST(a.colour_group_name AS VARCHAR), 'unknown') AS colour_group_name,
                COALESCE(CAST(a.department_name AS VARCHAR), 'unknown') AS department_name,
                COALESCE(CAST(a.detail_desc AS VARCHAR), '') AS detail_desc,
                COALESCE(AVG(w.price), 0.02) AS avg_price,
                COUNT(w.article_id) AS train_tx_count
            FROM raw_art a
            INNER JOIN (SELECT DISTINCT article_id FROM windowed_tx) act
                ON CAST(a.article_id AS VARCHAR) = act.article_id
            LEFT JOIN enriched_tx w
                ON CAST(a.article_id AS VARCHAR) = w.article_id AND w.t_dat < {train_end}
            GROUP BY 1, 2, 3, 4, 5, 6, 7
        ) TO '{cat_out}' (FORMAT PARQUET)
        """
    )
    counts["catalog"] = con.execute(f"SELECT COUNT(*) FROM '{cat_out}'").fetchone()[0]
    con.close()
    print("Split counts:", counts)
    return counts


def step2_precompute_minilm(catalog_df: pd.DataFrame) -> np.ndarray:
    """Pre-compute frozen 384-d MiniLM embeddings once for all active catalog items."""
    texts = (
        catalog_df["prod_name"]
        + " | "
        + catalog_df["product_type_name"]
        + " | "
        + catalog_df["department_name"]
        + " | "
        + catalog_df["detail_desc"]
    ).tolist()
    model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    embs = model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    return embs.astype(np.float32)


def eval_ranked_lists(gt_by_user: dict[str, set[str]], preds_by_user: dict[str, list[str]]) -> dict[str, float]:
    users = [u for u, gt in gt_by_user.items() if gt]
    recalls, ndcgs, mrrs = [], [], []
    for u in users:
        gt = gt_by_user[u]
        preds = preds_by_user.get(u, [])
        recalls.append(sum(1 for x in preds[:50] if x in gt) / len(gt))
        dcg = sum(1.0 / math.log2(i + 2) for i, x in enumerate(preds[:10]) if x in gt)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt), 10)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        mrr = next((1.0 / (i + 1) for i, x in enumerate(preds[:10]) if x in gt), 0.0)
        mrrs.append(mrr)
    return {
        "recall@50": float(np.mean(recalls)),
        "ndcg@10": float(np.mean(ndcgs)),
        "mrr@10": float(np.mean(mrrs)),
    }


def step3_popularity_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    top50 = train_df["article_id"].value_counts().head(50).index.tolist()
    gt = test_df.groupby("customer_id")["article_id"].apply(set).to_dict()
    preds = {u: top50 for u in gt}
    metrics = eval_ranked_lists(gt, preds)
    print("Popularity Baseline (Week 8 Test):", metrics)
    return metrics


def run_all() -> None:
    step1_temporal_split()
    catalog_df = pd.read_parquet(ART_DIR / "catalog.parquet")
    train_df = pd.read_parquet(WORK_DIR / "train.parquet")
    test_df = pd.read_parquet(WORK_DIR / "test.parquet")

    pop_metrics = step3_popularity_baseline(train_df, test_df)
    minilm_vecs = step2_precompute_minilm(catalog_df)
    np.save(WORK_DIR / "minilm_item_vecs.npy", minilm_vecs)

    summary = {"popularity_baseline": pop_metrics, "catalog_items": len(catalog_df)}
    (ART_DIR / "ablation_metrics.json").write_text(json.dumps(summary, indent=2))
    print("Stage-1 artifacts & baseline complete. Ready for Two-Tower & LightGBM loops.")


if __name__ == "__main__":
    run_all()
