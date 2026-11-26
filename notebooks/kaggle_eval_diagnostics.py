"""Eval-only Kaggle diagnostics script using pre-trained NextPick v6 artifacts (no retraining).

Computes:
1. 7-row controlled ablation on Week-8 test users (68,984):
   - popularity_baseline
   - repurchase_baseline
   - item_knn_baseline
   - two_tower_only
   - two_tower_plus_lightgbm (ANN-80 reranked by LightGBM)
   - multichannel_unranked (ANN-80 + Repurchase-20 + ItemKNN-20)
   - multichannel_plus_lightgbm (Full system)
2. Candidate-pool diagnostics (Candidate Recall@pool, Hit Rate, Avg Pool Size, Unique Channel Shares).
3. 1,000-sample Bootstrap 95% CIs (Recall@50, NDCG@10) + Paired Per-User Comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from src.baselines.classical import (
    fit_item_cooccurrence,
    recommend_item_knn,
    recommend_repurchase,
    score_item_knn_candidates,
)
from src.eval.diagnostics import (
    bootstrap_ci,
    evaluate_candidate_pool,
    paired_comparison,
    reciprocal_rank_fusion,
)
from src.eval.metrics import evaluate_recommendations, ndcg_at_k, recall_at_k
from src.models.stage2_features import compute_stage2_features_for_candidates

DATA_DIR = Path("/kaggle/input/h-and-m-personalized-fashion-recommendations")
ART_DIR = Path("./artifacts")
WORK_DIR = Path("./data/processed_eval")
MAX_SEQ_LEN = 20


def _split_train_test(data_dir: Path, work_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    work_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    tx_path = data_dir / "transactions_train.csv"
    art_path = data_dir / "articles.csv"

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
    train_df = con.execute(
        f"SELECT customer_id, t_dat, article_id FROM enriched_tx WHERE t_dat < {train_end}"
    ).fetchdf()
    test_df = con.execute(f"SELECT * FROM enriched_tx WHERE t_dat >= {val_end}").fetchdf()
    con.close()
    return train_df, test_df


def _encode_users_onnx(
    ort_sess: ort.InferenceSession,
    test_first: pd.DataFrame,
    item_embs: np.ndarray,
    art_to_idx: dict[str, int],
    batch_size: int = 512,
) -> np.ndarray:
    n = len(test_first)
    dim = item_embs.shape[1]
    out = np.empty((n, dim), dtype=np.float32)

    hist_lists = test_first["hist_article_ids"].tolist()
    channels = np.clip(test_first["sales_channel_id"].to_numpy(dtype=np.int64), 0, 3)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bs = end - start
        vecs = np.zeros((bs, MAX_SEQ_LEN, dim), dtype=np.float32)
        mask = np.zeros((bs, MAX_SEQ_LEN), dtype=np.float32)
        for i, hist in enumerate(hist_lists[start:end]):
            if hist is None or len(hist) == 0:
                continue
            recent = [art_to_idx[a] for a in reversed(hist) if a in art_to_idx][:MAX_SEQ_LEN]
            if recent:
                vecs[i, : len(recent)] = item_embs[recent]
                mask[i, : len(recent)] = 1.0
        u_emb = ort_sess.run(
            ["user_emb"],
            {"hist_item_vecs": vecs, "hist_mask": mask, "channel_id": channels[start:end]},
        )[0]
        out[start:end] = u_emb
    return out


def _retrieve_ann_topk(
    u_embs: np.ndarray,
    item_embs: np.ndarray,
    idx_to_art: list[str],
    top_k: int = 80,
    batch_size: int = 512,
) -> tuple[list[list[str]], list[list[float]]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    item_t = torch.from_numpy(item_embs).to(device)
    all_ids, all_scores = [], []
    for start in range(0, len(u_embs), batch_size):
        u_t = torch.from_numpy(u_embs[start : start + batch_size]).to(device)
        scores, indices = torch.topk(u_t @ item_t.T, k=top_k, dim=-1)
        for row_idx, row_sc in zip(indices.cpu().numpy(), scores.cpu().numpy()):
            all_ids.append([idx_to_art[j] for j in row_idx])
            all_scores.append(row_sc.tolist())
    return all_ids, all_scores


def run_eval(data_dir: Path = DATA_DIR, art_dir: Path = ART_DIR, out_path: Path | None = None) -> dict:
    catalog_df = pd.read_parquet(art_dir / "catalog.parquet")
    emb_df = pd.read_parquet(art_dir / "item_embeddings.parquet")
    ort_sess = ort.InferenceSession(str(art_dir / "user_tower.onnx"), providers=["CPUExecutionProvider"])
    booster = lgb.Booster(model_file=str(art_dir / "lgbm_ranker.txt"))

    emb_map = {str(r.article_id): np.asarray(r.embedding, dtype=np.float32) for r in emb_df.itertuples(index=False)}
    idx_to_art = catalog_df["article_id"].astype(str).tolist()
    art_to_idx = {a: i for i, a in enumerate(idx_to_art)}
    item_embs = np.vstack([emb_map[a] for a in idx_to_art]).astype(np.float32)
    cat_lookup = catalog_df.set_index("article_id").to_dict(orient="index")

    catalog_size = len(catalog_df)
    item_counts = dict(zip(catalog_df["article_id"].astype(str), catalog_df["train_tx_count"]))

    train_df, test_df = _split_train_test(data_dir, WORK_DIR)
    top50_popular = train_df["article_id"].value_counts().head(50).index.tolist()
    knn_neighbors = fit_item_cooccurrence(train_df, max_neighbors=50, min_cooccurrence=2)

    test_first = test_df.sort_values(["t_dat", "row_id"]).groupby("customer_id").head(1).reset_index(drop=True)
    gt = test_df.groupby("customer_id")["article_id"].apply(set).to_dict()
    users = test_first["customer_id"].tolist()

    u_embs = _encode_users_onnx(ort_sess, test_first, item_embs, art_to_idx)
    ann_ids_all, ann_scores_all = _retrieve_ann_topk(u_embs, item_embs, idx_to_art, top_k=80)

    preds_pop = {u: top50_popular for u in users}
    preds_rep: dict[str, list[str]] = {}
    preds_knn: dict[str, list[str]] = {}
    preds_tt_only: dict[str, list[str]] = {}
    preds_tt_lgb: dict[str, list[str]] = {}
    preds_mc_unranked: dict[str, list[str]] = {}
    preds_mc_lgb: dict[str, list[str]] = {}

    pool_ann: dict[str, list[str]] = {}
    pool_rep: dict[str, list[str]] = {}
    pool_knn: dict[str, list[str]] = {}

    for row, ann_ids, ann_scores, u_emb in zip(test_first.itertuples(), ann_ids_all, ann_scores_all, u_embs):
        uid = row.customer_id
        hist_arts = [str(x) for x in reversed(row.hist_article_ids)]
        hist_prices = list(reversed(row.hist_prices))
        hist_depts = list(reversed(row.hist_departments))
        hist_groups = list(reversed(row.hist_groups))

        preds_rep[uid] = recommend_repurchase(hist_arts, top50_popular, k=50)
        preds_knn[uid] = recommend_item_knn(hist_arts, knn_neighbors, top50_popular, k=50)
        preds_tt_only[uid] = ann_ids[:50]

        knn_scores = score_item_knn_candidates(hist_arts, knn_neighbors)
        rep_top20 = recommend_repurchase(hist_arts, top50_popular, k=20)
        knn_top20 = [aid for aid, _ in sorted(knn_scores.items(), key=lambda x: (-x[1], x[0]))[:20]]

        pool_ann[uid] = ann_ids
        pool_rep[uid] = rep_top20
        pool_knn[uid] = knn_top20

        # 1) Two-Tower + LightGBM (ANN-80 only, item_knn_score=0.0)
        tt_cands = [
            {
                "article_id": aid,
                "stage1_cosine_score": sc,
                "price": cat_lookup[aid]["avg_price"],
                "department_name": cat_lookup[aid]["department_name"],
                "product_group_name": cat_lookup[aid]["product_group_name"],
                "train_tx_count": cat_lookup[aid]["train_tx_count"],
                "item_knn_score": 0.0,
            }
            for aid, sc in zip(ann_ids, ann_scores)
        ]
        tt_feats = compute_stage2_features_for_candidates(
            hist_prices, hist_depts, hist_groups, row.sales_channel_id, tt_cands, hist_article_ids=hist_arts
        )
        tt_scores = booster.predict(tt_feats)
        preds_tt_lgb[uid] = [ann_ids[idx] for idx in np.argsort(-tt_scores)]

        # 2) Multi-Channel Unranked (Reciprocal Rank Fusion) & Multi-Channel + LightGBM
        score_map = dict(zip(ann_ids, ann_scores))
        merged_ids: list[str] = []
        seen: set[str] = set()
        for aid in ann_ids + rep_top20 + knn_top20:
            if aid not in seen and aid in cat_lookup:
                seen.add(aid)
                merged_ids.append(aid)
        preds_mc_unranked[uid] = reciprocal_rank_fusion([ann_ids, rep_top20, knn_top20], k=60, top_n=50)

        mc_cands = []
        for aid in merged_ids:
            cos_sc = score_map.get(aid)
            if cos_sc is None:
                idx = art_to_idx.get(aid)
                cos_sc = float(u_emb @ item_embs[idx]) if idx is not None else 0.0
            meta = cat_lookup[aid]
            mc_cands.append(
                {
                    "article_id": aid,
                    "stage1_cosine_score": cos_sc,
                    "price": meta["avg_price"],
                    "department_name": meta["department_name"],
                    "product_group_name": meta["product_group_name"],
                    "train_tx_count": meta["train_tx_count"],
                    "item_knn_score": float(knn_scores.get(aid, 0.0)),
                }
            )
        mc_feats = compute_stage2_features_for_candidates(
            hist_prices, hist_depts, hist_groups, row.sales_channel_id, mc_cands, hist_article_ids=hist_arts
        )
        mc_scores = booster.predict(mc_feats)
        preds_mc_lgb[uid] = [merged_ids[idx] for idx in np.argsort(-mc_scores)]

    models = {
        "popularity_baseline": preds_pop,
        "repurchase_baseline": preds_rep,
        "item_knn_baseline": preds_knn,
        "two_tower_only": preds_tt_only,
        "two_tower_plus_lightgbm": preds_tt_lgb,
        "multichannel_unranked": preds_mc_unranked,
        "multichannel_plus_lightgbm": preds_mc_lgb,
    }

    ablation_table = {
        name: evaluate_recommendations(
            gt, preds, recall_k=50, ndcg_k=10, mrr_k=10, catalog_size=catalog_size, item_counts=item_counts
        )
        for name, preds in models.items()
    }
    candidate_pool = evaluate_candidate_pool(gt, pool_ann, pool_rep, pool_knn)

    eval_users = [u for u in users if gt.get(u)]
    per_user_ndcg = {
        name: [ndcg_at_k(preds.get(u, ()), gt[u], k=10) for u in eval_users] for name, preds in models.items()
    }
    per_user_recall = {
        name: [recall_at_k(preds.get(u, ()), gt[u], k=50) for u in eval_users] for name, preds in models.items()
    }

    bootstrap_95ci = {
        name: {
            "ndcg@10": bootstrap_ci(per_user_ndcg[name], n_boot=1000, seed=42),
            "recall@50": bootstrap_ci(per_user_recall[name], n_boot=1000, seed=42),
        }
        for name in models
    }
    paired_vs_final_ndcg10 = {
        name: paired_comparison(per_user_ndcg["multichannel_plus_lightgbm"], per_user_ndcg[name])
        for name in models
        if name != "multichannel_plus_lightgbm"
    }

    result = {
        "ablation_table": ablation_table,
        "candidate_pool_diagnostics": candidate_pool,
        "bootstrap_95ci": bootstrap_95ci,
        "paired_vs_final_ndcg10": paired_vs_final_ndcg10,
    }
    target = out_path or (art_dir / "eval_diagnostics.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    print("Saved eval diagnostics to", target)
    return result


if __name__ == "__main__":
    run_eval()
