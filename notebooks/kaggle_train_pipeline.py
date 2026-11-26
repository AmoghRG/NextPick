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
import random
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset

from src.baselines.classical import (
    fit_item_cooccurrence,
    recommend_item_knn,
    recommend_repurchase,
    score_item_knn_candidates,
)
from src.eval.metrics import evaluate_recommendations
from src.models.stage2_features import (
    STAGE2_FEATURE_NAMES,
    compute_stage2_features_for_candidates,
)
from src.models.two_tower import (
    ItemTower,
    UserTower,
    export_user_tower_onnx,
    in_batch_logq_loss,
)

DATA_DIR = Path("/kaggle/input/h-and-m-personalized-fashion-recommendations")
WORK_DIR = Path("./data/processed")
ART_DIR = Path("./artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 20
EMBED_DIM = 128
SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    # Active SKUs known strictly BEFORE Week 8 starts (no future Week-8 catalog leakage)
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
                COALESCE(AVG(CASE WHEN w.t_dat < {train_end} THEN w.price END), 0.02) AS avg_price,
                COUNT(CASE WHEN w.t_dat < {train_end} THEN 1 END) AS train_tx_count
            FROM raw_art a
            INNER JOIN (
                SELECT DISTINCT article_id FROM enriched_tx WHERE t_dat < {val_end}
            ) act
                ON CAST(a.article_id AS VARCHAR) = act.article_id
            LEFT JOIN enriched_tx w
                ON CAST(a.article_id AS VARCHAR) = w.article_id AND w.t_dat < {val_end}
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


def eval_ranked_lists(
    gt_by_user: dict[str, set[str]],
    preds_by_user: dict[str, list[str]],
    catalog_size: int | None = None,
    item_counts: dict[str, int] | None = None,
) -> dict[str, float]:
    return evaluate_recommendations(
        gt_by_user,
        preds_by_user,
        recall_k=50,
        ndcg_k=10,
        mrr_k=10,
        catalog_size=catalog_size,
        item_counts=item_counts,
    )


def step3_popularity_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    catalog_size: int,
    item_counts: dict[str, int],
) -> dict[str, float]:
    top50 = train_df["article_id"].value_counts().head(50).index.tolist()
    gt = test_df.groupby("customer_id")["article_id"].apply(set).to_dict()
    preds = {u: top50 for u in gt}
    metrics = eval_ranked_lists(gt, preds, catalog_size=catalog_size, item_counts=item_counts)
    print("Popularity Baseline (Week 8 Test):", metrics)
    return metrics


def step3b_repurchase_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    catalog_size: int,
    item_counts: dict[str, int],
) -> dict[str, float]:
    top50 = train_df["article_id"].value_counts().head(50).index.tolist()
    test_first = test_df.sort_values(["t_dat", "row_id"]).groupby("customer_id").head(1)
    preds = {
        row.customer_id: recommend_repurchase(list(reversed(row.hist_article_ids)), top50, k=50)
        for row in test_first.itertuples()
    }
    gt = test_df.groupby("customer_id")["article_id"].apply(set).to_dict()
    metrics = eval_ranked_lists(gt, preds, catalog_size=catalog_size, item_counts=item_counts)
    print("User Repurchase Baseline (Week 8 Test):", metrics)
    return metrics


def step3c_item_knn_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    catalog_size: int,
    item_counts: dict[str, int],
) -> tuple[dict[str, float], dict[str, list[tuple[str, float]]]]:
    top50 = train_df["article_id"].value_counts().head(50).index.tolist()
    neighbors = fit_item_cooccurrence(train_df, max_neighbors=50, min_cooccurrence=2)
    test_first = test_df.sort_values(["t_dat", "row_id"]).groupby("customer_id").head(1)
    preds = {
        row.customer_id: recommend_item_knn(list(reversed(row.hist_article_ids)), neighbors, top50, k=50)
        for row in test_first.itertuples()
    }
    gt = test_df.groupby("customer_id")["article_id"].apply(set).to_dict()
    metrics = eval_ranked_lists(gt, preds, catalog_size=catalog_size, item_counts=item_counts)
    print("ItemKNN Co-Visitation Baseline (Week 8 Test):", metrics)
    return metrics, neighbors


def _encode_catalog_features(catalog_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Encode categorical columns and price quantiles into bounded integer IDs."""
    def _factorize(series: pd.Series, max_vocab: int) -> np.ndarray:
        codes, _ = pd.factorize(series, sort=True)
        return np.clip(codes + 1, 0, max_vocab - 1).astype(np.int64)

    prices = catalog_df["avg_price"].to_numpy(dtype=np.float32)
    quantiles = np.quantile(prices, np.linspace(0, 1, 11))
    price_bucket = np.clip(np.digitize(prices, quantiles[1:-1]), 0, 11).astype(np.int64)

    return {
        "group_id": _factorize(catalog_df["product_group_name"], 32),
        "dept_id": _factorize(catalog_df["department_name"], 512),
        "colour_id": _factorize(catalog_df["colour_group_name"], 64),
        "price_bucket": price_bucket,
    }


class InteractionDataset(Dataset):
    """PyTorch dataset yielding user history item indices and target positive item index."""

    def __init__(self, df: pd.DataFrame, art_to_idx: dict[str, int]) -> None:
        filtered = df[df["article_id"].isin(art_to_idx)].reset_index(drop=True)
        self.target_idx = np.array([art_to_idx[a] for a in filtered["article_id"]], dtype=np.int64)
        self.channels = np.clip(filtered["sales_channel_id"].to_numpy(dtype=np.int64), 0, 3)

        n = len(filtered)
        self.hist_idx = np.zeros((n, MAX_SEQ_LEN), dtype=np.int64)
        self.hist_mask = np.zeros((n, MAX_SEQ_LEN), dtype=np.float32)

        for i, hist in enumerate(filtered["hist_article_ids"]):
            if hist is None or len(hist) == 0:
                continue
            # Reverse so index 0 is the most recent interaction for exponential decay
            recent = [art_to_idx[a] for a in reversed(hist) if a in art_to_idx][:MAX_SEQ_LEN]
            if recent:
                self.hist_idx[i, : len(recent)] = recent
                self.hist_mask[i, : len(recent)] = 1.0

    def __len__(self) -> int:
        return len(self.target_idx)

    def __getitem__(self, idx: int):
        return self.hist_idx[idx], self.hist_mask[idx], self.channels[idx], self.target_idx[idx]


@torch.no_grad()
def compute_catalog_embeddings(
    item_tower: ItemTower,
    minilm_vecs: np.ndarray,
    cat_feats: dict[str, np.ndarray],
    batch_size: int = 1024,
) -> np.ndarray:
    item_tower.eval()
    out = []
    n = len(minilm_vecs)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        emb = item_tower(
            torch.from_numpy(minilm_vecs[start:end]).to(DEVICE),
            torch.from_numpy(cat_feats["group_id"][start:end]).to(DEVICE),
            torch.from_numpy(cat_feats["dept_id"][start:end]).to(DEVICE),
            torch.from_numpy(cat_feats["colour_id"][start:end]).to(DEVICE),
            torch.from_numpy(cat_feats["price_bucket"][start:end]).to(DEVICE),
        )
        out.append(emb.cpu().numpy())
    return np.vstack(out).astype(np.float32)


@torch.no_grad()
def retrieve_topk_for_df(
    user_tower: UserTower,
    item_embs: np.ndarray,
    df: pd.DataFrame,
    art_to_idx: dict[str, int],
    idx_to_art: list[str],
    top_k: int = 100,
    batch_size: int = 512,
    return_user_embs: bool = False,
) -> tuple[list[list[str]], list[list[float]]] | tuple[list[list[str]], list[list[float]], np.ndarray]:
    """Batched Stage-1 retrieval returning Top-K article_ids and cosine scores per row."""
    user_tower.eval()
    ds = InteractionDataset(df, art_to_idx)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    item_tensor = torch.from_numpy(item_embs).to(DEVICE)

    all_ids, all_scores, all_u_embs = [], [], []
    for hist_idx, hist_mask, chan, _ in loader:
        hist_vecs = item_tensor[hist_idx.to(DEVICE)]
        u_emb = user_tower(hist_vecs, hist_mask.to(DEVICE), chan.to(DEVICE))
        if return_user_embs:
            all_u_embs.append(u_emb.cpu().numpy())
        scores, indices = torch.topk(u_emb @ item_tensor.T, k=top_k, dim=-1)
        for row_idx, row_sc in zip(indices.cpu().numpy(), scores.cpu().numpy()):
            all_ids.append([idx_to_art[j] for j in row_idx])
            all_scores.append(row_sc.tolist())
    if return_user_embs:
        return all_ids, all_scores, np.vstack(all_u_embs).astype(np.float32)
    return all_ids, all_scores


def step4_train_two_tower(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    minilm_vecs: np.ndarray,
    apply_logq: bool,
    epochs: int = 3,
    batch_size: int = 512,
) -> tuple[UserTower, ItemTower, np.ndarray, dict[str, float]]:
    """Train Two-Tower retrieval model and evaluate Recall@50, NDCG@10, MRR@10 on Week 8."""
    idx_to_art = catalog_df["article_id"].tolist()
    art_to_idx = {a: i for i, a in enumerate(idx_to_art)}
    cat_feats = _encode_catalog_features(catalog_df)

    counts = catalog_df["train_tx_count"].to_numpy(dtype=np.float32) + 1.0
    log_q_table = torch.from_numpy(np.log(counts / counts.sum())).to(DEVICE)
    minilm_t = torch.from_numpy(minilm_vecs).to(DEVICE)
    group_t = torch.from_numpy(cat_feats["group_id"]).to(DEVICE)
    dept_t = torch.from_numpy(cat_feats["dept_id"]).to(DEVICE)
    colour_t = torch.from_numpy(cat_feats["colour_id"]).to(DEVICE)
    price_t = torch.from_numpy(cat_feats["price_bucket"]).to(DEVICE)

    item_tower = ItemTower().to(DEVICE)
    user_tower = UserTower().to(DEVICE)
    opt = torch.optim.AdamW(list(item_tower.parameters()) + list(user_tower.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ds = InteractionDataset(train_df, art_to_idx)
    if len(ds) == 0:
        raise ValueError("InteractionDataset is empty; cannot train Two-Tower.")
    effective_bs = min(batch_size, len(ds))
    gen = torch.Generator().manual_seed(SEED)
    loader = DataLoader(ds, batch_size=effective_bs, shuffle=True, drop_last=len(ds) > effective_bs, generator=gen)

    for epoch in range(1, epochs + 1):
        item_tower.train()
        user_tower.train()
        total_loss = 0.0
        for hist_idx, hist_mask, chan, target_idx in loader:
            hist_idx = hist_idx.to(DEVICE)
            target_idx = target_idx.to(DEVICE)

            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                # Encode positive items and historical items through ItemTower
                flat_hist = hist_idx.view(-1)
                hist_vecs = item_tower(
                    minilm_t[flat_hist], group_t[flat_hist], dept_t[flat_hist], colour_t[flat_hist], price_t[flat_hist]
                ).view(hist_idx.size(0), MAX_SEQ_LEN, EMBED_DIM)

                pos_vecs = item_tower(
                    minilm_t[target_idx], group_t[target_idx], dept_t[target_idx], colour_t[target_idx], price_t[target_idx]
                )
                u_vecs = user_tower(hist_vecs, hist_mask.to(DEVICE), chan.to(DEVICE))
                loss = in_batch_logq_loss(
                    u_vecs, pos_vecs, target_idx, log_q_table[target_idx], tau=0.07, apply_logq=apply_logq
                )

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()
        print(f"[TwoTower logQ={apply_logq}] Epoch {epoch}/{epochs} loss={total_loss / len(loader):.4f}")

    item_embs = compute_catalog_embeddings(item_tower, minilm_vecs, cat_feats)

    # Evaluate at user's FIRST Week 8 interaction (history strictly from Weeks 1-7 -> zero Week 8 leakage)
    test_first = test_df.sort_values(["t_dat", "row_id"]).groupby("customer_id").head(1)
    top_ids, _ = retrieve_topk_for_df(user_tower, item_embs, test_first, art_to_idx, idx_to_art, top_k=50)
    gt = test_df.groupby("customer_id")["article_id"].apply(set).to_dict()
    preds = dict(zip(test_first["customer_id"], top_ids))
    item_counts = dict(zip(catalog_df["article_id"], catalog_df["train_tx_count"]))
    metrics = eval_ranked_lists(gt, preds, catalog_size=len(catalog_df), item_counts=item_counts)
    print(f"Two-Tower (logQ={apply_logq}) Week 8 Test:", metrics)
    return user_tower, item_tower, item_embs, metrics


def _build_multichannel_candidates(
    hist_article_ids: list[str],
    ann_ids: list[str],
    ann_scores: list[float],
    u_emb: np.ndarray,
    item_embs: np.ndarray,
    art_to_idx: dict[str, int],
    cat_lookup: dict,
    knn_neighbors: dict[str, list[tuple[str, float]]],
    top_popular: list[str],
) -> tuple[list[str], list[dict]]:
    """Merge Two-Tower ANN (Top-80) + Repurchase (Top-20) + ItemKNN (Top-20) into one candidate pool."""
    knn_scores = score_item_knn_candidates(hist_article_ids, knn_neighbors)
    rep_ids = recommend_repurchase(hist_article_ids, top_popular, k=20)
    knn_top = [aid for aid, _ in sorted(knn_scores.items(), key=lambda x: (-x[1], x[0]))[:20]]

    score_map = dict(zip(ann_ids, ann_scores))
    merged_ids: list[str] = []
    seen: set[str] = set()
    for aid in ann_ids + rep_ids + knn_top:
        if aid not in seen and aid in cat_lookup:
            seen.add(aid)
            merged_ids.append(aid)

    cands: list[dict] = []
    for aid in merged_ids:
        cos_sc = score_map.get(aid)
        if cos_sc is None:
            idx = art_to_idx.get(aid)
            cos_sc = float(u_emb @ item_embs[idx]) if idx is not None else 0.0
        meta = cat_lookup[aid]
        cands.append(
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
    return merged_ids, cands


def step5_stage2_lightgbm(
    user_tower: UserTower,
    item_embs: np.ndarray,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    knn_neighbors: dict[str, list[tuple[str, float]]],
    top_popular: list[str],
    max_val_queries: int = 30000,
) -> tuple[dict[str, float], dict[str, float]]:
    """Mine multi-channel Stage-1 candidates, train lambdarank on Week 7 Val, evaluate on Week 8."""
    idx_to_art = catalog_df["article_id"].tolist()
    art_to_idx = {a: i for i, a in enumerate(idx_to_art)}
    cat_lookup = catalog_df.set_index("article_id").to_dict(orient="index")

    val_sample = val_df.sample(n=min(max_val_queries, len(val_df)), random_state=42).reset_index(drop=True)
    val_ids, val_scores, val_u_embs = retrieve_topk_for_df(
        user_tower, item_embs, val_sample, art_to_idx, idx_to_art, top_k=80, return_user_embs=True
    )

    X_rows, y_rows, groups = [], [], []
    for row, ann_ids, ann_scores, u_emb in zip(val_sample.itertuples(), val_ids, val_scores, val_u_embs):
        gt_item = row.article_id
        hist_arts = list(reversed(row.hist_article_ids))
        cand_ids, cands = _build_multichannel_candidates(
            hist_arts, ann_ids, ann_scores, u_emb, item_embs, art_to_idx, cat_lookup, knn_neighbors, top_popular
        )
        if gt_item not in cand_ids:
            continue

        feats = compute_stage2_features_for_candidates(
            list(reversed(row.hist_prices)),
            list(reversed(row.hist_departments)),
            list(reversed(row.hist_groups)),
            row.sales_channel_id,
            cands,
            hist_article_ids=hist_arts,
        )
        X_rows.append(feats)
        y_rows.extend([1 if aid == gt_item else 0 for aid in cand_ids])
        groups.append(len(cand_ids))

    print(
        f"Stage-2 Multi-Channel Val Recall@~100: {len(groups) / len(val_sample):.4f} "
        f"({len(groups)}/{len(val_sample)} queries retained)"
    )
    X_train = np.vstack(X_rows)
    y_train = np.array(y_rows, dtype=np.int32)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[10],
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
    )
    ranker.fit(X_train, y_train, group=groups)
    ranker.booster_.save_model(str(ART_DIR / "lgbm_ranker.txt"))
    gain_importance = {
        name: float(val)
        for name, val in zip(
            STAGE2_FEATURE_NAMES,
            ranker.booster_.feature_importance(importance_type="gain"),
        )
    }
    print(
        "Stage-2 LightGBM Gain Importance:",
        dict(sorted(gain_importance.items(), key=lambda x: -x[1])),
    )

    # Strict unmodified future-horizon evaluation at user's FIRST Week 8 interaction
    test_first = test_df.sort_values(["t_dat", "row_id"]).groupby("customer_id").head(1)
    test_ids, test_scores, test_u_embs = retrieve_topk_for_df(
        user_tower, item_embs, test_first, art_to_idx, idx_to_art, top_k=80, return_user_embs=True
    )

    preds_by_user = {}
    for row, ann_ids, ann_scores, u_emb in zip(test_first.itertuples(), test_ids, test_scores, test_u_embs):
        hist_arts = list(reversed(row.hist_article_ids))
        cand_ids, cands = _build_multichannel_candidates(
            hist_arts, ann_ids, ann_scores, u_emb, item_embs, art_to_idx, cat_lookup, knn_neighbors, top_popular
        )
        feats = compute_stage2_features_for_candidates(
            list(reversed(row.hist_prices)),
            list(reversed(row.hist_departments)),
            list(reversed(row.hist_groups)),
            row.sales_channel_id,
            cands,
            hist_article_ids=hist_arts,
        )
        lgb_scores = ranker.predict(feats)
        order = np.argsort(-lgb_scores)
        preds_by_user[row.customer_id] = [cand_ids[idx] for idx in order]

    gt = test_df.groupby("customer_id")["article_id"].apply(set).to_dict()
    item_counts = dict(zip(catalog_df["article_id"], catalog_df["train_tx_count"]))
    metrics = eval_ranked_lists(gt, preds_by_user, catalog_size=len(catalog_df), item_counts=item_counts)
    print("Two-Tower + LightGBM Stage 2 (Week 8 Test):", metrics)
    return metrics, gain_importance


def run_all() -> None:
    set_global_seed(SEED)
    step1_temporal_split()
    catalog_df = pd.read_parquet(ART_DIR / "catalog.parquet")
    train_df = pd.read_parquet(WORK_DIR / "train.parquet")
    val_df = pd.read_parquet(WORK_DIR / "val.parquet")
    test_df = pd.read_parquet(WORK_DIR / "test.parquet")

    catalog_size = len(catalog_df)
    item_counts = dict(zip(catalog_df["article_id"], catalog_df["train_tx_count"]))
    top50_popular = train_df["article_id"].value_counts().head(50).index.tolist()

    pop_metrics = step3_popularity_baseline(train_df, test_df, catalog_size, item_counts)
    repurchase_metrics = step3b_repurchase_baseline(train_df, test_df, catalog_size, item_counts)
    item_knn_metrics, knn_neighbors = step3c_item_knn_baseline(train_df, test_df, catalog_size, item_counts)
    minilm_vecs = step2_precompute_minilm(catalog_df)
    np.save(WORK_DIR / "minilm_item_vecs.npy", minilm_vecs)

    # Ablation 1: Naive In-Batch (no logQ)
    _, _, _, naive_metrics = step4_train_two_tower(
        train_df, test_df, catalog_df, minilm_vecs, apply_logq=False, epochs=3
    )
    # Ablation 2: Full In-Batch + logQ + Collision Masking + Self-Attention + Recency Decay
    user_tower, _, item_embs, logq_metrics = step4_train_two_tower(
        train_df, test_df, catalog_df, minilm_vecs, apply_logq=True, epochs=6
    )

    export_user_tower_onnx(user_tower, ART_DIR / "user_tower.onnx")
    emb_df = pd.DataFrame({"article_id": catalog_df["article_id"], "embedding": item_embs.tolist()})
    emb_df.to_parquet(ART_DIR / "item_embeddings.parquet", index=False)

    stage2_metrics, gain_importance = step5_stage2_lightgbm(
        user_tower, item_embs, val_df, test_df, catalog_df, knn_neighbors, top50_popular
    )

    ablation = {
        "popularity_baseline": pop_metrics,
        "repurchase_baseline": repurchase_metrics,
        "item_knn_baseline": item_knn_metrics,
        "two_tower_no_logq": naive_metrics,
        "two_tower_logq_decay": logq_metrics,
        "two_tower_plus_lightgbm": stage2_metrics,
        "stage2_feature_importance_gain": gain_importance,
    }
    (ART_DIR / "ablation_metrics.json").write_text(json.dumps(ablation, indent=2))
    print("All Kaggle artifacts & ablation metrics saved to ./artifacts/")


if __name__ == "__main__":
    run_all()
