# NextPick — 2-Stage Real-Time Product Recommendation System

Real-time e-commerce retrieval and ranking engine built on **PyTorch Two-Tower**, **ONNX Runtime**, **Qdrant (HNSW)**, **Redis**, **LightGBM (`lambdarank`)**, **FastAPI**, and a **Next.js Pipeline Inspector**.

![NextPick Storefront & Pipeline Inspector](docs/inspector.png)

---

## Architecture

```text
Next.js Storefront + Pipeline Inspector
        │
        ├── POST /event ──► Redis Bounded Window (LPUSH + LTRIM 0 19 + TTL)
        │
        └── GET /recommend
                ├── 0 clicks ──► Cold-Start Trending Cache
                └── >=1 click ─► ONNX User Tower (masked self-attention + recency decay)
                                      │
                                      ▼
                                 Qdrant HNSW (Top-100 + with_payload=True)
                                      │
                                      ▼
                                 LightGBM lambdarank (14 zero-skew features)
                                      │
                                      ▼
                                 Top-10 Ranked + Rank Deltas + Latency ms
```

---

## Key Engineering Decisions

- **Cold-Start-Safe Two-Tower:** Avoids pure ID embeddings; Item Tower combines frozen `all-MiniLM-L6-v2` text embeddings + categorical/price embeddings, while User Tower applies **4-head masked self-attention + exponential recency-decay pooling** ($w_k = \exp(-\lambda k)$) over the last 20 interactions.
- **Anti-Bias Training:** In-batch softmax with temperature ($\tau = 0.07$), **same-item collision masking** off-diagonal, and **`logQ` popularity correction**.
- **Zero Train-Serving Skew:** Offline history computed in **DuckDB** (`ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING`) matches the live **Redis** `LPUSH + LTRIM 0 19` window identically via shared feature logic in `src/models/stage2_features.py`.
- **Single-Hop Stage-2 Metadata:** Item metadata (`price`, `department_name`, `product_group_name`, `train_tx_count`, `image_url`) is stored inside Qdrant payloads (`with_payload=True`) — zero extra DB lookups per `/recommend`.

---

## Ablation & Benchmark Results (H&M 8-Week Temporal Split)

Evaluated on a **date-level temporal split** (`ORDER BY t_dat, row_id`, predicting 7-day Week 8 future-horizon purchases from each user's first Week-8 interaction using strictly Weeks 1–7 history) across **35,576 active SKUs** (`1,678,323` train / `255,241` val / `240,311` test transactions):

> **Temporal & evaluation notes:**
> - **Date-level ordering:** Interactions are ordered by date (`t_dat`) and a deterministic `row_id` tie-breaker because the H&M dataset lacks intra-day timestamps; same-day order is deterministic rather than true chronological order.
> - **Train vs. test query framing:** Stage-2 LightGBM trains on up to `30,000` sampled Week-7 single-positive rows, whereas Week-8 testing evaluates each user's first Week-8 interaction against all their Week-8 purchases (7-day future horizon).
> - **Catalog & JSON provenance:** `notebooks/kaggle_train_pipeline.py` filters `catalog.parquet` to active pre-Week-8 SKUs (whereas `src/data/temporal_split.py` keeps the full article catalog); `avg_price` is now aligned to `t_dat < train_end` in both (the exported v6 Kaggle JSON used pre-Week-8 `avg_price`), and the `"two_tower_plus_lightgbm"` key in `docs/ablation_metrics.json` holds the Multi-Channel Stage-1 + LightGBM Stage-2 metrics.

| Model / Pipeline Stage | Recall@50 | NDCG@10 (95% CI) | MRR@10 | Coverage@10 | Mean Pop@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Popularity Baseline (Weeks 1–6) | 0.0493 | 0.0088 `[0.0084, 0.0092]` | 0.0131 | 0.03% | 2,366.8 |
| User Repurchase Baseline | 0.0793 | 0.0337 `[0.0326, 0.0347]` | 0.0426 | 60.56% | 1,802.1 |
| ItemKNN Co-Visitation Baseline | 0.0611 | 0.0169 `[0.0162, 0.0175]` | 0.0222 | 44.70% | 1,208.4 |
| Two-Tower (No `logQ` — collapses popular items) | 0.0035 | 0.0006 | 0.0008 | 81.90% | 144.8 |
| Two-Tower Only (`In-Batch + logQ + Attention + Decay`) | 0.0768 | 0.0265 `[0.0257, 0.0274]` | 0.0328 | 43.56% | 837.2 |
| Two-Tower + LightGBM (ANN-80 reranked) | 0.0848 | 0.0342 `[0.0332, 0.0352]` | 0.0438 | 44.03% | 944.6 |
| Multi-Channel Unranked (ANN-80 + Rep-20 + KNN-20) | 0.0768 | 0.0265 `[0.0257, 0.0274]` | 0.0328 | 43.56% | 837.2 |
| **Multi-Channel Stage 1 + LightGBM Stage 2 (14 features)** | **0.0976** `[0.0957, 0.0994]` | **0.0349** `[0.0339, 0.0359]` | **0.0435** | **50.85%** | **992.4** |

> **Key takeaways from the full Week-8 evaluation (`68,984` users, [`docs/eval_diagnostics.json`](docs/eval_diagnostics.json)):**
> - **Beats every baseline on `Recall@50` and `NDCG@10`:** Full Multi-Channel + LightGBM hits **`0.0976` Recall@50** (+23.1% over Repurchase `0.0793`, +59.7% over ItemKNN `0.0611`) and **`0.0349` NDCG@10** (`[0.0339, 0.0359]`, non-overlapping 95% CI vs. Popularity & ItemKNN; paired `NDCG@10` win rate `73.8%` vs. Popularity, `65.8%` vs. ItemKNN, `65.3%` vs. Two-Tower only).
> - **Candidate Pool Diagnostics (avg `99.8` unique items/user):** `12.46%` Candidate Recall (`24.51%` user hit rate). Every channel contributes unique ground-truth hits: **Two-Tower ANN (`43.51%` unique hits)**, **Repurchase (`17.76%` unique hits)**, **ItemKNN (`4.51%` unique hits)**.
> - **Anti-bias & coverage:** Global Popularity covers just `0.03%` of the catalog (`Mean Pop@10 = 2,366.8`), whereas the full 2-stage pipeline reaches **`50.85%` catalog coverage** (`Mean Pop@10 = 992.4`).
> - **Top Stage-2 LightGBM Gain Importances:** `is_repurchase` (`189,394`), `item_knn_score` (`144,446`), `steps_since_same_dept` (`105,124`), `stage1_cosine_score` (`85,755`), `user_avg_price` (`37,200`), `item_log_popularity` (`35,391`), `item_price` (`30,604`), `price_abs_diff` (`21,831`), `price_zscore` (`21,172`).

### Serving Load Test (`locust` — 20 Concurrent Users, 60s, Single-Worker Uvicorn + Docker Qdrant/Redis)
- **Throughput:** `86.96 req/s` (`5,165` requests, **0.00% failures**)
- **`POST /event`:** `21 ms` median · `92 ms` p95
- **`GET /recommend` (2-stage personalized):** `~95 ms` median · `~260 ms` p95
- **Aggregated (`/event` + `/recommend`):** `75 ms` median · `240 ms` p95 · `480 ms` p99

### Evaluation & Reproducibility Audit
- **Raw Metrics & Diagnostics Output:** [`docs/ablation_metrics.json`](docs/ablation_metrics.json) · [`docs/eval_diagnostics.json`](docs/eval_diagnostics.json) (exported directly from Kaggle Week 8 evaluation)
- **Temporal Split & Causal History:** [`src/data/temporal_split.py`](src/data/temporal_split.py) (`ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING`)
- **Metric Definitions:** [`src/eval/metrics.py`](src/eval/metrics.py) (per-user `Recall@50`, `NDCG@10`, `MRR@10`)
- **Training & Stage-2 Script:** [`notebooks/kaggle_train_pipeline.py`](notebooks/kaggle_train_pipeline.py)
- **Load Test Script:** [`bench/locustfile.py`](bench/locustfile.py) (`make bench`)

---

## Quickstart

### 0. Instant 60-Second Demo (No Kaggle Required)
```bash
make demo   # Generates 500 demo SKUs + user_tower.onnx, boots Qdrant & Redis, indexes Qdrant
make api    # Starts FastAPI on :8000
make web    # Starts Next.js Storefront + Pipeline Inspector on :3000
```

### 1. Full H&M Training & Export (Kaggle T4 GPU)
Upload `notebooks/kaggle_train_pipeline.py` to a Kaggle notebook with the H&M Personalized Fashion dataset attached, run it, and place the exported files (`user_tower.onnx`, `item_embeddings.parquet`, `lgbm_ranker.txt`, `catalog.parquet`) into `./artifacts/`.

### 2. Start Qdrant, Redis & FastAPI Locally
```bash
docker compose up -d
PYTHONPATH=. python src/index/qdrant_loader.py
PYTHONPATH=. uvicorn src.serving.main:app --host 0.0.0.0 --port 8000
```

### 3. Run the Next.js Storefront + Pipeline Inspector
```bash
cd web
npm install
npm run dev
```

### 4. Run Tests & Load Benchmark
```bash
PYTHONPATH=. pytest -v
locust -f bench/locustfile.py --host http://localhost:8000
```
