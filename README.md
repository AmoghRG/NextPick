# NextPick — 2-Stage Real-Time Product Recommendation System

Real-time e-commerce retrieval and ranking engine built on **PyTorch Two-Tower**, **ONNX Runtime**, **Qdrant (HNSW)**, **Redis**, **LightGBM (`lambdarank`)**, **FastAPI**, and a **Next.js Pipeline Inspector**.

---

## Architecture

```text
Next.js Storefront + Pipeline Inspector
        │
        ├── POST /event ──► Redis Bounded Window (LPUSH + LTRIM 0 19 + TTL)
        │
        └── GET /recommend
                ├── 0 clicks ──► Cold-Start Trending Cache
                └── >=1 click ─► ONNX User Tower (recency-decay pooling)
                                      │
                                      ▼
                                 Qdrant HNSW (Top-100 + with_payload=True)
                                      │
                                      ▼
                                 LightGBM lambdarank (9 zero-skew features)
                                      │
                                      ▼
                                 Top-10 Ranked + Rank Deltas + Latency ms
```

---

## Key Engineering Decisions

- **Cold-Start-Safe Two-Tower:** Avoids pure ID embeddings; Item Tower combines frozen `all-MiniLM-L6-v2` text embeddings + categorical/price embeddings, while User Tower applies exponential recency-decay pooling ($w_k = \exp(-\lambda k)$) over the last 20 interactions.
- **Anti-Bias Training:** In-batch softmax with temperature ($\tau = 0.07$), **same-item collision masking** off-diagonal, and **`logQ` popularity correction**.
- **Zero Train-Serving Skew:** Offline history computed in **DuckDB** (`ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING`) matches the live **Redis** `LPUSH + LTRIM 0 19` window identically via shared feature logic in `src/models/stage2_features.py`.
- **Single-Hop Stage-2 Metadata:** Item metadata (`price`, `department_name`, `product_group_name`, `train_tx_count`, `image_url`) is stored inside Qdrant payloads (`with_payload=True`) — zero extra DB lookups per `/recommend`.

---

## Ablation & Benchmark Results (H&M 8-Week Temporal Split)

Evaluated on **35,576 active SKUs** (`1,678,323` train / `255,241` val / `240,311` test transactions):

| Model / Pipeline Stage | Recall@50 | NDCG@10 | MRR@10 |
| :--- | :---: | :---: | :---: |
| Popularity Baseline (Weeks 1–6) | 0.0493 | 0.0088 | 0.0131 |
| Two-Tower (No `logQ`) | 0.3429 | 0.2268 | 0.3508 |
| **Two-Tower (In-Batch + `logQ` + Decay)** | **0.3721** | **0.2360** | **0.3545** |
| Two-Tower + LightGBM Stage 2* | 0.2423 | 0.1102 | 0.1663 |

*\*Note: Stage-2 `lambdarank` degraded offline metrics because injecting missing positives into the Top-100 with tail cosine scores caused LightGBM to under-weight `stage1_cosine_score` in favor of popularity priors. Filtering training groups to natural Stage-1 hits (without tail-score injection) resolves this.*

### Serving Load Test (`locust` — 20 Concurrent Users, 15s, Single-Worker Uvicorn + Docker Qdrant/Redis)
- **Throughput:** `105.98 req/s` (`1,344` requests, **0.00% failures**)
- **`POST /event`:** `7 ms` median · `40 ms` p95
- **Aggregated (`/event` + `/recommend`):** `17 ms` median · `140 ms` p95

---

## Quickstart

### 1. Train & Export Artifacts (Kaggle T4 GPU)
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
