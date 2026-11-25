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

| Model / Pipeline Stage | Recall@50 | NDCG@10 | MRR@10 | p95 Latency |
| :--- | :---: | :---: | :---: | :---: |
| Popularity Baseline (Weeks 1–6) | *Pending Kaggle run* | *Pending Kaggle run* | *Pending Kaggle run* | — |
| Two-Tower (No `logQ`) | *Pending Kaggle run* | *Pending Kaggle run* | *Pending Kaggle run* | — |
| Two-Tower (In-Batch + `logQ` + Decay) | *Pending Kaggle run* | *Pending Kaggle run* | *Pending Kaggle run* | — |
| Two-Tower + LightGBM Stage 2 | *Pending Kaggle run* | *Pending Kaggle run* | *Pending Kaggle run* | *Pending Locust run* |

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
