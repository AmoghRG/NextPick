# Deployment, Local Quickstart & API Guide

## System Components

| Component | Technology | Required? | Purpose |
| :--- | :--- | :---: | :--- |
| **API** | FastAPI + Uvicorn | Yes | `/event`, `/recommend`, `/health` serving |
| **Model Inference** | ONNX Runtime | Optional (fallback) | `user_tower.onnx` inference (masked-mean fallback if absent) |
| **Vector Retrieval** | Qdrant (`v1.12.1`) | Yes | HNSW Top-100 ANN search + payload metadata |
| **Session State** | Redis (`7.4-alpine`) | Yes (or in-memory flag) | Bounded 20-item sliding window (`NEXTPICK_ALLOW_INMEMORY_REDIS=1` for dev) |
| **Ranking** | LightGBM | Optional (fallback) | `lgbm_ranker.txt` reranker (bounded heuristic fallback if absent) |
| **Frontend** | Next.js | Optional (UI) | Storefront + Pipeline Inspector on `:3000` |

---

## 1. Local 60-Second Demo (No Kaggle Required)

From the repository root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Generate 500 synthetic SKUs + user_tower.onnx, start Qdrant & Redis, index Qdrant
make demo

# 2. Start FastAPI on http://localhost:8000 (in-memory Redis fallback enabled for dev)
make api

# 3. Start Next.js Storefront + Pipeline Inspector on http://localhost:3000
make web
```

> **Synthetic Demo Notice:** `make demo` generates random 128-d vectors, synthetic SKUs, untrained `user_tower.onnx` weights, and `artifacts/demo_manifest.json`. The API and UI explicitly flag `SYNTHETIC DEMO — RANDOM EMBEDDINGS / UNTRAINED MODEL` so demo output is never confused with the Kaggle-trained model.

### Running Tests & Locust Benchmark

```bash
PYTHONPATH=. pytest -v
make bench   # 15s headless Locust run against http://127.0.0.1:8000 (requires running API)
```

---

## 2. Clean-Clone Docker Build (API)

The [`Dockerfile`](../Dockerfile) uses a **multi-stage build** so it succeeds on a clean `git clone` without pre-existing `artifacts/`:

```bash
docker build -t nextpick-api .
docker run -p 7860:7860 \
  -e QDRANT_URL=http://host.docker.internal:6333 \
  -e REDIS_URL=redis://host.docker.internal:6379/0 \
  nextpick-api
```

- **Ports:** Container exposes `7860` (Uvicorn); local `make api` runs on `8000`.
- **Using trained artifacts:** Mount your Kaggle `artifacts/` directory and set `NEXTPICK_ARTIFACTS_DIR=/app/artifacts` (ensure `demo_manifest.json` is absent).

---

## 3. API Endpoints

- `GET /health` — Returns overall status (`ok` vs `degraded`), `is_synthetic_demo`, and per-dependency status (`redis`, `qdrant`, `onnx`, `lightgbm`).
- `POST /event` — Pushes `{session_id, article_id, price, department_name, product_group_name}` to Redis (`LPUSH + LTRIM 0 19 + TTL`).
- `GET /recommend?session_id=...&top_k=10&product_group=...&max_price=...&sales_channel_id=2` — Returns ranked items with `stage1_rank`, `stage2_rank`, `rank_delta`, `stage1_cosine_score`, `stage2_logit`, and per-stage `latency_ms`.

---

## 4. Troubleshooting

- **Redis connection refused:** Start `docker compose up -d` or run with `NEXTPICK_ALLOW_INMEMORY_REDIS=1` (explicitly disabled by default).
- **Qdrant empty results:** Run `PYTHONPATH=. python src/index/qdrant_loader.py` after `catalog.parquet` and `item_embeddings.parquet` are in `artifacts/`.
- **Offline preview banner in UI:** Indicates the Next.js frontend cannot reach `NEXT_PUBLIC_API_URL` (`http://localhost:8000` by default) and is displaying static seed items.
