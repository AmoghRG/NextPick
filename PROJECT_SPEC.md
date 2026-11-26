# NextPick — 2-Stage Real-Time Product Recommendation System

## Architecture
- **Stage 1 (Retrieval):** PyTorch Two-Tower (side-feature Item Tower + masked self-attention & recency-decayed User Tower), trained with in-batch negatives, same-item collision masking, and `logQ` popularity correction → exported to **ONNX Runtime** + indexed in **Qdrant (HNSW)** with full metadata payload (`with_payload=True`), combined with short-horizon **User Repurchase** and **ItemKNN Co-Visitation** candidate channels in offline evaluation.
- **Live Session State:** **Redis** bounded sliding window (`LPUSH` + `LTRIM 0 19` + TTL) for real-time session-aware User Tower inference, with anonymous cold-start fallback.
- **Stage 2 (Ranking):** **LightGBM** (`lambdarank`, `ndcg@10`) re-scoring Stage-1 candidates to Top-10 using 14 zero-skew point-in-time cross-features.
- **UI:** **Next.js** storefront + **Pipeline Inspector** visualizing Stage-1 → Stage-2 rank movement, fallback/synthetic-demo state, and per-stage latency.

## Dataset & Temporal Split
- **Dataset:** H&M Personalized Fashion (8-week slice: `1,678,323` train / `255,241` val / `240,311` test transactions across `35,576` active SKUs).
- **Split:** Weeks 1–6 Train | Week 7 Validation | Week 8 Test (authoritative training pipeline in `notebooks/kaggle_train_pipeline.py`).
- **Rolling Features & Temporal Limitation:** Computed via **DuckDB** (`ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING`) ordered by `(t_dat, row_id)`. Because H&M transactions lack intra-day timestamps, same-day ordering is deterministic rather than true chronological order.
