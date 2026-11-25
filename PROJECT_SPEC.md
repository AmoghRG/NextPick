# NextPick — 2-Stage Real-Time Product Recommendation System

## Architecture
- **Stage 1 (Retrieval):** PyTorch Two-Tower (side-feature Item Tower + recency-decayed sequence User Tower), trained with in-batch negatives, same-item collision masking, and `logQ` popularity correction → exported to **ONNX Runtime** + indexed in **Qdrant (HNSW)** with full metadata payload (`with_payload=True`).
- **Live Session State:** **Redis** bounded sliding window (`LPUSH` + `LTRIM 0 19` + TTL) for real-time session-aware User Tower inference, with anonymous cold-start fallback.
- **Stage 2 (Ranking):** **LightGBM** (`lambdarank`, `ndcg@10`) re-scoring Top-100 Qdrant candidates to Top-10 using 9 zero-skew point-in-time cross-features.
- **UI:** **Next.js** storefront + **Pipeline Inspector** visualizing Stage-1 → Stage-2 rank movement (`#42 → #2`) and per-stage latency.

## Dataset & Temporal Split
- **Dataset:** H&M Personalized Fashion (8-week slice: ~2.5M–3M transactions, ~45K active SKUs).
- **Split:** Weeks 1–6 Train | Week 7 Validation | Week 8 Test.
- **Rolling Features:** Computed via **DuckDB** (`ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING`) strictly prior to timestamp `t`.
