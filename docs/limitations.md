# Known Limitations & Scope Boundaries

NextPick is a portfolio engineering project built to demonstrate a 2-stage real-time recommendation architecture. The following limitations are explicitly documented rather than hidden:

## 1. Experimental & Statistical Scope
- **Single temporal test window:** Results are reported on one chronological holdout (Week 8, `68,984` users) rather than multiple rolling-week retraining windows.
- **No multi-seed study:** PyTorch Two-Tower training used a single deterministic seed (`SEED = 42`) due to Kaggle runtime (~2 hours per full training run).
- **No multiple-comparison correction:** Paired bootstrap 95% CIs are reported against 6 baselines without Bonferroni/Holm adjustment.
- **Modest absolute NDCG lift over Repurchase:** While `Recall@50` jumps `+23.1%` (`0.0793 → 0.0976`), the paired `NDCG@10` improvement over User Repurchase is `+0.00118` (`95% CI [+0.00051, +0.00184]`).

## 2. Modeling & Data Constraints
- **Date-level ordering (`t_dat, row_id`):** H&M transactions lack intra-day timestamps; same-day ordering uses deterministic `row_id` rather than true chronological event order.
- **Train vs. test query mismatch:** Stage-2 LightGBM trains on sampled Week-7 single-item interactions (`<= 30,000` rows, conditioned on Stage-1 hit), whereas Week-8 testing evaluates from each user's first Week-8 interaction against all their Week-8 purchases over 7 days.
- **Candidate recall bottleneck:** The ~100-item Stage-1 pool achieves `12.46%` Candidate Recall (`24.51%` user hit rate), bounding downstream LightGBM performance.
- **Offline vs. live multi-channel parity:** Offline evaluation merges Two-Tower ANN + Repurchase + ItemKNN and populates `item_knn_score`, whereas live FastAPI serving queries Qdrant ANN alone and passes `item_knn_score=0.0`.

## 3. Demo & Production Readiness
- **Synthetic demo artifacts:** `make demo` generates random embeddings and an untrained `UserTower` ONNX file to verify pipeline wiring in 60 seconds—not recommendation relevance.
- **Not production-hardened:** Lacks authentication/authorization, API rate limiting, Redis/Qdrant TLS & ACLs, PII interaction logging governance, and automated alerting.
- **Dataset specificity:** Results on an 8-week H&M fashion slice do not automatically generalize to other domains, catalogs, or interaction horizons.
