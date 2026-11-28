"""NextPick FastAPI Serving Application (/event, /recommend, /health)."""

from __future__ import annotations

import gc
import json
import logging
import os
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import redis
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from src.index.qdrant_loader import COLLECTION_NAME
from src.serving.ranker import Stage2Ranker
from src.serving.retriever import Stage1Retriever
from src.serving.session_store import InMemoryRedis, SessionEvent, SessionStore


class EventRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    article_id: str = Field(..., min_length=1)
    price: float = Field(..., ge=0.0)
    department_name: str = "unknown"
    product_group_name: str = "unknown"


class AppServices:
    """Container holding initialized serving dependencies."""

    def __init__(
        self,
        session_store: SessionStore,
        retriever: Stage1Retriever,
        ranker: Stage2Ranker,
        trending_fallback: list[dict],
        is_synthetic_demo: bool = False,
        article_segments: dict[str, str] | None = None,
        segment_catalog: dict[str, list[dict]] | None = None,
        article_names: dict[str, str] | None = None,
    ) -> None:
        self.session_store = session_store
        self.retriever = retriever
        self.ranker = ranker
        self.trending_fallback = trending_fallback
        self.is_synthetic_demo = is_synthetic_demo
        self.article_segments = article_segments or {}
        self.segment_catalog = segment_catalog or {}
        self.article_names = article_names or {}


def build_default_services() -> AppServices:
    """Initialize Redis, Qdrant, ONNX, and LightGBM from environment / ./artifacts."""
    art_dir = Path(os.getenv("NEXTPICK_ARTIFACTS_DIR", "artifacts"))
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY") or None
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    allow_inmemory = os.getenv("NEXTPICK_ALLOW_INMEMORY_REDIS", "0") == "1"

    try:
        r_client = redis.from_url(redis_url, decode_responses=True)
        r_client.ping()
    except Exception:
        if not allow_inmemory:
            raise
        logging.getLogger("nextpick").warning(
            "Redis unavailable; falling back to process-local InMemoryRedis (dev/demo only)."
        )
        r_client = InMemoryRedis()

    q_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, check_compatibility=False)

    item_vectors: dict[str, np.ndarray] = {}
    emb_path = art_dir / "item_embeddings.parquet"
    if emb_path.exists():
        emb_df = pd.read_parquet(emb_path)
        item_vectors = {
            str(r.article_id): np.asarray(r.embedding, dtype=np.float32)
            for r in emb_df.itertuples(index=False)
        }
        del emb_df
        gc.collect()

    trending: list[dict] = []
    article_segments: dict[str, str] = {}
    segment_catalog: dict[str, list[dict]] = {}
    cat_path = art_dir / "catalog.parquet"
    if cat_path.exists():
        full_cat = pd.read_parquet(cat_path).sort_values("train_tx_count", ascending=False)
        # Keep the top-selling colorway per prod_name so cold-start doesn't repeat identical items
        dedup_cat = full_cat.drop_duplicates(subset=["prod_name"], keep="first")
        if "index_group_name" in full_cat.columns:
            article_segments = dict(
                zip(
                    full_cat["article_id"].astype(str),
                    full_cat["index_group_name"].astype(str),
                )
            )
            for seg_name in ("Menswear", "Ladieswear", "Divided", "Sport", "Baby/Children"):
                seg_pool = dedup_cat[dedup_cat["index_group_name"] == seg_name]
                seg_rows: list = []
                seg_dept_counts: dict[str, int] = {}
                seg_overflow: list = []
                for row in seg_pool.itertuples(index=False):
                    dept = str(row.department_name)
                    if seg_dept_counts.get(dept, 0) < 2 and len(seg_rows) < 40:
                        seg_rows.append(row)
                        seg_dept_counts[dept] = seg_dept_counts.get(dept, 0) + 1
                    else:
                        seg_overflow.append(row)
                if len(seg_rows) < 40:
                    seg_rows.extend(seg_overflow[: 40 - len(seg_rows)])
                segment_catalog[seg_name] = [
                    {
                        "article_id": str(r.article_id),
                        "prod_name": str(r.prod_name),
                        "product_group_name": str(r.product_group_name),
                        "index_group_name": str(getattr(r, "index_group_name", "unknown")),
                        "department_name": str(r.department_name),
                        "price": float(r.avg_price),
                        "train_tx_count": int(r.train_tx_count),
                        "stage1_rank": 101,
                        "stage2_rank": 101,
                        "rank_delta": 0,
                        "stage1_cosine_score": 0.0,
                        "stage2_logit": 0.0,
                    }
                    for r in seg_rows
                ]

        if "index_group_name" in dedup_cat.columns:
            # Build a diverse cold-start mix across Women, Divided, Men, Sport, and product groups
            selected_rows = []
            seen_ids: set[str] = set()
            seen_types: dict[str, int] = {}

            def _pick_diverse(pool: pd.DataFrame, limit: int, max_per_group: int = 2) -> None:
                for row in pool.itertuples(index=False):
                    if len(selected_rows) >= limit:
                        break
                    aid = str(row.article_id)
                    if aid in seen_ids:
                        continue
                    pg = str(row.product_group_name)
                    if seen_types.get(pg, 0) >= max_per_group:
                        continue
                    selected_rows.append(row)
                    seen_ids.add(aid)
                    seen_types[pg] = seen_types.get(pg, 0) + 1

            # Interleave 1 item at a time across key segments for Top-10 variety
            women_pool = dedup_cat[dedup_cat["index_group_name"] == "Ladieswear"]
            divided_pool = dedup_cat[dedup_cat["index_group_name"] == "Divided"]
            men_pool = dedup_cat[dedup_cat["index_group_name"] == "Menswear"]
            sport_pool = dedup_cat[dedup_cat["index_group_name"] == "Sport"]

            # Target balanced Top-10: Women (4), Divided (2), Men (3), Sport (1) across diverse categories
            _pick_diverse(women_pool, 2, max_per_group=1)
            _pick_diverse(divided_pool, 4, max_per_group=1)
            _pick_diverse(men_pool, 6, max_per_group=1)
            _pick_diverse(sport_pool, 7, max_per_group=1)
            _pick_diverse(women_pool, 9, max_per_group=2)
            _pick_diverse(men_pool, 10, max_per_group=2)

            # Ensure Socks & Tights also has at least 5 items in trending_fallback for filter usability
            socks_pool = dedup_cat[dedup_cat["product_group_name"] == "Socks & Tights"]
            for row in socks_pool.itertuples(index=False):
                if sum(1 for r in selected_rows if str(r.product_group_name) == "Socks & Tights") >= 5:
                    break
                aid = str(row.article_id)
                if aid not in seen_ids:
                    selected_rows.append(row)
                    seen_ids.add(aid)

            # Fill up to 60 diverse items so category/price filters always have plenty of results
            for row in dedup_cat.itertuples(index=False):
                if len(selected_rows) >= 60:
                    break
                aid = str(row.article_id)
                if aid not in seen_ids:
                    selected_rows.append(row)
                    seen_ids.add(aid)
            cat_rows = selected_rows
        else:
            cat_rows = list(dedup_cat.head(60).itertuples(index=False))

        for rank, r in enumerate(cat_rows, start=1):
            trending.append(
                {
                    "article_id": str(r.article_id),
                    "prod_name": str(r.prod_name),
                    "product_group_name": str(r.product_group_name),
                    "index_group_name": str(getattr(r, "index_group_name", "unknown")),
                    "department_name": str(r.department_name),
                    "price": float(r.avg_price),
                    "train_tx_count": int(r.train_tx_count),
                    "stage1_rank": rank,
                    "stage2_rank": rank,
                    "rank_delta": 0,
                    "stage1_cosine_score": 0.0,
                    "stage2_logit": 0.0,
                }
            )
        del full_cat, dedup_cat, cat_rows
        gc.collect()

    onnx_path = art_dir / "user_tower.onnx"
    lgbm_path = art_dir / "lgbm_ranker.txt"
    is_synthetic_demo = (art_dir / "demo_manifest.json").exists()

    return AppServices(
        session_store=SessionStore(r_client),
        retriever=Stage1Retriever(q_client, onnx_path, item_vectors, collection_name=COLLECTION_NAME),
        ranker=Stage2Ranker(lgbm_path),
        trending_fallback=trending,
        is_synthetic_demo=is_synthetic_demo,
        article_segments=article_segments,
        segment_catalog=segment_catalog,
    )


def _diversify_personalized(
    raw_ranked: list[dict],
    history: list[SessionEvent],
    svc: AppServices,
    top_k: int,
    product_group: str | None = None,
    max_price: float | None = None,
) -> list[dict]:
    """Ensure early-round segment alignment, cross-category variety, and no duplicate colorways."""
    clicked_ids = {e.article_id for e in history}
    raw_name_map = {
        str(x.get("article_id")): str(x.get("prod_name") or "").strip().lower()
        for x in (
            list(raw_ranked)
            + list(svc.trending_fallback)
            + [it for seg_items in svc.segment_catalog.values() for it in seg_items]
        )
        if x.get("prod_name")
    }
    clicked_names = {
        (svc.article_names.get(aid) or raw_name_map.get(aid) or "").strip().lower()
        for aid in clicked_ids
    } - {""}
    hist_segs = [
        svc.article_segments.get(e.article_id, "unknown")
        for e in history
        if svc.article_segments.get(e.article_id, "unknown") not in ("", "unknown")
    ]

    target_segs: set[str] | None = None
    if hist_segs:
        unique_segs = set(hist_segs)
        if unique_segs == {"Menswear"}:
            target_segs = {"Menswear"}
        elif unique_segs <= {"Ladieswear", "Divided"}:
            target_segs = {"Ladieswear", "Divided"}
        else:
            target_segs = unique_segs

    early_session = len(history) <= 2
    max_per_dept = 2 if early_session else 4
    max_per_group = None if product_group else (3 if early_session else 7)

    # Combine LightGBM-ranked Qdrant candidates with same-segment diverse catalog items (as tail exploration)
    pool: list[dict] = list(raw_ranked)
    if early_session and target_segs and svc.segment_catalog:
        seen_pool_ids = {str(x.get("article_id")) for x in pool}
        for seg in target_segs:
            for item in svc.segment_catalog.get(seg, []):
                aid = str(item.get("article_id"))
                if aid in seen_pool_ids:
                    continue
                if product_group and item.get("product_group_name") != product_group:
                    continue
                if max_price is not None and float(item.get("price", 0.0)) > max_price:
                    continue
                pool.append(dict(item))
                seen_pool_ids.add(aid)

    selected: list[dict] = []
    seen_names: set[str] = set()
    seen_ids: set[str] = set()
    dept_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}

    def _try_add(item: dict, *, require_segment: bool, enforce_caps: bool, skip_clicked: bool) -> bool:
        aid = str(item.get("article_id"))
        if aid in seen_ids:
            return False
        if skip_clicked and aid in clicked_ids:
            return False
        seg = str(item.get("index_group_name") or svc.article_segments.get(aid, "unknown"))
        if require_segment and target_segs and seg not in target_segs:
            return False
        pname = str(item.get("prod_name") or aid).strip().lower()
        if skip_clicked and pname in clicked_names:
            return False
        if pname in seen_names:
            return False
        dept = str(item.get("department_name") or "unknown")
        pg = str(item.get("product_group_name") or "unknown")
        if enforce_caps:
            if dept_counts.get(dept, 0) >= max_per_dept:
                return False
            if max_per_group is not None and group_counts.get(pg, 0) >= max_per_group:
                return False
        selected.append(dict(item))
        seen_ids.add(aid)
        seen_names.add(pname)
        dept_counts[dept] = dept_counts.get(dept, 0) + 1
        group_counts[pg] = group_counts.get(pg, 0) + 1
        return True

    # Pass 1: Matching segment + category caps + non-clicked
    for item in pool:
        if len(selected) >= top_k:
            break
        _try_add(item, require_segment=True, enforce_caps=True, skip_clicked=True)

    # Pass 2: Matching segment without category caps (still deduped & non-clicked)
    if len(selected) < top_k:
        for item in pool:
            if len(selected) >= top_k:
                break
            _try_add(item, require_segment=True, enforce_caps=False, skip_clicked=True)

    # Pass 3: Any segment with category caps (still deduped & non-clicked)
    if len(selected) < top_k:
        for item in pool:
            if len(selected) >= top_k:
                break
            _try_add(item, require_segment=False, enforce_caps=True, skip_clicked=True)

    # Pass 4: Fill any remaining slots from raw_ranked (including clicked/same-name if small test catalog)
    if len(selected) < top_k:
        for item in raw_ranked:
            if len(selected) >= top_k:
                break
            aid = str(item.get("article_id"))
            if aid not in seen_ids:
                selected.append(dict(item))
                seen_ids.add(aid)

    for new_rank, item in enumerate(selected, start=1):
        item["stage2_rank"] = new_rank
        item["rank_delta"] = int(item.get("stage1_rank", new_rank)) - new_rank
    return selected[:top_k]


def create_app(services: AppServices | None = None) -> FastAPI:
    app = FastAPI(title="NextPick API", version="1.0.0")
    cors_origins = [
        o.strip()
        for o in os.getenv("NEXTPICK_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
        if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )
    access_logger = logging.getLogger("nextpick.access")

    @app.middleware("http")
    async def request_context_logging(request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
        access_logger.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": duration_ms,
                }
            )
        )
        return response

    app.state.services = services

    def _svc() -> AppServices:
        if app.state.services is None:
            app.state.services = build_default_services()
        return app.state.services

    @app.get("/health")
    def health() -> dict:
        svc = _svc()
        redis_ok = False
        redis_inmemory = isinstance(svc.session_store._redis, InMemoryRedis)
        try:
            redis_ok = bool(svc.session_store._redis.ping())
        except Exception:
            redis_ok = False

        qdrant_ok = False
        try:
            qdrant_ok = bool(svc.retriever._qdrant.collection_exists(COLLECTION_NAME))
        except Exception:
            qdrant_ok = False

        onnx_ok = svc.retriever._ort is not None
        lgbm_ok = svc.ranker._booster is not None
        status = "ok" if (redis_ok and not redis_inmemory and qdrant_ok and onnx_ok and lgbm_ok) else "degraded"
        return {
            "status": status,
            "is_synthetic_demo": svc.is_synthetic_demo,
            "dependencies": {
                "redis": "in_memory_fallback" if redis_inmemory else ("ok" if redis_ok else "down"),
                "qdrant": "ok" if qdrant_ok else "down",
                "onnx": "ok" if onnx_ok else "fallback",
                "lightgbm": "ok" if lgbm_ok else "fallback",
            },
            "trending_cached": len(svc.trending_fallback),
        }

    @app.post("/event")
    def record_event(req: EventRequest) -> dict:
        svc = _svc()
        ev = SessionEvent(
            article_id=req.article_id,
            price=req.price,
            department_name=req.department_name,
            product_group_name=req.product_group_name,
        )
        history_len = svc.session_store.record_event(req.session_id, ev)
        return {"status": "ok", "session_id": req.session_id, "history_len": history_len}

    @app.get("/recommend")
    def recommend(
        session_id: str = Query(..., min_length=1),
        product_group: str | None = Query(default=None),
        max_price: float | None = Query(default=None),
        sales_channel_id: int = Query(default=2, ge=0, le=3),
        top_k: int = Query(default=10, ge=1, le=50),
    ) -> dict:
        svc = _svc()
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        history = svc.session_store.get_history(session_id)
        redis_ms = (time.perf_counter() - t0) * 1000.0

        # 0-click anonymous cold-start router
        if not history:
            items = svc.trending_fallback
            if product_group:
                items = [x for x in items if x.get("product_group_name") == product_group]
            if max_price is not None:
                items = [x for x in items if float(x.get("price", 0.0)) <= max_price]
            total_ms = (time.perf_counter() - t_start) * 1000.0
            return {
                "session_id": session_id,
                "mode": "cold_start_trending",
                "ranker_mode": svc.ranker.mode,
                "is_synthetic_demo": svc.is_synthetic_demo,
                "history_len": 0,
                "items": items[:top_k],
                "latency_ms": {
                    "redis": round(redis_ms, 3),
                    "onnx": 0.0,
                    "qdrant": 0.0,
                    "lightgbm": 0.0,
                    "total": round(total_ms, 3),
                },
            }

        t1 = time.perf_counter()
        user_vec = svc.retriever.encode_user([e.article_id for e in history], sales_channel_id=sales_channel_id)
        onnx_ms = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        candidates = svc.retriever.search_candidates(
            user_vector=user_vec,
            top_k=150,
            product_group=product_group,
            max_price=max_price,
        )
        qdrant_ms = (time.perf_counter() - t2) * 1000.0

        t3 = time.perf_counter()
        raw_ranked = svc.ranker.rank(
            history,
            candidates,
            sales_channel_id=sales_channel_id,
            top_k=len(candidates),
        )
        ranked = _diversify_personalized(
            raw_ranked=raw_ranked,
            history=history,
            svc=svc,
            top_k=top_k,
            product_group=product_group,
            max_price=max_price,
        )
        lgbm_ms = (time.perf_counter() - t3) * 1000.0

        total_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "session_id": session_id,
            "mode": "two_stage_personalized",
            "ranker_mode": svc.ranker.mode,
            "is_synthetic_demo": svc.is_synthetic_demo,
            "history_len": len(history),
            "items": ranked,
            "latency_ms": {
                "redis": round(redis_ms, 3),
                "onnx": round(onnx_ms, 3),
                "qdrant": round(qdrant_ms, 3),
                "lightgbm": round(lgbm_ms, 3),
                "total": round(total_ms, 3),
            },
        }

    return app


app = create_app()
