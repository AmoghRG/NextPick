"""Integration tests for Qdrant loader, Redis bounded session window, and FastAPI endpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from src.index.qdrant_loader import COLLECTION_NAME, load_catalog_to_qdrant
from src.models.two_tower import UserTower, export_user_tower_onnx
from src.serving.main import AppServices, create_app
from src.serving.ranker import Stage2Ranker
from src.serving.retriever import Stage1Retriever
from src.serving.session_store import InMemoryRedis, SessionEvent, SessionStore


def test_redis_sliding_window_bounded_to_20() -> None:
    store = SessionStore(InMemoryRedis(), max_len=20)
    for i in range(30):
        length = store.record_event(
            "s1",
            SessionEvent(
                article_id=f"art_{i:03d}",
                price=10.0 + i,
                department_name="DeptA",
                product_group_name="Grp1",
            ),
        )
    assert length <= 20
    hist = store.get_history("s1")
    assert len(hist) == 20
    # Index 0 must be the most recent click (art_029)
    assert hist[0].article_id == "art_029"
    assert hist[-1].article_id == "art_010"


def test_end_to_end_qdrant_onnx_fastapi_flow(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    n_items = 15
    vecs = rng.normal(size=(n_items, 128)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True)

    cat_df = pd.DataFrame(
        [
            {
                "article_id": f"art_{i:03d}",
                "prod_name": f"Product {i}",
                "product_type_name": "Hoodie",
                "product_group_name": "Upper" if i % 2 == 0 else "Lower",
                "department_name": "Menswear" if i < 5 else "Womenswear",
                "colour_group_name": "Black",
                "avg_price": 0.02 + 0.01 * (i % 4),
                "train_tx_count": 100 - i,
                "in_stock": True,
            }
            for i in range(n_items)
        ]
    )
    emb_df = pd.DataFrame({"article_id": cat_df["article_id"], "embedding": vecs.tolist()})
    cat_path = tmp_path / "catalog.parquet"
    emb_path = tmp_path / "item_embeddings.parquet"
    cat_df.to_parquet(cat_path, index=False)
    emb_df.to_parquet(emb_path, index=False)

    q_client = QdrantClient(":memory:")
    indexed = load_catalog_to_qdrant(q_client, cat_path, emb_path)
    assert indexed == n_items

    onnx_path = tmp_path / "user_tower.onnx"
    export_user_tower_onnx(UserTower().eval(), onnx_path)

    item_vectors = {f"art_{i:03d}": vecs[i] for i in range(n_items)}
    trending = [
        {
            "article_id": "art_000",
            "prod_name": "Product 0",
            "product_group_name": "Upper",
            "department_name": "Menswear",
            "price": 0.02,
            "train_tx_count": 100,
            "stage1_rank": 1,
            "stage2_rank": 1,
            "rank_delta": 0,
            "stage1_cosine_score": 0.0,
            "stage2_logit": 0.0,
        }
    ]

    services = AppServices(
        session_store=SessionStore(InMemoryRedis()),
        retriever=Stage1Retriever(q_client, onnx_path, item_vectors, collection_name=COLLECTION_NAME),
        ranker=Stage2Ranker(model_path=None),
        trending_fallback=trending,
    )
    client = TestClient(create_app(services))

    # 1. Health check
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    # 2. 0-click cold-start fallback
    r_cold = client.get("/recommend?session_id=new_user&top_k=5")
    assert r_cold.status_code == 200
    assert r_cold.json()["mode"] == "cold_start_trending"
    assert len(r_cold.json()["items"]) == 1

    # 3. Record click -> personalized 2-stage response
    r_ev = client.post(
        "/event",
        json={
            "session_id": "new_user",
            "article_id": "art_002",
            "price": 0.04,
            "department_name": "Menswear",
            "product_group_name": "Upper",
        },
    )
    assert r_ev.status_code == 200
    assert r_ev.json()["history_len"] == 1

    r_rec = client.get("/recommend?session_id=new_user&product_group=Upper&top_k=5")
    assert r_rec.status_code == 200
    body = r_rec.json()
    assert body["mode"] == "two_stage_personalized"
    assert body["history_len"] == 1
    assert len(body["items"]) == 5
    assert all(item["product_group_name"] == "Upper" for item in body["items"])
    assert "rank_delta" in body["items"][0]
    assert set(body["latency_ms"].keys()) == {"redis", "onnx", "qdrant", "lightgbm", "total"}

    # 4. Verify ONNX LRU cache hit on repeated identical sequence
    services.retriever.encode_user(["art_002"], sales_channel_id=2)
    cache_info = services.retriever._cached_encode.cache_info()
    assert cache_info.hits >= 1


def test_generate_demo_artifacts(tmp_path: Path) -> None:
    from scripts.generate_demo_artifacts import generate_demo_artifacts

    generate_demo_artifacts(tmp_path, num_items=25)
    assert (tmp_path / "catalog.parquet").exists()
    assert (tmp_path / "item_embeddings.parquet").exists()
    assert (tmp_path / "user_tower.onnx").exists()
    assert len(pd.read_parquet(tmp_path / "catalog.parquet")) == 25
