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

    # 1. Health check reports dependency status cleanly
    r_health = client.get("/health")
    assert r_health.status_code == 200
    h_json = r_health.json()
    assert h_json["status"] == "degraded"  # InMemoryRedis active in unit test
    assert h_json["dependencies"]["redis"] == "in_memory_fallback"
    assert h_json["dependencies"]["qdrant"] == "ok"
    assert h_json["dependencies"]["onnx"] == "ok"

    # 1b. Invalid sales_channel_id rejected at API boundary
    r_bad_chan = client.get("/recommend?session_id=new_user&sales_channel_id=99")
    assert r_bad_chan.status_code == 422

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
    assert body["ranker_mode"] == "heuristic_fallback"
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


def test_retriever_fallback_masked_pooling_and_unknown_items() -> None:
    q_client = QdrantClient(":memory:")
    v0 = np.array([1.0] + [0.0] * 127, dtype=np.float32)
    retriever = Stage1Retriever(q_client, onnx_path=None, item_vectors={"known": v0})

    # Empty or unknown-only history returns zero vector
    assert np.allclose(retriever.encode_user([]), np.zeros(128, dtype=np.float32))
    assert np.allclose(retriever.encode_user(["missing_id"]), np.zeros(128, dtype=np.float32))

    # Single known item with unknown item ignores unknown and normalizes cleanly
    emb = retriever.encode_user(["known", "missing_id"])
    assert np.allclose(emb, v0, atol=1e-5)


def test_qdrant_ensure_collection_is_non_destructive(tmp_path: Path) -> None:
    import pytest
    from qdrant_client.models import PointStruct
    from src.index.qdrant_loader import ensure_collection

    q_client = QdrantClient(":memory:")
    ensure_collection(q_client, collection="test_col", vector_size=128)
    q_client.upsert(
        collection_name="test_col",
        points=[PointStruct(id=1, vector=[0.1] * 128, payload={"article_id": "a1"})],
    )

    # Re-running ensure_collection without recreate=True must preserve existing points
    ensure_collection(q_client, collection="test_col", vector_size=128, recreate=False)
    assert q_client.get_collection("test_col").points_count == 1

    # Mismatched vector size must raise ValueError instead of silently corrupting
    with pytest.raises(ValueError, match="vector size mismatch"):
        ensure_collection(q_client, collection="test_col", vector_size=64, recreate=False)


def test_qdrant_stable_ids_and_stale_pruning(tmp_path: Path) -> None:
    from src.index.qdrant_loader import stable_point_id

    assert stable_point_id("SKU_A") == stable_point_id("SKU_A")
    assert stable_point_id("SKU_A") != stable_point_id("SKU_B")

    q_client = QdrantClient(":memory:")
    cat1 = pd.DataFrame([{"article_id": "SKU_A", "prod_name": "A"}, {"article_id": "SKU_B", "prod_name": "B"}])
    emb1 = pd.DataFrame({"article_id": ["SKU_A", "SKU_B"], "embedding": [[0.1] * 128, [0.2] * 128]})
    c1, e1 = tmp_path / "c1.parquet", tmp_path / "e1.parquet"
    cat1.to_parquet(c1, index=False)
    emb1.to_parquet(e1, index=False)

    assert load_catalog_to_qdrant(q_client, c1, e1) == 2
    assert q_client.get_collection(COLLECTION_NAME).points_count == 2

    # Re-index with SKU_B removed and SKU_C added -> stale SKU_B must be pruned
    cat2 = pd.DataFrame([{"article_id": "SKU_C", "prod_name": "C"}, {"article_id": "SKU_A", "prod_name": "A2"}])
    emb2 = pd.DataFrame({"article_id": ["SKU_C", "SKU_A"], "embedding": [[0.3] * 128, [0.1] * 128]})
    c2, e2 = tmp_path / "c2.parquet", tmp_path / "e2.parquet"
    cat2.to_parquet(c2, index=False)
    emb2.to_parquet(e2, index=False)

    assert load_catalog_to_qdrant(q_client, c2, e2, prune_stale=True) == 2
    assert q_client.get_collection(COLLECTION_NAME).points_count == 2
    pts, _ = q_client.scroll(collection_name=COLLECTION_NAME, limit=10, with_payload=True)
    ids_present = {p.payload["article_id"] for p in pts}
    assert ids_present == {"SKU_A", "SKU_C"}


def test_qdrant_loader_fails_fast_on_invalid_inputs(tmp_path: Path) -> None:
    import pytest

    q_client = QdrantClient(":memory:")
    cat_ok = pd.DataFrame([{"article_id": "SKU_A"}, {"article_id": "SKU_B"}])
    emb_ok = pd.DataFrame({"article_id": ["SKU_A", "SKU_B"], "embedding": [[0.1] * 128, [0.2] * 128]})
    c_ok, e_ok = tmp_path / "c_ok.parquet", tmp_path / "e_ok.parquet"
    cat_ok.to_parquet(c_ok, index=False)
    emb_ok.to_parquet(e_ok, index=False)
    assert load_catalog_to_qdrant(q_client, c_ok, e_ok) == 2

    # Missing embedding for SKU_B must fail fast WITHOUT deleting existing points
    emb_partial = pd.DataFrame({"article_id": ["SKU_A"], "embedding": [[0.1] * 128]})
    e_part = tmp_path / "e_part.parquet"
    emb_partial.to_parquet(e_part, index=False)
    with pytest.raises(ValueError, match="Missing embeddings"):
        load_catalog_to_qdrant(q_client, c_ok, e_part)
    assert q_client.get_collection(COLLECTION_NAME).points_count == 2

    # NaN embedding must fail fast
    emb_nan = pd.DataFrame({"article_id": ["SKU_A", "SKU_B"], "embedding": [[float("nan")] * 128, [0.2] * 128]})
    e_nan = tmp_path / "e_nan.parquet"
    emb_nan.to_parquet(e_nan, index=False)
    with pytest.raises(ValueError, match="Non-finite"):
        load_catalog_to_qdrant(q_client, c_ok, e_nan)

    # Zero-dimensional embedding and invalid batch_size must fail fast
    emb_empty_vec = pd.DataFrame({"article_id": ["SKU_A", "SKU_B"], "embedding": [[], [0.2] * 128]})
    e_zero = tmp_path / "e_zero.parquet"
    emb_empty_vec.to_parquet(e_zero, index=False)
    with pytest.raises(ValueError, match="Invalid embedding vector shape"):
        load_catalog_to_qdrant(q_client, c_ok, e_zero)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        load_catalog_to_qdrant(q_client, c_ok, e_ok, batch_size=0)

    # Null/blank article_id and negative avg_price must fail fast
    cat_null = pd.DataFrame([{"article_id": None, "avg_price": 0.02}, {"article_id": "SKU_B", "avg_price": 0.03}])
    c_null = tmp_path / "c_null.parquet"
    cat_null.to_parquet(c_null, index=False)
    with pytest.raises(ValueError, match="Null article_id"):
        load_catalog_to_qdrant(q_client, c_null, e_ok)

    cat_neg_price = pd.DataFrame([{"article_id": "SKU_A", "avg_price": -1.0}, {"article_id": "SKU_B", "avg_price": 0.03}])
    c_neg = tmp_path / "c_neg.parquet"
    cat_neg_price.to_parquet(c_neg, index=False)
    with pytest.raises(ValueError, match="avg_price"):
        load_catalog_to_qdrant(q_client, c_neg, e_ok)


def test_dependency_failure_modes(monkeypatch, tmp_path: Path) -> None:
    import pytest
    from src.serving.main import build_default_services

    # 1. Redis down + NEXTPICK_ALLOW_INMEMORY_REDIS=0 must raise
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:59999/0")
    monkeypatch.setenv("NEXTPICK_ALLOW_INMEMORY_REDIS", "0")
    with pytest.raises(Exception):
        build_default_services()

    # 2. Empty Qdrant collection + Qdrant down health status
    q_client = QdrantClient(":memory:")
    services = AppServices(
        session_store=SessionStore(InMemoryRedis()),
        retriever=Stage1Retriever(q_client, onnx_path=None, item_vectors={}),
        ranker=Stage2Ranker(model_path=None),
        trending_fallback=[],
    )
    client = TestClient(create_app(services))
    h = client.get("/health").json()
    assert h["status"] == "degraded"
    assert h["dependencies"]["qdrant"] == "down"

    # 3. Empty candidates returns [] cleanly
    assert services.ranker.rank([], []) == []


def test_api_contract_422_and_request_id_headers() -> None:
    q_client = QdrantClient(":memory:")
    services = AppServices(
        session_store=SessionStore(InMemoryRedis()),
        retriever=Stage1Retriever(q_client, onnx_path=None, item_vectors={}),
        ranker=Stage2Ranker(model_path=None),
        trending_fallback=[],
    )
    client = TestClient(create_app(services))

    # Invalid /event payload (negative price) -> 422 + X-Request-ID echoed
    r_bad_ev = client.post(
        "/event",
        json={"session_id": "s1", "article_id": "A1", "price": -1.0},
        headers={"X-Request-ID": "req-test-422"},
    )
    assert r_bad_ev.status_code == 422
    assert r_bad_ev.headers["X-Request-ID"] == "req-test-422"
    assert isinstance(r_bad_ev.json()["detail"], list)

    # Invalid /recommend query (top_k=0) -> 422 + auto-generated X-Request-ID
    r_bad_rec = client.get("/recommend", params={"session_id": "s1", "top_k": 0})
    assert r_bad_rec.status_code == 422
    assert len(r_bad_rec.headers["X-Request-ID"]) > 0


def test_api_empty_history_cold_start_filters() -> None:
    q_client = QdrantClient(":memory:")
    trending = [
        {
            "article_id": "T1",
            "prod_name": "Top Tee",
            "product_group_name": "Garment Upper body",
            "department_name": "Jersey",
            "price": 0.02,
            "train_tx_count": 100,
        },
        {
            "article_id": "T2",
            "prod_name": "Expensive Coat",
            "product_group_name": "Garment Upper body",
            "department_name": "Outerwear",
            "price": 0.15,
            "train_tx_count": 90,
        },
        {
            "article_id": "T3",
            "prod_name": "Jeans",
            "product_group_name": "Garment Lower body",
            "department_name": "Denim",
            "price": 0.02,
            "train_tx_count": 80,
        },
    ]
    services = AppServices(
        session_store=SessionStore(InMemoryRedis()),
        retriever=Stage1Retriever(q_client, onnx_path=None, item_vectors={}),
        ranker=Stage2Ranker(model_path=None),
        trending_fallback=trending,
    )
    client = TestClient(create_app(services))
    resp = client.get(
        "/recommend",
        params={"session_id": "cold_user", "product_group": "Garment Upper body", "max_price": 0.05},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "cold_start_trending"
    assert body["history_len"] == 0
    assert [x["article_id"] for x in body["items"]] == ["T1"]
    assert "X-Process-Time-Ms" in resp.headers


def test_api_unknown_sku_event_graceful_serving(tmp_path: Path) -> None:
    from qdrant_client.models import Distance, PointStruct, VectorParams

    q_client = QdrantClient(":memory:")
    vec = [1.0] + [0.0] * 127
    q_client.create_collection(COLLECTION_NAME, vectors_config=VectorParams(size=128, distance=Distance.COSINE))
    q_client.upsert(
        COLLECTION_NAME,
        points=[
            PointStruct(
                id=1,
                vector=vec,
                payload={
                    "article_id": "KNOWN_1",
                    "prod_name": "Known Tee",
                    "product_group_name": "Garment Upper body",
                    "department_name": "Jersey",
                    "price": 0.02,
                    "train_tx_count": 10,
                },
            )
        ],
    )
    services = AppServices(
        session_store=SessionStore(InMemoryRedis()),
        retriever=Stage1Retriever(q_client, onnx_path=None, item_vectors={"KNOWN_1": np.array(vec, dtype=np.float32)}),
        ranker=Stage2Ranker(model_path=None),
        trending_fallback=[],
    )
    client = TestClient(create_app(services))
    ev = client.post("/event", json={"session_id": "u_unk", "article_id": "UNKNOWN_SKU_999", "price": 0.03})
    assert ev.status_code == 200
    assert ev.json()["history_len"] == 1

    rec = client.get("/recommend", params={"session_id": "u_unk", "top_k": 5})
    assert rec.status_code == 200
    body = rec.json()
    assert body["mode"] == "two_stage_personalized"
    assert body["items"][0]["article_id"] == "KNOWN_1"
    assert set(body["latency_ms"].keys()) == {"redis", "onnx", "qdrant", "lightgbm", "total"}


def test_early_session_diversity_and_segment_alignment() -> None:
    from src.serving.main import _diversify_personalized

    services = AppServices(
        session_store=SessionStore(InMemoryRedis()),
        retriever=Stage1Retriever(QdrantClient(":memory:"), onnx_path=None, item_vectors={}),
        ranker=Stage2Ranker(model_path=None),
        trending_fallback=[],
        article_segments={
            "M_SH1": "Menswear",
            "M_SH2": "Menswear",
            "M_SH3": "Menswear",
            "W_DR1": "Ladieswear",
            "M_HD1": "Menswear",
            "M_TR1": "Menswear",
        },
        segment_catalog={
            "Menswear": [
                {
                    "article_id": "M_HD1",
                    "prod_name": "Men Hoodie",
                    "product_group_name": "Garment Upper body",
                    "index_group_name": "Menswear",
                    "department_name": "Jersey",
                    "price": 0.03,
                    "stage1_rank": 101,
                    "stage2_rank": 101,
                },
                {
                    "article_id": "M_TR1",
                    "prod_name": "Men Chino",
                    "product_group_name": "Garment Lower body",
                    "index_group_name": "Menswear",
                    "department_name": "Trousers",
                    "price": 0.04,
                    "stage1_rank": 101,
                    "stage2_rank": 101,
                },
            ]
        },
    )

    raw_ranked = [
        {"article_id": "M_SH1", "prod_name": "Dario Shorts", "product_group_name": "Garment Lower body", "index_group_name": "Menswear", "department_name": "Shorts", "price": 0.02, "stage1_rank": 1},
        {"article_id": "M_SH2", "prod_name": "Laurent Shorts", "product_group_name": "Garment Lower body", "index_group_name": "Menswear", "department_name": "Shorts", "price": 0.02, "stage1_rank": 2},
        {"article_id": "M_SH2_ALT", "prod_name": "Laurent Shorts", "product_group_name": "Garment Lower body", "index_group_name": "Menswear", "department_name": "Shorts", "price": 0.02, "stage1_rank": 3},
        {"article_id": "M_SH3", "prod_name": "Zen Shorts", "product_group_name": "Garment Lower body", "index_group_name": "Menswear", "department_name": "Shorts", "price": 0.02, "stage1_rank": 4},
        {"article_id": "M_SH4", "prod_name": "Extra Shorts", "product_group_name": "Garment Lower body", "index_group_name": "Menswear", "department_name": "Shorts", "price": 0.02, "stage1_rank": 5},
        {"article_id": "W_DR1", "prod_name": "Ladies Dress", "product_group_name": "Garment Full body", "index_group_name": "Ladieswear", "department_name": "Dress", "price": 0.03, "stage1_rank": 6},
    ]
    history = [SessionEvent(article_id="M_SH1", price=0.02, department_name="Shorts", product_group_name="Garment Lower body")]

    out = _diversify_personalized(raw_ranked, history, services, top_k=4)
    out_ids = [x["article_id"] for x in out]
    # Clicked item excluded, duplicate colorway excluded, >2 Shorts capped, Menswear Hoodie & Chino blended in, Ladieswear excluded
    assert "M_SH1" not in out_ids
    assert "M_SH2_ALT" not in out_ids
    assert "M_SH4" not in out_ids
    assert "W_DR1" not in out_ids
    assert "M_HD1" in out_ids
    assert "M_TR1" in out_ids
    assert [x["stage2_rank"] for x in out] == [1, 2, 3, 4]


def test_build_default_services_diversifies_trending_and_segments(tmp_path: Path, monkeypatch) -> None:
    from src.serving import main as serving_main

    rows = []
    segs = ["Ladieswear", "Divided", "Menswear", "Sport"]
    groups = ["Garment Upper body", "Garment Lower body", "Socks & Tights", "Garment Full body"]
    for i in range(24):
        rows.append(
            {
                "article_id": f"SKU_{i:03d}",
                "prod_name": f"Item {i // 2}",
                "product_group_name": groups[i % len(groups)],
                "index_group_name": segs[i % len(segs)],
                "department_name": f"Dept_{i % 6}",
                "avg_price": 0.02 + 0.001 * i,
                "train_tx_count": 100 - i,
            }
        )
    pd.DataFrame(rows).to_parquet(tmp_path / "catalog.parquet", index=False)
    pd.DataFrame({"article_id": ["SKU_000"], "embedding": [[0.1] * 128]}).to_parquet(
        tmp_path / "item_embeddings.parquet", index=False
    )

    monkeypatch.setenv("NEXTPICK_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("NEXTPICK_ALLOW_INMEMORY_REDIS", "1")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    monkeypatch.setattr(serving_main, "QdrantClient", lambda **kwargs: QdrantClient(":memory:"))

    svc = serving_main.build_default_services()
    assert len(svc.trending_fallback) >= 10
    assert len({x["prod_name"] for x in svc.trending_fallback}) == len(svc.trending_fallback)
    assert "Menswear" in svc.segment_catalog
    assert svc.article_segments["SKU_002"] == "Menswear"
