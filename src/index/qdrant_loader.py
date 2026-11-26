"""Batch loader for indexing NextPick item embeddings and metadata payloads into Qdrant."""

from __future__ import annotations

import argparse
import logging
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)

COLLECTION_NAME = "nextpick_items"


def stable_point_id(article_id: str) -> str:
    """Deterministic UUIDv5 derived from article_id so re-indexing never shifts IDs."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"nextpick:{article_id}"))


def build_image_url(article_id: str) -> str:
    """Standard H&M article image path helper (can be overridden by catalog image_url)."""
    padded = str(article_id).zfill(10)
    return f"https://images.hm.com/assets/{padded[:3]}/{padded}.jpg"


def ensure_collection(
    client: QdrantClient,
    collection: str = COLLECTION_NAME,
    vector_size: int = 128,
    recreate: bool = False,
) -> None:
    """Ensure Qdrant collection exists non-destructively (verify dimensions, upsert safely)."""
    if client.collection_exists(collection):
        if recreate:
            client.delete_collection(collection)
        else:
            info = client.get_collection(collection)
            existing_cfg = info.config.params.vectors
            existing_size = getattr(existing_cfg, "size", None)
            if existing_size is not None and existing_size != vector_size:
                raise ValueError(
                    f"Collection '{collection}' vector size mismatch: expected {vector_size}, got {existing_size}"
                )
            return

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
    )
    client.create_payload_index(collection, "product_group_name", PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection, "price", PayloadSchemaType.FLOAT)


def _delete_stale_points(client: QdrantClient, collection: str, valid_ids: set[str]) -> int:
    """Remove points in Qdrant that are no longer present in the current catalog."""
    stale_ids: list[str] = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        for pt in points:
            pid = str(pt.id)
            if pid not in valid_ids:
                stale_ids.append(pid)
        if next_offset is None:
            break
        offset = next_offset

    for start in range(0, len(stale_ids), 512):
        batch = stale_ids[start : start + 512]
        client.delete(collection_name=collection, points_selector=PointIdsList(points=batch))
    return len(stale_ids)


def validate_catalog_and_embeddings(cat_df: pd.DataFrame, emb_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fail-fast validation before any Qdrant writes or stale pruning."""
    if cat_df.empty or emb_df.empty:
        raise ValueError("Catalog or embeddings table is empty.")

    if cat_df["article_id"].isna().any() or emb_df["article_id"].isna().any():
        raise ValueError("Null article_id values found in catalog or embeddings.")

    cat_ids = cat_df["article_id"].astype(str).str.strip()
    emb_ids = emb_df["article_id"].astype(str).str.strip()
    if (cat_ids == "").any() or (emb_ids == "").any():
        raise ValueError("Empty or whitespace-only article_id values found.")
    if not cat_ids.is_unique:
        raise ValueError("Duplicate article_id values found in catalog.")
    if not emb_ids.is_unique:
        raise ValueError("Duplicate article_id values found in embeddings.")

    if "avg_price" in cat_df.columns:
        prices = pd.to_numeric(cat_df["avg_price"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(prices).all() or (prices < 0.0).any():
            raise ValueError("Invalid catalog avg_price values (must be finite and non-negative).")

    missing = set(cat_ids) - set(emb_ids)
    if missing:
        raise ValueError(f"Missing embeddings for {len(missing)} catalog items (e.g. {sorted(missing)[:3]}).")

    extra = set(emb_ids) - set(cat_ids)
    if extra:
        logging.getLogger("nextpick").warning(
            "Ignoring %d extra embedding IDs not present in catalog.", len(extra)
        )

    merged = cat_df.copy()
    merged["article_id"] = cat_ids
    emb_clean = emb_df.copy()
    emb_clean["article_id"] = emb_ids
    merged = merged.merge(emb_clean, on="article_id", how="inner")

    first_vec = np.asarray(merged["embedding"].iloc[0], dtype=np.float32)
    if first_vec.ndim != 1 or first_vec.shape[0] <= 0:
        raise ValueError(f"Invalid embedding vector shape: {first_vec.shape}")
    vec_size = int(first_vec.shape[0])
    for art_id, raw_vec in zip(merged["article_id"], merged["embedding"]):
        arr = np.asarray(raw_vec, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != vec_size:
            raise ValueError(f"Dimension mismatch for {art_id}: expected {vec_size}, got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite (NaN/Inf) embedding detected for {art_id}.")

    return merged, vec_size


def load_catalog_to_qdrant(
    client: QdrantClient,
    catalog_parquet: Path,
    embeddings_parquet: Path,
    collection: str = COLLECTION_NAME,
    batch_size: int = 256,
    recreate: bool = False,
    prune_stale: bool = True,
) -> int:
    """Validate inputs, then non-destructively upsert catalog + embeddings into Qdrant."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    cat_df = pd.read_parquet(catalog_parquet)
    emb_df = pd.read_parquet(embeddings_parquet)
    merged, vec_size = validate_catalog_and_embeddings(cat_df, emb_df)
    ensure_collection(client, collection=collection, vector_size=vec_size, recreate=recreate)

    total = 0
    valid_ids: set[str] = set()
    points: list[PointStruct] = []
    for row in merged.itertuples(index=False):
        art_id = str(row.article_id)
        pid = stable_point_id(art_id)
        valid_ids.add(pid)
        payload = {
            "article_id": art_id,
            "prod_name": str(getattr(row, "prod_name", "")),
            "product_type_name": str(getattr(row, "product_type_name", "unknown")),
            "product_group_name": str(getattr(row, "product_group_name", "unknown")),
            "department_name": str(getattr(row, "department_name", "unknown")),
            "colour_group_name": str(getattr(row, "colour_group_name", "unknown")),
            "price": float(getattr(row, "avg_price", 0.0)),
            "train_tx_count": int(getattr(row, "train_tx_count", 0)),
            "image_url": str(getattr(row, "image_url", "")) or build_image_url(art_id),
        }
        points.append(
            PointStruct(
                id=pid,
                vector=[float(x) for x in row.embedding],
                payload=payload,
            )
        )
        if len(points) >= batch_size:
            client.upsert(collection_name=collection, points=points)
            total += len(points)
            points.clear()

    if points:
        client.upsert(collection_name=collection, points=points)
        total += len(points)

    if prune_stale and not recreate:
        _delete_stale_points(client, collection, valid_ids)

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Index NextPick item embeddings into Qdrant.")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY", ""))
    parser.add_argument("--catalog", type=Path, default=Path("artifacts/catalog.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("artifacts/item_embeddings.parquet"))
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection before indexing.")
    parser.add_argument("--no-prune", action="store_true", help="Skip deleting stale points not in catalog.")
    args = parser.parse_args()

    client = QdrantClient(
        url=args.qdrant_url,
        api_key=args.qdrant_api_key or None,
        timeout=60,
        check_compatibility=False,
    )
    count = load_catalog_to_qdrant(
        client, args.catalog, args.embeddings, recreate=args.recreate, prune_stale=not args.no_prune
    )
    print(f"Indexed {count} items into Qdrant collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
