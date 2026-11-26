"""Batch loader for indexing NextPick item embeddings and full metadata payloads into Qdrant."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

COLLECTION_NAME = "nextpick_items"


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
    client.create_payload_index(collection, "in_stock", PayloadSchemaType.BOOL)
    client.create_payload_index(collection, "product_group_name", PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection, "price", PayloadSchemaType.FLOAT)


def load_catalog_to_qdrant(
    client: QdrantClient,
    catalog_parquet: Path,
    embeddings_parquet: Path,
    collection: str = COLLECTION_NAME,
    batch_size: int = 256,
    recreate: bool = False,
) -> int:
    """Join catalog metadata with 128-d embeddings and non-destructively upsert into Qdrant."""
    cat_df = pd.read_parquet(catalog_parquet)
    emb_df = pd.read_parquet(embeddings_parquet)
    merged = cat_df.merge(emb_df, on="article_id", how="inner")

    if merged.empty:
        return 0

    vec_size = len(merged["embedding"].iloc[0])
    ensure_collection(client, collection=collection, vector_size=vec_size, recreate=recreate)

    total = 0
    points: list[PointStruct] = []
    for idx, row in enumerate(merged.itertuples(index=False)):
        payload = {
            "article_id": str(row.article_id),
            "prod_name": str(getattr(row, "prod_name", "")),
            "product_type_name": str(getattr(row, "product_type_name", "unknown")),
            "product_group_name": str(getattr(row, "product_group_name", "unknown")),
            "department_name": str(getattr(row, "department_name", "unknown")),
            "colour_group_name": str(getattr(row, "colour_group_name", "unknown")),
            "price": float(getattr(row, "avg_price", 0.0)),
            "train_tx_count": int(getattr(row, "train_tx_count", 0)),
            "image_url": str(getattr(row, "image_url", "")) or build_image_url(str(row.article_id)),
            "in_stock": bool(getattr(row, "in_stock", True)),
        }
        points.append(
            PointStruct(
                id=idx + 1,
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

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Index NextPick item embeddings into Qdrant.")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY", ""))
    parser.add_argument("--catalog", type=Path, default=Path("artifacts/catalog.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("artifacts/item_embeddings.parquet"))
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection before indexing.")
    args = parser.parse_args()

    client = QdrantClient(
        url=args.qdrant_url,
        api_key=args.qdrant_api_key or None,
        timeout=60,
        check_compatibility=False,
    )
    count = load_catalog_to_qdrant(client, args.catalog, args.embeddings, recreate=args.recreate)
    print(f"Indexed {count} items into Qdrant collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
