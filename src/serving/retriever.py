"""Stage-1 Retriever: ONNX User Tower inference + single-hop Qdrant HNSW search."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

from src.index.qdrant_loader import COLLECTION_NAME


class Stage1Retriever:
    """Computes live user embeddings via ONNX Runtime and queries Qdrant with payload."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        onnx_path: Path | None,
        item_vectors: dict[str, np.ndarray],
        collection_name: str = COLLECTION_NAME,
        max_seq_len: int = 20,
        embed_dim: int = 128,
    ) -> None:
        self._qdrant = qdrant_client
        self._collection = collection_name
        self._item_vectors = item_vectors
        self._max_seq_len = max_seq_len
        self._embed_dim = embed_dim
        self._ort = (
            ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            if onnx_path and onnx_path.exists()
            else None
        )

    def encode_user(self, article_ids_recent_first: Sequence[str], sales_channel_id: int = 2) -> np.ndarray:
        """Assemble (1, L, 128) history tensor and run ONNX User Tower (or fallback mean)."""
        vecs = np.zeros((1, self._max_seq_len, self._embed_dim), dtype=np.float32)
        mask = np.zeros((1, self._max_seq_len), dtype=np.float32)

        valid = [self._item_vectors[a] for a in article_ids_recent_first if a in self._item_vectors][: self._max_seq_len]
        if valid:
            vecs[0, : len(valid)] = np.stack(valid, axis=0)
            mask[0, : len(valid)] = 1.0

        if self._ort is not None:
            chan = np.array([sales_channel_id], dtype=np.int64)
            out = self._ort.run(
                ["user_emb"],
                {"hist_item_vecs": vecs, "hist_mask": mask, "channel_id": chan},
            )[0]
            return out[0].astype(np.float32)

        # Fallback if ONNX artifact isn't loaded (e.g. lightweight unit test)
        pooled = vecs[0].mean(axis=0)
        norm = float(np.linalg.norm(pooled))
        return (pooled / norm).astype(np.float32) if norm > 1e-6 else pooled

    def search_candidates(
        self,
        user_vector: np.ndarray,
        top_k: int = 100,
        product_group: str | None = None,
        max_price: float | None = None,
    ) -> list[dict]:
        """Single-hop Qdrant HNSW search returning Top-K candidates with full payload."""
        must_filters: list[FieldCondition] = [
            FieldCondition(key="in_stock", match=MatchValue(value=True)),
        ]
        if product_group:
            must_filters.append(FieldCondition(key="product_group_name", match=MatchValue(value=product_group)))
        if max_price is not None:
            must_filters.append(FieldCondition(key="price", range=Range(lte=float(max_price))))

        res = self._qdrant.query_points(
            collection_name=self._collection,
            query=user_vector.tolist(),
            query_filter=Filter(must=must_filters),
            limit=top_k,
            with_payload=True,
        )
        candidates: list[dict] = []
        for rank_idx, pt in enumerate(res.points, start=1):
            payload = dict(pt.payload or {})
            payload["stage1_rank"] = rank_idx
            payload["stage1_cosine_score"] = float(pt.score)
            candidates.append(payload)
        return candidates
