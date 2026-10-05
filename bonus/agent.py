"""Small hybrid episodic memory joined with a Feast stable profile."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feast import FeatureStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from app.embeddings import Embedder


class HybridMemoryAgent:
    """Store per-user episodes in Qdrant and read stable features from Feast."""

    PROFILE_FEATURES = [
        "user_profile_features:reading_speed_wpm",
        "user_profile_features:preferred_language",
        "user_profile_features:topic_affinity",
        "query_velocity_features:queries_last_hour",
        "query_velocity_features:distinct_topics_24h",
    ]

    def __init__(self, feast_repo_path: str | Path | None = None) -> None:
        self.feast_repo_path = Path(feast_repo_path or ROOT / "app" / "feast_repo").resolve()
        self.store = FeatureStore(repo_path=str(self.feast_repo_path))
        self.embedder = Embedder()
        self.client = QdrantClient(":memory:")
        self.collection = "episodic_memory"
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
        )
        self._memories: list[dict[str, str]] = []

    @staticmethod
    def _tokens(text: str) -> list[str]:
        stop_words = {"và", "là", "của", "trong", "với", "cho", "này", "đó",
                      "một", "khi", "có", "từ", "đến", "ở", "the", "and", "of", "to"}
        tokens = [token.strip(".,;:!?()[]{}\"'") for token in text.casefold().split()]
        return [token for token in tokens if token and token not in stop_words]

    def remember(self, text: str, user_id: str) -> str:
        """Embed and store one episode; return its stable memory identifier."""
        text = text.strip()
        user_id = user_id.strip()
        if not text or not user_id:
            raise ValueError("text and user_id must be non-empty")
        memory_id = str(uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        vector = next(self.embedder.embed([text])).tolist()
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=memory_id,
                vector=vector,
                payload={"memory_id": memory_id, "user_id": user_id,
                         "text": text, "created_at": created_at},
            )],
        )
        self._memories.append({
            "memory_id": memory_id, "user_id": user_id,
            "text": text, "created_at": created_at,
        })
        return memory_id

    def _keyword_ranks(self, query: str, user_id: str) -> list[str]:
        memories = [m for m in self._memories if m["user_id"] == user_id]
        if not memories:
            return []
        index = BM25Okapi([self._tokens(m["text"]) for m in memories])
        scores = index.get_scores(self._tokens(query))
        ordered = sorted(range(len(memories)), key=lambda i: -scores[i])
        return [memories[i]["memory_id"] for i in ordered if scores[i] > 0]

    def _profile(self, user_id: str) -> dict[str, object]:
        values = self.store.get_online_features(
            features=self.PROFILE_FEATURES,
            entity_rows=[{"user_id": user_id}],
        ).to_dict()
        return {
            key: rows[0]
            for key, rows in values.items()
            if key != "user_id" and rows and rows[0] is not None
        }

    def recall(self, query: str, user_id: str, top_k: int = 5,
               rrf_k: int = 60) -> dict[str, object]:
        """Fuse user-filtered vector and BM25 episodes, then attach Feast profile."""
        query = query.strip()
        user_id = user_id.strip()
        if not query or not user_id:
            raise ValueError("query and user_id must be non-empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_vector = next(self.embedder.embed([query])).tolist()
        result = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=Filter(must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]),
            limit=max(top_k * 5, 20),
        )
        scores: dict[str, float] = {}
        metadata: dict[str, dict[str, str | float]] = {}
        for rank, point in enumerate(result.points, start=1):
            memory_id = str(point.payload["memory_id"])
            scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (rrf_k + rank)
            metadata[memory_id] = {
                "memory_id": memory_id,
                "text": str(point.payload["text"]),
                "created_at": str(point.payload["created_at"]),
                "vector_score": float(point.score),
            }
        for rank, memory_id in enumerate(self._keyword_ranks(query, user_id), start=1):
            scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (rrf_k + rank)
            metadata.setdefault(memory_id, next(
                {"memory_id": m["memory_id"], "text": m["text"],
                 "created_at": m["created_at"]} for m in self._memories
                if m["memory_id"] == memory_id
            ))
        memories = []
        for memory_id, score in sorted(scores.items(), key=lambda pair: -pair[1])[:top_k]:
            memories.append({**metadata[memory_id], "rrf_score": score})
        return {
            "user_id": user_id,
            "query": query,
            "memories": memories,
            "profile": self._profile(user_id),
        }
