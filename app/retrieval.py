from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from app.config import Settings
from app.embeddings import Embedder
from app.indexes import SEARCH_INDEX_NAME, VECTOR_INDEX_NAME

logger = logging.getLogger(__name__)

RRF_K = 60


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    title: str
    text: str
    metadata: dict
    score: float = 0.0
    extra: dict = field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[Chunk]: ...


def _parse_major_minor(version: str) -> tuple[int, int]:
    parts = version.split(".")
    try:
        return int(parts[0]), int(parts[1] if len(parts) > 1 else "0")
    except ValueError:
        return 0, 0


def _supports_rank_fusion(version: str) -> bool:
    major, minor = _parse_major_minor(version)
    return (major, minor) >= (8, 0)


def _to_chunk(doc: dict, score: float | None = None) -> Chunk:
    return Chunk(
        id=str(doc.get("_id", doc.get("id", ""))),
        title=str(doc.get("title", "")),
        text=str(doc.get("text", "")),
        metadata=dict(doc.get("metadata") or {}),
        score=float(score if score is not None else doc.get("score", 0.0)),
    )


class MongoRetriever:
    """Hybrid (vector + lexical) retriever for MongoDB Atlas.

    Picks `$rankFusion` on MongoDB 8.0+ (including Atlas M0) and falls back to
    `$unionWith`-style reciprocal rank fusion executed client-side on older
    servers (< 8.0).
    """

    def __init__(self, coll: Any, embedder: Embedder, cfg: Settings):
        self._coll = coll
        self._embedder = embedder
        self._cfg = cfg

        version = coll.database.command("buildInfo").get("version", "0.0")
        self._use_rank_fusion = _supports_rank_fusion(version)
        path = "rankFusion" if self._use_rank_fusion else "unionWith+rrf"
        logger.info("retrieval: %s", path)

    def retrieve(self, query: str, k: int) -> list[Chunk]:
        qvec = self._embedder.embed_query(query)
        if self._use_rank_fusion:
            return self._rank_fusion(query, qvec, k)
        return self._union_with_rrf(query, qvec, k)

    # ---- $rankFusion path (MongoDB 8.0+, incl. Atlas M0) ----

    def _rank_fusion(self, query: str, qvec: list[float], k: int) -> list[Chunk]:
        candidates = max(50, k * 10)
        per_branch = max(k * 4, 20)
        pipeline = [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": {
                            "vector": [
                                {
                                    "$vectorSearch": {
                                        "index": VECTOR_INDEX_NAME,
                                        "path": "embedding",
                                        "queryVector": qvec,
                                        "numCandidates": candidates,
                                        "limit": per_branch,
                                    }
                                }
                            ],
                            "lexical": [
                                {
                                    "$search": {
                                        "index": SEARCH_INDEX_NAME,
                                        "text": {
                                            "query": query,
                                            "path": ["title", "text"],
                                        },
                                    }
                                },
                                {"$limit": per_branch},
                            ],
                        }
                    },
                    "combination": {
                        "weights": {
                            "vector": self._cfg.hybrid_vector_weight,
                            "lexical": self._cfg.hybrid_lexical_weight,
                        }
                    },
                }
            },
            {"$limit": k},
            {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "score"},
                }
            },
        ]
        return [_to_chunk(d) for d in self._coll.aggregate(pipeline)][:k]

    # ---- $unionWith + RRF fallback (MongoDB < 8.0) ----

    def _union_with_rrf(self, query: str, qvec: list[float], k: int) -> list[Chunk]:
        per_branch = max(k * 4, 20)
        candidates = max(50, k * 10)

        vector_pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": qvec,
                    "numCandidates": candidates,
                    "limit": per_branch,
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "text": 1,
                    "metadata": 1,
                }
            },
        ]
        lexical_pipeline = [
            {
                "$search": {
                    "index": SEARCH_INDEX_NAME,
                    "text": {"query": query, "path": ["title", "text"]},
                }
            },
            {"$limit": per_branch},
            {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "text": 1,
                    "metadata": 1,
                }
            },
        ]

        vector_hits = list(self._coll.aggregate(vector_pipeline))
        lexical_hits = list(self._coll.aggregate(lexical_pipeline))

        return _fuse_rrf(
            vector_hits,
            lexical_hits,
            w_v=self._cfg.hybrid_vector_weight,
            w_l=self._cfg.hybrid_lexical_weight,
            k=k,
        )


def _fuse_rrf(
    vector_hits: Iterable[dict],
    lexical_hits: Iterable[dict],
    *,
    w_v: float,
    w_l: float,
    k: int,
) -> list[Chunk]:
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, doc in enumerate(vector_hits, start=1):
        doc_id = str(doc.get("_id", doc.get("id", "")))
        if not doc_id:
            continue
        scores[doc_id] = scores.get(doc_id, 0.0) + w_v / (RRF_K + rank)
        docs.setdefault(doc_id, doc)

    for rank, doc in enumerate(lexical_hits, start=1):
        doc_id = str(doc.get("_id", doc.get("id", "")))
        if not doc_id:
            continue
        scores[doc_id] = scores.get(doc_id, 0.0) + w_l / (RRF_K + rank)
        docs.setdefault(doc_id, doc)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [_to_chunk(docs[doc_id], score=score) for doc_id, score in ranked]


__all__ = ("Chunk", "MongoRetriever", "Retriever")
