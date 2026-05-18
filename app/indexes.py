from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

VECTOR_INDEX_NAME = "kb_vector"
SEARCH_INDEX_NAME = "kb_search"


class IndexTimeout(RuntimeError):
    pass


def _vector_index_model(dimensions: int) -> dict:
    return {
        "name": VECTOR_INDEX_NAME,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": dimensions,
                    "similarity": "cosine",
                }
            ]
        },
    }


def _search_index_model() -> dict:
    return {
        "name": SEARCH_INDEX_NAME,
        "type": "search",
        "definition": {
            "mappings": {
                "dynamic": False,
                "fields": {
                    "title": {"type": "string", "analyzer": "lucene.standard"},
                    "text": {"type": "string", "analyzer": "lucene.standard"},
                },
            }
        },
    }


def _existing_index_names(coll: Any) -> dict[str, dict]:
    return {idx["name"]: idx for idx in coll.list_search_indexes()}


def _ensure_collection_exists(coll: Any) -> None:
    """Atlas refuses to list/create search indexes on a missing namespace.

    The collection is implicitly created on first write, but ingest needs the
    indexes BEFORE writing. Insert + delete a placeholder doc to materialise
    the namespace if it doesn't exist yet."""
    try:
        db = coll.database
        if coll.name in db.list_collection_names():
            return
        db.create_collection(coll.name)
    except Exception:  # noqa: BLE001 — fall back to insert-driven creation
        coll.insert_one({"_id": "__bootstrap__"})
        coll.delete_one({"_id": "__bootstrap__"})


def ensure_indexes(
    coll: Any,
    dimensions: int,
    *,
    poll_interval: float = 2.0,
    timeout_s: float = 180.0,
) -> None:
    """Create the vector + Atlas Search indexes if missing, then wait until both are queryable."""
    _ensure_collection_exists(coll)
    existing = _existing_index_names(coll)

    if VECTOR_INDEX_NAME not in existing:
        coll.create_search_index(_vector_index_model(dimensions))
        logger.info("created vector index %s", VECTOR_INDEX_NAME)
    if SEARCH_INDEX_NAME not in existing:
        coll.create_search_index(_search_index_model())
        logger.info("created search index %s", SEARCH_INDEX_NAME)

    deadline = time.monotonic() + timeout_s
    while True:
        statuses = _existing_index_names(coll)
        ready = all(
            statuses.get(name, {}).get("queryable") is True
            for name in (VECTOR_INDEX_NAME, SEARCH_INDEX_NAME)
        )
        if ready:
            return
        if time.monotonic() >= deadline:
            details = [
                (n, statuses.get(n, {}).get("status"))
                for n in (VECTOR_INDEX_NAME, SEARCH_INDEX_NAME)
            ]
            raise IndexTimeout(f"indexes not ready within {timeout_s}s: {details}")
        time.sleep(poll_interval)


__all__ = (
    "IndexTimeout",
    "VECTOR_INDEX_NAME",
    "SEARCH_INDEX_NAME",
    "ensure_indexes",
)
