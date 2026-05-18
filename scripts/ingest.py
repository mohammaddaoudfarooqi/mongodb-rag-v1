"""Ingest knowledge_base.json → MongoDB Atlas (kb_chunks).

Usage:
    uv run python scripts/ingest.py
    uv run python scripts/ingest.py --path /custom/path/to/knowledge_base.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pymongo import UpdateOne  # noqa: E402

from app.config import Settings  # noqa: E402
from app.embeddings import Embedder, make_embedder  # noqa: E402
from app.indexes import ensure_indexes  # noqa: E402

logger = logging.getLogger("ingest")


def build_chunk_id(record: dict) -> str:
    source = (record.get("metadata") or {}).get("source", "")
    title = record.get("title", "")
    digest = hashlib.sha1(f"{source}|{title}".encode()).hexdigest()
    return digest


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list at {path}, got {type(data).__name__}")
    for r in data:
        if "title" not in r or "text" not in r:
            raise ValueError(f"record missing required fields: {r!r}")
    return data


def _batches(items: list[dict], batch_size: int) -> Iterator[list[dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def ingest_records(
    coll: Any,
    embedder: Embedder,
    records: Iterable[dict],
    *,
    batch_size: int = 16,
) -> int:
    records = list(records)
    n_total = 0
    for batch in _batches(records, batch_size):
        texts = [f"{r.get('title', '')}\n\n{r.get('text', '')}".strip() for r in batch]
        embeddings = embedder.embed_documents(texts)
        ops: list[UpdateOne] = []
        for record, vector in zip(batch, embeddings, strict=True):
            doc_id = build_chunk_id(record)
            doc = {
                "_id": doc_id,
                "title": record.get("title"),
                "text": record.get("text"),
                "metadata": record.get("metadata") or {},
                "embedding": vector,
            }
            ops.append(UpdateOne({"_id": doc_id}, {"$set": doc}, upsert=True))
        if ops:
            coll.bulk_write(ops, ordered=False)
            n_total += len(ops)
    return n_total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "knowledge_base.json",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    cfg = Settings.load()
    records = load_records(args.path)
    logger.info("loaded %d records from %s", len(records), args.path)

    from pymongo import MongoClient  # imported here to keep module import side-effect free

    client = MongoClient(cfg.mongo_uri)
    coll = client[cfg.database]["kb_chunks"]
    ensure_indexes(coll, dimensions=cfg.embedding_dimensions)

    embedder = make_embedder(cfg)
    n = ingest_records(coll, embedder, records, batch_size=args.batch_size)
    logger.info("ingested %d chunks into %s.kb_chunks", n, cfg.database)
    return 0


if __name__ == "__main__":
    sys.exit(main())
