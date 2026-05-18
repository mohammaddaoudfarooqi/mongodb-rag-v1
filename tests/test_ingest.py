from __future__ import annotations

from typing import Any

from scripts.ingest import build_chunk_id, ingest_records


class _FakeColl:
    def __init__(self):
        self.bulk_calls: list[list[Any]] = []
        self.indexes_ensured = 0

    def bulk_write(self, ops, ordered=False):
        self.bulk_calls.append(list(ops))

        class _R:
            upserted_count = len(ops)
            modified_count = 0

        return _R()


class _FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(i)] * 4 for i, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 4


def test_build_chunk_id_is_stable_and_unique():
    a = {"title": "Returns", "metadata": {"source": "policies/returns.pdf"}, "text": "x"}
    b = {"title": "Returns", "metadata": {"source": "policies/returns.pdf"}, "text": "y"}
    c = {"title": "Returns", "metadata": {"source": "policies/other.pdf"}, "text": "x"}
    assert build_chunk_id(a) == build_chunk_id(b)
    assert build_chunk_id(a) != build_chunk_id(c)


def test_ingest_records_upserts_with_embeddings():
    coll = _FakeColl()
    embedder = _FakeEmbedder()
    records = [
        {"title": "T1", "text": "Body 1", "metadata": {"source": "a.pdf"}},
        {"title": "T2", "text": "Body 2", "metadata": {"source": "b.pdf"}},
    ]

    n = ingest_records(coll, embedder, records, batch_size=10)
    assert n == 2
    assert len(coll.bulk_calls) == 1
    assert len(coll.bulk_calls[0]) == 2


def test_ingest_records_batches_large_inputs():
    coll = _FakeColl()
    embedder = _FakeEmbedder()
    records = [
        {"title": f"T{i}", "text": f"Body {i}", "metadata": {"source": f"{i}.pdf"}}
        for i in range(7)
    ]
    n = ingest_records(coll, embedder, records, batch_size=3)
    assert n == 7
    assert [len(b) for b in coll.bulk_calls] == [3, 3, 1]
