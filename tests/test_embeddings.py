from __future__ import annotations

from dataclasses import replace

import pytest

from app.config import Settings
from app.embeddings import EmbeddingDimensionMismatch, make_embedder


def _settings(provider: str, dim: int = 4) -> Settings:
    return Settings(
        mongo_uri="mongodb+srv://x:y@c.mongodb.net/",
        database="knowledge_base",
        embedding_provider=provider,
        embedding_dimensions=dim,
        embedding_doc_model="voyage-4",
        embedding_query_model="voyage-4-lite",
        hybrid_vector_weight=0.7,
        hybrid_lexical_weight=0.3,
        retrieval_k=5,
        aws_region="us-east-1",
        bedrock_model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        atlas_api_key="atlas-key" if provider == "atlas" else None,
        atlas_embedding_url=None,
        voyage_api_key="voyage-key" if provider == "voyage" else None,
    )


class _FakeAtlasTransport:
    def __init__(self, vectors: list[list[float]]):
        self.vectors = vectors
        self.calls: list[dict] = []

    def post(self, url: str, json: dict, headers: dict, timeout: float) -> dict:
        self.calls.append({"url": url, "json": json, "headers": headers})
        n = len(json.get("input", []))
        return {"data": [{"embedding": v} for v in self.vectors[:n]]}


class _FakeVoyageClient:
    def __init__(self, vectors: list[list[float]]):
        self.vectors = vectors
        self.calls: list[dict] = []

    def embed(self, texts, model: str, input_type: str):
        self.calls.append({"texts": list(texts), "model": model, "input_type": input_type})

        class _R:
            embeddings = self.vectors[: len(texts)]

        return _R()


def test_factory_dispatches_atlas():
    cfg = _settings("atlas")
    transport = _FakeAtlasTransport([[0.1, 0.2, 0.3, 0.4]])
    emb = make_embedder(cfg, atlas_transport=transport)
    out = emb.embed_query("hello")
    assert out == [0.1, 0.2, 0.3, 0.4]
    assert transport.calls, "atlas transport must be called"


def test_factory_dispatches_voyage():
    cfg = _settings("voyage")
    fake = _FakeVoyageClient([[1.0, 2.0, 3.0, 4.0]])
    emb = make_embedder(cfg, voyage_client=fake)
    out = emb.embed_query("hello")
    assert out == [1.0, 2.0, 3.0, 4.0]
    assert fake.calls[0]["model"] == "voyage-4-lite"
    assert fake.calls[0]["input_type"] == "query"


def test_voyage_uses_doc_model_for_documents():
    cfg = _settings("voyage")
    fake = _FakeVoyageClient([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    emb = make_embedder(cfg, voyage_client=fake)
    out = emb.embed_documents(["doc-a", "doc-b"])
    assert len(out) == 2
    assert fake.calls[0]["model"] == "voyage-4"
    assert fake.calls[0]["input_type"] == "document"


def test_dimension_mismatch_raises_on_query():
    cfg = _settings("voyage", dim=4)
    fake = _FakeVoyageClient([[0.1, 0.2, 0.3]])  # only 3 dims
    emb = make_embedder(cfg, voyage_client=fake)
    with pytest.raises(EmbeddingDimensionMismatch):
        emb.embed_query("hi")


def test_dimension_mismatch_raises_on_documents():
    cfg = _settings("atlas", dim=4)
    transport = _FakeAtlasTransport([[1.0, 2.0]])
    emb = make_embedder(cfg, atlas_transport=transport)
    with pytest.raises(EmbeddingDimensionMismatch):
        emb.embed_documents(["doc"])


def test_unknown_provider_raises():
    cfg = _settings("atlas")
    cfg = replace(cfg, embedding_provider="other")
    with pytest.raises(ValueError, match="provider"):
        make_embedder(cfg)
