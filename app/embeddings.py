from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from app.config import Settings


class EmbeddingDimensionMismatch(RuntimeError):
    pass


@runtime_checkable
class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def _check_dim(vec: list[float], expected: int) -> list[float]:
    if len(vec) != expected:
        raise EmbeddingDimensionMismatch(
            f"embedding has {len(vec)} dimensions, expected {expected}"
        )
    return vec


class _DimGuard:
    def __init__(self, inner: Embedder, expected_dim: int):
        self._inner = inner
        self._expected = expected_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = self._inner.embed_documents(texts)
        for v in out:
            _check_dim(v, self._expected)
        return out

    def embed_query(self, text: str) -> list[float]:
        return _check_dim(self._inner.embed_query(text), self._expected)


# ---------- Atlas backend ----------


_DEFAULT_ATLAS_URL = "https://ai.mongodb.com/v1/embeddings"


class _RequestsTransport:
    def post(self, url: str, json: dict, headers: dict, timeout: float) -> dict:
        import requests

        resp = requests.post(url, json=json, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()


class _AtlasEmbedder:
    def __init__(self, cfg: Settings, transport: Any | None):
        if not cfg.atlas_api_key:
            raise ValueError("MDB_ATLAS_API_KEY required for atlas embedder")
        self._url = cfg.atlas_embedding_url or _DEFAULT_ATLAS_URL
        self._api_key = cfg.atlas_api_key
        self._doc_model = cfg.embedding_doc_model
        self._query_model = cfg.embedding_query_model
        self._dim = cfg.embedding_dimensions
        self._transport = transport or _RequestsTransport()

    def _embed(self, inputs: list[str], model: str, input_type: str) -> list[list[float]]:
        payload = {
            "model": model,
            "input": inputs,
            "input_type": input_type,
            "output_dimension": self._dim,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = self._transport.post(self._url, json=payload, headers=headers, timeout=30.0)
        return [item["embedding"] for item in body["data"]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, self._doc_model, "document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], self._query_model, "query")[0]


# ---------- Voyage backend ----------


class _VoyageEmbedder:
    def __init__(self, cfg: Settings, client: Any | None):
        if client is None:
            import voyageai

            client = voyageai.Client(api_key=cfg.voyage_api_key)
        self._client = client
        self._doc_model = cfg.embedding_doc_model
        self._query_model = cfg.embedding_query_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(texts, model=self._doc_model, input_type="document")
        return [list(v) for v in result.embeddings]

    def embed_query(self, text: str) -> list[float]:
        result = self._client.embed([text], model=self._query_model, input_type="query")
        return list(result.embeddings[0])


# ---------- Factory ----------


def make_embedder(
    cfg: Settings,
    *,
    atlas_transport: Any | None = None,
    voyage_client: Any | None = None,
) -> Embedder:
    if cfg.embedding_provider == "atlas":
        inner: Embedder = _AtlasEmbedder(cfg, atlas_transport)
    elif cfg.embedding_provider == "voyage":
        inner = _VoyageEmbedder(cfg, voyage_client)
    else:
        raise ValueError(f"unknown embedding provider: {cfg.embedding_provider!r}")
    return _DimGuard(inner, cfg.embedding_dimensions)


__all__: Iterable[str] = (
    "Embedder",
    "EmbeddingDimensionMismatch",
    "make_embedder",
)
