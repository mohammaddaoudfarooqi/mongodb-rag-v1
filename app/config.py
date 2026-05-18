from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

_VALID_PROVIDERS = ("atlas", "voyage")


@dataclass(frozen=True, slots=True)
class Settings:
    mongo_uri: str
    database: str
    embedding_provider: str
    embedding_dimensions: int
    embedding_doc_model: str
    embedding_query_model: str
    hybrid_vector_weight: float
    hybrid_lexical_weight: float
    retrieval_k: int
    aws_region: str
    bedrock_model_id: str
    atlas_api_key: str | None
    atlas_embedding_url: str | None
    voyage_api_key: str | None

    @classmethod
    def load(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = env if env is not None else os.environ

        mongo_uri = env.get("MDB_MCP_CONNECTION_STRING", "").strip()
        if not mongo_uri:
            raise ValueError("MDB_MCP_CONNECTION_STRING is required")

        provider = env.get("EMBEDDING_PROVIDER", "atlas").strip().lower()
        if provider not in _VALID_PROVIDERS:
            raise ValueError(
                f"EMBEDDING_PROVIDER must be one of {_VALID_PROVIDERS}; got {provider!r}"
            )

        atlas_key = (env.get("MDB_ATLAS_API_KEY") or "").strip() or None
        voyage_key = (env.get("VOYAGE_API_KEY") or "").strip() or None
        if provider == "atlas" and not atlas_key:
            raise ValueError("MDB_ATLAS_API_KEY is required when EMBEDDING_PROVIDER=atlas")
        if provider == "voyage" and not voyage_key:
            raise ValueError("VOYAGE_API_KEY is required when EMBEDDING_PROVIDER=voyage")

        return cls(
            mongo_uri=mongo_uri,
            database=env.get("MDB_DATABASE", "knowledge_base"),
            embedding_provider=provider,
            embedding_dimensions=int(env.get("EMBEDDING_DIMENSIONS", "1024")),
            embedding_doc_model=env.get("EMBEDDING_DOC_MODEL", "voyage-4"),
            embedding_query_model=env.get("EMBEDDING_QUERY_MODEL", "voyage-4-lite"),
            hybrid_vector_weight=float(env.get("HYBRID_VECTOR_WEIGHT", "0.7")),
            hybrid_lexical_weight=float(env.get("HYBRID_LEXICAL_WEIGHT", "0.3")),
            retrieval_k=int(env.get("RETRIEVAL_K", "5")),
            aws_region=env.get("AWS_REGION", "us-east-1"),
            bedrock_model_id=env.get(
                "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
            ),
            atlas_api_key=atlas_key,
            atlas_embedding_url=(env.get("MDB_ATLAS_EMBEDDING_URL") or "").strip() or None,
            voyage_api_key=voyage_key,
        )
