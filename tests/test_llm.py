from __future__ import annotations

import importlib
import sys


class _FakeBedrockClient:
    def __init__(self, deltas: list[str]):
        self.deltas = deltas
        self.calls: list[dict] = []

    def converse_stream(self, **kwargs):
        self.calls.append(kwargs)
        events = [{"contentBlockDelta": {"delta": {"text": d}}} for d in self.deltas] + [
            {"messageStop": {"stopReason": "end_turn"}}
        ]
        return {"stream": iter(events)}


def _settings():
    from app.config import Settings

    return Settings(
        mongo_uri="mongodb+srv://x:y@c.mongodb.net/",
        database="knowledge_base",
        embedding_provider="atlas",
        embedding_dimensions=1024,
        embedding_doc_model="voyage-4",
        embedding_query_model="voyage-4-lite",
        hybrid_vector_weight=0.7,
        hybrid_lexical_weight=0.3,
        retrieval_k=5,
        aws_region="us-east-1",
        bedrock_model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        atlas_api_key="atlas-key",
        atlas_embedding_url=None,
        voyage_api_key=None,
    )


def test_stream_yields_concatenated_deltas():
    from app.llm import BedrockClient

    fake = _FakeBedrockClient(["Hello", ", ", "world"])
    client = BedrockClient(_settings(), boto_client=fake)
    out = list(client.stream("sys", [{"role": "user", "content": [{"text": "hi"}]}]))
    assert "".join(out) == "Hello, world"


def test_stream_passes_system_with_cache_point():
    from app.llm import BedrockClient

    fake = _FakeBedrockClient(["ok"])
    client = BedrockClient(_settings(), boto_client=fake)
    list(client.stream("system prompt", [{"role": "user", "content": [{"text": "hi"}]}]))
    kwargs = fake.calls[0]
    assert kwargs["modelId"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert kwargs["system"][0]["text"] == "system prompt"
    # cache point present so Bedrock prompt caching is enabled on the system prompt
    assert any("cachePoint" in block for block in kwargs["system"])


def test_import_does_not_create_boto_client():
    sys.modules.pop("app.llm", None)
    sentinel: list[str] = []

    class _Watch:
        def client(self, *args, **kwargs):
            sentinel.append("created")
            raise AssertionError("boto3.client must not be called at import time")

    real_boto3 = sys.modules.get("boto3")
    sys.modules["boto3"] = _Watch()  # type: ignore[assignment]
    try:
        importlib.import_module("app.llm")
    finally:
        if real_boto3 is not None:
            sys.modules["boto3"] = real_boto3
        else:
            sys.modules.pop("boto3", None)
    assert sentinel == []
