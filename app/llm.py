from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from app.config import Settings


@runtime_checkable
class LLMClient(Protocol):
    def stream(self, system: str, messages: list[dict]) -> Iterator[str]: ...


class BedrockClient:
    def __init__(self, cfg: Settings, *, boto_client: Any | None = None):
        self._model_id = cfg.bedrock_model_id
        self._region = cfg.aws_region
        self._client = boto_client  # constructed lazily below

    def _ensure_client(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self._region)
        return self._client

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        client = self._ensure_client()
        resp = client.converse_stream(
            modelId=self._model_id,
            system=[
                {"text": system},
                {"cachePoint": {"type": "default"}},
            ],
            messages=messages,
            inferenceConfig={"maxTokens": 1024, "temperature": 0.0},
        )
        for event in resp["stream"]:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                text = delta.get("text", "")
                if text:
                    yield text
            elif "messageStop" in event:
                return


__all__ = ("BedrockClient", "LLMClient")
