from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.rag import CANNED_NO_INFO
from app.retrieval import Chunk


class FakeRetriever:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[Chunk]:
        self.calls.append((query, k))
        return self._chunks[:k]


class FakeLLM:
    """Tiny rule-driven LLM for golden / fallback tests.

    If any chunk has metadata.source matching one referenced by the user
    prompt, echo the source path; otherwise emit the canned no-info phrase.
    """

    def __init__(self, *, response: str | None = None):
        self.response = response
        self.calls: list[tuple[str, list[dict]]] = []

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        self.calls.append((system, messages))
        if self.response is not None:
            yield self.response
            return
        # Heuristic: if context is empty, return the canned no-info phrase.
        text = messages[0]["content"][0]["text"]
        if "(no relevant context retrieved)" in text:
            yield CANNED_NO_INFO
            return
        # Otherwise, echo the first source path found in context, if any.
        marker = "source="
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.find("\n", start)
            source = text[start:end].strip() if end > start else text[start:].strip()
            yield f"Per {source}: see context above."
            return
        yield CANNED_NO_INFO


@pytest.fixture
def returns_chunk() -> Chunk:
    return Chunk(
        id="returns-1",
        title="Standard Return Policy",
        text=(
            "Unopened merchandise may be returned within 30 days of delivery for a "
            "full refund. Opened electronics may be returned within 15 days subject "
            "to a 15% restocking fee."
        ),
        metadata={
            "source": "policies/returns.pdf",
            "category": "returns",
            "audience": "agent",
        },
        score=0.9,
    )


@pytest.fixture
def fake_retriever(returns_chunk: Chunk) -> FakeRetriever:
    return FakeRetriever([returns_chunk])


@pytest.fixture
def empty_retriever() -> FakeRetriever:
    return FakeRetriever([])


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
