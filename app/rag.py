"""Pure RAG pipeline: retrieve → format → stream answer.

Kept Streamlit-free so it can be exercised directly from tests and the smoke
script."""

from __future__ import annotations

from collections.abc import Iterator

from app.llm import LLMClient
from app.retrieval import Chunk, Retriever

CANNED_NO_INFO = "I don't have that information."

SYSTEM_PROMPT = (
    "You are a customer-support assistant. Answer the user's question using "
    "ONLY the information in the <context> block below. If the context does "
    f"not contain the answer, reply with exactly: {CANNED_NO_INFO}\n"
    "Do not invent facts. Cite the source (metadata.source) when helpful, but "
    "never fabricate a citation."
)


def format_context(chunks: list[Chunk]) -> str:
    if not chunks:
        return "<context>\n(no relevant context retrieved)\n</context>"
    parts = ["<context>"]
    for i, c in enumerate(chunks, start=1):
        source = c.metadata.get("source", "unknown")
        parts.append(f"[{i}] title={c.title} | source={source}")
        parts.append(c.text)
        parts.append("")
    parts.append("</context>")
    return "\n".join(parts)


def build_messages(question: str, chunks: list[Chunk]) -> list[dict]:
    user_text = f"{format_context(chunks)}\n\nQuestion: {question}"
    return [{"role": "user", "content": [{"text": user_text}]}]


def answer_stream(
    retriever: Retriever,
    llm: LLMClient,
    question: str,
    k: int,
) -> tuple[list[Chunk], Iterator[str]]:
    chunks = retriever.retrieve(question, k=k)
    messages = build_messages(question, chunks)
    return chunks, llm.stream(SYSTEM_PROMPT, messages)


def answer(retriever: Retriever, llm: LLMClient, question: str, k: int) -> tuple[list[Chunk], str]:
    chunks, stream = answer_stream(retriever, llm, question, k)
    return chunks, "".join(stream)


__all__ = (
    "CANNED_NO_INFO",
    "SYSTEM_PROMPT",
    "answer",
    "answer_stream",
    "build_messages",
    "format_context",
)
