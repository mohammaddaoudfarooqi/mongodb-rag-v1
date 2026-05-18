"""End-to-end golden + fallback paths against in-memory fakes."""

from __future__ import annotations

from app.rag import CANNED_NO_INFO, answer


def test_golden_path_cites_returns_pdf(fake_retriever, fake_llm):
    chunks, text = answer(fake_retriever, fake_llm, "What is the return window?", k=5)

    assert len(chunks) == 1
    assert chunks[0].metadata["source"] == "policies/returns.pdf"
    assert "policies/returns.pdf" in text


def test_fallback_path_returns_canned_when_no_context(empty_retriever, fake_llm):
    chunks, text = answer(
        empty_retriever, fake_llm, "What is the airspeed of an unladen swallow?", k=5
    )
    assert chunks == []
    assert text == CANNED_NO_INFO


def test_retriever_invoked_with_requested_k(fake_retriever, fake_llm):
    answer(fake_retriever, fake_llm, "anything", k=3)
    assert fake_retriever.calls == [("anything", 3)]


def test_system_prompt_carries_canned_directive(fake_retriever, fake_llm):
    answer(fake_retriever, fake_llm, "anything", k=5)
    system, _ = fake_llm.calls[0]
    assert CANNED_NO_INFO in system
