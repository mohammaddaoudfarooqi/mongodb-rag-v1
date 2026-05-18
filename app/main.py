"""Streamlit entrypoint. Imports MUST be side-effect free; production wiring
runs only inside ``main()``."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.config import Settings  # noqa: E402
from app.llm import LLMClient  # noqa: E402
from app.rag import answer_stream  # noqa: E402
from app.retrieval import Retriever  # noqa: E402

PAGE_TITLE = "Hybrid Search RAG (MongoDB Atlas)"


def get_app(retriever: Retriever, llm: LLMClient, *, k: int = 5) -> Callable[[], None]:
    """Return a render function bound to the given retriever + LLM.

    Streamlit is imported lazily so unit tests can invoke this factory without
    pulling Streamlit's runtime."""

    def render() -> None:
        import streamlit as st

        st.set_page_config(page_title=PAGE_TITLE, page_icon="🔎", layout="centered")
        st.title(PAGE_TITLE)
        st.caption("Customer-support answers grounded in your knowledge base.")

        with st.form("ask"):
            question = st.text_input("Ask a question", placeholder="What is the return window?")
            submitted = st.form_submit_button("Submit")

        if not submitted or not question.strip():
            return

        chunks, stream = answer_stream(retriever, llm, question.strip(), k=k)
        st.write_stream(stream)

        with st.expander(f"Sources ({len(chunks)})", expanded=False):
            if not chunks:
                st.info("No sources retrieved.")
            for i, chunk in enumerate(chunks, start=1):
                source = chunk.metadata.get("source", "unknown")
                st.markdown(
                    f"**[{i}] {chunk.title}**  \n"
                    f"`source`: {source}  \n"
                    f"`fusion score`: {chunk.score:.4f}"
                )

    return render


def main() -> None:
    """Production wiring: load settings, connect Mongo, ensure indexes, run UI."""
    import streamlit as st
    from pymongo import MongoClient

    from app.embeddings import make_embedder
    from app.indexes import ensure_indexes
    from app.llm import BedrockClient
    from app.retrieval import MongoRetriever

    cfg = Settings.load()

    @st.cache_resource(show_spinner=False)
    def _bootstrap(_cfg_signature: str) -> tuple[Any, Any, Any]:
        client = MongoClient(cfg.mongo_uri)
        coll = client[cfg.database]["kb_chunks"]
        embedder = make_embedder(cfg)
        with st.spinner("Preparing search indexes…"):
            ensure_indexes(coll, dimensions=cfg.embedding_dimensions)
        retriever = MongoRetriever(coll, embedder, cfg)
        llm = BedrockClient(cfg)
        return retriever, llm, client

    retriever, llm, _client = _bootstrap(cfg.mongo_uri + cfg.embedding_provider)
    get_app(retriever, llm, k=cfg.retrieval_k)()


if __name__ == "__main__":
    main()
