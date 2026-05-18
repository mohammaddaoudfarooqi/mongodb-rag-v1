"""End-to-end smoke test against live Atlas + Bedrock + embedding provider.

Runs the real production wiring twice:

    1. Golden path:    "What is the return window?"
                       → asserts the answer cites `policies/returns.pdf`.
    2. Out-of-scope:   "What is the airspeed of an unladen swallow?"
                       → asserts the answer is the canned phrase.

Prints both transcripts. Exits 0 on success, 1 on any assertion or runtime
failure. No UI clicks, no fakes.

Prerequisites:
    - .env populated with all required vars (see .env.sample).
    - Knowledge base ingested:  uv run python scripts/ingest.py
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
from pymongo import MongoClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.embeddings import make_embedder  # noqa: E402
from app.indexes import ensure_indexes  # noqa: E402
from app.llm import BedrockClient  # noqa: E402
from app.rag import CANNED_NO_INFO, answer  # noqa: E402
from app.retrieval import MongoRetriever  # noqa: E402

logger = logging.getLogger("smoke")

GOLDEN_QUESTION = "What is the standard return window for unopened merchandise?"
GOLDEN_EXPECTED_CITATION = "policies/returns.pdf"

OUT_OF_SCOPE_QUESTION = (
    "What is the airspeed of an unladen European swallow during the autumn migration?"
)


def _format_sources(chunks: Iterable) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(
            f"  [{i}] {c.title} — source={c.metadata.get('source', 'unknown')} score={c.score:.4f}"
        )
    return "\n".join(lines) if lines else "  (no chunks)"


def main() -> int:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    cfg = Settings.load()
    client = MongoClient(cfg.mongo_uri)
    coll = client[cfg.database]["kb_chunks"]

    ensure_indexes(coll, dimensions=cfg.embedding_dimensions)

    embedder = make_embedder(cfg)
    retriever = MongoRetriever(coll, embedder, cfg)
    llm = BedrockClient(cfg)

    failures: list[str] = []

    # ---- Golden path ----------------------------------------------------
    print("\n" + "=" * 72)
    print(f"GOLDEN  Q: {GOLDEN_QUESTION}")
    print("=" * 72)
    chunks, text = answer(retriever, llm, GOLDEN_QUESTION, k=cfg.retrieval_k)
    print(f"A: {text}\n")
    print("Sources:")
    print(_format_sources(chunks))

    cited = any(c.metadata.get("source") == GOLDEN_EXPECTED_CITATION for c in chunks)
    text_cites = GOLDEN_EXPECTED_CITATION in text
    if not (cited or text_cites):
        failures.append(
            f"golden path did not cite {GOLDEN_EXPECTED_CITATION!r} in chunks or answer"
        )
    else:
        print(f"\n✓ golden path cited {GOLDEN_EXPECTED_CITATION}")

    # ---- Fallback path --------------------------------------------------
    print("\n" + "=" * 72)
    print(f"FALLBACK  Q: {OUT_OF_SCOPE_QUESTION}")
    print("=" * 72)
    chunks, text = answer(retriever, llm, OUT_OF_SCOPE_QUESTION, k=cfg.retrieval_k)
    print(f"A: {text}\n")
    print("Sources:")
    print(_format_sources(chunks))

    if CANNED_NO_INFO not in text.strip():
        failures.append(
            f"fallback path did not return canned reply {CANNED_NO_INFO!r}; got: {text!r}"
        )
    else:
        print(f"\n✓ fallback path returned canned reply: {CANNED_NO_INFO!r}")

    # ---- Verdict --------------------------------------------------------
    print("\n" + "=" * 72)
    if failures:
        for f in failures:
            print(f"✗ {f}")
        print(f"FAIL ({len(failures)} assertion(s))")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
