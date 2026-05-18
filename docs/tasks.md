# Tasks — Hybrid Search RAG with MongoDB

Sequenced for TDD; each task lists tests first.

## T0 — Bootstrap project scaffold

- pyproject.toml (uv-managed): python `>=3.11`, deps: `streamlit`, `pymongo`,
  `boto3`, `voyageai`, `python-dotenv`, `pytest`, `ruff`.
- `.gitignore` (`.env`, `.mcp.json`, `__pycache__/`, `.venv/`).
- `.env.sample` with every variable from prompt.md.
- ruff config (line length 100, default lints).
- Acceptance: `uv sync && uv run pytest -q` runs (no tests yet → 0 collected).

## T1 — Verify boundaries (live)

- MongoDB MCP: `list-databases`.
- Bedrock: one minimal ConverseStream call.
- Stop on either failing.

## T2 — `app/config.py`

- Tests first: missing required var raises; provider switch resolved.
- Implements `Settings.load()` from env.
- Maps to: REQ-060, REQ-061, REQ-063.

## T3 — `app/embeddings.py`

- Tests first: dim mismatch raises; factory dispatches by provider; query vs
  doc model selection.
- AtlasEmbedder + VoyageEmbedder + factory.
- Maps to: REQ-010 .. REQ-013.

## T4 — `app/llm.py`

- Tests first: stream yields concatenated deltas; system prompt carries cache
  point; `import app.llm` does not construct boto3 client.
- BedrockClient.stream via ConverseStream.
- Maps to: REQ-040 .. REQ-044, REQ-064.

## T5 — `app/indexes.py`

- Tests first: ensure functions are idempotent on a fake collection; poll
  exits when index reports READY/queryable; raises on timeout.
- Vector and Atlas Search index creation + polling.
- Maps to: REQ-020 .. REQ-022.

## T6 — `app/retrieval.py`

- Tests first: version >= 8.1 picks `$rankFusion` path; < 8.1 picks unionWith;
  startup logs exactly one `retrieval: …` line; RRF math correct on a small
  hand-crafted result set; weights 0.7 / 0.3 from env override correctly.
- MongoRetriever.
- Maps to: REQ-030 .. REQ-034, REQ-070.

## T7 — `app/main.py`

- Tests first: `import app.main` creates no clients (REQ-064);
  `get_app(fake_retriever, fake_llm)` is callable; sources panel data
  assembled correctly from chunks.
- Streamlit page uses get_app factory.
- Maps to: REQ-050 .. REQ-053, REQ-064.

## T8 — `scripts/ingest.py`

- Tests first: idempotent upsert key derivation; ensures indexes before write.
- One-shot ingest using doc embedder.
- Maps to: REQ-001 .. REQ-004.

## T9 — `tests/test_e2e_fakes.py` (golden + fallback)

- Golden-path: question about returns → answer cites `policies/returns.pdf`,
  using FakeRetriever returning the returns chunk and FakeLLM that echoes
  the source.
- Fallback: out-of-scope question with FakeRetriever returning unrelated
  chunks → answer is exactly *"I don't have that information."*
- Maps to: REQ-NF-003, REQ-042, REQ-043.

## T10 — CI: `.github/workflows/ci.yml`

- ruff + pytest matrix on Python 3.11. Pin actions to commit SHAs.
- No live Atlas / Bedrock / Voyage secrets needed.
- Maps to: REQ-NF-001, REQ-NF-002.

## T11 — `scripts/smoke.py`

- Production wiring. Run twice: golden (assert citation
  `policies/returns.pdf`) + out-of-scope (assert exact canned reply). Print
  transcripts.
- Maps to: REQ-NF-003 (live).

## T12 — README

- 5-min quickstart, env vars table, ingest, run, CI, troubleshooting.

## T13 — Final verification

- `uv run streamlit run app/main.py` (confirm boot, no traceback).
- `uv run python scripts/smoke.py` (capture transcripts).

## Traceability matrix

| Req | Test IDs | Status |
| --- | --- | --- |
| REQ-001 .. REQ-004 | T8 | Not started |
| REQ-010 .. REQ-013 | T3 | Not started |
| REQ-020 .. REQ-022 | T5 | Not started |
| REQ-030 .. REQ-034 | T6 | Not started |
| REQ-040 .. REQ-044 | T4 | Not started |
| REQ-050 .. REQ-053 | T7 | Not started |
| REQ-060 .. REQ-064 | T2, T7 | Not started |
| REQ-070 | T6 | Not started |
| REQ-NF-001 .. REQ-NF-003 | T9, T10, T11 | Not started |
