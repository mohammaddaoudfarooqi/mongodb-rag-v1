# MongoDB Hybrid Search RAG

A small, production-shaped retrieval-augmented generation app built on
**MongoDB Atlas hybrid search** (vector + full-text rank fusion), Anthropic
**Claude Haiku 4.5 on Bedrock**, and **Voyage** embeddings. The user types a
customer-support question; the answer is grounded in a seed knowledge base and
streamed back with cited sources.

```text
Streamlit UI ── Retriever (Atlas $rankFusion / $unionWith+RRF) ── Bedrock (ConverseStream)
                    │                                                  │
              embeddings (voyage-4 docs / voyage-4-lite queries)   prompt-cached system prompt
```

## Quickstart (≈ 5 minutes)

You need an Atlas M0 cluster, AWS credentials with Bedrock access in
`us-east-1`, and a [Voyage / Atlas Model API key].

```bash
# 1. clone & install (uv handles Python 3.11 + deps)
git clone <this-repo> && cd mongodb-rag
uv sync

# 2. configure
cp .env.sample .env
# then edit .env: paste your MDB_MCP_CONNECTION_STRING and either
# MDB_ATLAS_API_KEY (default) or VOYAGE_API_KEY (set EMBEDDING_PROVIDER=voyage)

# 3. ingest the seed knowledge base into kb_chunks
uv run python scripts/ingest.py

# 4. live smoke test (golden-path + fallback transcripts, real services)
uv run python scripts/smoke.py

# 5. run the Streamlit UI
uv run streamlit run app/main.py
```

Open the printed URL, ask `What is the standard return window?`, and you
should see the answer stream in plus a Sources panel that cites
`policies/returns.pdf`.

[Voyage / Atlas Model API key]: https://www.mongodb.com/docs/voyageai/management/api-keys/

## What it demonstrates

- **Hybrid search via rank fusion** of `$vectorSearch` (semantic) and
  `$search` (lexical, standard analyzer). Default weights `0.7 / 0.3`
  (vector-leaning), tunable via env.
- **MongoDB version detection.** On startup the app inspects
  `db.command("buildInfo").version`. ≥ 8.0 → native `$rankFusion`
  (Atlas M0 runs 8.0+, so it uses this path); otherwise `$unionWith` +
  reciprocal rank fusion (`rrf_k = 60`). One log
  line names the chosen path: `retrieval: rankFusion` or
  `retrieval: unionWith+rrf`.
- **Asymmetric embeddings** at 1024 dims: `voyage-4` for documents,
  `voyage-4-lite` for queries. The first embedding is dimension-checked;
  mismatches raise immediately.
- **Pluggable embedding backends** behind a single `Embedder` protocol —
  switch between Atlas's hosted embedding API and the Voyage SDK by changing
  `EMBEDDING_PROVIDER`. The rest of the code does not branch on provider.
- **Bedrock prompt caching** on the system prompt via the ConverseStream
  `cachePoint` block. No Anthropic SDK; just `boto3 bedrock-runtime`.
- **Test-friendly seams.** `Retriever` and `LLMClient` are protocols;
  `app.main.get_app(retriever, llm)` is a factory that pytest substitutes
  with in-memory fakes. `import app` is side-effect free — no module-level
  `MongoClient(...)` or `boto3.client(...)`.

## Project layout

```text
app/
  config.py        Settings.load() — env → frozen dataclass, fail-fast on missing
  embeddings.py    Embedder protocol + Atlas / Voyage backends + dim guard
  retrieval.py     Retriever protocol + MongoRetriever (rankFusion / unionWith+RRF)
  llm.py           LLMClient protocol + BedrockClient (ConverseStream + caching)
  indexes.py       Ensure vector + Atlas Search indexes; poll until READY
  rag.py           Pure pipeline: retrieve → format → stream answer
  main.py          Streamlit page + get_app(retriever, llm) factory
scripts/
  ingest.py        Load knowledge_base.json → kb_chunks (idempotent SHA-1 _id)
  smoke.py         Golden + fallback transcripts against real Atlas / Bedrock
tests/             pytest, in-memory fakes, side-effect-free import test
docs/              requirements.md, design.md, tasks.md
.github/workflows/ ci.yml — ruff + pytest, no live secrets
```

## Configuration

All values come from `.env` (or the process environment). Every variable is
listed in `.env.sample`.

| Var | Default | Notes |
| --- | --- | --- |
| `MDB_MCP_CONNECTION_STRING` | — (required) | Atlas SRV. Same value the MCP server uses (single source of truth). |
| `MDB_DATABASE` | `knowledge_base` | Database name. |
| `EMBEDDING_PROVIDER` | `atlas` | `atlas` or `voyage`. |
| `EMBEDDING_DIMENSIONS` | `1024` | App fails fast if model output differs. |
| `EMBEDDING_DOC_MODEL` | `voyage-4` | Used by ingest. |
| `EMBEDDING_QUERY_MODEL` | `voyage-4-lite` | Used at query time. |
| `MDB_ATLAS_API_KEY` | — | Required when `EMBEDDING_PROVIDER=atlas`. |
| `MDB_ATLAS_EMBEDDING_URL` | `https://ai.mongodb.com/v1/embeddings` | Override only for non-default region. |
| `VOYAGE_API_KEY` | — | Required when `EMBEDDING_PROVIDER=voyage`. |
| `HYBRID_VECTOR_WEIGHT` | `0.7` | Vector branch weight in fusion. |
| `HYBRID_LEXICAL_WEIGHT` | `0.3` | Lexical branch weight in fusion. |
| `RETRIEVAL_K` | `5` | Top-k chunks passed to the LLM. |
| `AWS_REGION` | `us-east-1` | Bedrock region. |
| `BEDROCK_MODEL_ID` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | |

AWS credentials use the standard chain (env vars, profile, SSO, role) — they
are not read from `.env` directly.

`.env` and `.mcp.json` are gitignored; never commit secrets.

## Development

```bash
uv run pytest -q                  # unit + e2e fakes (no live services)
uv run ruff check .               # lint
uv run ruff format --check .      # format
```

CI runs the same three commands on push and PR. See
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Specifications

- [`docs/requirements.md`](docs/requirements.md) — EARS requirements + premortem
- [`docs/design.md`](docs/design.md) — module layout + boundary inventory
- [`docs/tasks.md`](docs/tasks.md) — task breakdown + traceability matrix

## Non-goals

Auth, multi-tenant, conversation memory, document upload UI.
