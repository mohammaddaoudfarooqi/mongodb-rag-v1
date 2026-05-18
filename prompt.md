# Hybrid Search RAG with MongoDB

Build a small, production-shaped RAG app demonstrating **hybrid search**
(vector + full-text) on MongoDB Atlas.

## Skills

MongoDB MCP server is available. Invoke as needed:
`spec-driven-tdd` (plan + execute), `mongodb-search-and-ai` (indexes +
hybrid query), `mongodb-mcp-setup` (only if creds missing),
`github-actions` (CI), `doc-authoring` (README).

## Goal

Streamlit app: user asks a customer-support question, gets an answer
grounded in the seed knowledge base, retrieved via **rank fusion of
`$vectorSearch` and `$search`**.

## Stack

- Python 3.11+ with **`uv`**, Streamlit, PyMongo, MongoDB Atlas M0.
- **Embeddings — asymmetric, pluggable.** Ingest `voyage-4`, query
  `voyage-4-lite`. Both pinned to **1024 dimensions** (verify on
  startup, fail fast on mismatch). Backend selected by
  `EMBEDDING_PROVIDER`:
  - `atlas` (default) — MongoDB Atlas AI embedding API.
  - `voyage` — Voyage SDK direct.
  Single `Embedder` interface in `app/embeddings.py` with a factory; the
  rest of the code does not branch on provider.
- **LLM — Claude Haiku 4.5 on Bedrock.** `boto3` `bedrock-runtime`
  ConverseStream (no Anthropic SDK). Enable Bedrock prompt caching on the
  system prompt.
  - `BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0`
  - `AWS_REGION=us-east-1`
- **Tests:** pytest. **CI:** GitHub Actions (ruff + pytest).
- **Test seams (mandatory).** Define two protocols in `app/`:
  - `Retriever` — `retrieve(query: str, k: int) -> list[Chunk]`.
  - `LLMClient` — `stream(system: str, messages: list[dict]) -> Iterator[str]`.
  The Streamlit app and any callable entrypoint accept both via dependency
  injection (constructor args or a `get_app(retriever, llm)` factory).
  Pytest substitutes in-memory fakes; the production wiring picks the real
  PyMongo + Bedrock implementations. **No module-level `boto3.client(...)`,
  `MongoClient(...)`, or index-bootstrap calls** — those run inside
  `main()` / `get_app()` so `import app` is side-effect free and CI never
  needs live credentials.

## Data and indexes

Seed: `knowledge_base.json` at repo root. Schema per doc:
`{title, text, metadata: {source, category, audience, last_updated}}`.

DB `knowledge_base` (set `MDB_DATABASE=knowledge_base`), collection
`kb_chunks`. One doc per chunk; no splitting.

The **app and ingest script must bootstrap indexes** if missing, then poll
until READY before serving queries / writes:

- Vector index on `embedding` (cosine, `numDimensions` from the model).
- Atlas Search index on `text` and `title` (standard analyzer).

Show a "Preparing search indexes…" spinner on first load.

## Behavior

1. Single text input + Submit.
2. Hybrid retrieval with **vector-leaning weights: 0.7 vector / 0.3
   lexical** (default, configurable via env). **Default implementation
   is `$unionWith` + reciprocal-rank fusion**, since Atlas M0 is on
   MongoDB 8.0 and `$rankFusion` requires 8.1+. On startup, check
   `db.command("buildInfo").version`; if ≥ 8.1, use the `$rankFusion`
   path. Log one line at startup naming which path was chosen
   (`retrieval: rankFusion` / `retrieval: unionWith+rrf`).
3. Top-k chunks (default `k=5`) → Bedrock with a system prompt that says:
   *answer only from the provided context; otherwise say "I don't have that
   information."*
4. Stream the answer.
5. Expandable "Sources" panel: `title`, `metadata.source`, fusion score.

Non-goals: auth, multi-tenant, conversation memory, document upload UI.

## Environment variables

| Var | Notes |
| --- | --- |
| `MDB_MCP_CONNECTION_STRING` | Atlas SRV. One source of truth (MCP + app). |
| `MDB_DATABASE` | Default `knowledge_base`. |
| `EMBEDDING_DIMENSIONS` | Default `1024`. App fails fast if model output differs. |
| `HYBRID_VECTOR_WEIGHT` | Default `0.7` (vector-leaning). |
| `HYBRID_LEXICAL_WEIGHT` | Default `0.3` (lexical complement). |
| `EMBEDDING_PROVIDER` | `atlas` (default) or `voyage`. |
| `EMBEDDING_DOC_MODEL` | Default `voyage-4`. |
| `EMBEDDING_QUERY_MODEL` | Default `voyage-4-lite`. |
| `MDB_ATLAS_API_KEY` | Required when `EMBEDDING_PROVIDER=atlas`. |
| `MDB_ATLAS_EMBEDDING_URL` | Override only for non-default region. |
| `VOYAGE_API_KEY` | Required when `EMBEDDING_PROVIDER=voyage`. |
| `AWS_REGION` | Default `us-east-1`. |
| `BEDROCK_MODEL_ID` | Default `us.anthropic.claude-haiku-4-5-20251001-v1:0`. |

AWS credentials via the standard chain (env, profile, SSO, role).

Load all of the above from a local `.env` (use `python-dotenv` or
`uv run --env-file .env`). Maintain a committed `.env.sample` with every
variable listed (no real values). `.env` and `.mcp.json` must be gitignored. Never echo
secrets in logs, errors, or the UI.

## Workflow

1. Plan with `spec-driven-tdd`. Confirm the spec before coding.
2. Verify MongoDB via MCP and Bedrock via one Converse call. Stop if either
   is missing.
3. Build ingest → retrieval → generation → UI, one module at a time, with
   red-green-refactor.
4. pytest coverage: golden path (known Q cites expected source) +
   fallback path (out-of-scope Q returns the canned response), against
   a **mocked** retrieval layer.
5. CI: ruff and pytest — no live Atlas in CI.
6. README via `doc-authoring`, with a 5-minute quickstart.
7. Run the app with `uv run streamlit run app/main.py` to confirm it
   starts. Then, in a separate script (`scripts/smoke.py`), import the
   retrieval + generation function directly with the production wiring
   and call it twice: once with the golden-path question (assert the
   answer cites `policies/returns.pdf`), once with an out-of-scope
   question (assert the response is the canned *"I don't have that
   information."*). Print both transcripts. Do not claim to have
   clicked anything in the UI.
8. Push only when the user supplies a repo URL. No force push, no extra
   branches.

After each step post one line: `Step N done — [what changed]`. Stop only on
blockers requiring a decision not covered above.
