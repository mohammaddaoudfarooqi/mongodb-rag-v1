# Workshop Steps — From Prompt to Production

Follow along with the live demo. Everything here runs from this folder.

> Goal: ship a Streamlit RAG app that uses MongoDB Atlas hybrid search
> (`$rankFusion` over `$vectorSearch` + `$search`) and Claude Haiku 4.5
> on Bedrock — built end-to-end by Claude Code using MongoDB Agent
> Skills + the MongoDB MCP server.

---

## 0. Before you start

### Required

The build won't run without these.

- [ ] **Claude Code** installed and signed in — [claude.com/claude-code](https://claude.com/claude-code)
- [ ] **MongoDB Atlas M0** cluster (free) with IP allow-listed and a DB user — [setup guide](https://www.mongodb.com/docs/atlas/tutorial/create-new-cluster/)
- [ ] **AWS account** with **Anthropic Claude Haiku 4.5** enabled in Bedrock (`us-east-1`, *Model access*) **and** an IAM principal granted `bedrock:InvokeModel` + `bedrock:InvokeModelWithResponseStream` on that model
- [ ] **AWS credentials** on the standard chain (`aws configure`, SSO, or env vars)
- [ ] **Embeddings** — either an Atlas Model API key (default) **or** a Voyage API key
- [ ] **Python 3.11+** with [`uv`](https://docs.astral.sh/uv/)
- [ ] **Node.js** on PATH — Claude Code itself ships as an npm package, the MongoDB MCP server is launched per-session via `npx -y mongodb-mcp-server@latest`, and the skill installs in §1 use `npx`

### Optional

Only needed if you want to push the result to GitHub at the end (per
`prompt.md` step 8, the push only happens if you supply a repo URL).

- [ ] **git** installed locally
- [ ] An **empty GitHub repo** to push to, **with an SSH key** registered on your account (or a PAT if you prefer HTTPS) — the demo uses `git@github.com:...`

> Heads-up: the build creates a `knowledge_base` database with a
> `kb_chunks` collection (≈35 documents + two search indexes) inside
> *your* Atlas cluster. Use a dedicated demo project / cluster if you
> need isolation.

---

## 1. Install the skills

Run once, globally:

```bash
# MongoDB agent skills
npx skills add mongodb/agent-skills -g --all -y

# Engineering skills (spec-driven-tdd, doc-authoring, github-actions, ...)
npx skills add mohammaddaoudfarooqi/agent-engineering-skills -g --all -y
```

Verify in a fresh Claude Code session — type `/skills` and check that all of
these are listed and enabled:

**MongoDB skills**

- `mongodb-search-and-ai` — Atlas Search, Vector Search, hybrid search
  (used directly by `prompt.md`)
- `mongodb-mcp-setup` — bootstrap the MCP server credentials when missing
  (used directly by `prompt.md` *only if* you skip §2 and the connection
  string isn't already configured)
- `mongodb-connection` — connection-pool / driver tuning
- `mongodb-natural-language-querying` — translate plain-English asks
  into `find` / aggregation pipelines
- `mongodb-query-optimizer` — explain plans, indexing, slow-query triage
- `mongodb-schema-design` — embed-vs-reference, anti-patterns, migrations
- `atlas-stream-processing` — Atlas Stream Processing workflows (ASP)

**Engineering skills**

- `spec-driven-tdd` — plan + execute (used directly by `prompt.md`)
- `doc-authoring` — README / reference / how-to (used directly by `prompt.md`)
- `github-actions` — CI/CD workflow authoring (used directly by `prompt.md`)

The five skills `prompt.md` invokes by name are `spec-driven-tdd`,
`mongodb-search-and-ai`, `mongodb-mcp-setup`, `github-actions`, and
`doc-authoring`. The rest are available to Claude Code throughout the
build for query-shaping, index advice, and schema review.

---

## 2. Configure the MongoDB MCP server

```bash
cp mcp.json.sample .mcp.json
```

Open `.mcp.json` and replace `MDB_MCP_CONNECTION_STRING` with your Atlas
SRV connection string. Claude Code auto-discovers `.mcp.json` from the
working directory.

Sanity check — in Claude Code:

> *"List databases via the MongoDB MCP server."*

You should see your Atlas databases.

---

## 3. Create your `.env`

In this folder, create a `.env` with:

```bash
# MongoDB
MDB_MCP_CONNECTION_STRING="mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"
MDB_DATABASE="knowledge_base"

# Embeddings — pick ONE provider
EMBEDDING_PROVIDER="atlas"            # or "voyage"
EMBEDDING_DOC_MODEL="voyage-4"
EMBEDDING_QUERY_MODEL="voyage-4-lite"
EMBEDDING_DIMENSIONS="1024"
MDB_ATLAS_API_KEY=""                  # required if EMBEDDING_PROVIDER=atlas
# VOYAGE_API_KEY=""                   # required if EMBEDDING_PROVIDER=voyage

# Bedrock (AWS creds via standard chain)
AWS_REGION="us-east-1"
AWS_ACCESS_KEY_ID="...."
AWS_SECRET_ACCESS_KEY="...."
BEDROCK_MODEL_ID="us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Hybrid search
HYBRID_VECTOR_WEIGHT="0.7"
HYBRID_LEXICAL_WEIGHT="0.3"
RETRIEVAL_K="5"
```

`.env` and `.mcp.json` are gitignored. Never paste secrets into chat.

The committed `.env.sample` mirrors this list for new collaborators.

---

## 4. Hand the prompt to Claude Code

In a fresh Claude Code session in an empty folder as the working directory,
copy-paste files: prompt.md, .mcp.json and knowledge_base.json

Prompt to Claude code:

```text
@prompt.md
```

Claude Code will:

1. Plan the build with `spec-driven-tdd` and emit
   `docs/{requirements,design,tasks}.md` (you'll be asked to confirm
   the spec, including which embedding provider should be the default).
2. Verify MongoDB via the MCP server (`list-databases`) and Bedrock via
   one minimal `ConverseStream` call. Stop if either fails.
3. Scaffold the `uv` project: `pyproject.toml`, `.gitignore`,
   `.env.sample`, ruff config.
4. Build modules TDD-style, one at a time —
   `app/{config,embeddings,llm,indexes,retrieval,rag,main}.py` —
   each with red → green → refactor and matching tests.
5. Bootstrap Atlas Search + Vector Search indexes from PyMongo
   (`create_search_index`) and poll until they report queryable;
   ingest the seed corpus with `scripts/ingest.py` (idempotent —
   `_id` is a stable hash of source + title).
6. Detect `db.command("buildInfo").version` at app startup and log
   one line naming the chosen retrieval path: `retrieval: rankFusion`
   (8.1+) or `retrieval: unionWith+rrf` (M0 today is 8.0).
7. Write the pytest suite — golden path + fallback path against
   in-memory fakes — plus a side-effect-free import test that fails
   if any module touches `MongoClient(...)` or `boto3.client(...)`
   at import time.
8. Add a GitHub Actions workflow (ruff check + ruff format --check +
   pytest, Python 3.11, uv, SHA-pinned actions, least-privilege
   permissions) and a Dependabot config.
9. Generate `README.md` with `doc-authoring` (5-minute quickstart).
10. Run `scripts/smoke.py` — production wiring against live Atlas +
    Bedrock — twice: golden cites `policies/returns.pdf`, fallback
    returns the canned phrase.
11. Boot the Streamlit app headless and hit `/_stcore/health` to
    confirm clean startup.
12. Push to your repo.

Expect ~15–20 minutes. Stay in the loop — Claude Code will pause for
confirmation after the spec phase, after the live-boundary check, and
again before pushing to GitHub.

---

## 5. Verify locally

The smoke script is the fastest way to prove the whole pipeline works
against real services:

```bash
# One-shot ingest (idempotent — safe to re-run)
uv run --env-file .env python scripts/ingest.py

# Live smoke — golden + fallback transcripts
uv run --env-file .env python scripts/smoke.py
```

Expected output:

```text
GOLDEN  Q: What is the standard return window for unopened merchandise?
A: According to the Standard Return Policy, unopened merchandise may be
   returned within 30 days of delivery for a full refund...
✓ golden path cited policies/returns.pdf

FALLBACK  Q: What is the airspeed of an unladen European swallow...
A: I don't have that information.
✓ fallback path returned canned reply

PASS
```

Then run the UI:

```bash
uv run --env-file .env streamlit run app/main.py
```

The first load shows a *"Preparing search indexes…"* spinner while
indexes are confirmed READY. Try the **golden path**:

> *"What is the return window for an opened laptop?"*

Expected: a streamed answer citing `policies/returns.pdf` with the
15-day, 15%-restocking-fee rule.

Try an **out-of-scope** question:

> *"What's the weather in Tokyo?"*

Expected fallback: *"I don't have that information."*

Open the **Sources** panel — you should see title, source, and the
fusion score for each retrieved chunk.

---

## 6. Watch CI go green

Your push triggers GitHub Actions:

- `ruff check`
- `ruff format --check`
- `pytest` (in-memory fakes — no live Atlas / Bedrock / Voyage creds in CI)

Open the run on GitHub and confirm all jobs pass.

---

## What to look for during the demo

- The **retrieval-path log line** at startup — `retrieval: rankFusion`
  on 8.1+, `retrieval: unionWith+rrf` on M0 (8.0). The same code handles
  both; the version check picks the path.
- The **`$rankFusion` pipeline** (or `$unionWith` + RRF fallback) in
  `app/retrieval.py` — vector + lexical fused with 0.7 / 0.3 weights.
- **PyMongo `create_search_index` calls** that bootstrap the Atlas
  Search and Vector Search indexes — no clicking around in the Atlas
  UI, and the app polls until both are queryable before serving.
- **Bedrock prompt caching** on the system prompt via the
  ConverseStream `cachePoint` block — second turn is noticeably
  cheaper / faster.
- The **fail-fast dimension check** — both Voyage models pinned to
  1024 dims; a wrong-dimension response crashes on startup, not at
  query time.
- The **side-effect-free import test** — `import app.main` triggers
  no `MongoClient` or `boto3` construction; production wiring lives
  inside `main()` / `get_app(retriever, llm)`.
- The **pytest suite** — golden-path and fallback tests run against
  in-memory fakes, so CI doesn't need live Atlas creds.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `$rankFusion` not recognized | Atlas M0 ships MongoDB 8.0; the app falls back to `$unionWith` + RRF automatically. The startup log line tells you which path is active. |
| `Collection 'knowledge_base.kb_chunks' does not exist` | Atlas refuses to create search indexes on a missing namespace. Run `scripts/ingest.py` first, or rely on `app/indexes.py` which auto-creates the collection. |
| Vector index "still building" | M0 indexes can take 1–3 minutes. The ingest script polls; if it gives up, re-run it. |
| `ModuleNotFoundError: No module named 'app'` when running scripts directly | Already handled — `scripts/*.py` and `app/main.py` prepend the repo root to `sys.path`. If you reorganise, preserve that. |
| `AccessDeniedException` from Bedrock | Request access to **Anthropic · Claude Haiku 4.5** in *Model access*. IAM needs `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream`. |
| `ValidationException` on the model ID | The `us.` prefix is an inference profile — model must be enabled in a US region (default `us-east-1`). |
| Atlas embedding `401` | The Atlas Model API key is separate from a Voyage-direct key. Generate one under *Atlas → Services → Model API Keys*. Voyage-direct keys do **not** work against `ai.mongodb.com`. |
| Voyage API errors | Model names update periodically — see the [Voyage blog](https://blog.voyageai.com/). |
| pytest can't find modules in CI | Make sure the workflow runs `uv sync` before `uv run pytest`; the `github-actions` skill sets this up — re-invoke it if you customised the workflow. |

---

## Reference links

- [MongoDB Agent Skills](https://github.com/mongodb/agent-skills)
- [Agent Engineering Skills](https://github.com/mohammaddaoudfarooqi/agent-engineering-skills)
- [MongoDB MCP Server](https://github.com/mongodb-js/mongodb-mcp-server)
- [Hybrid search docs](https://www.mongodb.com/docs/vector-search/hybrid-search/vector-search-with-full-text-search/)
- [Atlas M0 setup](https://www.mongodb.com/docs/atlas/tutorial/create-new-cluster/)
- [Voyage 4 announcement](https://blog.voyageai.com/2026/01/15/voyage-4/)
- [Embedding & reranking on Atlas](https://www.mongodb.com/company/blog/product-release-announcements/introducing-the-embedding-and-reranking-api-on-mongodb-atlas)
- [Amazon Bedrock — Claude Haiku 4.5](https://aws.amazon.com/bedrock/anthropic/)
