# Design — Hybrid Search RAG with MongoDB

## Module layout

```text
app/
  __init__.py        # empty; no side effects
  config.py          # env loading, dataclass of resolved settings
  embeddings.py      # Embedder protocol + AtlasEmbedder, VoyageEmbedder, factory
  retrieval.py       # Retriever protocol + MongoRetriever (rankFusion or unionWith+RRF)
  llm.py             # LLMClient protocol + BedrockClient (ConverseStream)
  indexes.py         # ensure & poll vector + Atlas Search indexes
  main.py            # get_app(retriever, llm) factory + Streamlit page
scripts/
  ingest.py          # one-shot: load knowledge_base.json → kb_chunks
  smoke.py           # production-wired golden + fallback transcripts
tests/
  conftest.py        # in-memory fakes
  test_embeddings.py
  test_retrieval.py
  test_llm.py
  test_main.py
docs/
  requirements.md
  design.md
  tasks.md
.github/workflows/ci.yml
.env.sample
.gitignore
pyproject.toml
README.md
```

## Key components

### Embedder (`app/embeddings.py`)

```python
class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...

def make_embedder(cfg: Settings) -> Embedder: ...   # factory, dispatches on EMBEDDING_PROVIDER
```

- `AtlasEmbedder` — POST to Atlas AI embedding URL; batches per call.
- `VoyageEmbedder` — uses the `voyageai` SDK with `model=voyage-4` for docs and
  `voyage-4-lite` for queries.
- Both wrap a dimension assertion: first embedding's length must equal
  `cfg.embedding_dimensions`; otherwise raise `EmbeddingDimensionMismatch`.

### Retriever (`app/retrieval.py`)

```python
@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    text: str
    metadata: dict
    score: float

class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[Chunk]: ...
```

`MongoRetriever`:

1. On construction, calls `db.command("buildInfo")["version"]`. Picks
   `_rank_fusion` if `>= 8.1`, otherwise `_union_with_rrf`. Logs the choice
   exactly once via `logging.info("retrieval: %s", path)`.
2. `_rank_fusion(query, k)` — single `$rankFusion` stage with two pipelines
   (vector + text), weighted 0.7 / 0.3.
3. `_union_with_rrf(query, k)` — runs `$vectorSearch` and `$search`
   pipelines via `$unionWith`, ranks each, computes
   `score = w_v / (rrf_k + rank_v) + w_t / (rrf_k + rank_t)` (RRF constant
   `rrf_k = 60`), groups by chunk id, sorts desc, takes top `k`.

### LLMClient (`app/llm.py`)

```python
class LLMClient(Protocol):
    def stream(self, system: str, messages: list[dict]) -> Iterator[str]: ...
```

`BedrockClient`:

- `boto3.client("bedrock-runtime", region_name=cfg.aws_region)` constructed
  lazily in `__init__`.
- Calls `converse_stream` with `system=[{"text": system, "cachePoint": {"type": "default"}}]`
  to enable prompt caching on the system prompt.
- Yields text deltas from the `messageStream` events.

### Indexes (`app/indexes.py`)

- `ensure_vector_index(coll, dims)` — creates if missing, polls
  `coll.list_search_indexes()` until queryable.
- `ensure_search_index(coll)` — creates a standard-analyzer index over `text`
  and `title`.
- `wait_until_ready(coll, names, timeout=180s)` — blocking poll.

### UI (`app/main.py`)

```python
def get_app(retriever: Retriever, llm: LLMClient) -> Callable[[], None]:
    def render() -> None:
        # Streamlit page: text_input + submit
        # On submit: chunks = retriever.retrieve(q, k); stream llm answer; sources panel
    return render

def main() -> None:
    cfg = Settings.load()
    client = MongoClient(cfg.mongo_uri)
    coll = client[cfg.database]["kb_chunks"]
    with st.spinner("Preparing search indexes…"):
        ensure_indexes(coll, cfg.embedding_dimensions)
    retriever = MongoRetriever(coll, make_embedder(cfg), cfg)
    llm = BedrockClient(cfg)
    get_app(retriever, llm)()
```

`import app.main` constructs nothing — `main()` runs all side effects.

## Configuration

`app/config.py` resolves env into a frozen `Settings` dataclass. Every env var
in `.env.sample` is reflected; defaults match `prompt.md`. Missing required
values (e.g. `MDB_MCP_CONNECTION_STRING`, AWS creds for Bedrock,
`VOYAGE_API_KEY` when provider is voyage) raise on `Settings.load()`.

## Boundary inventory

| # | Boundary | From | To | Acceptance test |
| - | --- | --- | --- | --- |
| 1 | Embedding API | `Embedder` impl | Voyage / Atlas HTTP | TC-PARITY-EMB (dimension contract test, real one-shot in smoke) |
| 2 | MongoDB Atlas | `MongoRetriever` | Atlas M0 cluster | TC-INTEG-RETRIEVAL (smoke run via scripts/smoke.py) |
| 3 | Bedrock Runtime | `BedrockClient` | AWS bedrock-runtime | TC-INTEG-LLM (smoke run via scripts/smoke.py) |
| 4 | Streamlit UI | browser | `get_app(retriever, llm)` | Manual: `uv run streamlit run app/main.py` confirms boot; smoke covers business logic. |
| 5 | Side-effect-free import | pytest | `import app.main` | TC-IMPORT-CLEAN (no env vars; assert no MongoClient / boto3 client created) |

## Test infrastructure

- pytest + ruff. `pyproject.toml` configures both.
- In-memory `FakeRetriever` and `FakeLLM` in `tests/conftest.py`.
- Mock-parity check on `Embedder` dimension guard (see Premortem #1).
- CI runs ruff + pytest on push; no live secrets.

## Decisions / trade-offs

- **Default embedding provider** — kept as `atlas` per spec, but the factory
  makes switching a one-line env change. If `MDB_ATLAS_API_KEY` is unset and
  provider is `atlas`, `Settings.load()` raises before any work.
- **Index creation timing** — done in `main()` and `scripts/ingest.py`. The
  spinner appears only on first boot (subsequent runs short-circuit because
  indexes already exist).
- **Idempotent ingest key** — SHA-1 of `metadata.source + "|" + title`. Stable
  enough for this seed; can change later without schema migration since it is
  the `_id`.
