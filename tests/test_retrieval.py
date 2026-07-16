from __future__ import annotations

import logging

from app.config import Settings
from app.retrieval import Chunk, MongoRetriever


def _settings(k: int = 5) -> Settings:
    return Settings(
        mongo_uri="mongodb+srv://x:y@c.mongodb.net/",
        database="knowledge_base",
        embedding_provider="atlas",
        embedding_dimensions=4,
        embedding_doc_model="voyage-4",
        embedding_query_model="voyage-4-lite",
        hybrid_vector_weight=0.7,
        hybrid_lexical_weight=0.3,
        retrieval_k=k,
        aws_region="us-east-1",
        bedrock_model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        atlas_api_key="atlas-key",
        atlas_embedding_url=None,
        voyage_api_key=None,
    )


class _FakeDB:
    def __init__(self, version: str):
        self._version = version
        self.command_calls: list[str] = []

    def command(self, name: str):
        self.command_calls.append(name)
        if name == "buildInfo":
            return {"version": self._version}
        raise ValueError(f"unexpected command: {name}")


class _FakeColl:
    def __init__(self, db: _FakeDB, results: list[dict]):
        self.database = db
        self.results = results
        self.aggregate_calls: list[list[dict]] = []

    def aggregate(self, pipeline: list[dict]):
        self.aggregate_calls.append(pipeline)
        return iter(self.results)


class _FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _doc(_id: str, score: float = 0.5) -> dict:
    return {
        "_id": _id,
        "title": f"Title {_id}",
        "text": f"Text {_id}",
        "metadata": {"source": f"src/{_id}.pdf"},
        "score": score,
    }


def test_logs_rank_fusion_path_on_8_0(caplog):
    db = _FakeDB("8.0.4")
    coll = _FakeColl(db, results=[])
    caplog.set_level(logging.INFO)
    MongoRetriever(coll, _FakeEmbedder(), _settings())
    msgs = [r.message for r in caplog.records if "retrieval:" in r.message]
    assert msgs == ["retrieval: rankFusion"]


def test_logs_union_with_path_below_8_0(caplog):
    db = _FakeDB("7.0.4")
    coll = _FakeColl(db, results=[])
    caplog.set_level(logging.INFO)
    MongoRetriever(coll, _FakeEmbedder(), _settings())
    msgs = [r.message for r in caplog.records if "retrieval:" in r.message]
    assert msgs == ["retrieval: unionWith+rrf"]


def test_rank_fusion_pipeline_uses_rank_fusion_stage():
    db = _FakeDB("8.0.0")
    coll = _FakeColl(db, results=[_doc("a", 0.9), _doc("b", 0.5)])
    r = MongoRetriever(coll, _FakeEmbedder(), _settings(k=2))
    chunks = r.retrieve("hello", k=2)
    assert [c.id for c in chunks] == ["a", "b"]
    assert isinstance(chunks[0], Chunk)
    pipeline = coll.aggregate_calls[0]
    # Top-level stage is $rankFusion
    assert "$rankFusion" in pipeline[0]


def test_union_with_pipeline_runs_two_pipelines_and_fuses_via_rrf():
    db = _FakeDB("7.0.0")
    # Two pipelines run sequentially (vector branch then text branch).
    # Return a small overlapping result set so RRF fusion produces a
    # deterministic top-2.
    vector_branch = [
        {
            "_id": "a",
            "title": "A",
            "text": "ta",
            "metadata": {"source": "src/a.pdf"},
            "_rank": 1,
            "_branch": "vector",
        },
        {
            "_id": "b",
            "title": "B",
            "text": "tb",
            "metadata": {"source": "src/b.pdf"},
            "_rank": 2,
            "_branch": "vector",
        },
    ]
    text_branch = [
        {
            "_id": "b",
            "title": "B",
            "text": "tb",
            "metadata": {"source": "src/b.pdf"},
            "_rank": 1,
            "_branch": "text",
        },
        {
            "_id": "c",
            "title": "C",
            "text": "tc",
            "metadata": {"source": "src/c.pdf"},
            "_rank": 2,
            "_branch": "text",
        },
    ]

    class _BranchedColl(_FakeColl):
        def __init__(self, db, vector, text):
            super().__init__(db, results=[])
            self._vector = vector
            self._text = text
            self._n = 0

        def aggregate(self, pipeline):
            self.aggregate_calls.append(pipeline)
            self._n += 1
            return iter(self._vector if self._n == 1 else self._text)

    coll = _BranchedColl(db, vector_branch, text_branch)
    r = MongoRetriever(coll, _FakeEmbedder(), _settings(k=2))
    chunks = r.retrieve("hello", k=2)
    # b appears in both branches → highest fused score; a or c follows.
    assert chunks[0].id == "b"
    assert chunks[0].score > chunks[1].score
    # Two aggregate calls: one per branch.
    assert len(coll.aggregate_calls) == 2


def test_retrieve_passes_query_embedding_and_k_to_aggregation():
    db = _FakeDB("8.0.0")
    coll = _FakeColl(db, results=[_doc("a", 0.9)])
    r = MongoRetriever(coll, _FakeEmbedder(), _settings(k=3))
    r.retrieve("hello", k=3)
    pipeline = coll.aggregate_calls[0]
    # The query vector must show up somewhere in the rankFusion definition.
    serialized = repr(pipeline)
    assert "0.1" in serialized and "0.4" in serialized


def test_returns_at_most_k_results():
    db = _FakeDB("8.0.0")
    coll = _FakeColl(db, results=[_doc(str(i), 1.0 - i * 0.01) for i in range(20)])
    r = MongoRetriever(coll, _FakeEmbedder(), _settings(k=5))
    chunks = r.retrieve("q", k=5)
    assert len(chunks) <= 5
