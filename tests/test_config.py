import pytest

from app.config import Settings


def _base_env(provider: str = "atlas") -> dict:
    env = {
        "MDB_MCP_CONNECTION_STRING": "mongodb+srv://x:y@cluster.mongodb.net/",
        "MDB_DATABASE": "knowledge_base",
        "EMBEDDING_PROVIDER": provider,
        "EMBEDDING_DIMENSIONS": "1024",
        "EMBEDDING_DOC_MODEL": "voyage-4",
        "EMBEDDING_QUERY_MODEL": "voyage-4-lite",
        "HYBRID_VECTOR_WEIGHT": "0.7",
        "HYBRID_LEXICAL_WEIGHT": "0.3",
        "RETRIEVAL_K": "5",
        "AWS_REGION": "us-east-1",
        "BEDROCK_MODEL_ID": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    }
    if provider == "atlas":
        env["MDB_ATLAS_API_KEY"] = "atlas-key"
    elif provider == "voyage":
        env["VOYAGE_API_KEY"] = "voyage-key"
    return env


def test_settings_load_with_atlas_provider():
    s = Settings.load(_base_env("atlas"))
    assert s.embedding_provider == "atlas"
    assert s.embedding_dimensions == 1024
    assert s.hybrid_vector_weight == 0.7
    assert s.hybrid_lexical_weight == 0.3
    assert s.retrieval_k == 5


def test_settings_load_with_voyage_provider():
    s = Settings.load(_base_env("voyage"))
    assert s.embedding_provider == "voyage"
    assert s.voyage_api_key == "voyage-key"


def test_missing_mongo_uri_raises():
    env = _base_env()
    del env["MDB_MCP_CONNECTION_STRING"]
    with pytest.raises(ValueError, match="MDB_MCP_CONNECTION_STRING"):
        Settings.load(env)


def test_atlas_provider_requires_atlas_key():
    env = _base_env("atlas")
    del env["MDB_ATLAS_API_KEY"]
    with pytest.raises(ValueError, match="MDB_ATLAS_API_KEY"):
        Settings.load(env)


def test_voyage_provider_requires_voyage_key():
    env = _base_env("voyage")
    del env["VOYAGE_API_KEY"]
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        Settings.load(env)


def test_unknown_provider_raises():
    env = _base_env()
    env["EMBEDDING_PROVIDER"] = "bogus"
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        Settings.load(env)


def test_defaults_applied_when_missing():
    s = Settings.load(
        {
            "MDB_MCP_CONNECTION_STRING": "mongodb+srv://x:y@c.mongodb.net/",
            "MDB_ATLAS_API_KEY": "k",
        }
    )
    assert s.database == "knowledge_base"
    assert s.embedding_provider == "atlas"
    assert s.embedding_dimensions == 1024
    assert s.hybrid_vector_weight == 0.7
    assert s.hybrid_lexical_weight == 0.3
    assert s.retrieval_k == 5
    assert s.aws_region == "us-east-1"
    assert s.bedrock_model_id == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
