from __future__ import annotations

import importlib
import sys


def test_import_main_does_not_construct_clients(monkeypatch):
    """REQ-064: import side-effect free. No MongoClient / boto3 client at import."""
    sys.modules.pop("app.main", None)

    monkeypatch.delenv("MDB_MCP_CONNECTION_STRING", raising=False)
    monkeypatch.delenv("MDB_ATLAS_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    # Tripwire: any call to these constructors fails the test.
    class _Tripwire:
        def __init__(self, name: str):
            self._name = name

        def __call__(self, *args, **kwargs):
            raise AssertionError(f"{self._name} must not be called at import time")

    import boto3
    import pymongo

    monkeypatch.setattr(boto3, "client", _Tripwire("boto3.client"))
    monkeypatch.setattr(pymongo, "MongoClient", _Tripwire("MongoClient"))

    importlib.import_module("app.main")  # MUST NOT raise


def test_get_app_returns_callable(fake_retriever, fake_llm):
    from app.main import get_app

    render = get_app(fake_retriever, fake_llm)
    assert callable(render)


def test_get_app_does_not_call_retriever_or_llm_until_render(fake_retriever, fake_llm):
    from app.main import get_app

    get_app(fake_retriever, fake_llm)
    assert fake_retriever.calls == []
    assert fake_llm.calls == []
