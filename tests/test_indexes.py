from __future__ import annotations

import pytest

from app.indexes import IndexTimeout, ensure_indexes


class _FakeDB:
    def __init__(self, has_collection: bool = True):
        self._has_collection = has_collection
        self.created_collections: list[str] = []

    def list_collection_names(self) -> list[str]:
        return ["kb_chunks"] if self._has_collection else []

    def create_collection(self, name: str) -> None:
        self.created_collections.append(name)
        self._has_collection = True


class _FakeCollection:
    name = "kb_chunks"

    def __init__(self, ready_after: int = 1, existing: list[str] | None = None):
        self.database = _FakeDB(has_collection=True)
        self._existing = list(existing or [])
        self._ready_after = ready_after
        self._calls_to_list = 0
        self.created: list[dict] = []

    def list_search_indexes(self):
        self._calls_to_list += 1
        statuses = []
        for name in self._existing:
            queryable = self._calls_to_list >= self._ready_after
            statuses.append(
                {
                    "name": name,
                    "status": "READY" if queryable else "PENDING",
                    "queryable": queryable,
                }
            )
        return iter(statuses)

    def create_search_index(self, model):
        spec = model if isinstance(model, dict) else getattr(model, "document", model)
        name = spec.get("name") if isinstance(spec, dict) else None
        if not name and hasattr(model, "document"):
            name = model.document.get("name")
        self.created.append({"name": name, "model": spec})
        self._existing.append(name)
        return name


def test_ensure_indexes_creates_when_missing():
    coll = _FakeCollection(ready_after=1)
    ensure_indexes(coll, dimensions=4, poll_interval=0.0, timeout_s=2.0)
    names = [c["name"] for c in coll.created]
    assert "kb_vector" in names
    assert "kb_search" in names


def test_ensure_indexes_is_idempotent():
    coll = _FakeCollection(ready_after=1, existing=["kb_vector", "kb_search"])
    ensure_indexes(coll, dimensions=4, poll_interval=0.0, timeout_s=2.0)
    assert coll.created == []  # both already exist; nothing created


def test_ensure_indexes_polls_until_ready():
    coll = _FakeCollection(ready_after=3, existing=["kb_vector", "kb_search"])
    ensure_indexes(coll, dimensions=4, poll_interval=0.0, timeout_s=2.0)
    assert coll._calls_to_list >= 3


def test_ensure_indexes_times_out():
    class _NeverReady(_FakeCollection):
        def list_search_indexes(self):
            return iter(
                [
                    {"name": "kb_vector", "status": "PENDING", "queryable": False},
                    {"name": "kb_search", "status": "PENDING", "queryable": False},
                ]
            )

    coll = _NeverReady(existing=["kb_vector", "kb_search"])
    with pytest.raises(IndexTimeout):
        ensure_indexes(coll, dimensions=4, poll_interval=0.0, timeout_s=0.05)
