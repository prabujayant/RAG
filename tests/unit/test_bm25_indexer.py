"""Unit tests for the Postgres-tsvector BM25Indexer.

A fake session stands in for PostgreSQL — no network, no docker. It stores
postings in memory and emulates the WHERE-clause semantics the indexer
relies on (document_id ANY / NOT ANY). Real ranking fidelity (ts_rank_cd)
is covered by the live-PG smoke test in tests/integration/.
"""

from __future__ import annotations

import pytest
from app.ingestion.chunker import Chunk, content_hash
from app.retrieval import bm25 as bm25_mod
from app.retrieval.bm25 import BM25Indexer

# ------------------------------------------------------------------- fakes


class _FakeScalarResult:
    def __init__(self, value: object) -> None:
        self._value = value

    def scalar(self) -> object:
        return self._value


class _FakeMappingsResult:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def mappings(self) -> _FakeMappingsResult:
        return self

    def all(self) -> list[dict]:
        return self._rows


def _tokens(text: str) -> list[str]:
    return [t for t in (text or "").lower().split() if t]


def _where_matches(src: dict, clause: object) -> bool:
    """Evaluate a simple SQLAlchemy WHERE clause against an in-memory row.

    Handles ``AND`` of ``column == value`` / ``column.in_(values)`` — the
    only shapes the indexer generates.
    """
    from sqlalchemy.sql import operators

    op = getattr(clause, "operator", None)
    if op is operators.and_:
        return all(_where_matches(src, c) for c in clause.clauses)  # type: ignore[attr-defined]
    left = getattr(clause, "left", None)
    right = getattr(clause, "right", None)
    key = getattr(left, "key", None)
    value = getattr(right, "value", None)
    if key is None:
        return True
    if op is operators.in_op:
        values = value if isinstance(value, (list, tuple, set)) else [value]
        return src.get(key) in values
    return src.get(key) == value


class _FakeSession:
    """In-memory stand-in for a SQLAlchemy session."""

    def __init__(self) -> None:
        self.postings: dict[str, dict] = {}
        self.statements: list[str] = []

    # context-manager protocol (session_factory returns self)
    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def commit(self) -> None:
        return None

    # -- ORM surface used by the indexer -------------------------------
    def merge(self, obj: object) -> None:
        self.postings[obj.chunk_id] = {  # type: ignore[attr-defined]
            "chunk_id": obj.chunk_id,  # type: ignore[attr-defined]
            "document_id": obj.document_id,  # type: ignore[attr-defined]
            "text": obj.text,  # type: ignore[attr-defined]
            "source": obj.source,  # type: ignore[attr-defined]
            "page_number": obj.page_number,  # type: ignore[attr-defined]
            "section": obj.section,  # type: ignore[attr-defined]
        }

    def get(self, _model: object, key: str) -> object | None:
        from types import SimpleNamespace

        src = self.postings.get(key)
        return SimpleNamespace(**src) if src is not None else None

    def query(self, _model: object) -> _FakeQuery:
        return _FakeQuery(self)

    # -- Core surface ----------------------------------------------------
    def execute(self, stmt: object, params: dict | None = None):
        sql = str(getattr(stmt, "text", stmt))
        self.statements.append(sql)
        params = params or {}
        if "ts_rank_cd" in sql:
            return _FakeMappingsResult(self._search(params))
        if "COUNT" in sql.upper():
            where = getattr(stmt, "whereclause", None)
            rows = [
                src for src in self.postings.values()
                if where is None or _where_matches(src, where)
            ]
            return _FakeScalarResult(len(rows))
        # DDL / anything else: succeed silently.
        return _FakeMappingsResult([])

    # -- emulation --------------------------------------------------------
    def _search(self, params: dict) -> list[dict]:
        query = params.get("q", "")
        wanted = params.get("f")
        banned = set(params.get("e") or [])
        k = params.get("k", 20)
        q_tokens = set(_tokens(query))
        scored = []
        for pid, src in self.postings.items():
            if wanted is not None and src["document_id"] not in wanted:
                continue
            if src["document_id"] in banned:
                continue
            hay = set(_tokens(src["text"])) | set(_tokens(src.get("section") or ""))
            overlap = len(q_tokens & hay)
            if overlap == 0:
                continue
            scored.append((overlap, pid, src))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [
            {
                "chunk_id": pid,
                "document_id": src["document_id"],
                "text": src["text"],
                "source": src["source"],
                "page_number": src["page_number"],
                "section": src["section"],
                "score": float(overlap),
            }
            for overlap, pid, src in scored[:k]
        ]


class _FakeQuery:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session
        self._filters: list = []

    def filter(self, *criteria) -> _FakeQuery:
        self._filters.extend(criteria)
        return self

    def _matches(self, src: dict) -> bool:
        return all(_where_matches(src, crit) for crit in self._filters)

    def delete(self, synchronize_session: bool = False) -> int:  # noqa: ARG002
        doomed = [pid for pid, src in self._session.postings.items() if self._matches(src)]
        for pid in doomed:
            del self._session.postings[pid]
        return len(doomed)


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def fake_session() -> _FakeSession:
    bm25_mod._ENSURED.clear()
    return _FakeSession()


@pytest.fixture
def indexer(fake_session) -> BM25Indexer:
    return BM25Indexer(settings=None, session_factory=lambda: fake_session)


def _make_chunks(document_id: str, texts: list[str]) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{document_id}:{i}",
            document_id=document_id,
            document_name=document_id,
            text=t,
            source=f"md/{document_id}.md",
            page_number=None,
            section=None,
            index=i,
            content_hash=content_hash(t),
        )
        for i, t in enumerate(texts)
    ]


# ---------------------------------------------------------------- tests


def test_ensure_index_runs_tsvector_ddl(fake_session, indexer) -> None:
    indexer.ensure_index()
    joined = "\n".join(fake_session.statements)
    assert "tsvector" in joined
    assert "GIN" in joined
    assert "keyword_postings_tsv_trigger" in joined


def test_ensure_index_idempotent(fake_session, indexer) -> None:
    indexer.ensure_index()
    n = len(fake_session.statements)
    indexer.ensure_index()  # memoized: no new statements
    assert len(fake_session.statements) == n


def test_index_documents_upserts_postings(indexer, fake_session) -> None:
    chunks = _make_chunks("auth", ["OAuth tokens expire after 60 minutes.", "Rate limits are hourly."])
    indexer.index_documents(chunks)
    assert set(fake_session.postings.keys()) == {"auth:0", "auth:1"}
    assert fake_session.postings["auth:0"]["text"] == "OAuth tokens expire after 60 minutes."


def test_index_documents_empty_noop(indexer, fake_session) -> None:
    indexer.index_documents([])
    assert fake_session.postings == {}


def test_upsert_returns_count(indexer) -> None:
    chunks = _make_chunks("auth", ["a", "b", "c"])
    assert indexer.upsert(chunks) == 3


def test_search_uses_ts_rank(indexer, fake_session) -> None:
    indexer.index_documents(_make_chunks("a", ["hello world"]))
    indexer.search("hello")
    joined = "\n".join(fake_session.statements)
    assert "ts_rank_cd" in joined
    assert "websearch_to_tsquery" in joined


def test_search_basic(indexer) -> None:
    chunks = _make_chunks(
        "auth",
        [
            "OAuth access tokens expire after 60 minutes.",
            "Rate limits reset every hour.",
            "Use HTTPS for all production traffic.",
        ],
    )
    indexer.index_documents(chunks)
    res = indexer.search("OAuth tokens")
    assert len(res) >= 1
    assert res[0]["id"] == "auth:0"
    assert res[0]["score"] > 0
    assert "OAuth" in res[0]["payload"]["text"]


def test_search_blank_query_short_circuits(indexer, fake_session) -> None:
    indexer.index_documents(_make_chunks("a", ["hello world"]))
    n_before = len(fake_session.statements)
    assert indexer.search("   ") == []
    assert len(fake_session.statements) == n_before  # no SQL issued


def test_search_filter_document_ids(indexer) -> None:
    indexer.index_documents(_make_chunks("a", ["OAuth tokens"]))
    indexer.index_documents(_make_chunks("b", ["OAuth tokens"]))
    res = indexer.search("OAuth", filter_document_ids=["a"])
    assert {r["payload"]["document_id"] for r in res} == {"a"}


def test_search_exclude_document_ids(indexer) -> None:
    indexer.index_documents(_make_chunks("a", ["OAuth tokens"]))
    indexer.index_documents(_make_chunks("b", ["OAuth tokens"]))
    res = indexer.search("OAuth", exclude_document_ids=["b"])
    assert {r["payload"]["document_id"] for r in res} == {"a"}


def test_search_section_weighted(indexer) -> None:
    chunks = _make_chunks("a", ["plain body", "another body"])
    chunks[1].section = "OAuth troubleshooting"
    indexer.index_documents(chunks)
    res = indexer.search("OAuth")
    assert res[0]["id"] == "a:1"


def test_count_and_by_document(indexer) -> None:
    indexer.index_documents(_make_chunks("a", ["x", "y"]))
    indexer.index_documents(_make_chunks("b", ["z"]))
    assert indexer.count() == 3
    assert indexer.count(document_id="a") == 2


def test_get_returns_posting(indexer) -> None:
    chunks = _make_chunks("a", ["hello world"])
    indexer.index_documents(chunks)
    src = indexer.get("a:0")
    assert src is not None
    assert src["text"] == "hello world"


def test_get_missing_returns_none(indexer) -> None:
    assert indexer.get("nope") is None


def test_delete_by_document(indexer, fake_session) -> None:
    indexer.index_documents(_make_chunks("a", ["x", "y"]))
    indexer.index_documents(_make_chunks("b", ["z"]))
    assert indexer.delete_by_document("a") == 2
    assert fake_session.postings.keys() == {"b:0"}


def test_delete_by_ids(indexer, fake_session) -> None:
    indexer.index_documents(_make_chunks("a", ["x", "y", "z"]))
    assert indexer.delete_by_ids(["a:0", "a:2"]) == 2
    assert set(fake_session.postings.keys()) == {"a:1"}


def test_delete_by_ids_empty(indexer) -> None:
    assert indexer.delete_by_ids([]) == 0


def test_delete_index_clears_postings(indexer, fake_session) -> None:
    indexer.ensure_index()
    indexer.index_documents(_make_chunks("a", ["x"]))
    indexer.delete_index()
    assert fake_session.postings == {}
