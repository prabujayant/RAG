"""Unit tests for the SQLAlchemy ORM models (Phase 2).

These validate schema structure and default behaviour against an in-memory
SQLite engine. No PostgreSQL instance is required for unit tests.
"""

from __future__ import annotations

import uuid

import pytest
from app.db.models import (
    Base,
    Chunk,
    Document,
    EvaluationRun,
    IngestionJob,
    QueryRecord,
    gen_uuid,
)
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

TABLES = {
    "documents",
    "chunks",
    "ingestion_jobs",
    "queries",
    "evaluation_runs",
}

@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()

@pytest.fixture
def session(engine) -> Session:
    with Session(engine) as s:
        yield s

def test_all_tables_created(engine) -> None:
    names = set(inspect(engine).get_table_names())
    assert TABLES.issubset(names)

def test_document_pk_and_columns(engine) -> None:
    cols = {c["name"]: c for c in inspect(engine).get_columns("documents")}
    assert cols["id"]["primary_key"]  # primary key column
    assert "title" in cols
    assert "module" in cols
    assert "format" in cols

def test_chunk_foreign_key(engine) -> None:
    fks = inspect(engine).get_foreign_keys("chunks")
    assert any(fk["constrained_columns"] == ["document_id"] for fk in fks)
    cols = {c["name"] for c in inspect(engine).get_columns("chunks")}
    assert {"id", "chunk_index", "text", "content_hash", "section"}.issubset(cols)

def test_ingestion_job_columns(engine) -> None:
    cols = {c["name"] for c in inspect(engine).get_columns("ingestion_jobs")}
    assert {"status", "error", "chunk_count", "document_id"}.issubset(cols)

def test_query_record_columns(engine) -> None:
    cols = {c["name"] for c in inspect(engine).get_columns("queries")}
    assert {
        "query_id",
        "question",
        "answer",
        "grounded",
        "confidence",
        "retrieved_chunk_ids",
        "citation_ids",
    }.issubset(cols)

def test_document_roundtrip(session) -> None:
    doc = Document(
        document_id="auth-guide",
        title="Auth Guide",
        module="auth",
        source="data/corpus/auth-guide.md",
        format="md",
    )
    session.add(doc)
    session.commit()

    fetched = session.get(Document, doc.id)
    assert fetched is not None
    assert fetched.document_id == "auth-guide"
    assert fetched.format == "md"

def test_document_chunk_relationship(session) -> None:
    doc = Document(
        document_id="auth-guide",
        title="Auth Guide",
        module="auth",
        source="s",
        format="md",
    )
    doc.chunks.append(
        Chunk(
            id="auth-guide:0",
            document_id="auth-guide",
            chunk_index=0,
            text="hello",
            source="s",
            content_hash="abc",
        )
    )
    session.add(doc)
    session.commit()

    db_doc = session.get(Document, doc.id)
    assert len(db_doc.chunks) == 1
    assert db_doc.chunks[0].id == "auth-guide:0"
    assert db_doc.chunks[0].text == "hello"

def test_ingestion_job_default_status(session) -> None:
    job = IngestionJob(document_id="auth-guide")
    session.add(job)
    session.flush()
    assert job.id  # auto uuid assigned on flush
    assert job.status == "pending"
    session.commit()
    assert session.get(IngestionJob, job.id).status == "pending"

def test_query_record_json_defaults(session) -> None:
    q = QueryRecord(query_id="q-1", question="What is X?")
    session.add(q)
    session.flush()  # Python-side defaults populate at flush
    assert q.retrieved_chunk_ids == []
    assert q.citation_ids == []
    session.commit()
    db_q = session.get(QueryRecord, q.id)
    assert db_q.retrieved_chunk_ids == []
    assert db_q.citation_ids == []
    assert db_q.query_id == "q-1"

def test_evaluation_run_defaults(session) -> None:
    run = EvaluationRun(run_id="run-1", experiment="A")
    assert run.passed is None
    session.add(run)
    session.commit()
    db_run = session.get(EvaluationRun, run.id)
    assert db_run.experiment == "A"
    assert db_run.config == {}

def test_gen_uuid_returns_unique() -> None:
    a, b = gen_uuid(), gen_uuid()
    assert uuid.UUID(a)  # parseable as uuid
    assert a != b

def test_document_id_generated_on_flush(session) -> None:
    doc = Document(
        document_id="rate-limits",
        title="Rate Limits",
        module="api",
        source="s",
        format="md",
    )
    session.add(doc)
    session.flush()
    assert doc.id is not None
    assert uuid.UUID(doc.id)
