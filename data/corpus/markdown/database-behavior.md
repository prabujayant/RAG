---
doc_id: database-behavior
title: Database Behavior
version: 1.0
module: database
last_updated: 2026-09-06
---


## Database Behavior (#database-behavior)

The platform stores metadata in PostgreSQL 16. The connection pool has a maximum of 100 connections per instance.

The database stores documents, chunks, ingestion jobs, queries, and evaluation runs. Vector embeddings are stored in Qdrant, not in PostgreSQL.

Backups are taken every 12 hours and retained for 30 days.

