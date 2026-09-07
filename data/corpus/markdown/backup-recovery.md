---
doc_id: backup-recovery
title: Backup and Recovery
version: 1.0
module: database
last_updated: 2026-09-06
---


## Database Behavior (#database-behavior)

The platform stores metadata in PostgreSQL 16. The connection pool has a maximum of 100 connections per instance.

The database stores documents, chunks, ingestion jobs, queries, and evaluation runs. Vector embeddings are stored in Qdrant, not in PostgreSQL.

Backups are taken every 12 hours and retained for 30 days.

## Database Migrations (#database-migrations)

Schema changes are applied through versioned migrations. Migrations run automatically at deploy time before new code is released.

Downgrades are not supported automatically. Always back up before deploying a migration.

