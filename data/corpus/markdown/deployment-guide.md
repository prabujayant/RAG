---
doc_id: deployment-guide
title: Deployment Guide
version: 1.0
module: deployment
last_updated: 2026-09-06
---


## Deployment Overview (#deployment-overview)

The AskMyDocs platform has a guaranteed uptime SLA of 99.95% for the Enterprise plan.

The standard topology deploys the API service, the ingestion worker, and the Identity Service. The ingestion worker processes documents asynchronously from a queue.

| Component | Purpose |
| --- | --- |
| api | Serves the REST API |
| ingestion-worker | Parses and indexes documents |
| identity | OAuth tokens and user management |
| postgres | Application metadata |
| qdrant | Vector search |
| opensearch | BM25 full-text search |

## Environments (#environments)

The platform supports three deployment environments.

| Environment | Purpose | Data isolation |
| --- | --- | --- |
| development | Local iteration | Full isolation |
| staging | Pre-production validation | Shared test data |
| production | Live traffic | Full isolation |

> **Warning**: Do not use production credentials in development or staging.

## Database Migrations (#database-migrations)

Schema changes are applied through versioned migrations. Migrations run automatically at deploy time before new code is released.

Downgrades are not supported automatically. Always back up before deploying a migration.

