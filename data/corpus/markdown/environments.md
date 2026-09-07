---
doc_id: environments
title: Environments
version: 1.0
module: deployment
last_updated: 2026-09-06
---


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

