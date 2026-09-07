-- AskMyDocs PostgreSQL bootstrap script (runs once on first container start).
-- Creates the schema extension used by identifier generation.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";