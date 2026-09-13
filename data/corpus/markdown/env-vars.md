---
doc_id: env-vars
title: Environment Variables Reference
version: 1.0
module: configuration
last_updated: 2026-09-12
---


## Environment Variables (#environment-variables)

The platform is configured through environment variables. Key variables:

| Variable | Default | Description |
| --- | --- | --- |
| APP_ENV | development | Runtime environment |
| DATABASE_URL |  | PostgreSQL connection string |
| QDRANT_URL | http://localhost:6333 | Qdrant vector database |
| OPENSEARCH_URL | http://localhost:9200 | OpenSearch BM25 index |
| EMBEDDING_MODEL | BAAI/bge-m3 | Embedding model |
| RERANKER_MODEL | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | Cross-encoder reranker |
| OPENROUTER_API_KEY |  | LLM provider key |
| CHUNK_SIZE | 512 | Chunk size (tokens) |
| CHUNK_OVERLAP | 64 | Chunk overlap |

> **Tip**: Never commit secrets to version control. Use a secret manager in production.

