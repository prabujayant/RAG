---
doc_id: configuration-guide
title: Configuration Guide
version: 1.0
module: configuration
last_updated: 2026-09-06
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
| RERANKER_MODEL | BAAI/bge-reranker-v2-m3 | Cross-encoder reranker |
| OPENROUTER_API_KEY |  | LLM provider key |
| CHUNK_SIZE | 512 | Chunk size (tokens) |
| CHUNK_OVERLAP | 64 | Chunk overlap |

> **Tip**: Never commit secrets to version control. Use a secret manager in production.

## Feature Flags (#feature-flags)

Flags are evaluated server-side. Flags are boolean by default; some flags accept a percentage rollout or a targeting rule.

| Flag | Default | Effect |
| --- | --- | --- |
| rag.hybrid_search | true | Enable hybrid BM25 + vector retrieval |
| rag.reranker | true | Enable cross-encoder reranking |
| citations.require | true | Require citations on answers |
| experimental.new_parser | false | Use new document parser |

