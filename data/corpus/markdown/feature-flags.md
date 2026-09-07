---
doc_id: feature-flags
title: Feature Flags
version: 1.0
module: configuration
last_updated: 2026-09-06
---


## Feature Flags (#feature-flags)

Flags are evaluated server-side. Flags are boolean by default; some flags accept a percentage rollout or a targeting rule.

| Flag | Default | Effect |
| --- | --- | --- |
| rag.hybrid_search | true | Enable hybrid BM25 + vector retrieval |
| rag.reranker | true | Enable cross-encoder reranking |
| citations.require | true | Require citations on answers |
| experimental.new_parser | false | Use new document parser |

