---
doc_id: troubleshooting-guide
title: Troubleshooting Guide
version: 1.0
module: troubleshooting
last_updated: 2026-09-06
---


## Common Issues (#common-issues)

This section covers frequently reported issues and their resolutions.

| Symptom | Likely cause | Resolution |
| --- | --- | --- |
| 401 from all endpoints | Expired access token | Refresh the token before expiry |
| Documents not searchable after upload | Ingestion job failed or still running | Check ingestion status; retry the job |
| Slow retrieval | Reranker enabled on large candidate sets | Reduce BM25/VECTOR_TOP_K |
| 429 responses | Rate limit exceeded | Back off using Retry-After |
| Answers marked ungrounded | Evidence insufficient or citation mismatch | Re-ask with more specific wording |

## Rotating Secrets (#rotating-secrets)

Webhook secrets can be rotated from the Admin Console. After rotation, signatures generated with the old secret fail verification immediately. Rotate during low-traffic windows.

API keys are single-secret; rotating an API key invalidates the previous key immediately.

