---
doc_id: api-idempotency
title: Idempotency Guide
version: 1.0
module: api
last_updated: 2026-09-12
---


## Idempotency (#idempotency)

Write endpoints accept an ``Idempotency-Key`` header. If a request with the same key is retried, the server returns the original response without applying the change twice.

Idempotency keys are honored for 24 hours. Use UUID v4 values.

## Common Headers (#common-headers)

| Header | Required | Description |
| --- | --- | --- |
| Authorization | Yes | Bearer token |
| Idempotency-Key | Write ops | Prevent duplicate writes |
| X-Request-ID | No | Correlate logs; echoed in response |
| Accept-Version | No | Pin API version (default v1) |

