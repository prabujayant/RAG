---
doc_id: rate-limits
title: Rate Limits and Quotas
version: 1.0
module: rate-limits
last_updated: 2026-09-12
---


## Rate Limits (#rate-limits)

The default rate limit is 100 requests per minute.Short bursts up to 300 requests per minute are allowed.

Rate limits apply per API key, per endpoint. Responses include ``X-RateLimit-Limit``, ``X-RateLimit-Remaining``, and ``X-RateLimit-Reset`` headers.

> **Warning**: Hitting the rate limit returns HTTP 429 with a ``Retry-After`` header. Respect the header; do not retry immediately.

## Quotas by Plan (#quotas-by-plan)

Rate limits and quotas vary by plan.

| Plan | Default rate | Burst | Max documents |
| --- | --- | --- | --- |
| Free | 30 req/min | 60 req/min | 100 |
| Pro | 100 requests per minute | 300 requests per minute | 10,000 |
| Enterprise | 1,000 req/min | 2,000 req/min | Unlimited |

