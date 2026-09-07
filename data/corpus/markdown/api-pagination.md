---
doc_id: api-pagination
title: Pagination Guide
version: 1.0
module: api
last_updated: 2026-09-06
---


## Pagination (#pagination)

List endpoints paginate using cursor-based pagination. The response contains a ``next_cursor`` field; pass it as the ``cursor`` parameter to fetch the next page.

| Parameter | Type | Description |
| --- | --- | --- |
| limit | int | Max items per page (1-100, default 20) |
| cursor | string | Opaque pagination cursor |

## Common Headers (#common-headers)

| Header | Required | Description |
| --- | --- | --- |
| Authorization | Yes | Bearer token |
| Idempotency-Key | Write ops | Prevent duplicate writes |
| X-Request-ID | No | Correlate logs; echoed in response |
| Accept-Version | No | Pin API version (default v1) |

