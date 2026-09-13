---
doc_id: api-errors
title: API Error Reference
version: 1.0
module: api
last_updated: 2026-09-12
---


## Error Response Format (#error-response-format)

All errors use a consistent JSON envelope:

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit exceeded. Retry after 5 seconds.",
    "request_id": "req_9f1c",
    "details": {}
  }
}
```

The ``code`` field is a stable machine-readable string. The ``message`` is human-readable and may change.

## Error Codes (#error-codes)

| HTTP | Code | Common cause |
| --- | --- | --- |
| 400 | invalid_request | Malformed request body |
| 401 | unauthenticated | Missing or invalid token |
| 403 | permission_denied | Token lacks required scope |
| 404 | not_found | Resource does not exist |
| 409 | conflict | Resource already exists or state conflict |
| 422 | validation_failed | Request body failed validation |
| 429 | rate_limit_exceeded | Too many requests |
| 5xx | internal_error | Server-side failure |

Retryable errors are those in the 5xx range, 429, and 408. Client code should only retry those statuses, with exponential backoff and jitter.

## Troubleshooting Common Errors (#troubleshooting-common-errors)

**401 unauthenticated**: verify the token is not expired and is sent in the Authorization header.

**403 permission_denied**: the token lacks the required scope. Request the scope in the OAuth consent screen.

**429 rate_limit_exceeded**: inspect the Retry-After header and back off.

**5xx internal_error**: check the status page; retry with exponential backoff.

