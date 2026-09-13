---
doc_id: webhooks-guide
title: Webhooks Guide
version: 1.0
module: webhooks
last_updated: 2026-09-12
---


## Webhooks (#webhooks)

Webhooks notify your system about events such as document processed, document failed, and ingestion complete.

Each delivery is signed with HMAC-SHA256 signature using the webhook secret. The signature is sent in the ``X-AskMyDocs-Signature`` header as ``t=<timestamp>,v1=<hex>``.

Delivery is retried up to 5 retries, exponential backoff, max 10 minutes.

```python
# Verify a webhook signature
import hashlib, hmac

def verify(secret, body, header):
    timestamp, signature = header.split(',')
    digest = hmac.new(secret.encode(), f'{timestamp}.{body}'.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, signature.split('=')[1])
```

## Webhook Events (#webhook-events)

The platform emits the following events.

| Event | Trigger |
| --- | --- |
| document.uploaded | A document was uploaded |
| document.ingested | Ingestion completed successfully |
| document.failed | Ingestion failed |
| query.completed | A query completed with citations |

## Registering Endpoints (#registering-endpoints)

Register up to 10 webhook endpoints per project from the Admin Console. Each endpoint can filter events by type.

Endpoints that fail 8 consecutive deliveries are automatically disabled and an alert is raised.

