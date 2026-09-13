---
doc_id: webhook-signing
title: Webhook Signature Verification
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

