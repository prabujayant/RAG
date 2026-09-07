---
doc_id: rotating-secrets
title: Rotating Secrets
version: 1.0
module: troubleshooting
last_updated: 2026-09-06
---


## Rotating Secrets (#rotating-secrets)

Webhook secrets can be rotated from the Admin Console. After rotation, signatures generated with the old secret fail verification immediately. Rotate during low-traffic windows.

API keys are single-secret; rotating an API key invalidates the previous key immediately.

