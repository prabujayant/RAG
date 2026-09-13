---
doc_id: api-keys
title: API Keys Management
version: 1.0
module: authentication
last_updated: 2026-09-12
---


## API Keys (#api-keys)

API keys are the recommended credential for server-to-server integrations. Keys are prefixed with ``amsk_`` and are shown only once at creation time.

An API key can be scoped to a set of permissions using the permissions claim. Scoping keys is strongly recommended.

Keys expire after 90 days and can be revoked at any time from the Admin Console.

```bash
# Example: authenticate with an API key
curl -H "Authorization: Bearer amsk_live_abc123" \
  https://api.askmydocs.example/v1/projects
```

## Basic Authentication for Legacy Integrations (#basic-authentication-for-legacy-integrations)

Basic authentication is supported only for legacy integrations and is disabled by default. Use OAuth 2.0 or API keys for new integrations.

> **Note**: Basic auth is not available for accounts enrolled in SSO.

