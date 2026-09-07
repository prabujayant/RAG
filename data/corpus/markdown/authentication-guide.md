---
doc_id: authentication-guide
title: Authentication Guide
version: 1.0
module: authentication
last_updated: 2026-09-06
---


This guide describes how authentication works for the AskMyDocs platform.

Every API request must present credentials. AskMyDocs supports OAuth 2.0 bearer tokens as the primary mechanism for service accounts and API keys for long-running integrations.

The platform issues tokens through a dedicated Identity Service. The Identity Service is the only component allowed to mint or refresh tokens. Other services validate tokens against the Identity Service's public key endpoint.

## Token Lifecycle (#token-lifecycle)

Access tokens are short-lived. By default, an access token expires after 60 minutes. Refresh tokens are long-lived and expire after 30 days.

| Token type | Lifetime | Transport |
| --- | --- | --- |
| Access token (JWT) | 60 minutes | Authorization header |
| Refresh token (opaque) | 30 days | Secure cookie or POST body |

When an access token expires, the client must use the refresh token to obtain a new one. The refresh flow is described in the OAuth guide.

> **Warning**: Never store refresh tokens in browser local storage. Use a secure, HttpOnly cookie.

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

## Authentication Errors (#authentication-errors)

The Identity Service returns the following error codes:

| Error code | HTTP status | Meaning |
| --- | --- | --- |
| invalid_client | 401 | Client id or secret is invalid |
| invalid_grant | 400 | Refresh token is invalid or expired |
| invalid_token | 401 | Access token is invalid or expired |
| insufficient_scope | 403 | Token lacks required scope |
| temporarily_unavailable | 503 | Identity Service is unavailable |

> **Tip**: Clients should treat 401 responses as a signal to re-authenticate, not to retry the same request.

