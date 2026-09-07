---
doc_id: security-guide
title: Security Guide
version: 1.0
module: security
last_updated: 2026-09-06
---


## Encryption (#encryption)

All data at rest is encrypted with AES-256 at rest. Data in transit uses TLS 1.2 or later.

Customer encryption keys are supported for enterprise plans. If you provision a customer key, AskMyDocs cannot recover data if the key is lost.

## Audit Logging (#audit-logging)

Security-relevant events are recorded in the audit log: sign-in, sign-out, permission changes, API key creation/revocation, and webhook secret rotation.

Audit logs are retained for 180 days by default and can be exported to an external SIEM through the Admin Console.

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

