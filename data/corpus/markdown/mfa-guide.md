---
doc_id: mfa-guide
title: Multi-Factor Authentication
version: 1.0
module: authentication
last_updated: 2026-09-06
---


## Multi-Factor Authentication (MFA) (#multi-factor-authentication-mfa)

MFA adds a second verification factor on top of a password. Supported factors: TOTP authenticator apps, SMS, and WebAuthn security keys.

MFA is required for administrators by default. Users can enroll multiple devices, but only the most recently enrolled device is active at any time.

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

