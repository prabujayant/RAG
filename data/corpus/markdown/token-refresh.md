---
doc_id: token-refresh
title: Token Refresh and Rotation
version: 1.0
module: authentication
last_updated: 2026-09-06
---


## Refresh Token Rotation (#refresh-token-rotation)

Refresh tokens expire after 30 days. AskMyDocs rotates refresh tokens on every use: the previous refresh token is invalidated immediately.

If a rotated refresh token is used again (token replay), the entire session is revoked and the user is forced to re-authenticate. Clients must therefore persist each new refresh token before using the access token it returns.

## Token Lifecycle (#token-lifecycle)

Access tokens are short-lived. By default, an access token expires after 60 minutes. Refresh tokens are long-lived and expire after 30 days.

| Token type | Lifetime | Transport |
| --- | --- | --- |
| Access token (JWT) | 60 minutes | Authorization header |
| Refresh token (opaque) | 30 days | Secure cookie or POST body |

When an access token expires, the client must use the refresh token to obtain a new one. The refresh flow is described in the OAuth guide.

> **Warning**: Never store refresh tokens in browser local storage. Use a secure, HttpOnly cookie.

