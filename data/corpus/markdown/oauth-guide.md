---
doc_id: oauth-guide
title: OAuth 2.0 Integration Guide
version: 1.0
module: oauth
last_updated: 2026-09-12
---


## Overview (#overview)

The AskMyDocs platform implements OAuth 2.0 to let third-party applications access customer resources on the user's behalf.

The authorization server is the Identity Service. It supports the Authorization Code grant flow with Proof Key for Code Exchange (PKCE), the Client Credentials grant, and the Refresh Token grant.

## Supported Flows (#supported-flows)

| Flow | Use case | Actor |
| --- | --- | --- |
| Authorization Code + PKCE | Browser, mobile, SPAs | End user |
| Client Credentials | Server-to-server | Service account |
| Refresh Token | Rotate access token | End user or service |

For browser-based applications, the Authorization Code flow with PKCE is recommended. The implicit flow is not supported.

## Scopes (#scopes)

Scopes are space-delimited strings. The platform defines the following standard scopes:

| Scope | Grants |
| --- | --- |
| openid | Identity information |
| profile | Read user profile |
| documents:read | Read documents |
| documents:write | Create, edit, delete documents |
| admin | Admin Console access (superuser) |

Scopes are enforced by the API gateway. A token without the ``documents:write`` scope receives 403 for write operations.

## Token Claims (#token-claims)

Access tokens are JWTs signed with RS256. Standard claims: ``iss``, ``sub``, ``aud``, ``exp``, ``iat``, ``scope``, and ``permissions``.

The ``permissions`` claim is an array of permission strings such as ``documents:read``. Services must validate the signature using the Identity Service JWKS endpoint before trusting any claim.

## Discovery (#discovery)

OAuth configuration is published at the well-known endpoint:

```text
https://identity.askmydocs.example/.well-known/openid-configuration
```

The discovery document lists authorization, token, revocation, and JWKS endpoints. Clients must fetch this document at startup and honor the ``expiration`` hints.

## Refresh Token Rotation (#refresh-token-rotation)

Refresh tokens expire after 30 days. AskMyDocs rotates refresh tokens on every use: the previous refresh token is invalidated immediately.

If a rotated refresh token is used again (token replay), the entire session is revoked and the user is forced to re-authenticate. Clients must therefore persist each new refresh token before using the access token it returns.

