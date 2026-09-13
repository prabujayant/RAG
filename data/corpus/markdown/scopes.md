---
doc_id: scopes
title: OAuth Scopes and Permissions
version: 1.0
module: oauth
last_updated: 2026-09-12
---


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

