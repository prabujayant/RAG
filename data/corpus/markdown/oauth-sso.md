---
doc_id: oauth-sso
title: SSO and Federation
version: 1.0
module: oauth
last_updated: 2026-09-12
---


## SSO and Federation (#sso-and-federation)

Enterprise SSO is available on the Enterprise plan. AskMyDocs supports SAML 2.0 and OpenID Connect identity providers.

When SSO is enabled, password authentication is disabled for all users in that organization. SCIM is supported for automatic user provisioning and deprovisioning.

## Discovery (#discovery)

OAuth configuration is published at the well-known endpoint:

```text
https://identity.askmydocs.example/.well-known/openid-configuration
```

The discovery document lists authorization, token, revocation, and JWKS endpoints. Clients must fetch this document at startup and honor the ``expiration`` hints.

