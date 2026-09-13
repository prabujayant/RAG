---
doc_id: user-management
title: User Management
version: 1.0
module: users
last_updated: 2026-09-12
---


## Roles (#roles)

Permissions are assigned through roles. The platform ships with four default roles.

| Role | Permissions |
| --- | --- |
| owner | Full access including billing |
| admin | Manage users, projects, and settings |
| editor | Create and edit documents |
| viewer | Read-only access |

Roles can be scoped per project. Custom roles can be defined with an arbitrary combination of permissions.

## SCIM Provisioning (#scim-provisioning)

SCIM 2.0 is supported on the Enterprise plan. When users are deprovisioned in the identity provider, their access is revoked within 5 minutes.

