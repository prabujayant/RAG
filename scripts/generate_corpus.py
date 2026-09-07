"""Generate the AskMyDocs enterprise documentation corpus.

Produces a mixed-format tree under ``data/corpus/``:

    data/corpus/
    ├── markdown/    ~30 docs
    ├── pdf/         5 docs (authentication-guide, api-reference, security-guide,
                     deployment-guide, troubleshooting-guide)
    ├── docx/        ~5 docs
    └── html/        ~6 docs

Run:
    python scripts/generate_corpus.py [--out data/corpus] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_content import FACTS, create_auth_context, doc  # noqa: E402
from corpus_shims import (  # noqa: E402
    code_block,
    h2,
    note_box,
    para,
    table,
    write_doc,
)
from pdf_docx_render import render_docx, render_html, render_pdf  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"


# --------------------------------------------------------------------------
# Section builders (return markdown fragments)
# --------------------------------------------------------------------------


def sec_auth_intro() -> str:
    return (
        para("This guide describes how authentication works for the AskMyDocs platform.")
        + para(
            "Every API request must present credentials. AskMyDocs supports "
            "OAuth 2.0 bearer tokens as the primary mechanism for service "
            "accounts and API keys for long-running integrations."
        )
        + para(
            "The platform issues tokens through a dedicated Identity Service. "
            "The Identity Service is the only component allowed to mint or "
            "refresh tokens. Other services validate tokens against the "
            "Identity Service's public key endpoint."
        )
    )


def sec_auth_tokens() -> str:
    ctx = create_auth_context()
    return (
        h2("Token Lifecycle")
        + para(
            f"Access tokens are short-lived. By default, an access token expires after "
            f"{ctx['token_expiry']}. Refresh tokens are long-lived and expire after "
            f"{ctx['refresh_token_expiry']}."
        )
        + table(
            ["Token type", "Lifetime", "Transport"],
            [
                ["Access token (JWT)", ctx["token_expiry"], "Authorization header"],
                ["Refresh token (opaque)", ctx["refresh_token_expiry"], "Secure cookie or POST body"],
            ],
        )
        + para(
            "When an access token expires, the client must use the refresh token "
            "to obtain a new one. The refresh flow is described in the OAuth "
            "guide."
        )
        + note_box(
            "warning",
            "Never store refresh tokens in browser local storage. Use a secure, "
            "HttpOnly cookie.",
        )
    )


def sec_auth_api_keys() -> str:
    return (
        h2("API Keys")
        + para(
            "API keys are the recommended credential for server-to-server "
            "integrations. Keys are prefixed with "
            f"``{FACTS['api_key_prefix']}`` and are shown only once at creation time."
        )
        + para(
            "An API key can be scoped to a set of permissions using the "
            "permissions claim. Scoping keys is strongly recommended."
        )
        + para(
            f"Keys expire after {FACTS['api_key_lifetime']} and can be revoked at any time from the Admin Console."
        )
        + code_block(
            "bash",
            "# Example: authenticate with an API key\n"
            'curl -H "Authorization: Bearer amsk_live_abc123" \\\n'
            "  https://api.askmydocs.example/v1/projects",
        )
    )


def sec_auth_basic() -> str:
    return (
        h2("Basic Authentication for Legacy Integrations")
        + para(
            "Basic authentication is supported only for legacy integrations and "
            "is disabled by default. Use OAuth 2.0 or API keys for new "
            "integrations."
        )
        + note_box(
            "note",
            "Basic auth is not available for accounts enrolled in SSO.",
        )
    )


def sec_auth_errors() -> str:
    return (
        h2("Authentication Errors")
        + para("The Identity Service returns the following error codes:")
        + table(
            ["Error code", "HTTP status", "Meaning"],
            [
                ["invalid_client", "401", "Client id or secret is invalid"],
                ["invalid_grant", "400", "Refresh token is invalid or expired"],
                ["invalid_token", "401", "Access token is invalid or expired"],
                ["insufficient_scope", "403", "Token lacks required scope"],
                ["temporarily_unavailable", "503", "Identity Service is unavailable"],
            ],
        )
        + note_box(
            "tip",
            "Clients should treat 401 responses as a signal to re-authenticate, "
            "not to retry the same request.",
        )
    )


def sec_mfa() -> str:
    return (
        h2("Multi-Factor Authentication (MFA)")
        + para(
            "MFA adds a second verification factor on top of a password. "
            "Supported factors: TOTP authenticator apps, SMS, and WebAuthn "
            "security keys."
        )
        + para(
            "MFA is required for administrators by default. Users can enroll "
            "multiple devices, but only the most recently enrolled device is "
            "active at any time."
        )
    )


def sec_security_encryption() -> str:
    return (
        h2("Encryption")
        + para(
            f"All data at rest is encrypted with {FACTS['pii_encryption']}. "
            "Data in transit uses TLS 1.2 or later."
        )
        + para(
            "Customer encryption keys are supported for enterprise plans. "
            "If you provision a customer key, AskMyDocs cannot recover data if "
            "the key is lost."
        )
    )


def sec_security_audit() -> str:
    return (
        h2("Audit Logging")
        + para(
            "Security-relevant events are recorded in the audit log: sign-in, "
            "sign-out, permission changes, API key creation/revocation, and "
            "webhook secret rotation."
        )
        + para(
            "Audit logs are retained for 180 days by default and can be "
            "exported to an external SIEM through the Admin Console."
        )
    )


def sec_oauth_intro() -> str:
    return (
        h2("Overview")
        + para(
            f"The AskMyDocs platform implements {FACTS['oauth_provider']} to let "
            "third-party applications access customer resources on the user's "
            "behalf."
        )
        + para(
            "The authorization server is the Identity Service. It supports the "
            "Authorization Code grant flow with Proof Key for Code Exchange "
            "(PKCE), the Client Credentials grant, and the Refresh Token grant."
        )
    )


def sec_oauth_flows() -> str:
    return (
        h2("Supported Flows")
        + table(
            ["Flow", "Use case", "Actor"],
            [
                ["Authorization Code + PKCE", "Browser, mobile, SPAs", "End user"],
                ["Client Credentials", "Server-to-server", "Service account"],
                ["Refresh Token", "Rotate access token", "End user or service"],
            ],
        )
        + para(
            "For browser-based applications, the Authorization Code flow with "
            "PKCE is recommended. The implicit flow is not supported."
        )
    )


def sec_oauth_scopes() -> str:
    return (
        h2("Scopes")
        + para(
            "Scopes are space-delimited strings. The platform defines the "
            "following standard scopes:"
        )
        + table(
            ["Scope", "Grants"],
            [
                ["openid", "Identity information"],
                ["profile", "Read user profile"],
                ["documents:read", "Read documents"],
                ["documents:write", "Create, edit, delete documents"],
                ["admin", "Admin Console access (superuser)"],
            ],
        )
        + para(
            "Scopes are enforced by the API gateway. A token without the "
            "``documents:write`` scope receives 403 for write operations."
        )
    )


def sec_oauth_claims() -> str:
    return (
        h2("Token Claims")
        + para(
            "Access tokens are JWTs signed with RS256. Standard claims: "
            "``iss``, ``sub``, ``aud``, ``exp``, ``iat``, ``scope``, and "
            "``permissions``."
        )
        + para(
            "The ``permissions`` claim is an array of permission strings such "
            "as ``documents:read``. Services must validate the signature using "
            "the Identity Service JWKS endpoint before trusting any claim."
        )
    )


def sec_oauth_discovery() -> str:
    return (
        h2("Discovery")
        + para("OAuth configuration is published at the well-known endpoint:")
        + code_block(
            "text",
            "https://identity.askmydocs.example/.well-known/openid-configuration",
        )
        + para(
            "The discovery document lists authorization, token, revocation, and "
            "JWKS endpoints. Clients must fetch this document at startup and "
            "honor the ``expiration`` hints."
        )
    )


def sec_oauth_refresh() -> str:
    return (
        h2("Refresh Token Rotation")
        + para(
            f"Refresh tokens expire after {FACTS['refresh_token_expiry']}. "
            "AskMyDocs rotates refresh tokens on every use: the previous "
            "refresh token is invalidated immediately."
        )
        + para(
            "If a rotated refresh token is used again (token replay), the "
            "entire session is revoked and the user is forced to "
            "re-authenticate. Clients must therefore persist each new refresh "
            "token before using the access token it returns."
        )
    )


def sec_oauth_sso() -> str:
    return (
        h2("SSO and Federation")
        + para(
            "Enterprise SSO is available on the Enterprise plan. AskMyDocs "
            f"supports {FACTS['sso_saml']} and {FACTS['sso_oidc']} identity "
            "providers."
        )
        + para(
            "When SSO is enabled, password authentication is disabled for all "
            "users in that organization. SCIM is supported for automatic "
            "user provisioning and deprovisioning."
        )
    )


def sec_api_terminal() -> str:
    return (
        h2("API Terminal")
        + para("All API requests must use HTTPS.")
        + code_block(
            "text",
            "https://api.askmydocs.example/v1",
        )
        + para(
            "The API is versioned in the URL path. The current version is v1. "
            "Breaking changes are announced at least 6 months in advance."
        )
    )


def sec_api_auth_header() -> str:
    return (
        h2("Authentication")
        + para(
            "Every request must include an Authorization header with a bearer "
            "token:"
        )
        + code_block(
            "http",
            'GET /v1/projects HTTP/1.1\nHost: api.askmydocs.example\nAuthorization: Bearer eyJhbGciOiJSUzI1NiIs...',
        )
        + para(
            f"Tokens expire after {FACTS['token_expiry']}. Use the refresh "
            "flow before expiry to avoid 401 responses."
        )
    )


def sec_api_pagination() -> str:
    return (
        h2("Pagination")
        + para(
            "List endpoints paginate using cursor-based pagination. The response "
            "contains a ``next_cursor`` field; pass it as the ``cursor`` "
            "parameter to fetch the next page."
        )
        + table(
            ["Parameter", "Type", "Description"],
            [
                ["limit", "int", "Max items per page (1-100, default 20)"],
                ["cursor", "string", "Opaque pagination cursor"],
            ],
        )
    )


def sec_api_idempotency() -> str:
    return (
        h2("Idempotency")
        + para(
            "Write endpoints accept an ``Idempotency-Key`` header. If a "
            "request with the same key is retried, the server returns the "
            "original response without applying the change twice."
        )
        + para(
            "Idempotency keys are honored for 24 hours. Use UUID v4 values."
        )
    )


def sec_api_rate_limits() -> str:
    return (
        h2("Rate Limits")
        + para(
            f"The default rate limit is {FACTS['rate_limit_default']}."
            f"Short bursts up to {FACTS['rate_limit_burst']} are allowed."
        )
        + para(
            "Rate limits apply per API key, per endpoint. Responses include "
            "``X-RateLimit-Limit``, ``X-RateLimit-Remaining``, and "
            "``X-RateLimit-Reset`` headers."
        )
        + note_box(
            "warning",
            "Hitting the rate limit returns HTTP 429 with a ``Retry-After`` "
            "header. Respect the header; do not retry immediately.",
        )
    )


def sec_api_webhooks() -> str:
    return (
        h2("Webhooks")
        + para(
            "Webhooks notify your system about events such as document "
            "processed, document failed, and ingestion complete."
        )
        + para(
            f"Each delivery is signed with {FACTS['webhook_delivery']} using "
            "the webhook secret. The signature is sent in the "
            "``X-AskMyDocs-Signature`` header as ``t=<timestamp>,v1=<hex>``."
        )
        + para(
            f"Delivery is retried up to {FACTS['webhook_retry']}."
        )
        + code_block(
            "python",
            "# Verify a webhook signature\n"
            "import hashlib, hmac\n\n"
            "def verify(secret, body, header):\n"
            "    timestamp, signature = header.split(',')\n"
            "    digest = hmac.new(secret.encode(), f'{timestamp}.{body}'.encode(), hashlib.sha256).hexdigest()\n"
            "    return hmac.compare_digest(digest, signature.split('=')[1])",
        )
    )


def sec_api_headers() -> str:
    return (
        h2("Common Headers")
        + table(
            ["Header", "Required", "Description"],
            [
                ["Authorization", "Yes", "Bearer token"],
                ["Idempotency-Key", "Write ops", "Prevent duplicate writes"],
                ["X-Request-ID", "No", "Correlate logs; echoed in response"],
                ["Accept-Version", "No", "Pin API version (default v1)"],
            ],
        )
    )


def sec_errors_summary() -> str:
    return (
        h2("Error Response Format")
        + para("All errors use a consistent JSON envelope:")
        + code_block(
            "json",
            '{\n  "error": {\n    "code": "rate_limit_exceeded",\n'
            '    "message": "Rate limit exceeded. Retry after 5 seconds.",\n'
            '    "request_id": "req_9f1c",\n    "details": {}\n  }\n}',
        )
        + para(
            "The ``code`` field is a stable machine-readable string. The "
            "``message`` is human-readable and may change."
        )
    )


def sec_errors_codes() -> str:
    return (
        h2("Error Codes")
        + table(
            ["HTTP", "Code", "Common cause"],
            [
                ["400", "invalid_request", "Malformed request body"],
                ["401", "unauthenticated", "Missing or invalid token"],
                ["403", "permission_denied", "Token lacks required scope"],
                ["404", "not_found", "Resource does not exist"],
                ["409", "conflict", "Resource already exists or state conflict"],
                ["422", "validation_failed", "Request body failed validation"],
                ["429", "rate_limit_exceeded", "Too many requests"],
                ["5xx", "internal_error", "Server-side failure"],
            ],
        )
        + para(
            "Retryable errors are those in the 5xx range, 429, and 408. "
            "Client code should only retry those statuses, with exponential "
            "backoff and jitter."
        )
    )


def sec_errors_troubleshoot() -> str:
    return (
        h2("Troubleshooting Common Errors")
        + para("**401 unauthenticated**: verify the token is not expired and is sent in the Authorization header.")
        + para("**403 permission_denied**: the token lacks the required scope. Request the scope in the OAuth consent screen.")
        + para("**429 rate_limit_exceeded**: inspect the Retry-After header and back off.")
        + para("**5xx internal_error**: check the status page; retry with exponential backoff.")
    )


def sec_config_env() -> str:
    return (
        h2("Environment Variables")
        + para("The platform is configured through environment variables. Key variables:")
        + table(
            ["Variable", "Default", "Description"],
            [
                ["APP_ENV", "development", "Runtime environment"],
                ["DATABASE_URL", "", "PostgreSQL connection string"],
                ["QDRANT_URL", "http://localhost:6333", "Qdrant vector database"],
                ["OPENSEARCH_URL", "http://localhost:9200", "OpenSearch BM25 index"],
                ["EMBEDDING_MODEL", "BAAI/bge-m3", "Embedding model"],
                ["RERANKER_MODEL", "BAAI/bge-reranker-v2-m3", "Cross-encoder reranker"],
                ["OPENROUTER_API_KEY", "", "LLM provider key"],
                ["CHUNK_SIZE", "512", "Chunk size (tokens)"],
                ["CHUNK_OVERLAP", "64", "Chunk overlap"],
            ],
        )
        + note_box(
            "tip",
            "Never commit secrets to version control. Use a secret manager in "
            "production.",
        )
    )


def sec_config_flags() -> str:
    return (
        h2("Feature Flags")
        + para(
            "Flags are evaluated server-side. Flags are boolean by default; "
            "some flags accept a percentage rollout or a targeting rule."
        )
        + table(
            ["Flag", "Default", "Effect"],
            [
                ["rag.hybrid_search", "true", "Enable hybrid BM25 + vector retrieval"],
                ["rag.reranker", "true", "Enable cross-encoder reranking"],
                ["citations.require", "true", "Require citations on answers"],
                ["experimental.new_parser", "false", "Use new document parser"],
            ],
        )
    )


def sec_deploy_overview() -> str:
    return (
        h2("Deployment Overview")
        + para(
            f"The AskMyDocs platform has a guaranteed uptime SLA of {FACTS['sla_uptime']} "
            "for the Enterprise plan."
        )
        + para(
            "The standard topology deploys the API service, the ingestion "
            "worker, and the Identity Service. The ingestion worker processes "
            "documents asynchronously from a queue."
        )
        + table(
            ["Component", "Purpose"],
            [
                ["api", "Serves the REST API"],
                ["ingestion-worker", "Parses and indexes documents"],
                ["identity", "OAuth tokens and user management"],
                ["postgres", "Application metadata"],
                ["qdrant", "Vector search"],
                ["opensearch", "BM25 full-text search"],
            ],
        )
    )


def sec_deploy_envs() -> str:
    return (
        h2("Environments")
        + para("The platform supports three deployment environments.")
        + table(
            ["Environment", "Purpose", "Data isolation"],
            [
                ["development", "Local iteration", "Full isolation"],
                ["staging", "Pre-production validation", "Shared test data"],
                ["production", "Live traffic", "Full isolation"],
            ],
        )
        + note_box(
            "warning",
            "Do not use production credentials in development or staging.",
        )
    )


def sec_deploy_migrations() -> str:
    return (
        h2("Database Migrations")
        + para(
            "Schema changes are applied through versioned migrations. "
            "Migrations run automatically at deploy time before new code is "
            "released."
        )
        + para(
            "Downgrades are not supported automatically. Always back up before "
            "deploying a migration."
        )
    )


def sec_monitoring_metrics() -> str:
    return (
        h2("Metrics")
        + para("The platform exposes Prometheus metrics at ``/metrics``.")
        + table(
            ["Metric", "Type", "Description"],
            [
                ["api_requests_total", "counter", "Total API requests by route and status"],
                ["api_request_duration_seconds", "histogram", "Request latency"],
                ["ingestion_documents_processed_total", "counter", "Documents ingested"],
                ["retrieval_latency_seconds", "histogram", "Retrieval pipeline latency"],
                ["verification_success_rate", "histogram", "Citation validation pass rate"],
            ],
        )
        + note_box(
            "note",
            "Metrics are retained for 30 days. Alerts should aggregate over 5-minute windows.",
        )
    )


def sec_monitoring_alerts() -> str:
    return (
        h2("Alerts")
        + para("Recommended alerting thresholds:")
        + table(
            ["Alert", "Threshold", "Severity"],
            [
                ["Error rate", "> 1% over 5 minutes", "critical"],
                ["p95 latency", "> 2s over 5 minutes", "warning"],
                ["Queue depth", "> 5000 messages", "critical"],
                ["Verification success rate", "< 95% over 1 hour", "warning"],
            ],
        )
    )


def sec_users_roles() -> str:
    return (
        h2("Roles")
        + para(
            "Permissions are assigned through roles. The platform ships with "
            "four default roles."
        )
        + table(
            ["Role", "Permissions"],
            [
                ["owner", "Full access including billing"],
                ["admin", "Manage users, projects, and settings"],
                ["editor", "Create and edit documents"],
                [FACTS["rbac_default_role"], "Read-only access"],
            ],
        )
        + para(
            "Roles can be scoped per project. Custom roles can be defined with "
            "an arbitrary combination of permissions."
        )
    )


def sec_users_scim() -> str:
    return (
        h2("SCIM Provisioning")
        + para(
            "SCIM 2.0 is supported on the Enterprise plan. When users are "
            "deprovisioned in the identity provider, their access is revoked "
            "within 5 minutes."
        )
    )


def sec_db_behavior() -> str:
    return (
        h2("Database Behavior")
        + para(
            f"The platform stores metadata in {FACTS['database_engine']}. "
            "The connection pool has a maximum of "
            f"{FACTS['db_connection_max']} connections per instance."
        )
        + para(
            "The database stores documents, chunks, ingestion jobs, queries, "
            "and evaluation runs. Vector embeddings are stored in Qdrant, not "
            "in PostgreSQL."
        )
        + para(
            "Backups are taken every 12 hours and retained for "
            f"{FACTS['backup_retention']}."
        )
    )


def sec_troubleshooting_common() -> str:
    return (
        h2("Common Issues")
        + para(
            "This section covers frequently reported issues and their "
            "resolutions."
        )
        + table(
            ["Symptom", "Likely cause", "Resolution"],
            [
                [
                    "401 from all endpoints",
                    "Expired access token",
                    "Refresh the token before expiry",
                ],
                [
                    "Documents not searchable after upload",
                    "Ingestion job failed or still running",
                    "Check ingestion status; retry the job",
                ],
                [
                    "Slow retrieval",
                    "Reranker enabled on large candidate sets",
                    "Reduce BM25/VECTOR_TOP_K",
                ],
                [
                    "429 responses",
                    "Rate limit exceeded",
                    "Back off using Retry-After",
                ],
                [
                    "Answers marked ungrounded",
                    "Evidence insufficient or citation mismatch",
                    "Re-ask with more specific wording",
                ],
            ],
        )
    )


def sec_troubleshooting_secrets() -> str:
    return (
        h2("Rotating Secrets")
        + para(
            "Webhook secrets can be rotated from the Admin Console. After "
            "rotation, signatures generated with the old secret fail "
            "verification immediately. Rotate during low-traffic windows."
        )
        + para(
            "API keys are single-secret; rotating an API key invalidates the "
            "previous key immediately."
        )
    )


def sec_limits_quotas() -> str:
    return (
        h2("Quotas by Plan")
        + para("Rate limits and quotas vary by plan.")
        + table(
            ["Plan", "Default rate", "Burst", "Max documents"],
            [
                ["Free", "30 req/min", "60 req/min", "100"],
                ["Pro", FACTS["rate_limit_default"], FACTS["rate_limit_burst"], "10,000"],
                ["Enterprise", "1,000 req/min", "2,000 req/min", "Unlimited"],
            ],
        )
    )


def sec_webhooks_events() -> str:
    return (
        h2("Webhook Events")
        + para("The platform emits the following events.")
        + table(
            ["Event", "Trigger"],
            [
                ["document.uploaded", "A document was uploaded"],
                ["document.ingested", "Ingestion completed successfully"],
                ["document.failed", "Ingestion failed"],
                ["query.completed", "A query completed with citations"],
            ],
        )
    )


def sec_webhooks_endpoints() -> str:
    return (
        h2("Registering Endpoints")
        + para(
            "Register up to 10 webhook endpoints per project from the Admin "
            "Console. Each endpoint can filter events by type."
        )
        + para(
            "Endpoints that fail 8 consecutive deliveries are automatically "
            "disabled and an alert is raised."
        )
    )


# --------------------------------------------------------------------------
# Document assembly
# --------------------------------------------------------------------------


def build_documents() -> dict[str, dict[str, object]]:
    """Return ``{doc_id: {"title", "module", "content"}}`` for all documents."""
    docs: dict[str, dict[str, object]] = {}

    def add(doc_id: str, title: str, module: str, *sections: Callable[[], str]) -> None:
        docs[doc_id] = {
            "title": title,
            "module": module,
            "content": doc(doc_id, title, "1.0", module, *sections),
        }

    # --- Authentication & security ---
    add(
        "authentication-guide",
        "Authentication Guide",
        "authentication",
        lambda: (sec_auth_intro() + sec_auth_tokens() + sec_auth_api_keys() + sec_auth_basic() + sec_auth_errors()),
    )
    add("mfa-guide", "Multi-Factor Authentication", "authentication", lambda: (sec_mfa() + sec_auth_errors()))
    add("security-guide", "Security Guide", "security", lambda: (sec_security_encryption() + sec_security_audit() + sec_auth_errors()))
    add("api-keys", "API Keys Management", "authentication", lambda: (sec_auth_api_keys() + sec_auth_basic()))
    add("token-refresh", "Token Refresh and Rotation", "authentication", lambda: (sec_oauth_refresh() + sec_auth_tokens()))

    # --- OAuth / SSO ---
    add(
        "oauth-guide",
        "OAuth 2.0 Integration Guide",
        "oauth",
        lambda: (sec_oauth_intro() + sec_oauth_flows() + sec_oauth_scopes() + sec_oauth_claims() + sec_oauth_discovery() + sec_oauth_refresh()),
    )
    add("sso-saml", "SAML 2.0 Enterprise SSO", "oauth", lambda: (sec_oauth_sso() + sec_oauth_discovery()))
    add("oauth-sso", "SSO and Federation", "oauth", lambda: (sec_oauth_sso() + sec_oauth_discovery()))
    add("scopes", "OAuth Scopes and Permissions", "oauth", lambda: (sec_oauth_scopes() + sec_oauth_claims()))

    # --- API ---
    add(
        "api-reference",
        "API Reference",
        "api",
        lambda: (sec_api_terminal() + sec_api_auth_header() + sec_api_pagination() + sec_api_idempotency() + sec_api_rate_limits() + sec_api_webhooks() + sec_api_headers()),
    )
    add(
        "api-errors",
        "API Error Reference",
        "api",
        lambda: (sec_errors_summary() + sec_errors_codes() + sec_errors_troubleshoot()),
    )
    add("api-pagination", "Pagination Guide", "api", lambda: (sec_api_pagination() + sec_api_headers()))
    add("api-idempotency", "Idempotency Guide", "api", lambda: (sec_api_idempotency() + sec_api_headers()))

    # --- Configuration ---
    add(
        "configuration-guide",
        "Configuration Guide",
        "configuration",
        lambda: (sec_config_env() + sec_config_flags()),
    )
    add("env-vars", "Environment Variables Reference", "configuration", lambda: (sec_config_env()))
    add("feature-flags", "Feature Flags", "configuration", lambda: (sec_config_flags()))

    # --- Deployment ---
    add(
        "deployment-guide",
        "Deployment Guide",
        "deployment",
        lambda: (sec_deploy_overview() + sec_deploy_envs() + sec_deploy_migrations()),
    )
    add("environments", "Environments", "deployment", lambda: (sec_deploy_envs() + sec_deploy_migrations()))

    # --- Monitoring ---
    add(
        "monitoring-guide",
        "Monitoring Guide",
        "monitoring",
        lambda: (sec_monitoring_metrics() + sec_monitoring_alerts()),
    )
    add("alerts-guide", "Alerting Guide", "monitoring", lambda: (sec_monitoring_alerts()))

    # --- User management ---
    add(
        "user-management",
        "User Management",
        "users",
        lambda: (sec_users_roles() + sec_users_scim()),
    )
    add("roles-permissions", "Roles and Permissions", "users", lambda: (sec_users_roles()))

    # --- Database ---
    add("database-behavior", "Database Behavior", "database", lambda: (sec_db_behavior()))
    add("backup-recovery", "Backup and Recovery", "database", lambda: (sec_db_behavior() + sec_deploy_migrations()))

    # --- Troubleshooting ---
    add(
        "troubleshooting-guide",
        "Troubleshooting Guide",
        "troubleshooting",
        lambda: (sec_troubleshooting_common() + sec_troubleshooting_secrets()),
    )
    add("rotating-secrets", "Rotating Secrets", "troubleshooting", lambda: (sec_troubleshooting_secrets()))

    # --- Rate limits ---
    add("rate-limits", "Rate Limits and Quotas", "rate-limits", lambda: (sec_api_rate_limits() + sec_limits_quotas()))

    # --- Webhooks ---
    add(
        "webhooks-guide",
        "Webhooks Guide",
        "webhooks",
        lambda: (sec_api_webhooks() + sec_webhooks_events() + sec_webhooks_endpoints()),
    )
    add("webhook-signing", "Webhook Signature Verification", "webhooks", lambda: (sec_api_webhooks()))

    return docs


# --------------------------------------------------------------------------
# Multi-format output
# --------------------------------------------------------------------------


def emit_files(docs: dict[str, dict[str, object]], out_root: Path) -> dict[str, Path]:
    """Write all documents to the four format directories.

    Returns a mapping of ``doc_id -> chosen primary path`` for the manifest.
    """
    markdown_dir = out_root / "markdown"
    pdf_dir = out_root / "pdf"
    docx_dir = out_root / "docx"
    html_dir = out_root / "html"
    for d in (markdown_dir, pdf_dir, docx_dir, html_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Which documents become PDF/DOCX/HTML?
    pdf_set = {
        "authentication-guide",
        "api-reference",
        "security-guide",
        "deployment-guide",
        "troubleshooting-guide",
    }
    docx_set = {"oauth-guide", "configuration-guide", "monitoring-guide", "user-management", "webhooks-guide"}
    html_set = {"api-errors", "rate-limits", "environments", "roles-permissions", "sso-saml", "database-behavior"}

    manifest: dict[str, Path] = {}
    emitted: list[Path] = []
    for doc_id, meta in docs.items():
        content = str(meta["content"])
        md_path = markdown_dir / f"{doc_id}.md"
        write_doc(md_path, content)
        emitted.append(md_path)
        manifest[doc_id] = md_path

        if doc_id in pdf_set:
            emitted.append(render_pdf(content, md_path, output_dir=pdf_dir))
        if doc_id in docx_set:
            emitted.append(render_docx(content, md_path, output_dir=docx_dir))
        if doc_id in html_set:
            emitted.append(render_html(content, md_path, output_dir=html_dir))

    # Manifest for ingestion tests / reproducibility
    manifest_path = out_root / "manifest.json"
    manifest_json = {k: str(v.relative_to(out_root)) for k, v in sorted(manifest.items())}
    manifest_path.write_text(
        __import__("json").dumps(manifest_json, indent=2) + "\n", encoding="utf-8"
    )

    # Remove stale files from a previous run that are no longer produced.
    expected_names = {p.name for p in emitted}
    for d in (markdown_dir, pdf_dir, docx_dir, html_dir):
        for f in list(d.glob("*")):
            if f.is_file() and f.name not in expected_names:
                f.unlink()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the AskMyDocs enterprise documentation corpus.")
    parser.add_argument("--out", default=str(DATA_DIR), help="Output directory (default: data/corpus)")
    args = parser.parse_args()
    out_root = Path(args.out).resolve()
    docs = build_documents()
    emit_files(docs, out_root)
    print(f"Generated {len(docs)} documents under {out_root}")
    for fmt in ("markdown", "pdf", "docx", "html"):
        d = out_root / fmt
        print(f"  {fmt}: {len(list(d.glob('*')))} file(s)")
    print(f"Manifest written to {out_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())