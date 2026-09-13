---
doc_id: monitoring-guide
title: Monitoring Guide
version: 1.0
module: monitoring
last_updated: 2026-09-12
---


## Metrics (#metrics)

The platform exposes Prometheus metrics at ``/metrics``.

| Metric | Type | Description |
| --- | --- | --- |
| api_requests_total | counter | Total API requests by route and status |
| api_request_duration_seconds | histogram | Request latency |
| ingestion_documents_processed_total | counter | Documents ingested |
| retrieval_latency_seconds | histogram | Retrieval pipeline latency |
| verification_success_rate | histogram | Citation validation pass rate |

> **Note**: Metrics are retained for 30 days. Alerts should aggregate over 5-minute windows.

## Alerts (#alerts)

Recommended alerting thresholds:

| Alert | Threshold | Severity |
| --- | --- | --- |
| Error rate | > 1% over 5 minutes | critical |
| p95 latency | > 2s over 5 minutes | warning |
| Queue depth | > 5000 messages | critical |
| Verification success rate | < 95% over 1 hour | warning |

