---
doc_id: alerts-guide
title: Alerting Guide
version: 1.0
module: monitoring
last_updated: 2026-09-12
---


## Alerts (#alerts)

Recommended alerting thresholds:

| Alert | Threshold | Severity |
| --- | --- | --- |
| Error rate | > 1% over 5 minutes | critical |
| p95 latency | > 2s over 5 minutes | warning |
| Queue depth | > 5000 messages | critical |
| Verification success rate | < 95% over 1 hour | warning |

