# Operational Runbooks Index

## Overview
This directory contains operational runbooks for managing Cryptofeed's Kafka backend infrastructure, including migration procedures, health monitoring, and incident response.

**Audience**: Operations teams, SREs, on-call engineers
**Purpose**: Provide step-by-step procedures for common operational tasks

---

## Runbooks

### Core Operational Runbooks

#### [Kafka Backend Deprecation](kafka-backend-deprecation.md)
**Purpose**: Manage deprecation lifecycle of legacy Kafka backend
**Use when**: Monitoring migration progress, handling user reports, executing rollbacks
**Key procedures**:
- Monitoring deprecation warnings
- Tracking migration progress
- Handling user reports
- Emergency rollback procedures

**Estimated time**: 15-30 minutes per procedure

---

#### [Kafka Backend Health Monitoring](kafka-backend-health-monitoring.md)
**Purpose**: Monitor Kafka backend health and maintain high availability
**Use when**: Daily health checks, investigating performance issues, responding to alerts
**Key procedures**:
- Manual and automated health checks
- Performance monitoring
- Alerting configuration and response
- Common health issue resolution

**Estimated time**: 10-20 minutes per check

---

#### [Kafka Migration Execution](kafka-migration-execution.md)
**Purpose**: Execute production migration from legacy to modern backend
**Use when**: Deploying modern backend to production
**Key procedures**:
- Pre-migration checklist
- Blue-green deployment
- Canary rollout
- Validation and verification
- Rollback procedures

**Estimated time**: 2-4 hours per deployment

---

#### [Kafka Incident Response](kafka-incident-response.md)
**Purpose**: Respond to Kafka backend incidents
**Use when**: Critical alerts, service degradation, production outages
**Key procedures**:
- Incident classification (SEV-1 to SEV-4)
- Critical incident response (0-30 minutes)
- High priority incident response (15-60 minutes)
- Common incident scenarios
- Post-incident procedures

**Estimated time**: 5 minutes - 2 hours (severity-dependent)

---

## Quick Reference Guide

### Daily Operations
```bash
# Health check
python tools/kafka-maintenance-scheduler.py run-once \
  --config config/daily-health.yaml

# Status check
python tools/kafka-maintenance-scheduler.py status \
  --config config/maintenance-tasks.yaml

# Review alerts
# Check PagerDuty and Slack #kafka-alerts
```

### Emergency Procedures
```bash
# Quick rollback
kubectl apply -f config/prod/service-rollback-to-legacy.yaml

# Emergency health check
from cryptofeed.backends.kafka.maintenance.integration import MaintenanceCoordinator
coordinator = MaintenanceCoordinator()
result = coordinator.handle_health_check(
    bootstrap_servers=['kafka:9092'],
    implementation='modern'
)
print(f"Status: {'OK' if result.success else 'FAILED'}")
```

### Common Commands
```bash
# View Kafka topics
kafka-topics --bootstrap-server kafka:9092 --list

# Check consumer lag
kafka-consumer-groups --bootstrap-server kafka:9092 \
  --group cryptofeed-consumers --describe

# Test message flow
python scripts/test-message-flow.py --count=100

# Generate performance report
python scripts/generate-performance-report.py \
  --start-date "2025-11-01" --end-date "2025-11-30"
```

---

## Runbook Selection Guide

### "Which runbook should I use?"

**Scenario**: I need to check if the system is healthy
→ **Runbook**: [Health Monitoring](kafka-backend-health-monitoring.md)

**Scenario**: Migration progress seems slow
→ **Runbook**: [Deprecation Management](kafka-backend-deprecation.md)

**Scenario**: Planning production migration deployment
→ **Runbook**: [Migration Execution](kafka-migration-execution.md)

**Scenario**: PagerDuty alert fired
→ **Runbook**: [Incident Response](kafka-incident-response.md)

**Scenario**: User reported issue with migration
→ **Runbook**: [Deprecation Management](kafka-backend-deprecation.md) → "Handling User Reports"

**Scenario**: High error rate detected
→ **Runbook**: [Incident Response](kafka-incident-response.md) → SEV-1 or SEV-2 response

**Scenario**: Performance degradation
→ **Runbook**: [Health Monitoring](kafka-backend-health-monitoring.md) → "Performance Monitoring"

---

## Related Documentation

### Kafka Backend Documentation
- [Knowledge Transfer Guide](../kafka/KNOWLEDGE_TRANSFER.md) - Comprehensive onboarding
- [Architecture](../kafka/architecture.md) - System architecture
- [Migration Guide](../kafka/MIGRATION_GUIDE.md) - User migration guide
- [Troubleshooting](../kafka/TROUBLESHOOTING.md) - Detailed troubleshooting
- [API Reference](../kafka/API_REFERENCE.md) - Complete API docs

### Tools
- [Maintenance Scheduler](../../tools/kafka-maintenance-scheduler.py) - Automated tasks
- [Configuration Migration](../../tools/migrate-kafka-config.py) - Config translation

### Monitoring
- **Grafana Dashboards**:
  - Kafka Backend Health & Performance
  - Kafka Migration Progress
  - Kafka Error Rates & Latency
- **Slack Channels**:
  - #kafka-migration - Progress updates
  - #kafka-alerts - Automated alerts
  - #incident-kafka - Incident coordination

---

## Runbook Development Guidelines

### When to Create a New Runbook
- Repeated manual procedures (>3 occurrences)
- Complex multi-step operations
- Time-sensitive incident response
- Knowledge transfer requirements

### Runbook Structure
```markdown
# [Runbook Title]

## Overview
- Purpose
- Audience
- Estimated time
- Prerequisites

## Procedures
### Procedure Name
1. Step 1
2. Step 2
...

### Expected Outcomes
### Troubleshooting

## Success Criteria
## Related Documentation
## Changelog
```

### Runbook Maintenance
- Review quarterly or after major incidents
- Update based on incident learnings
- Test procedures in staging
- Version control all changes
- Archive obsolete runbooks

---

## Success Metrics

### Runbook Effectiveness
- **Usage**: Runbooks consulted during 100% of incidents
- **Accuracy**: <5% procedures fail when followed correctly
- **Currency**: All runbooks updated within 30 days of major changes
- **Coverage**: All common scenarios documented

### Operational Excellence
- **MTTD** (Mean Time To Detect): <5 minutes
- **MTTA** (Mean Time To Acknowledge): <2 minutes
- **MTTR** (Mean Time To Resolve): <30 minutes (SEV-1), <4 hours (SEV-2)
- **Incident recurrence**: 0% (same root cause)

---

## Feedback and Improvements

### How to Contribute
1. Identify gap or improvement opportunity
2. Create Jira ticket with label `runbook-improvement`
3. Submit PR with proposed changes
4. Request review from SRE team
5. Update changelog after merge

### Feedback Channels
- Slack: #kafka-migration
- Email: sre-team@example.com
- Jira: Project `RUNBOOKS`

---

## Changelog
- 2025-11-26: Initial runbook index created (Task 6.3)
- 2025-11-26: Added 4 core operational runbooks
