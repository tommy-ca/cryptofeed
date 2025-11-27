# Kafka Backend Knowledge Transfer Guide

## Overview
This document provides comprehensive knowledge transfer for maintaining and operating the Kafka backend infrastructure, including both legacy and modern implementations during the migration period.

**Audience**: New team members, rotating on-call engineers, support staff
**Learning time**: 2-4 hours (reading), 1-2 weeks (hands-on)
**Prerequisites**: Basic Kafka knowledge, Python programming, Kubernetes familiarity

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Key Components](#key-components)
3. [Operational Procedures](#operational-procedures)
4. [Troubleshooting Guide](#troubleshooting-guide)
5. [Team Responsibilities](#team-responsibilities)
6. [Resources and References](#resources-and-references)

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Exchange Feeds (Binance, Coinbase, Kraken, etc.)                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Cryptofeed Data Ingestion Layer                                 │
│  ├─ Exchange connectors (WebSocket, REST)                       │
│  ├─ Data normalization (Trade, OrderBook, etc.)                 │
│  └─ Protobuf serialization                                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌─────────────────┐           ┌─────────────────┐
│ Legacy Backend  │           │ Modern Backend  │
│ (Deprecated)    │           │ (Production)    │
│                 │           │                 │
│ - BackendQueue  │           │ - KafkaCallback │
│ - JSON only     │           │ - Protobuf      │
│ - Per-type      │           │ - Consolidated  │
│   topics        │           │   topics        │
└────────┬────────┘           └────────┬────────┘
         │                             │
         └──────────────┬──────────────┘
                        ▼
              ┌──────────────────┐
              │ Kafka Cluster    │
              │ (Production)     │
              └────────┬─────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │Consumer │  │Consumer │  │Consumer │
   │ Flink   │  │ Python  │  │ Custom  │
   └─────────┘  └─────────┘  └─────────┘
```

### Component Relationships

- **Legacy Backend** (`cryptofeed.backends.kafka`): Original implementation, being phased out
- **Modern Backend** (`cryptofeed.backends.kafka.*`): New modular implementation with advanced features
- **Maintenance System** (`cryptofeed.backends.kafka.maintenance`): Orchestrates migration and operations

### Data Flow

1. **Exchange Feeds** → Raw market data (WebSocket/REST)
2. **Cryptofeed** → Normalized data objects (Trade, OrderBook, etc.)
3. **Protobuf Serialization** → Binary message format
4. **Kafka Backend** → Publishes to Kafka topics
5. **Kafka Cluster** → Distributes messages to consumers
6. **Consumers** → Process and store data (Iceberg, DuckDB, etc.)

---

## Key Components

### 1. Legacy Backend (Deprecated)

**Location**: `cryptofeed/backends/kafka.py`

**Purpose**: Original Kafka backend implementation

**Key Classes**:
- `TradeKafka` - Trade message publisher
- `BookKafka` - Order book publisher
- `TickerKafka` - Ticker data publisher
- `FundingKafka` - Funding rate publisher

**Current Status**: Deprecated, emits warnings, no new features

**When to Use**: Only for existing deployments during migration period

**Example**:
```python
from cryptofeed import FeedHandler
from cryptofeed.backends.kafka import TradeKafka  # Deprecated!

fh = FeedHandler()
fh.add_feed(
    'Binance',
    channels=['trades'],
    symbols=['BTC-USDT'],
    callbacks={'trades': TradeKafka(bootstrap='kafka:9092')}
)
fh.run()
```

---

### 2. Modern Backend (Production)

**Location**: `cryptofeed/backends/kafka/`

**Purpose**: Modern, maintainable Kafka backend with protobuf support

**Key Modules**:
- `callback.py` - `KafkaCallback`, `KafkaProtobufCallback` main interfaces
- `config.py` - Configuration models (Pydantic)
- `producer.py` - Kafka producer implementation
- `partition.py` - Partition strategy implementations
- `health.py` - Health check system

**Key Features**:
- Protobuf serialization (63% smaller messages)
- Consolidated topics (O(20) vs O(10K))
- Multiple partition strategies
- Exactly-once semantics
- Comprehensive error handling

**Example**:
```python
from cryptofeed import FeedHandler
from cryptofeed.backends.kafka import KafkaProtobufCallback
from cryptofeed.backends.kafka.config import KafkaConfig

config = KafkaConfig(
    bootstrap_servers=['kafka:9092'],
    topic_prefix='crypto-data',
    partition_strategy='composite',
    enable_idempotence=True,
)

callback = KafkaProtobufCallback(config)

fh = FeedHandler()
fh.add_feed(
    'Binance',
    channels=['trades', 'l2_book'],
    symbols=['BTC-USDT', 'ETH-USDT'],
    callbacks={
        'trades': callback,
        'l2_book': callback,
    }
)
fh.run()
```

---

### 3. Maintenance System

**Location**: `cryptofeed/backends/kafka/maintenance/`

**Purpose**: Orchestrate migration, monitoring, and operational excellence

**Key Modules**:
- `integration.py` - `MaintenanceCoordinator` unified interface
- `scheduler.py` - Automated task scheduling
- `deprecation_system.py` - Deprecation warning management
- `shim_removal.py` - Compatibility shim lifecycle

**Key Classes**:
- `MaintenanceCoordinator` - Central coordination point
- `MaintenanceScheduler` - Automated maintenance tasks
- `DeprecationWarningSystem` - Warning emission and tracking
- `HealthCheckSystem` - Health monitoring

**Example**:
```python
from cryptofeed.backends.kafka.maintenance.integration import MaintenanceCoordinator

coordinator = MaintenanceCoordinator()

# Track legacy usage
coordinator.handle_legacy_usage('TradeKafka', {'exchange': 'binance'})

# Get migration analytics
analytics = coordinator.deprecation_monitoring.get_migration_analytics()
print(f"Migration progress: {analytics['migration_percentage']}%")

# Check health
result = coordinator.handle_health_check(
    bootstrap_servers=['kafka:9092'],
    implementation='modern'
)
print(f"Health: {'OK' if result.success else 'FAILED'}")
```

---

## Operational Procedures

### Daily Operations

1. **Monitor Health (5 minutes)**
   ```bash
   # Run automated health checks
   python tools/kafka-maintenance-scheduler.py run-once \
     --config config/daily-health.yaml

   # Review Grafana dashboard
   # Navigate to: Kafka Backend Health & Performance
   ```

2. **Check Migration Progress (5 minutes)**
   ```bash
   # View migration analytics
   python tools/kafka-maintenance-scheduler.py status \
     --config config/maintenance-tasks.yaml --json daily-status.json

   # Review trends in Grafana
   # Dashboard: Kafka Migration Progress
   ```

3. **Review Alerts (5 minutes)**
   ```bash
   # Check PagerDuty for any incidents
   pd incident list --status=triggered,acknowledged

   # Review Slack #kafka-alerts channel
   ```

### Weekly Operations

1. **Generate Progress Report (15 minutes)**
   ```python
   from cryptofeed.backends.kafka.maintenance.integration import MaintenanceCoordinator

   coordinator = MaintenanceCoordinator()
   analytics = coordinator.deprecation_monitoring.get_migration_analytics()

   # Generate report
   report = f"""
   ## Weekly Migration Report

   **Migration Progress**: {analytics['migration_percentage']:.1f}%
   **Legacy Usage**: {analytics['legacy_usage']['total_legacy_usage']} calls
   **Modern Usage**: {analytics['modern_usage']['total_modern_usage']} calls
   **Velocity**: {analytics.get('weekly_velocity', 'N/A')}% per week

   **Timeline Recommendation**: {analytics['timeline_recommendation'].should_extend_timeline}
   """

   print(report)
   # Post to Slack #kafka-migration
   ```

2. **Review User Issues (15 minutes)**
   ```bash
   # Check Jira for migration-related tickets
   jira issue list --jql "labels=kafka-migration AND status!=Closed"

   # Review support ticket backlog
   ```

3. **Update Documentation (30 minutes)**
   - Review and update runbooks based on incidents
   - Update FAQ based on user questions
   - Keep migration guide current

### Monthly Operations

1. **Review and Adjust Timeline (1 hour)**
   ```python
   from cryptofeed.backends.kafka.deprecation import DeprecationTimeline

   timeline = DeprecationTimeline()

   # Review milestones
   for milestone in timeline.milestones:
       print(f"{milestone.name}: {milestone.status} (Due: {milestone.target_date})")

   # Adjust if needed based on analytics
   ```

2. **Stakeholder Communication (1 hour)**
   - Send monthly progress email
   - Update confluence documentation
   - Present at team all-hands if significant progress

3. **Performance Review (1 hour)**
   ```bash
   # Generate performance report
   python scripts/generate-performance-report.py \
     --start-date "2025-11-01" \
     --end-date "2025-11-30" \
     --output monthly-performance.pdf

   # Review trends, identify optimizations
   ```

---

## Troubleshooting Guide

### Quick Reference

| Symptom | Likely Cause | First Action | Runbook |
|---------|-------------|--------------|---------|
| High error rate | Kafka broker down | Check cluster health | [Incident Response](../runbooks/kafka-incident-response.md) |
| High latency | Network issues | Check connectivity | [Health Monitoring](../runbooks/kafka-backend-health-monitoring.md) |
| Deprecation warnings flooding logs | Legacy usage | Add warning filter | [Deprecation](../runbooks/kafka-backend-deprecation.md) |
| Migration stuck | User blockers | Contact heavy users | [Deprecation](../runbooks/kafka-backend-deprecation.md) |
| Consumer lag | Partition imbalance | Increase partitions | [Incident Response](../runbooks/kafka-incident-response.md) |

### Common Issues and Solutions

**Issue 1: "No such module: cryptofeed.backends.kafka.callback"**
```python
# Old import (wrong)
from cryptofeed.backends.kafka.callback import KafkaCallback

# New import (correct)
from cryptofeed.backends.kafka import KafkaCallback
```

**Issue 2: Configuration not applying**
```bash
# Verify configuration loaded
kubectl get configmap cryptofeed-config -o yaml

# Check environment variables
kubectl exec -it cryptofeed-pod -- env | grep KAFKA

# Restart deployment to apply
kubectl rollout restart deployment/cryptofeed
```

**Issue 3: Messages not appearing in topics**
```bash
# Check producer connectivity
kafka-console-producer --bootstrap-server kafka:9092 --topic test

# Verify topic exists
kafka-topics --bootstrap-server kafka:9092 --list | grep cryptofeed

# Check ACLs
kafka-acls --bootstrap-server kafka:9092 --list --topic cryptofeed.trades
```

---

## Team Responsibilities

### Backend Team
- **Owns**: Kafka backend code, maintenance system, migration execution
- **Responsible for**: Code changes, bug fixes, feature development
- **On-call rotation**: Primary escalation for SEV-1/SEV-2 incidents

### Platform Team
- **Owns**: Kafka cluster infrastructure, deployment pipelines
- **Responsible for**: Cluster health, scaling, capacity planning
- **On-call rotation**: Infrastructure-level incidents

### Support Team
- **Owns**: User documentation, ticket triage, initial response
- **Responsible for**: User communication, known issue tracking
- **Escalation**: Backend team for code issues, Platform team for infrastructure

### SRE Team
- **Owns**: Monitoring, alerting, runbooks, incident response
- **Responsible for**: Operational excellence, reliability improvements
- **On-call rotation**: Incident coordination, post-mortems

---

## Resources and References

### Documentation
- [Architecture](architecture.md) - System architecture and design
- [Migration Guide](MIGRATION_GUIDE.md) - Step-by-step migration instructions
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Best Practices](BEST_PRACTICES.md) - Production deployment guidelines
- [Troubleshooting](TROUBLESHOOTING.md) - Detailed troubleshooting guide

### Runbooks
- [Deprecation Management](../runbooks/kafka-backend-deprecation.md)
- [Health Monitoring](../runbooks/kafka-backend-health-monitoring.md)
- [Migration Execution](../runbooks/kafka-migration-execution.md)
- [Incident Response](../runbooks/kafka-incident-response.md)

### Tools
- `tools/kafka-maintenance-scheduler.py` - Automated maintenance scheduler
- `tools/migrate-kafka-config.py` - Configuration migration tool
- `scripts/test-message-flow.py` - End-to-end testing
- `scripts/generate-performance-report.py` - Performance analysis

### Monitoring
- **Grafana Dashboards**:
  - Kafka Backend Health & Performance
  - Kafka Migration Progress
  - Kafka Error Rates & Latency
- **PagerDuty**: On-call schedule and incident management
- **Slack Channels**:
  - #kafka-migration - Migration progress and updates
  - #kafka-alerts - Automated alerts and warnings
  - #incident-kafka - Active incident coordination

### Contact Information
- **Backend Team Lead**: [Contact info]
- **Platform Team Lead**: [Contact info]
- **SRE On-call**: PagerDuty rotation
- **Support Team**: [Support email/slack]

---

## Learning Path

### Week 1: Fundamentals
- [ ] Read this knowledge transfer guide
- [ ] Review architecture documentation
- [ ] Set up local development environment
- [ ] Run example code for legacy and modern backends

### Week 2: Operations
- [ ] Shadow daily operations (health checks, monitoring)
- [ ] Practice using maintenance scheduler tool
- [ ] Review recent incident reports
- [ ] Participate in weekly progress meeting

### Week 3: Troubleshooting
- [ ] Work through common troubleshooting scenarios
- [ ] Shadow on-call engineer during their shift
- [ ] Review runbooks and practice procedures
- [ ] Create test incident and practice response

### Week 4: Independence
- [ ] Take on-call shift with backup support
- [ ] Generate weekly progress report
- [ ] Respond to user issues independently
- [ ] Contribute to documentation improvements

---

## Success Criteria

### Individual Readiness
- [ ] Can explain system architecture
- [ ] Can perform daily operations independently
- [ ] Can troubleshoot common issues
- [ ] Can respond to SEV-2 incidents with support
- [ ] Confident taking on-call shifts

### Team Readiness
- [ ] All team members completed knowledge transfer
- [ ] Runbooks tested and validated
- [ ] On-call rotation fully staffed
- [ ] Backup coverage identified for all roles

---

## Changelog
- 2025-11-26: Initial knowledge transfer guide created (Task 6.3)
