# Kafka Backend Incident Response Runbook

## Overview
This runbook provides procedures for responding to Kafka backend incidents during the migration period and beyond.

**Audience**: On-call engineers, SREs, Incident commanders
**Estimated time**: Variable (5 minutes - 2 hours)
**Prerequisites**: On-call access, monitoring tools, escalation contacts

## Table of Contents
1. [Incident Classification](#incident-classification)
2. [Critical Incident Response](#critical-incident-response)
3. [High Priority Incident Response](#high-priority-incident-response)
4. [Common Incident Scenarios](#common-incident-scenarios)
5. [Post-Incident Procedures](#post-incident-procedures)

---

## Incident Classification

### Purpose
Quickly triage incidents to determine appropriate response level and escalation path.

### Severity Levels

**SEV-1 (Critical)**
- Production down or severely degraded
- Data loss occurring
- >50% error rate
- Security breach
- Response time: Immediate (0-5 minutes)

**SEV-2 (High)**
- Partial service degradation
- Elevated error rate (1-50%)
- Performance degradation
- Response time: 15 minutes

**SEV-3 (Medium)**
- Minor issues, workarounds available
- Isolated component failures
- Warning thresholds exceeded
- Response time: 2 hours

**SEV-4 (Low)**
- Cosmetic issues
- Documentation gaps
- Non-urgent improvements
- Response time: Next business day

### Classification Decision Tree

```
Is production traffic impacted?
├── Yes → Is service completely unavailable?
│   ├── Yes → SEV-1 (Critical)
│   └── No → Is >10% of traffic affected?
│       ├── Yes → SEV-2 (High)
│       └── No → SEV-3 (Medium)
└── No → Is there potential for user impact?
    ├── Yes → SEV-3 (Medium)
    └── No → SEV-4 (Low)
```

---

## Critical Incident Response

### Purpose
Rapidly respond to critical (SEV-1) incidents with potential for service outage or data loss.

### Immediate Actions (0-5 minutes)

1. **Acknowledge and Assess**
   ```bash
   # Acknowledge PagerDuty alert
   pd incident acknowledge --id <incident-id>

   # Join incident Slack channel
   # /incident create "Kafka Backend Critical Issue"

   # Quick health check
   python tools/kafka-maintenance-scheduler.py status \
     --config config/prod-health.yaml
   ```

2. **Declare Incident**
   ```text
   # In #incident-kafka Slack channel
   @here SEV-1 incident declared for Kafka backend

   Impact: [Describe user-facing impact]
   Started: [Timestamp]
   Incident Commander: [Your name]
   Comms Lead: [Tag team member]
   Tech Lead: [Tag team member]

   Status updates every 15 minutes
   ```

3. **Immediate Stabilization**
   ```bash
   # Option 1: Rollback to last known good (fastest)
   kubectl apply -f config/prod/service-rollback-to-legacy.yaml

   # Option 2: Scale up replicas (if capacity issue)
   kubectl scale deployment/cryptofeed --replicas=10

   # Option 3: Emergency rate limiting (if overload)
   kubectl apply -f config/prod/rate-limit-emergency.yaml
   ```

4. **Verify Stabilization**
   ```bash
   # Check error rate
   curl -s 'http://prometheus:9090/api/v1/query?query=rate(kafka_publish_errors_total[1m])'

   # Check service availability
   for i in {1..10}; do
     curl -s http://cryptofeed-service/health | jq .status
     sleep 1
   done
   ```

### Investigation (5-30 minutes)

1. **Gather Evidence**
   ```bash
   # Recent logs (last 15 minutes)
   kubectl logs -l app=cryptofeed --since=15m --tail=1000 > incident-logs.txt

   # Error logs only
   kubectl logs -l app=cryptofeed --since=15m | grep -i error > incident-errors.txt

   # Metrics snapshot
   curl -s 'http://prometheus:9090/api/v1/query_range?query=kafka_publish_latency_seconds&start=...' > incident-metrics.json
   ```

2. **Identify Root Cause**
   - Check recent deployments: `kubectl rollout history deployment/cryptofeed`
   - Review configuration changes: `kubectl get configmap cryptofeed-config -o yaml`
   - Examine Kafka broker health: `kafka-broker-api-versions --bootstrap-server kafka:9092`
   - Analyze error patterns: `grep -i "exception" incident-errors.txt | sort | uniq -c | sort -rn`

3. **Status Update (15 min)**
   ```text
   # Slack update
   Update [15 min]: Incident stabilized via [action taken]

   Current status: [Operational / Degraded / Down]
   Root cause hypothesis: [Brief description]
   Next steps: [What we're doing next]
   ETA to resolution: [Estimate]
   ```

### Resolution (30+ minutes)

1. **Implement Fix**
   ```bash
   # Apply patch
   kubectl apply -f config/prod/kafka-backend-hotfix.yaml

   # Verify fix
   kubectl rollout status deployment/cryptofeed

   # Test message flow
   python scripts/test-message-flow.py --count=100
   ```

2. **Validate Resolution**
   ```bash
   # Monitor for 15 minutes
   watch -n 30 'python tools/kafka-maintenance-scheduler.py status --config config/prod-health.yaml'

   # Check metrics stable
   curl -s 'http://prometheus:9090/api/v1/query?query=kafka_publish_errors_total'
   ```

3. **Close Incident**
   ```text
   # Slack announcement
   @here Incident RESOLVED

   Duration: [Start time] - [End time]
   Root cause: [Brief description]
   Resolution: [What was done]
   Follow-up: Incident report to be published within 24h

   Thank you team!
   ```

### Expected Outcomes
- Service restored within 30 minutes
- No data loss
- Impact minimized
- Root cause identified

---

## High Priority Incident Response

### Purpose
Respond to high-priority (SEV-2) incidents with degraded service but not complete outage.

### Response Procedure (15-60 minutes)

1. **Acknowledge and Triage**
   ```bash
   # Acknowledge alert
   pd incident acknowledge --id <incident-id>

   # Assess impact
   python scripts/assess-impact.py --start-time "15 minutes ago"
   ```

2. **Investigate**
   ```bash
   # Focused log analysis
   kubectl logs -l app=cryptofeed --since=30m | grep -C5 "error"

   # Performance analysis
   python scripts/analyze-performance.py --window=30m
   ```

3. **Mitigate**
   ```bash
   # Gradual rollback (if deployment-related)
   kubectl apply -f config/prod/service-blue-green-75percent.yaml
   sleep 300
   kubectl apply -f config/prod/service-blue-green-50percent.yaml

   # Or targeted fix
   kubectl set env deployment/cryptofeed KAFKA_BATCH_SIZE=1000
   ```

4. **Monitor and Validate**
   ```bash
   # 30-minute observation period
   watch -n 60 'echo "=== $(date) ===" && python tools/kafka-maintenance-scheduler.py status --config config/prod-health.yaml'
   ```

5. **Document and Close**
   - Create Jira ticket with full details
   - Update incident log
   - Schedule debrief if lessons learned

### Expected Outcomes
- Degradation resolved within 1 hour
- Minimal user impact
- Prevention measures identified

---

## Common Incident Scenarios

### Scenario 1: Kafka Broker Outage

**Symptoms**:
- Connection refused errors
- Timeout exceptions
- Health checks failing

**Response**:
```bash
# Verify Kafka cluster health
kubectl get pods -n kafka

# Check Kafka broker logs
kubectl logs -n kafka kafka-broker-0 --tail=100

# Test connectivity
telnet kafka.prod.example.com 9092

# If single broker down: Wait for auto-recovery
# If multiple brokers down: Escalate to infrastructure team

# Temporary mitigation: Route to backup Kafka cluster
kubectl apply -f config/prod/kafka-backup-cluster.yaml
```

---

### Scenario 2: Message Schema Incompatibility

**Symptoms**:
- Serialization errors
- Consumer unable to deserialize messages
- Schema registry errors

**Response**:
```bash
# Check schema versions
curl http://schema-registry:8081/subjects/cryptofeed.trades-value/versions

# Identify problematic messages
kafka-console-consumer --bootstrap-server kafka:9092 \
  --topic cryptofeed.trades \
  --property print.key=true \
  --from-beginning --max-messages 10

# Rollback to compatible schema version
curl -X DELETE http://schema-registry:8081/subjects/cryptofeed.trades-value/versions/latest

# Deploy schema-compatible producer
kubectl apply -f config/prod/kafka-backend-schema-v1.yaml
```

---

### Scenario 3: Partition Rebalancing Storm

**Symptoms**:
- Consumer lag spikes
- Frequent rebalancing events
- Throughput degradation

**Response**:
```bash
# Check consumer group status
kafka-consumer-groups --bootstrap-server kafka:9092 \
  --group cryptofeed-consumers --describe

# Stabilize consumer group
kubectl scale deployment/cryptofeed-consumers --replicas=5  # Reduce if too many

# Adjust session timeout
kubectl set env deployment/cryptofeed-consumers \
  KAFKA_SESSION_TIMEOUT_MS=30000 \
  KAFKA_HEARTBEAT_INTERVAL_MS=10000

# Monitor for stability
watch -n 10 'kafka-consumer-groups --bootstrap-server kafka:9092 --group cryptofeed-consumers --describe'
```

---

### Scenario 4: Memory Leak in Producer

**Symptoms**:
- Memory usage increasing over time
- OOMKilled pod restarts
- Gradual performance degradation

**Response**:
```bash
# Immediate: Restart affected pods
kubectl delete pod -l app=cryptofeed-producers

# Gather heap dump for analysis
kubectl exec -it cryptofeed-producer-xyz -- jcmd 1 GC.heap_dump /tmp/heap.hprof
kubectl cp cryptofeed-producer-xyz:/tmp/heap.hprof ./heap.hprof

# Temporary mitigation: Increase memory limit
kubectl patch deployment cryptofeed-producers -p '{"spec":{"template":{"spec":{"containers":[{"name":"cryptofeed","resources":{"limits":{"memory":"4Gi"}}}]}}}}'

# Long-term: Identify and fix leak
# Analyze heap dump with VisualVM or Eclipse MAT
```

---

## Post-Incident Procedures

### Purpose
Learn from incidents and prevent recurrence.

### Post-Incident Review (Within 24 hours)

1. **Create Incident Report**
   ```bash
   # Generate template
   python scripts/generate-incident-report.py \
     --incident-id <pagerduty-id> \
     --output incident-reports/2025-11-26-kafka-outage.md
   ```

2. **Incident Report Template**
   ```markdown
   # Incident Report: [Title]

   **Date**: 2025-11-26
   **Duration**: [Start] - [End] ([Total time])
   **Severity**: SEV-X
   **Impact**: [Description]

   ## Timeline
   - HH:MM - Incident detected
   - HH:MM - Response initiated
   - HH:MM - Mitigation applied
   - HH:MM - Incident resolved

   ## Root Cause
   [Detailed explanation]

   ## Resolution
   [What was done to fix]

   ## Prevention
   - [ ] Action item 1 (Owner: X, Due: DATE)
   - [ ] Action item 2 (Owner: Y, Due: DATE)

   ## Lessons Learned
   - What went well
   - What could be improved
   ```

3. **Blameless Post-Mortem Meeting**
   - Schedule within 48 hours
   - Invite all responders
   - Focus on process improvements, not blame
   - Document action items with owners and deadlines

4. **Update Runbooks**
   ```bash
   # If new scenario discovered
   git add docs/runbooks/kafka-incident-response.md
   git commit -m "docs: add scenario [X] to incident runbook"

   # Update alert thresholds if false positive
   git add config/alerts/kafka-backend.yaml
   git commit -m "fix: adjust [metric] alert threshold based on incident"
   ```

5. **Track Prevention Items**
   ```bash
   # Create Jira tickets for action items
   python scripts/create-incident-action-items.py \
     --incident-report incident-reports/2025-11-26-kafka-outage.md
   ```

### Expected Outcomes
- Comprehensive incident documentation
- Action items tracked and assigned
- Runbooks and alerts updated
- Team learning captured

---

## Success Criteria

### Response Metrics
- **Detection**: <5 minutes (automated alerts)
- **Acknowledgement**: <2 minutes (on-call)
- **Stabilization**: <15 minutes (SEV-1), <60 minutes (SEV-2)
- **Resolution**: <30 minutes (SEV-1), <4 hours (SEV-2)

### Prevention Metrics
- **Repeat Incidents**: 0 (same root cause)
- **Action Item Completion**: >90% within 30 days
- **Runbook Coverage**: All common scenarios documented

### Communication Metrics
- **Status Updates**: Every 15 min (SEV-1), Every hour (SEV-2)
- **Stakeholder Notification**: Within 5 minutes of detection
- **Post-Incident Report**: Within 24 hours

## Related Documentation
- [Health Monitoring Runbook](kafka-backend-health-monitoring.md)
- [Migration Execution Runbook](kafka-migration-execution.md)
- [Troubleshooting Guide](../kafka/TROUBLESHOOTING.md)
- [Rollback Procedures](../kafka/rollback-procedures.md)

## Escalation Contacts
- **On-call Engineer**: PagerDuty rotation
- **Backend Team Lead**: [Contact]
- **Kafka Infrastructure Team**: [Contact]
- **Engineering Manager**: [Contact]

## Changelog
- 2025-11-26: Initial runbook created (Task 6.3)
