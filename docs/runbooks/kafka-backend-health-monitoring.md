# Kafka Backend Health Monitoring Runbook

## Overview
This runbook provides operational procedures for monitoring Kafka backend health, detecting issues early, and maintaining high availability during the migration period.

**Audience**: SREs, Platform engineers, On-call personnel
**Estimated time**: 10-20 minutes per check
**Prerequisites**: Access to Kafka cluster, monitoring tools, production logs

## Table of Contents
1. [Health Check Procedures](#health-check-procedures)
2. [Performance Monitoring](#performance-monitoring)
3. [Alerting and Escalation](#alerting-and-escalation)
4. [Common Health Issues](#common-health-issues)

---

## Health Check Procedures

### Purpose
Validate that both legacy and modern Kafka backends can successfully connect to and interact with Kafka clusters.

### Procedure

1. **Manual Health Check**
   ```python
   from cryptofeed.backends.kafka.maintenance.integration import MaintenanceCoordinator

   coordinator = MaintenanceCoordinator()

   # Check legacy backend health
   legacy_result = coordinator.handle_health_check(
       bootstrap_servers=["kafka.prod.example.com:9092"],
       implementation="legacy"
   )
   print(f"Legacy health: {'OK' if legacy_result.success else 'FAILED'}")

   # Check modern backend health
   modern_result = coordinator.handle_health_check(
       bootstrap_servers=["kafka.prod.example.com:9092"],
       implementation="modern"
   )
   print(f"Modern health: {'OK' if modern_result.success else 'FAILED'}")
   ```

2. **Automated Health Check**
   ```bash
   # Using maintenance scheduler
   python tools/kafka-maintenance-scheduler.py run-once \
     --config config/health-checks.yaml

   # Check results
   python tools/kafka-maintenance-scheduler.py status \
     --config config/health-checks.yaml
   ```

3. **Interpret Results**
   - **Healthy**: Connection successful, latency < 100ms
   - **Degraded**: Connection successful, latency 100-500ms (warning)
   - **Unhealthy**: Connection failed or latency > 500ms (critical)

4. **Log Findings**
   ```bash
   # Health check logs
   tail -f /var/log/cryptofeed/health-checks.log

   # Metrics
   curl http://localhost:9090/metrics | grep kafka_health
   ```

### Expected Outcomes
- Both implementations report healthy status
- Latency within acceptable thresholds
- No connection errors

### Troubleshooting
**Issue**: Connection timeout
**Solution**: Check network connectivity, verify bootstrap servers, check firewall rules

**Issue**: Authentication errors
**Solution**: Verify credentials, check SASL configuration, validate SSL certificates

---

## Performance Monitoring

### Purpose
Monitor Kafka backend performance metrics to detect degradation and ensure SLAs are met.

### Procedure

1. **Key Metrics to Monitor**
   - **Latency**: p50, p95, p99 message publish latency
   - **Throughput**: Messages per second
   - **Error rate**: Failed publishes / total attempts
   - **Resource usage**: CPU, memory, network I/O

2. **Check Latency**
   ```bash
   # Query Prometheus metrics
   curl -s 'http://prometheus:9090/api/v1/query?query=kafka_publish_latency_seconds{quantile="0.99"}'

   # Expected values:
   # p99 < 5ms: Excellent
   # p99 < 50ms: Good
   # p99 < 100ms: Acceptable
   # p99 > 100ms: Investigate
   ```

3. **Check Throughput**
   ```bash
   # Messages per second
   curl -s 'http://prometheus:9090/api/v1/query?query=rate(kafka_messages_published_total[1m])'

   # Expected: >10,000 msg/s for production workloads
   ```

4. **Check Error Rate**
   ```bash
   # Error percentage
   curl -s 'http://prometheus:9090/api/v1/query?query=rate(kafka_publish_errors_total[5m]) / rate(kafka_publish_attempts_total[5m]) * 100'

   # Target: <0.1% errors
   ```

5. **Check Resource Usage**
   ```bash
   # CPU and memory
   kubectl top pods -l app=cryptofeed

   # Network I/O
   kubectl exec -it cryptofeed-pod -- netstat -i
   ```

6. **Grafana Dashboards**
   - Navigate to `Kafka Backend Performance` dashboard
   - Review panels:
     - Latency percentiles (7-day trend)
     - Throughput by implementation
     - Error rate timeline
     - Resource utilization

### Expected Outcomes
- All metrics within acceptable ranges
- No sudden spikes or degradation
- Resource usage stable and predictable

### Troubleshooting
**Issue**: Latency spike
**Solution**: Check Kafka broker health, review network latency, analyze slow queries

**Issue**: Throughput drop
**Solution**: Check producer configuration, verify partition count, review batch settings

---

## Alerting and Escalation

### Purpose
Configure and respond to automated alerts for Kafka backend health issues.

### Alert Configuration

1. **Critical Alerts** (Page on-call immediately)
   ```yaml
   # Prometheus alert rules
   - alert: KafkaBackendUnhealthy
     expr: kafka_health_check_success == 0
     for: 5m
     annotations:
       summary: "Kafka backend health check failing"
       description: "{{ $labels.implementation }} backend unable to connect to Kafka"

   - alert: KafkaHighErrorRate
     expr: rate(kafka_publish_errors_total[5m]) / rate(kafka_publish_attempts_total[5m]) > 0.01
     for: 10m
     annotations:
       summary: "High Kafka publish error rate"
       description: "Error rate: {{ $value }}% for {{ $labels.implementation }}"
   ```

2. **Warning Alerts** (Notify team, investigate during business hours)
   ```yaml
   - alert: KafkaHighLatency
     expr: kafka_publish_latency_seconds{quantile="0.99"} > 0.1
     for: 15m
     annotations:
       summary: "High Kafka publish latency"
       description: "P99 latency: {{ $value }}s for {{ $labels.implementation }}"

   - alert: KafkaThroughputDrop
     expr: rate(kafka_messages_published_total[5m]) < 1000
     for: 15m
     annotations:
       summary: "Low Kafka throughput"
       description: "Current rate: {{ $value }} msg/s"
   ```

### Escalation Path

1. **Tier 1 - On-call Engineer** (0-15 minutes)
   - Acknowledge alert
   - Run health checks
   - Check recent changes (deployments, config)
   - Attempt basic remediation (restart, rollback)

2. **Tier 2 - Backend Team** (15-30 minutes)
   - Deep dive into logs and metrics
   - Identify root cause
   - Implement fix or workaround
   - Update incident documentation

3. **Tier 3 - Core Maintainers** (30+ minutes)
   - Architectural decisions
   - Code fixes
   - Long-term solutions
   - Post-incident review

### Response Procedures

**Critical Alert Response**:
1. Acknowledge alert in PagerDuty
2. Join incident Slack channel
3. Run health check runbook
4. Execute rollback if needed
5. Post status updates every 15 minutes
6. Document all actions in incident log

**Warning Alert Response**:
1. Create Jira ticket
2. Investigate during next shift
3. Document findings
4. Schedule fix if needed
5. Update alert thresholds if false positive

### Expected Outcomes
- Incidents detected within 5 minutes
- Response initiated within 15 minutes
- Resolution within 1 hour (for critical)
- No data loss or availability impact

### Troubleshooting
**Issue**: Alert fatigue (too many false positives)
**Solution**: Tune alert thresholds, add aggregation windows, use rate-of-change alerts

**Issue**: Missed incidents (no alert fired)
**Solution**: Review alert coverage, add missing metrics, test alerting pipeline

---

## Common Health Issues

### Issue 1: Kafka Connection Failures

**Symptoms**:
- Health checks failing
- "Connection refused" errors in logs
- Timeout exceptions

**Diagnosis**:
```bash
# Test Kafka connectivity
telnet kafka.prod.example.com 9092

# Check DNS resolution
nslookup kafka.prod.example.com

# Verify network path
traceroute kafka.prod.example.com
```

**Resolution**:
1. Verify Kafka broker is running: `kubectl get pods -l app=kafka`
2. Check network policies and firewall rules
3. Validate security group configurations
4. Review Kafka broker logs for errors

---

### Issue 2: High Message Lag

**Symptoms**:
- Messages delayed in topics
- Consumer lag increasing
- Throughput degradation

**Diagnosis**:
```bash
# Check consumer lag
kafka-consumer-groups --bootstrap-server kafka:9092 \
  --group cryptofeed-consumers --describe

# Check topic partition count
kafka-topics --bootstrap-server kafka:9092 \
  --topic cryptofeed.trades --describe
```

**Resolution**:
1. Increase partition count for hot topics
2. Add more consumer instances
3. Optimize producer batch settings
4. Review partition key distribution

---

### Issue 3: Message Delivery Failures

**Symptoms**:
- Publish errors in logs
- Error rate alerts firing
- Data gaps in downstream systems

**Diagnosis**:
```python
# Check error details
from cryptofeed.backends.kafka.maintenance.integration import MaintenanceCoordinator

coordinator = MaintenanceCoordinator()
status = coordinator.get_system_status()

print(f"Recent errors: {status.get('error_count', 0)}")
```

**Resolution**:
1. Review exactly-once semantics configuration
2. Check Kafka broker disk space
3. Validate message schema compatibility
4. Review producer retry settings

---

## Success Criteria

### Health Targets
- **Availability**: >99.9% uptime for both implementations
- **Latency**: p99 < 5ms for message publish
- **Throughput**: >100,000 msg/s sustained
- **Error Rate**: <0.1% failed publishes

### Alert SLAs
- **Detection**: <5 minutes
- **Response**: <15 minutes
- **Resolution**: <1 hour (critical), <24 hours (warning)

### Monitoring Coverage
- All Kafka topics monitored
- Both legacy and modern implementations tracked
- All error scenarios have corresponding alerts

## Related Documentation
- [Kafka Backend Architecture](../kafka/architecture.md)
- [Troubleshooting Guide](../kafka/TROUBLESHOOTING.md)
- [Deprecation Runbook](kafka-backend-deprecation.md)

## Changelog
- 2025-11-26: Initial runbook created (Task 6.3)
