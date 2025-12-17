# Staging Deployment Runbook - Task 20
## KafkaCallback Deployment to Staging Environment

**Feature**: Market Data Kafka Producer - Phase 5 Week 1
**Task**: 20 - Deploy new KafkaCallback to staging environment
**Environment**: Staging
**Estimated Duration**: 6 hours

---

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Deployment Steps](#deployment-steps)
3. [Post-Deployment Validation](#post-deployment-validation)
4. [Monitoring & Health Checks](#monitoring--health-checks)
5. [Rollback Procedure](#rollback-procedure)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Success Criteria](#success-criteria)
8. [Time Estimates](#time-estimates)

---

## Pre-Deployment Checklist

**Duration**: 30 minutes

### Prerequisites

- [ ] All Week 0 blockers resolved (Tasks 19.2-19.7 complete)
- [ ] Security validation passed: `./scripts/validate-security-prerequisites.sh`
- [ ] Environment variables configured: `.env.production` file created and validated
- [ ] Kafka cluster operational (3+ brokers, all healthy)
- [ ] Monitoring infrastructure ready (Prometheus, Grafana)
- [ ] Consumer applications notified and ready
- [ ] Team on-call and available for entire deployment window

### Environment Validation

Run pre-deployment validation:

```bash
./scripts/validate-staging-deployment.sh
```

**Expected Output**: All checks pass with green checkmarks

**If any checks fail**: DO NOT proceed. Fix issues first.

### Team Readiness

- [ ] On-call engineer available for 6+ hours
- [ ] Backup engineer on standby
- [ ] Stakeholders notified of deployment window
- [ ] Communication channel established (Slack, etc.)
- [ ] Rollback team identified and ready

---

## Deployment Steps

### Overview

**Strategy**: Canary Rollout (10% → 50% → 100%)
**Total Duration**: ~6 hours (2h + 2h + 30min + monitoring)

### Step 1: Pre-Deployment Validation (30 min, T-30)

**Actions**:

1. Run security validation:
   ```bash
   ./scripts/validate-security-prerequisites.sh
   ```

2. Run environment validation:
   ```bash
   ./scripts/validate-environment.sh
   ```

3. Run staging validation:
   ```bash
   ./scripts/validate-staging-deployment.sh
   ```

4. Review configuration file:
   ```bash
   cat deployment/staging/kafka-callback-config.yaml
   ```

**Success Criteria**:
- All validation scripts pass
- Configuration file valid
- Team confirms readiness

**If failures**: Stop and resolve issues before proceeding.

---

### Step 2: Create Backup Point (10 min, T+0)

**Actions**:

1. Start deployment script (it will create backups):
   ```bash
   ./scripts/deploy-staging-kafka-callback.sh
   ```

2. Verify backup created:
   ```bash
   ls -la backups/deployment_*
   ```

**Success Criteria**:
- Backup directory created
- Configuration files backed up
- Kafka state recorded

---

### Step 3: Create Consolidated Topics (15 min, T+10)

**Actions**:

The deployment script will automatically create topics if they don't exist:

- `cryptofeed.trades` (12 partitions, RF=3)
- `cryptofeed.orderbook` (12 partitions, RF=3)
- `cryptofeed.ticker` (12 partitions, RF=3)
- `cryptofeed.candle` (12 partitions, RF=3)
- `cryptofeed.funding` (12 partitions, RF=3)
- `cryptofeed.liquidation` (12 partitions, RF=3)
- `cryptofeed.index` (12 partitions, RF=3)
- `cryptofeed.openinterest` (12 partitions, RF=3)

**Manual Verification** (if needed):

```bash
kafka-topics.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS --list | grep cryptofeed
```

**Success Criteria**:
- All 8 topics created (or already exist)
- Each topic has 12 partitions
- Replication factor = 3
- Min in-sync replicas = 2

---

### Step 4: Deploy to 10% of Instances (2 hours monitoring, T+25)

**Actions**:

1. Deploy KafkaCallback to 10% of producer instances using your deployment tool:
   - Kubernetes: `kubectl set image deployment/cryptofeed-producer ...`
   - Docker Swarm: `docker service update --image ... cryptofeed-producer`
   - Manual: Update 10% of instances with new configuration

2. Verify deployment:
   ```bash
   # Kubernetes example
   kubectl rollout status deployment/cryptofeed-producer

   # Docker example
   docker service ps cryptofeed-producer
   ```

3. Start continuous health monitoring:
   ```bash
   ./scripts/health-check-staging.sh --interval 30 --duration 2
   ```

4. Monitor metrics in Grafana:
   - Error rate (target: <0.1%)
   - Latency p99 (target: <5ms)
   - Broker CPU/Memory (target: <80%)
   - Message throughput

**Success Criteria**:
- 10% of instances running new KafkaCallback
- Health checks passing for 2 hours
- Error rate <0.1%
- Latency p99 <5ms
- Broker metrics stable

**If failures**: Run rollback immediately.

---

### Step 5: Expand to 50% of Instances (2 hours monitoring, T+2h 25min)

**Actions**:

1. Expand deployment to 50% of instances:
   ```bash
   # Update deployment to 50%
   ```

2. Verify expansion:
   ```bash
   # Check that 50% of instances are updated
   ```

3. Continue health monitoring:
   ```bash
   ./scripts/health-check-staging.sh --interval 30 --duration 2
   ```

4. Monitor metrics (same targets as Step 4)

**Success Criteria**:
- 50% of instances running new KafkaCallback
- Health checks passing for 2 hours
- Error rate <0.1%
- Latency p99 <5ms
- Broker metrics stable

**If failures**: Run rollback immediately.

---

### Step 6: Complete Rollout to 100% (30 min, T+4h 25min)

**Actions**:

1. Complete deployment to 100% of instances:
   ```bash
   # Update deployment to 100%
   ```

2. Verify complete rollout:
   ```bash
   # Check all instances updated
   ```

3. Run final health check:
   ```bash
   ./scripts/health-check-staging.sh --interval 30 --duration 1
   ```

**Success Criteria**:
- 100% of instances running new KafkaCallback
- Health checks passing
- No errors or warnings

---

## Post-Deployment Validation

**Duration**: 1 hour (T+4h 55min)

### Automated Validation

Run post-deployment validation script:

```bash
./scripts/validate-post-deployment.sh
```

**Validates**:
- Message format and headers
- Protobuf serialization
- Message latency <5ms
- Error rate <0.1%
- Broker metrics stable

**Expected Output**: All checks pass

### Manual Validation

1. **Message Headers Validation**:
   ```bash
   # Consume a sample message and check headers
   kafka-console-consumer.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
     --topic cryptofeed.trades \
     --max-messages 1 \
     --property print.headers=true
   ```

   **Verify headers present**:
   - `exchange:coinbase` (or other exchange)
   - `symbol:BTC-USD` (or other symbol)
   - `data_type:trades`
   - `schema_version:1.0.0`

2. **Protobuf Serialization Validation**:
   ```python
   # Run Python script to verify protobuf deserialization
   python3 << EOF
   from kafka import KafkaConsumer
   from cryptofeed.backends.protobuf_helpers import deserialize_protobuf_message

   consumer = KafkaConsumer('cryptofeed.trades', ...)
   for message in consumer:
       data_type = dict(message.headers).get('data_type').decode('utf-8')
       deserialized = deserialize_protobuf_message(message.value, data_type)
       print("✓ Protobuf deserialization successful")
       break
   EOF
   ```

3. **Consumer Application Testing**:
   - Verify consumer applications can deserialize messages
   - Verify consumer lag is minimal (<5 seconds)
   - Verify no consumer errors

---

## Monitoring & Health Checks

### Continuous Monitoring (2-4 hours post-deployment)

**Run continuous health checks**:

```bash
./scripts/health-check-staging.sh --interval 30 --alert-on-failure
```

**Monitor in Grafana**:

Dashboard URL: `${GRAFANA_URL}/d/kafka-producer-staging`

**Key Metrics**:

1. **Error Rate** (target: <0.1%)
   - Query: `rate(kafka_producer_errors_total[5m])`
   - Alert threshold: >0.1% for 5 minutes

2. **Latency p99** (target: <5ms)
   - Query: `histogram_quantile(0.99, rate(kafka_producer_latency_seconds_bucket[5m]))`
   - Alert threshold: >5ms for 10 minutes

3. **Broker CPU** (target: <80%)
   - Query: `avg(kafka_broker_cpu_percent)`
   - Alert threshold: >80% for 15 minutes

4. **Broker Memory** (target: <80%)
   - Query: `avg(kafka_broker_memory_percent)`
   - Alert threshold: >80% for 15 minutes

5. **Message Throughput**
   - Query: `rate(kafka_producer_messages_sent_total[5m])`
   - Baseline: Compare to pre-deployment levels

6. **Topic Lag** (for consumer validation)
   - Query: `kafka_consumer_lag_seconds`
   - Target: <5 seconds

---

## Rollback Procedure

**Trigger Conditions** (immediate rollback):

1. Error rate exceeds 0.1% for >5 minutes
2. Latency p99 exceeds 5ms for >10 minutes
3. Broker CPU exceeds 90% for >15 minutes
4. Message delivery failures detected
5. Consumer applications unable to deserialize messages

**Rollback Steps**:

1. **Initiate Rollback**:
   ```bash
   ./scripts/rollback-staging-deployment.sh
   ```

2. **Follow Prompts**:
   - Script will guide through rollback process
   - Confirm each step

3. **Verify Rollback**:
   - Check that new producer instances are stopped
   - Check that connections are drained
   - Verify no new messages being produced

4. **Notify Team**:
   - Send notification to Slack/PagerDuty
   - Create incident ticket
   - Schedule post-mortem

**Data Preservation**:
- ✓ All Kafka topics preserved (no deletion)
- ✓ Consumer offsets preserved
- ✓ All messages preserved

**Estimated Rollback Time**: <5 minutes

---

## Troubleshooting Guide

### Issue: Validation Scripts Fail

**Symptoms**: Pre-deployment validation returns errors

**Diagnosis**:
```bash
# Re-run with verbose output
./scripts/validate-staging-deployment.sh 2>&1 | tee validation.log

# Check specific failures
grep "✗" validation.log
```

**Common Causes**:
1. Environment variables not set → Source `.env.production`
2. Kafka cluster not accessible → Check network/firewall
3. Insufficient brokers → Scale up cluster to 3+ brokers
4. Security validation fails → Check certificates/credentials

**Resolution**: Fix the specific issue identified and re-run validation

---

### Issue: Topics Already Exist with Data

**Symptoms**: Deployment script warns about existing topics

**Diagnosis**:
```bash
# Check topic message counts
kafka-run-class.sh kafka.tools.GetOffsetShell \
  --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic cryptofeed.trades \
  --time -1
```

**Decision Tree**:
- **If topics are empty**: Proceed (topics will be reused)
- **If topics have test data**: Proceed (will be overwritten)
- **If topics have production data**: STOP - investigate why production data exists in staging

**Resolution**: Confirm data is expected before proceeding

---

### Issue: Health Checks Failing

**Symptoms**: `health-check-staging.sh` shows FAIL status

**Diagnosis**:
```bash
# Check specific failure
./scripts/health-check-staging.sh --interval 10

# Check Kafka connectivity
kafka-broker-api-versions.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS

# Check producer logs
# (depends on your deployment platform)
```

**Common Causes**:
1. Kafka cluster down → Check broker status
2. Network issues → Check connectivity
3. Configuration errors → Review kafka-callback-config.yaml
4. Resource exhaustion → Check broker CPU/memory

**Resolution**: Fix the underlying issue, then re-run health checks

---

### Issue: High Latency (>5ms)

**Symptoms**: Latency p99 exceeds 5ms threshold

**Diagnosis**:
```bash
# Query Prometheus for latency breakdown
curl -s "$PROMETHEUS_URL/api/v1/query?query=histogram_quantile(0.99,%20rate(kafka_producer_latency_seconds_bucket[5m]))"

# Check broker metrics
curl -s "$PROMETHEUS_URL/api/v1/query?query=kafka_broker_cpu_percent"
```

**Common Causes**:
1. Broker CPU/memory saturation → Scale brokers or reduce load
2. Network latency → Check broker-to-broker and client-to-broker latency
3. Large batch sizes → Reduce `batch_size` or `linger_ms` in config
4. Disk I/O bottleneck → Check broker disk performance

**Resolution**: Tune configuration or scale infrastructure

---

### Issue: Error Rate Exceeds 0.1%

**Symptoms**: Error rate above acceptable threshold

**Diagnosis**:
```bash
# Check error metrics
curl -s "$PROMETHEUS_URL/api/v1/query?query=rate(kafka_producer_errors_total[5m])"

# Check producer logs for error details
# (depends on your deployment platform)

# Check Kafka broker logs
# ssh to broker and check /var/log/kafka/
```

**Common Causes**:
1. Serialization errors → Check protobuf schema compatibility
2. Network timeouts → Increase `request_timeout_ms`
3. Broker rejecting messages → Check topic configuration
4. Authentication/authorization errors → Check SASL/SSL configuration

**Resolution**: Fix the specific error cause, then re-run validation

---

## Success Criteria

### Exit Criteria for Task 20

All of the following must be TRUE to mark Task 20 as complete:

- [ ] Staging deployment complete (100% rollout)
- [ ] All consolidated topics created and operational
- [ ] Messages have correct format and headers (exchange, symbol, data_type, schema_version)
- [ ] Protobuf serialization validated (~63% size reduction)
- [ ] Message latency p99 <5ms for 2 hours
- [ ] Error rate <0.1% for 2 hours
- [ ] Broker metrics stable (CPU <80%, Memory <80%) for 2-4 hours
- [ ] Consumer applications successfully deserializing messages
- [ ] Monitoring dashboard operational and showing healthy metrics
- [ ] Team signoff obtained

### Measurable Targets

| Metric | Target | Measurement | Duration |
|--------|--------|-------------|----------|
| Error Rate | <0.1% | Prometheus query | 2 hours |
| Latency p99 | <5ms | Prometheus query | 2 hours |
| Broker CPU | <80% | Prometheus query | 2-4 hours |
| Broker Memory | <80% | Prometheus query | 2-4 hours |
| Message Loss | Zero | Topic offset comparison | Continuous |
| Consumer Lag | <5 seconds | Consumer group lag | Continuous |

---

## Time Estimates

### Deployment Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Pre-Deployment Validation | 30 min | T+0:30 |
| Backup & Topic Creation | 25 min | T+0:55 |
| Deploy to 10% (monitoring) | 2 hours | T+2:55 |
| Expand to 50% (monitoring) | 2 hours | T+4:55 |
| Complete to 100% | 30 min | T+5:25 |
| Post-Deployment Validation | 1 hour | T+6:25 |
| **Total Deployment Time** | **~6.5 hours** | - |

### Monitoring Timeline (Post-Deployment)

- **Minimum**: 2 hours continuous monitoring
- **Recommended**: 4 hours continuous monitoring
- **Extended**: 24 hours intermittent monitoring

### Rollback Timeline (If Needed)

- **Detection**: <5 minutes (automated alerts)
- **Decision**: <5 minutes (team decision)
- **Execution**: <5 minutes (rollback script)
- **Verification**: <5 minutes (health checks)
- **Total Rollback Time**: <20 minutes

---

## Sign-off

### Deployment Sign-off

**Deployment Date**: _______________
**Deployment Start Time**: _______________
**Deployment End Time**: _______________
**Total Duration**: _______________

**Deployment Team**:
- On-Call Engineer: _______________
- Backup Engineer: _______________
- Team Lead: _______________

**Success Criteria Met**:
- [ ] All health checks passing
- [ ] Error rate <0.1%
- [ ] Latency p99 <5ms
- [ ] Broker metrics stable
- [ ] Consumer applications validated
- [ ] 2-4 hours monitoring complete

**Sign-off**:
- Engineer: _______________ Date: _______________
- Team Lead: _______________ Date: _______________

**Notes/Issues Encountered**:
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________

---

## Appendix

### A. Environment Variables Reference

See `.env.production.template` for complete list.

**Critical Variables**:
- `KAFKA_BOOTSTRAP_SERVERS`
- `KAFKA_SASL_USERNAME`
- `KAFKA_SASL_PASSWORD`
- `KAFKA_SSL_CERT`
- `KAFKA_SSL_KEY`
- `KAFKA_SSL_CA`
- `PROMETHEUS_URL`
- `GRAFANA_URL`

### B. Kafka Topic Configuration

```yaml
Topic: cryptofeed.{data_type}
Partitions: 12
Replication Factor: 3
Min In-Sync Replicas: 2
Cleanup Policy: delete
Retention: 7 days (staging only)
Compression: lz4
```

### C. Useful Commands

**Check topic lag**:
```bash
kafka-consumer-groups.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --group <consumer-group> --describe
```

**Consume messages with headers**:
```bash
kafka-console-consumer.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic cryptofeed.trades \
  --property print.headers=true \
  --max-messages 10
```

**Check broker health**:
```bash
kafka-broker-api-versions.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS
```

### D. Contact Information

**Escalation Path**:
1. On-Call Engineer (primary)
2. Backup Engineer (if primary unavailable)
3. Team Lead (critical decisions)
4. Engineering Manager (extended outages)

**Communication Channels**:
- Slack: #cryptofeed-deployments
- PagerDuty: cryptofeed-oncall
- Email: engineering@company.com

---

**Document Version**: 1.0
**Last Updated**: 2025-11-26
**Maintained By**: Cryptofeed Engineering Team
