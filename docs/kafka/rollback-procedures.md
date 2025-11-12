# Kafka Rollback Procedures

This document describes emergency rollback procedures for reverting from Phase 2 KafkaCallback to the legacy Kafka backend if critical issues are encountered during or after migration.

---

## Quick Rollback (< 5 minutes)

Use this procedure if Phase 2 must be disabled immediately due to critical issues.

### Symptoms of Critical Issues

- Kafka brokers unable to keep up with producer rate (consistent queue backing up)
- Message loss detected (consolidated topics have fewer messages than legacy)
- Severe latency increase (p99 > 100ms, typical is 2-5ms)
- Producer crashes or frequent reconnects
- Downstream consumers unable to process messages

### Rollback Steps

#### 1. Stop Phase 2 Producer (immediately)

```python
# Option A: In Python (if running in-process)
await phase2_kafka_callback.close()

# Option B: Via configuration (if using feature flag)
ENABLE_PHASE2_KAFKA = False  # Disable in config

# Option C: Via Kubernetes (if containerized)
kubectl delete deployment kafka-producer-phase2
```

#### 2. Restart Legacy Producer

```python
# Re-enable legacy kafka backend
from cryptofeed.backends.kafka import TradeKafka

legacy_kafka = TradeKafka(
    bootstrap_servers=['kafka:9092'],
    topic_prefix='cryptofeed',
    acks='1'
)

feed.add_callback(TradeKafka, legacy_kafka)
await producer.start()
```

#### 3. Verify Legacy Producer is Running

```bash
# Check message flow into legacy topics
kafkacat -b kafka:9092 -t cryptofeed.trades.binance.btc-usd -c 10

# Should see messages appearing within seconds
# Each message should be valid JSON
```

#### 4. Alert Operations Team

Send alert to on-call team:
- What: Phase 2 Kafka disabled due to critical issue
- When: [timestamp]
- Why: [error description]
- Impact: Now using legacy Kafka backend
- Next steps: Investigation required

---

## Data Recovery After Rollback

After reverting to legacy backend, verify no data gaps in critical systems.

### Step 1: Check Consumer Offsets

```bash
# For each critical consumer group
kafka-consumer-groups.sh --bootstrap-server kafka:9092 \
  --describe --group <consumer_group_name>

# Expected output:
# TOPIC                          PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
# cryptofeed.trades.binance...   0          12345           12346           1
```

Key metrics:
- **Current Offset**: Last processed message
- **LOG-END-OFFSET**: Latest message in topic
- **LAG**: Should be < 100 (not falling behind)

If LAG > 100: Consumers are behind, may miss data.

### Step 2: Verify No Consumer Lag Spike

```bash
# Check if consumer lag increased during rollback
# (Compare to pre-migration baseline)

# Historical data (if monitoring with Prometheus)
# Query: rate(kafka_consumer_lag_total[5m])
# Should show normal rate, not spiking
```

If lag is spiking: Consumers may not be processing messages.

### Step 3: Verify Downstream Data Consistency

```python
# For each critical downstream system
# Example: DuckDB table containing trades

import duckdb

con = duckdb.connect('trades.db')

# Count trades in DuckDB
duckdb_count = con.execute(
    'SELECT COUNT(*) FROM trades'
).fetchall()[0][0]

# Count trades in legacy Kafka
legacy_kafka_count = consumer.fetch_count(
    'cryptofeed.trades.binance.btc-usd'
)

# Should match (or DuckDB slightly behind)
assert abs(duckdb_count - legacy_kafka_count) < 100, \
    f"Data gap detected: DuckDB={duckdb_count}, Kafka={legacy_kafka_count}"
```

If counts mismatch significantly: Data was lost.

### Step 4: Handle Dead-Letter Queue (DLQ)

If Phase 2 produced error messages, they may be in DLQ:

```bash
# Check if DLQ topic exists
kafkacat -b kafka:9092 -L | grep -i dlq

# Sample error messages
kafkacat -b kafka:9092 -t kafka-dlq -c 10 | jq .

# Analyze errors
# - Serialization errors: Protobuf schema mismatch
# - Broker errors: Broker unavailable/down
# - Timeout errors: Producer lag behind
```

### Step 5: Mark Recovery Point

For full recovery if needed:

```bash
# Reset consumer offset to known good point
# (only if previous steps identified data loss)

kafka-consumer-groups.sh --bootstrap-server kafka:9092 \
  --group <consumer_group_name> \
  --reset-offsets \
  --to-offset <last_known_good_offset> \
  --execute \
  --topic cryptofeed.trades.binance.btc-usd

# Then consumer will replay from last known good point
```

---

## Extended Rollback (Full Data Cleanup)

Use this if Phase 2 topics need to be removed or if extended investigation required.

### When to Use Extended Rollback

- Phase 2 data was corrupted (protobuf parsing errors)
- Large data gaps detected
- Need to delete consolidated topics to test fresh migration
- Investigating subtle data consistency issues

### Extended Rollback Steps

#### 1. Create Backup of Phase 2 Topics

```bash
# Archive Phase 2 topics to external storage (Minio, S3, etc.)
# for later investigation

# Using kafka-mirror-maker
kafka-mirror-maker.sh \
  --source-brokers kafka:9092 \
  --target-brokers backup-kafka:9092 \
  --groups cryptofeed-group \
  --topics 'cryptofeed.*'
```

#### 2. Stop All Producers

```python
await phase2_kafka.close()      # Stop Phase 2
await legacy_kafka.close()      # Stop legacy
```

#### 3. Reset Consumer Groups

```bash
# For each consumer group affected by migration
for group in cryptofeed-consumers duckdb-ingest; do
  kafka-consumer-groups.sh --bootstrap-server kafka:9092 \
    --group $group \
    --reset-offsets \
    --to-earliest \
    --topic cryptofeed.trades \
    --execute
done
```

#### 4. Delete Phase 2 Topics (Optional)

```bash
# Only if Phase 2 topics are corrupted and need to be recreated
# WARNING: This deletes all data in Phase 2 topics!

for topic in cryptofeed.trades cryptofeed.orderbook cryptofeed.ticker; do
  kafka-topics.sh --bootstrap-server kafka:9092 \
    --delete \
    --topic $topic
done
```

#### 5. Restart Producers in Legacy Mode

```python
# Restart with legacy backend only
legacy_kafka = TradeKafka(bootstrap_servers=['kafka:9092'])
feed.add_callback(TradeKafka, legacy_kafka)
await producer.start()
```

#### 6. Verify Data Flow Recovery

```bash
# Check that messages are flowing into legacy topics again
watch -n 1 'kafka-consumer-groups.sh --bootstrap-server kafka:9092 \
  --describe --group cryptofeed-consumers | grep LAG'

# LAG should start at 0 and stay < 100
```

---

## Health Check Verification Post-Rollback

Use these checks to verify the system is healthy after rollback.

### Checklist

```
[ ] Legacy Kafka producer running (verified by message count)
[ ] Consumer lag < 100 (for all critical consumers)
[ ] Downstream data consistent (DuckDB, etc.)
[ ] No errors in application logs
[ ] Alerts cleared (latency normal, errors <0.1%)
[ ] Message rate matches pre-migration baseline
[ ] No memory leaks in producer
```

### Automated Verification Script

```python
#!/usr/bin/env python3
"""Post-rollback health check."""

import subprocess
import sys

def check_kafka_producer():
    """Verify messages are flowing into legacy topics."""
    result = subprocess.run([
        'kafkacat', '-b', 'kafka:9092',
        '-t', 'cryptofeed.trades.binance.btc-usd',
        '-c', '1', '-J'
    ], capture_output=True, timeout=5)

    if result.returncode == 0:
        print("✓ Kafka producer healthy (messages flowing)")
        return True
    else:
        print("✗ Kafka producer unhealthy (no messages)")
        return False

def check_consumer_lag():
    """Verify consumers are keeping up."""
    result = subprocess.run([
        'kafka-consumer-groups.sh',
        '--bootstrap-server', 'kafka:9092',
        '--describe', '--group', 'cryptofeed-consumers'
    ], capture_output=True, text=True)

    # Parse LAG column
    lines = result.stdout.strip().split('\n')
    total_lag = 0
    for line in lines:
        if 'TOPIC' not in line:
            parts = line.split()
            lag = int(parts[-1])
            total_lag += lag

    if total_lag < 100:
        print(f"✓ Consumer lag healthy ({total_lag} messages)")
        return True
    else:
        print(f"✗ Consumer lag too high ({total_lag} messages)")
        return False

def check_downstream_data():
    """Verify downstream systems received data."""
    import duckdb
    con = duckdb.connect('trades.db')
    count = con.execute(
        'SELECT COUNT(*) FROM trades WHERE created_at > NOW() - INTERVAL 5 MINUTE'
    ).fetchall()[0][0]

    if count > 0:
        print(f"✓ Downstream data healthy ({count} trades in last 5 min)")
        return True
    else:
        print("✗ Downstream data stale (no recent trades)")
        return False

if __name__ == '__main__':
    checks = [
        check_kafka_producer(),
        check_consumer_lag(),
        check_downstream_data(),
    ]

    if all(checks):
        print("\n✓ All health checks passed")
        sys.exit(0)
    else:
        print("\n✗ Some health checks failed")
        sys.exit(1)
```

---

## Investigation Steps

Before considering permanent rollback, investigate the root cause.

### Identify the Issue

#### Symptom: High Latency (p99 > 50ms)

```bash
# Check broker metrics
kafka-broker-api-versions.sh --bootstrap-server kafka:9092

# Check network latency
ping -c 10 kafka:9092

# Check broker CPU/memory
kubectl describe node kafka-0

# Review producer config
# - batch_size: too large? (causes buffering)
# - linger_ms: too high? (causes delay)
# - compression: too aggressive? (CPU bottleneck)
```

**Fix Options**:
- Increase batch_size (if waiting for full batch)
- Decrease linger_ms (send sooner)
- Use faster compression (snappy instead of zstd)
- Add more partitions (if broker CPU high)

#### Symptom: Message Loss

```bash
# Compare message counts between legacy and Phase 2 topics
legacy_count=$(kafkacat -b kafka:9092 -t cryptofeed.trades.binance.btc-usd -c -1 -q | wc -l)
phase2_count=$(kafkacat -b kafka:9092 -t cryptofeed.trades -c -1 -q | jq 'select(.exchange == "binance" and .symbol == "btc-usd")' | wc -l)

echo "Legacy: $legacy_count, Phase 2: $phase2_count"

# If Phase 2 < Legacy:
# 1. Check producer logs for errors
# 2. Check Kafka broker logs for rejections
# 3. Verify brokers have sufficient disk space
# 4. Check network connectivity during migration
```

**Fix Options**:
- Enable `idempotence: true` (prevent duplicates on retry)
- Increase `retries` (retry more aggressively)
- Use `acks: all` (ensure all replicas acknowledge)

#### Symptom: Downstream Consumer Errors

```bash
# Check consumer logs
kubectl logs deployment/duckdb-consumer | grep -i error | tail -20

# Check if deserialization failing
# (May indicate protobuf schema mismatch)

# If protobuf error:
# 1. Verify proto schema matches Phase 2 version
# 2. Check message format in Phase 2 topics
# 3. Verify serialization config matches consumer
```

**Fix Options**:
- Verify protobuf schema versions match
- Check serialization format (protobuf vs JSON)
- Ensure consumer understands Phase 2 message format

---

## Post-Rollback Analysis

### Generate Incident Report

```markdown
# Kafka Phase 2 Migration - Incident Report

## Timeline
- [start_time]: Migration begins (Phase 2 deployed)
- [detection_time]: Issue detected (symptom)
- [rollback_time]: Rollback initiated
- [recovery_time]: System recovered

## Root Cause
[Detailed description of what went wrong]

## Impact
- Duration: [minutes/hours] of degraded service
- Messages lost: [count] (if any)
- Affected systems: [list]

## Lessons Learned
1. What should have been done differently?
2. What monitoring would have detected this earlier?
3. How do we prevent this in future?

## Action Items
- [ ] Fix root cause
- [ ] Add monitoring for this failure mode
- [ ] Test mitigation before retry
- [ ] Document in runbook
- [ ] Schedule post-incident review

## Retry Plan
[Plan for attempting Phase 2 migration again]
```

### Determine Retry Approach

**Option 1: Fix and Retry (1-3 days)**
- Identify root cause
- Apply fix to Phase 2 code/config
- Test fix in staging
- Retry migration with Strategy B (dual-write, validation)

**Option 2: Investigate Further (3-7 days)**
- Run extended diagnostics
- Test Phase 2 in load testing environment
- Stress test at production scale
- Modify configuration based on findings
- Retry migration with additional monitoring

**Option 3: Abandon Phase 2 (indefinite)**
- Conclude Phase 2 not suitable for current deployment
- Continue using legacy backend
- Explore alternative architectures (external Kafka service, etc.)
- Document decision and rationale

---

## Backup and Recovery

### Preserving Phase 2 Data for Analysis

If needing to preserve Phase 2 topics for investigation:

```bash
# Dump consolidated topics to files for analysis
for topic in cryptofeed.trades cryptofeed.orderbook cryptofeed.ticker; do
  kafkacat -b kafka:9092 -t $topic -c -1 -q > /backup/$topic.jsonl &
done
wait

# Compress for storage
tar -czf /backup/phase2-topics-$(date +%Y%m%d).tar.gz /backup/*.jsonl

# Upload to S3
aws s3 cp /backup/phase2-topics-*.tar.gz s3://backup-bucket/kafka/
```

### Recovering from Backup

```bash
# If need to replay Phase 2 data later
zcat /backup/phase2-topics-*.tar.gz | kafkacat -P -b kafka:9092 -t cryptofeed.trades

# Verify recovery
kafkacat -b kafka:9092 -t cryptofeed.trades -c 10
```

---

## Testing Rollback

### Practice Rollback Before Production

Test the rollback procedure in staging environment:

```bash
# 1. Deploy Phase 2 in staging
# 2. Run for 30 minutes
# 3. Trigger rollback
# 4. Verify legacy backend takes over
# 5. Check for data gaps
# 6. Verify consumers recover
```

This ensures the procedure is smooth when needed in production.

---

## Support and Escalation

### Who to Contact

- **Technical Issue**: On-call Kafka engineer
- **Data Loss**: Database/DW team lead
- **Production Impact**: Engineering manager
- **Post-Incident**: Schedule blameless postmortem

### Useful Resources

- **Kafka Logs**: `/var/log/kafka/server.log` (on brokers)
- **Producer Logs**: Application container logs
- **Consumer Logs**: Consumer group container logs
- **Monitoring**: Prometheus/Grafana dashboards
- **Documentation**: `docs/kafka/` directory

---

## Conclusion

Rollback procedures are in place to handle critical issues. However, with proper testing (Strategy B: dual-write) and validation before production cutover, critical issues should be rare.

Key takeaway: **Test migration in staging, use dual-write in production, monitor closely during transition.**

For questions or additional scenarios, contact the Kafka infrastructure team.
