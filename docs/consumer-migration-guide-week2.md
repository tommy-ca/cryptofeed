# Consumer Migration Guide - Phase 5 Week 2

**Status**: Ready for Production Migration
**Version**: 1.0.0
**Last Updated**: November 13, 2025
**Target Audience**: Data Engineers, Analytics Engineers

---

## Overview

This guide provides step-by-step instructions for migrating consumer applications from legacy per-symbol Kafka topics to the new consolidated topic format. The migration is **non-disruptive** and can be executed incrementally by exchange.

### What's Changing

| Aspect | Legacy (Per-Symbol) | New (Consolidated) |
|--------|---------------------|-------------------|
| Topics | `cryptofeed.trades.coinbase.btc-usd` | `cryptofeed.trades` |
| Count | 10,000+ topics | ~20 topics |
| Message Format | JSON | Protobuf (63% smaller) |
| Headers | None | exchange, symbol, data_type, schema_version |
| Ordering | Per-symbol per-topic | By partition (exchange-symbol) |

### Migration Timeline

- **Week 1**: Infrastructure setup (new topics, producer canary)
- **Week 2**: Consumer templates ready, monitoring deployed ← **YOU ARE HERE**
- **Week 3**: Per-exchange migration (Coinbase → Binance → Others)
- **Week 4**: Stabilization and legacy cleanup

---

## Step 1: Prepare Your Consumer Code

### Option A: Update Existing Consumer (Recommended)

If you have an existing consumer reading from per-symbol topics:

#### Before (Legacy Topic Subscription)

```python
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'cryptofeed.trades.coinbase.btc-usd',
    'cryptofeed.trades.coinbase.eth-usd',
    'cryptofeed.trades.coinbase.etc-usd',
    bootstrap_servers=['kafka1:9092'],
    group_id='my-consumer',
    auto_offset_reset='earliest',
)

for message in consumer:
    # Process message
    symbol = message.topic.split('.')[-1]  # Extract from topic name
```

#### After (Consolidated Topic with Headers)

```python
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'cryptofeed.trades',  # Single consolidated topic
    bootstrap_servers=['kafka1:9092'],
    group_id='my-consumer-v2',  # Use new group for fresh offset
    auto_offset_reset='earliest',
)

for message in consumer:
    # Extract metadata from headers
    headers = dict(message.headers or [])
    exchange = headers.get(b'exchange', b'unknown').decode()
    symbol = headers.get(b'symbol', b'unknown').decode()

    # Process message (same logic, different source)
```

**Key Changes**:
1. Topic subscription: Single pattern instead of explicit topics
2. Symbol/exchange: Extract from headers instead of topic name
3. Consumer group: New group name (optional, but recommended for fresh offset)
4. Message format: Protobuf deserialization (more compact)

### Option B: Deploy New Consumer (Alternative)

If you prefer to run new and old consumers in parallel for validation:

```python
# Create new consumer group reading consolidated topics
consumer_new = KafkaConsumer(
    'cryptofeed.trades',  # New consolidated topic
    bootstrap_servers=['kafka1:9092'],
    group_id='my-consumer-v2-new',  # Separate group
    auto_offset_reset='earliest',
)

# Keep old consumer running in parallel
consumer_old = KafkaConsumer(
    'cryptofeed.trades.coinbase.*',  # Old pattern (regex)
    bootstrap_servers=['kafka1:9092'],
    group_id='my-consumer-v2-old',
    auto_offset_reset='earliest',
)

# Dual-consume for validation period (24 hours)
# Then switch primary traffic to new consumer
# Finally decommission old consumer
```

**Benefits**:
- Zero downtime validation
- Compare message counts and timestamps
- Easy rollback if issues arise

**Drawbacks**:
- Operational complexity (running 2 consumers)
- Higher resource usage
- Requires careful offset management

### Recommendation

Use **Option A** (update existing consumer) for most cases:
- Simpler operational model
- No resource duplication
- Faster deployment
- All data validated before switching

---

## Step 2: Test in Staging Environment

Before deploying to production, validate everything works in staging.

### Pre-Stage Checklist

- [ ] Code changes reviewed and tested locally
- [ ] Staging Kafka cluster has consolidated topics
- [ ] Staging has sufficient storage/database

### Staging Deployment

```bash
# 1. Deploy updated consumer to staging
kubectl apply -f k8s/staging/consumer-v2.yaml

# 2. Verify consumer started
kubectl logs deployment/consumer-v2 -n staging | grep "Consumer started"

# 3. Monitor consumer lag
kafka-consumer-groups.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --describe --group my-consumer-v2
# Expected: Current Lag < 100 messages (decreasing over time)

# 4. Validate message count
# Legacy: SELECT COUNT(*) FROM staging.trades WHERE exchange='coinbase'
# New:    SELECT COUNT(*) FROM staging.trades_v2 WHERE exchange='coinbase'
# Expected: Same count (±0.1%)

# 5. Run integration tests
pytest tests/integration/test_consumer_migration.py -v

# 6. Monitor for 24 hours
# Check:
# - Latency (p50, p95, p99)
# - Error rate (should be 0)
# - Consumer lag (should be <5 seconds)
# - Data integrity (spot-check 100 messages)
```

### Staging Validation Checklist

- [ ] Consumer starts without errors
- [ ] Consumer lag: <5 seconds within 5 minutes
- [ ] Message count matches legacy topics (±0.1%)
- [ ] No deserialization errors in logs
- [ ] Consumer group offset committed correctly
- [ ] Data completeness: All exchanges present
- [ ] Data integrity: Spot-check 100 messages (fields match)
- [ ] No duplicates in downstream storage
- [ ] All integration tests pass
- [ ] Latency <2% increase (p99 remains <5ms)

**Success Criteria**: All 10 items checked ✓

---

## Step 3: Deploy to Production

### Pre-Production Checklist

- [ ] Staging validation passed (all 10 items)
- [ ] Production deployment plan approved
- [ ] On-call team scheduled for Week 3
- [ ] Rollback procedure tested and documented
- [ ] Stakeholders notified (Slack, email)

### Production Canary Deployment

Production deployment uses **canary rollout** for safety:

#### Phase 1: Canary (10% of instances)

```bash
# Deploy to 10% of instances
kubectl set image deployment/consumer \
  consumer=myregistry/consumer:v2 \
  --record -n production

# Wait 2 hours and monitor
kubectl rollout pause deployment/consumer -n production

# Check metrics
curl $PROMETHEUS_URL/metrics | grep consumer_lag

# Decision: Continue or Rollback
if [ "lag < 5 seconds" ]; then
  echo "✓ Canary healthy, proceeding to 50%"
else
  echo "✗ Canary issues, rolling back"
  kubectl rollout undo deployment/consumer
fi
```

#### Phase 2: Ramp (50% of instances)

```bash
# Gradually increase to 50%
kubectl set image deployment/consumer \
  consumer=myregistry/consumer:v2 \
  --record -n production

# Wait 2 hours and monitor
kubectl rollout pause deployment/consumer -n production

# Validation
kafka-consumer-groups.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --describe --group my-consumer-v2 | grep LAG

# Decision
if [ "error_rate < 0.1%" ] && [ "lag < 5s" ]; then
  echo "✓ Ramp healthy, proceeding to 100%"
else
  echo "✗ Issues detected, rolling back"
  kubectl rollout undo deployment/consumer
fi
```

#### Phase 3: Complete (100% of instances)

```bash
# Deploy to all instances
kubectl rollout resume deployment/consumer -n production

# Wait for deployment to complete
kubectl rollout status deployment/consumer -n production

# Final validation
kubectl get pods -n production | grep consumer
# Expected: All pods running and ready
```

### Production Monitoring During Migration

```bash
# Monitor consumer lag (continuous)
watch -n 5 'kafka-consumer-groups.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --describe --group my-consumer-v2 | head -20'

# Monitor error rate
curl -s $PROMETHEUS_URL/api/v1/query \
  --data-urlencode 'query=rate(consumer_errors_total[5m])'

# Monitor latency
curl -s $PROMETHEUS_URL/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.99, consumer_latency_bucket)'

# Check logs for errors
tail -f deployment.log | grep -i error
```

---

## Step 4: Decommission Old Consumer

**Timeline**: After Week 3 migration complete + 1 week of monitoring

Once all exchanges are migrated and monitoring shows stable metrics:

```bash
# 1. Verify new consumer is healthy
# - Lag < 5 seconds
# - Error rate < 0.1%
# - Data completeness verified

# 2. Stop old consumer
kubectl delete deployment/consumer-v1 -n production

# 3. Delete old consumer group (if not needed for history)
kafka-consumer-groups.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --delete --group my-consumer

# 4. Archive code/documentation
# - Move v1 consumer to legacy/
# - Update runbooks to reference v2
# - Add migration notes to codebase

# 5. Update documentation
# - Remove legacy consumer references
# - Update deployment guides
# - Archive old topic subscriptions

# 6. Communicate to team
# - Slack: "@all Old consumer decommissioned"
# - Update monitoring dashboards
# - Update runbooks
```

---

## Step 5: Rollback Plan

If issues arise at any point, execute this rollback procedure (<5 minutes):

### Immediate Rollback (Within 5 Minutes)

```bash
# 1. Pause production deployment (T+0min)
kubectl rollout pause deployment/consumer -n production

# 2. Rollback to previous version (T+1min)
kubectl rollout undo deployment/consumer -n production

# 3. Redeploy old image (T+2min)
kubectl rollout resume deployment/consumer -n production

# 4. Verify consumer restarted with old code (T+3min)
kubectl logs deployment/consumer -n production | grep "Consumer started"
kubectl get pods -n production | grep consumer

# 5. Monitor metrics recovery (T+4-5min)
kafka-consumer-groups.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --describe --group my-consumer

# Expected: Lag decreasing, no new errors
```

### Post-Rollback Analysis

```bash
1. Document what went wrong
   - Consumer errors in logs?
   - Protobuf deserialization failures?
   - Consumer group coordination issues?
   - Resource exhaustion?

2. Fix issue in staging
   - Apply code fix
   - Test thoroughly
   - Get approval

3. Reschedule migration for next day
   - Update Slack/email
   - Adjust timeframe
   - Notify stakeholders

4. Root cause analysis
   - Investigate logs and metrics
   - Identify code/config issue
   - Implement permanent fix
```

---

## Header-Based Filtering Examples

### Example 1: Filter by Single Exchange

```python
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'cryptofeed.trades',
    bootstrap_servers=['kafka1:9092'],
    group_id='my-consumer-coinbase-only',
)

for message in consumer:
    headers = dict(message.headers or [])
    exchange = headers.get(b'exchange', b'').decode()

    # Only process Coinbase
    if exchange == 'coinbase':
        process_message(message)
```

### Example 2: Route by Data Type

```python
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'cryptofeed.trades',
    'cryptofeed.orderbook',  # Multiple data types
    bootstrap_servers=['kafka1:9092'],
    group_id='my-consumer-all-types',
)

for message in consumer:
    headers = dict(message.headers or [])
    data_type = headers.get(b'data_type', b'unknown').decode()

    if data_type == 'trades':
        process_trade(message)
    elif data_type == 'orderbook':
        process_orderbook(message)
```

### Example 3: Cross-Exchange Arbitrage

```python
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'cryptofeed.trades',
    bootstrap_servers=['kafka1:9092'],
    group_id='my-consumer-arb',
)

# Aggregate trades by symbol across all exchanges
trades_by_symbol = {}

for message in consumer:
    headers = dict(message.headers or [])
    exchange = headers.get(b'exchange', b'').decode()
    symbol = headers.get(b'symbol', b'').decode()

    key = f"{symbol}"  # Group by symbol only
    if key not in trades_by_symbol:
        trades_by_symbol[key] = []

    trades_by_symbol[key].append({
        'exchange': exchange,
        'price': message.value.price,  # After deserialization
    })

    # Check for arbitrage opportunity
    if len(set(t['exchange'] for t in trades_by_symbol[key])) > 1:
        prices = sorted([t['price'] for t in trades_by_symbol[key]])
        spread = prices[-1] - prices[0]
        if spread > ARBITRAGE_THRESHOLD:
            execute_arb_trade(trades_by_symbol[key])
```

---

## Message Header Reference

### Standard Headers (All Messages)

| Header | Type | Example | Usage |
|--------|------|---------|-------|
| `exchange` | bytes | `b"coinbase"` | Routing, filtering |
| `symbol` | bytes | `b"BTC-USD"` | Routing, filtering |
| `data_type` | bytes | `b"trades"` | Type discrimination |
| `schema_version` | bytes | `b"v1"` | Deserialization |

### Using Headers

```python
# Extract headers from message
headers = dict(message.headers or [])

# Decode bytes to strings
exchange = headers.get(b'exchange', b'unknown').decode()
symbol = headers.get(b'symbol', b'unknown').decode()

# Handling missing headers
data_type = headers.get(b'data_type', b'unknown').decode()
schema_version = headers.get(b'schema_version', b'v1').decode()
```

---

## Success Metrics

Track these metrics during migration to confirm success:

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Consumer Lag** | <5 seconds | `kafka-consumer-groups.sh --describe` |
| **Error Rate** | <0.1% | `rate(consumer_errors_total[5m])` |
| **Latency (p99)** | <5ms | `histogram_quantile(0.99, consumer_latency_bucket)` |
| **Message Count** | ±0.1% vs legacy | `SELECT COUNT(*) FROM ...` |
| **Data Integrity** | 100% | Hash validation of 100 messages |
| **Duplicates** | 0 | Unique ID validation |

---

## Troubleshooting

### Consumer Lag Exceeds 5 Seconds

**Symptoms**:
- Lag metric > 5000 messages
- Lag not decreasing over time

**Diagnosis**:
```bash
# Check consumer group status
kafka-consumer-groups.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --describe --group my-consumer-v2

# Check broker status
kafka-broker-api-versions.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS

# Check consumer resources
kubectl top pods -l app=consumer
# CPU: Should be <80%
# Memory: Should be <80%
```

**Solutions**:
1. **Low resources**: Scale up consumer (more replicas, more CPU/memory)
2. **Slow deserialization**: Profile and optimize protobuf parsing
3. **Broker issues**: Check broker load, disk space, network
4. **Consumer coordination**: Increase `max.poll.records`, tune timeouts

### Deserialization Errors

**Symptoms**:
- Logs show `ParseFromString()` errors
- Consumer lag increasing
- Error messages in logs

**Diagnosis**:
```bash
# Check error logs
kubectl logs deployment/consumer -n production | grep -i error

# Check message format
kafka-console-consumer.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --topic cryptofeed.trades --from-beginning --max-messages 1 | xxd
# Should look like binary protobuf (not JSON)
```

**Solutions**:
1. **Wrong schema**: Ensure protobuf schema matches producer
2. **Message corruption**: Check producer logs for serialization errors
3. **Old messages**: Check if offset is pointing to old JSON messages
4. **Schema version mismatch**: Use correct schema version from headers

### Consumer Group Rebalancing

**Symptoms**:
- Frequent "rebalancing" log messages
- Lag spikes every few minutes
- High latency during rebalancing

**Diagnosis**:
```bash
# Check consumer group status
kafka-consumer-groups.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS \
  --describe --group my-consumer-v2

# Check for network issues
ping $KAFKA_BOOTSTRAP_SERVERS[0]
```

**Solutions**:
1. **Heartbeat timeout**: Increase `session.timeout.ms` (30000 default)
2. **Poll frequency**: Increase `max.poll.interval.ms` if processing slow
3. **Network issues**: Fix network connectivity, reduce latency
4. **Resource exhaustion**: Scale up consumer, reduce batch size

---

## Next Steps

1. **Week 2 (You are here)**:
   - Review this guide
   - Prepare consumer code changes
   - Test in staging environment

2. **Week 3**:
   - Execute per-exchange migration (Coinbase → Binance → Others)
   - Validate each exchange before proceeding
   - Monitor lag and error rates continuously

3. **Week 4**:
   - Stabilization period (monitor 72 hours)
   - Legacy topic cleanup and archival
   - Post-migration validation and reporting

---

## Support

**Questions?** Slack: `#data-engineering`
**Issues?** Create Jira ticket: `KIRO-` prefix
**Emergency**: Page L1 SRE on-call (PagerDuty)

---

**Version**: 1.0.0
**Last Updated**: November 13, 2025
**Status**: Ready for Production
