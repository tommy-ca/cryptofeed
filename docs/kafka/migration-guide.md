# Kafka Migration Guide: Legacy → Phase 2

## Executive Summary

This guide helps operators migrate from the legacy `cryptofeed.backends.kafka` implementation to the new Phase 2 `KafkaCallback` producer. The Phase 2 implementation provides:

- **Consolidated Topics**: Reduce topic count from O(symbols × exchanges) to O(data_types) ~20 topics
- **Flexible Partitioning**: 4 configurable partition strategies (composite, symbol, exchange, round-robin)
- **Message Headers**: Structured routing metadata (exchange, symbol, data_type, schema_version)
- **Exactly-Once Semantics**: Idempotent producer with broker-level deduplication
- **Production Monitoring**: Prometheus metrics, Grafana dashboards, alerting rules
- **Better Error Handling**: Comprehensive exception boundaries, no silent failures

**Migration Status**: Required. Legacy backend deprecated as of Nov 10, 2025. Will be removed in v0.2.0.

**Estimated Timeline**: 1-3 days for production migration (including dual-write validation).

---

## Architecture Comparison

### Legacy Implementation (cryptofeed.backends.kafka)

```
Exchange Feed → TradeKafka/BookKafka → Per-Symbol Topics
                                        ├─ cryptofeed.trades.binance.btc-usd
                                        ├─ cryptofeed.trades.binance.eth-usd
                                        ├─ cryptofeed.trades.coinbase.btc-usd
                                        └─ ... (thousands of topics)

Characteristics:
- Per-symbol topic naming: cryptofeed.{type}.{exchange}.{symbol}
- Fixed round-robin partitioning (no ordering guarantees)
- No message headers (metadata lost)
- Basic error handling (may silently drop messages)
- No monitoring/metrics
```

### Phase 2 Implementation (KafkaCallback)

```
Exchange Feed → KafkaCallback → Consolidated Topics
                               ├─ cryptofeed.trades (exchange, symbol in partition key)
                               ├─ cryptofeed.orderbook
                               ├─ cryptofeed.ticker
                               └─ ... (~20 topics)

Characteristics:
- Consolidated topic naming: cryptofeed.{data_type}
- 4 configurable partition strategies (composite, symbol, exchange, round-robin)
- Message headers: content-type, exchange, symbol, data_type, schema_version
- Comprehensive error handling (no silent failures)
- Built-in Prometheus metrics (messages sent, latency, errors)
- Optional per-symbol topic mode for backward compatibility
```

### Key Differences

| Feature | Legacy | Phase 2 |
|---------|--------|---------|
| Topic Count | O(symbols × exchanges) | O(data_types) |
| Default Partitioning | Round-robin | Composite (exchange-symbol) |
| Ordering Guarantees | None | Per-symbol (configurable) |
| Message Headers | No | Yes (mandatory + optional) |
| Exactly-Once | No | Yes (idempotent producer) |
| Error Handling | Basic | Comprehensive |
| Monitoring | None | Prometheus + Grafana |
| Config Format | Simple | Structured (Pydantic models) |

---

## Migration Strategies

### Strategy A: Big Bang Cutover (Fast, Risky)

**Timeline**: 1-2 hours
**Risk**: High (immediate customer impact if issues arise)
**Recovery**: Rollback to legacy implementation

#### Steps:
1. Configure Phase 2 KafkaCallback in parallel config file
2. Deploy with feature flag (disable legacy, enable Phase 2)
3. Validate data in new consolidated topics
4. Switch consumers to new topics (requires changes)
5. Remove legacy configuration after 1-2 weeks of success

#### Pros:
- Fastest migration
- Simplest deployment (one switch)
- Immediate benefit from new features

#### Cons:
- Highest risk (no validation period)
- Requires coordinated consumer updates
- No fallback if issues arise during deployment

---

### Strategy B: Dual-Write (Safe, Complex)

**Timeline**: 3-5 days
**Risk**: Low (can validate before cutover)
**Recovery**: Switch off Phase 2 and revert to legacy

#### Steps:

1. **Deploy Phase 2 alongside legacy** (Week 1)
   - Run both KafkaCallback and TradeKafka in parallel
   - Produce to both consolidated and per-symbol topics
   - No consumer changes required yet

2. **Validate data quality** (Week 1)
   - Compare message counts between legacy and Phase 2
   - Verify protobuf serialization (if enabled)
   - Check message headers are present
   - Validate partition key distribution

3. **Update consumers gradually** (Week 2)
   - Start with non-critical consumers
   - Switch to consolidated topics one by one
   - Monitor each consumer for data gaps

4. **Disable legacy backend** (Week 2)
   - Stop writing to per-symbol topics
   - Keep per-symbol topics available for replay (30 days)
   - Archive legacy topics after verification period

#### Pros:
- Low risk (full validation before cutover)
- Time to update all consumers gradually
- Easy rollback (just disable Phase 2)
- Can detect subtle issues before customer impact

#### Cons:
- More complex deployment
- Temporary storage overhead (dual topics)
- Longer timeline (3-5 days)
- Requires data validation tooling

---

### Strategy C: Gradual Feed Migration (Safest, Slowest)

**Timeline**: 1-4 weeks
**Risk**: Very Low (validate each feed independently)
**Recovery**: Revert individual feeds to legacy

#### Steps:

1. **Start with low-traffic feeds** (Week 1)
   - Pick 2-3 feeds (e.g., secondary exchanges)
   - Enable Phase 2 KafkaCallback for these feeds only
   - Leave other feeds on legacy backend
   - Validate data quality for 1 week

2. **Move medium-traffic feeds** (Week 2)
   - Migrate feeds handling 10-50% of message volume
   - Monitor for latency, errors, dropped messages
   - Validate consumer processing

3. **Move high-traffic feeds** (Week 3-4)
   - Migrate main feeds (binance, coinbase, kraken)
   - Full capacity testing under production load
   - Monitor metrics closely (latency p99, error rate)

4. **Complete migration** (Week 4)
   - All feeds on Phase 2
   - Disable legacy backend
   - Archive per-symbol topics

#### Pros:
- Very low risk (isolated validation)
- Can revert individual feeds if issues arise
- Time to understand Phase 2 behavior
- Easiest to debug problems
- Can validate at scale gradually

#### Cons:
- Longest timeline (1-4 weeks)
- Temporary mixed deployment complexity
- Consumer must handle topic migration
- Ongoing operational overhead

---

## Configuration Translation

### Mapping Legacy to Phase 2

```yaml
# LEGACY FORMAT
bootstrap_servers:
  - kafka1:9092
  - kafka2:9092
  - kafka3:9092
topic_prefix: production    # → topic.prefix
acks: '1'                    # → acks (preserved)
retries: 5                   # → retries
retry_backoff_ms: 100        # → retry_backoff_ms
batch_size: 32768           # → batch_size
linger_ms: 20               # → linger_ms
compression_type: snappy    # → compression_type
```

```yaml
# PHASE 2 FORMAT (auto-translated)
bootstrap_servers:
  - kafka1:9092
  - kafka2:9092
  - kafka3:9092

topic:
  strategy: per_symbol           # "consolidated" for modern deployments
  prefix: production             # from topic_prefix
  partitions_per_topic: 3        # auto-set (can override)
  replication_factor: 3          # auto-set (can override)

partition:
  strategy: composite            # auto-set (can override)

acks: '1'
idempotence: false             # set based on acks value
retries: 5
retry_backoff_ms: 100
batch_size: 32768
linger_ms: 20
compression_type: snappy
```

### Default Behavior

- **Topic Strategy**: Defaults to `per_symbol` for backward compatibility (generates old topic names)
- **Partition Strategy**: Defaults to `composite` (per-exchange-symbol ordering)
- **Idempotence**: Auto-enabled if `acks: 'all'`, disabled otherwise
- **New Fields**: `partition.strategy`, `topic.strategy`, `idempotence` (all have sensible defaults)

---

## Dual-Write Implementation

Run both legacy and Phase 2 producers simultaneously:

```python
from cryptofeed.backends.kafka import TradeKafka
from cryptofeed.kafka_callback import KafkaCallback, KafkaConfig

# Create legacy producer (deprecated)
legacy_kafka = TradeKafka(
    bootstrap_servers=['kafka:9092'],
    topic_prefix='cryptofeed',
    acks='1'
)

# Create Phase 2 producer
phase2_config = KafkaConfig.from_yaml('config/phase2_kafka.yaml')
phase2_kafka = KafkaCallback(kafka_config=phase2_config)

# Both receive the same feed data
feed.add_callback({
    TradeKafka: legacy_kafka,      # Legacy (deprecated)
    KafkaCallback: phase2_kafka,   # Phase 2 (new)
})
```

### Data Validation

```python
# Compare message counts
legacy_count = consumer.fetch_count('cryptofeed.trades.binance.btc-usd')
phase2_count = consumer.fetch_count('cryptofeed.trades')  # with symbol filter

assert legacy_count == phase2_count, "Message count mismatch!"

# Verify protobuf deserialization
for msg in phase2_consumer.fetch('cryptofeed.trades'):
    trade = Trade()
    trade.ParseFromString(msg.value)
    assert trade.exchange == msg.headers['exchange']
```

---

## Validation Procedures

### Pre-Cutover Validation Checklist

- [ ] **Topic Creation**: Verify consolidated topics created (cryptofeed.trades, etc.)
- [ ] **Message Count**: Compare legacy vs Phase 2 message counts match
- [ ] **Message Content**: Spot-check message deserialization
- [ ] **Headers**: Verify message headers present and correct
- [ ] **Latency**: Confirm latency p99 < 50ms (typical: 2-5ms)
- [ ] **Error Rate**: Check error rate < 1% (typical: <0.1%)
- [ ] **Partition Distribution**: Verify keys distribute across partitions
- [ ] **Memory Usage**: Monitor producer memory (typical: <100MB)
- [ ] **Kafka Broker Health**: Check broker load, network I/O

### Consumer Migration Validation

For each consumer:
- [ ] Subscribe to new consolidated topics
- [ ] Verify message offset progression (no gaps)
- [ ] Compare record counts from old and new sources
- [ ] Validate downstream data (aggregations, inserts, etc.)
- [ ] Monitor consumer lag (should be < 60 seconds)

### Monitoring During Migration

```yaml
# Key metrics to watch
critical:
  - messages_sent_total        # Should match legacy count
  - kafka_produce_errors_total # Should be ~0
  - produce_latency_p99        # Should be <50ms

warnings:
  - produce_queue_depth        # >10K messages indicates slow broker
  - produce_latency_p95        # >20ms indicates network/broker issues

info:
  - messages_by_exchange       # Distribution across exchanges
  - messages_by_data_type      # Distribution across data types
  - dlq_messages_total         # Messages in dead-letter queue
```

---

## Rollback Procedures

### Quick Rollback (< 5 minutes)

If issues detected, quickly disable Phase 2 and revert to legacy:

```python
# Change feed configuration
feed.remove_callback(KafkaCallback)    # Remove Phase 2
feed.add_callback(TradeKafka, legacy)  # Re-enable legacy

# Restart producer (in-process)
await producer.stop()
await legacy_producer.start()
```

### Data Recovery After Rollback

Legacy and Phase 2 topics are independent:

1. **Verify consumer offsets** in legacy topics
2. **Replay from last checkpoint** (if using offset tracking)
3. **Verify no data gaps** in downstream systems
4. **Check DLQ for error messages**

### Investigation Steps

If issues occurred during migration:

```bash
# Check for errors
kubectl logs kafka-producer | grep ERROR

# Verify broker health
kafkacat -b kafka:9092 -L

# Check consumer lag
kafka-consumer-groups.sh --bootstrap-server kafka:9092 \
  --describe --group cryptofeed-consumers

# Sample messages from both topics
kafkacat -b kafka:9092 -t cryptofeed.trades -c 10 | jq .
kafkacat -b kafka:9092 -t cryptofeed.trades.binance.btc-usd -c 10 | jq .
```

---

## Monitoring During Migration

### Prometheus Metrics Setup

Phase 2 exports Prometheus metrics on `http://producer:9090/metrics`:

```yaml
# Metrics to scrape
- job_name: 'kafka-producer'
  static_configs:
    - targets: ['localhost:9090']
  scrape_interval: 10s
  scrape_timeout: 5s
```

### Key Metrics

```
# Messages produced
kafka_messages_sent_total{exchange="binance", data_type="trades"}
kafka_messages_sent_total{exchange="coinbase", data_type="trades"}

# Latency percentiles
kafka_produce_latency_p50{exchange="binance"}
kafka_produce_latency_p95{exchange="binance"}
kafka_produce_latency_p99{exchange="binance"}

# Errors
kafka_produce_errors_total{exchange="binance", error_type="broker_unavailable"}

# Queue depth
kafka_producer_queue_depth{exchange="binance"}
```

### Grafana Dashboard

Import dashboard from `docs/kafka/grafana-dashboard.json` to visualize:
- Messages sent over time (by exchange, data type)
- Latency percentiles (p50, p95, p99)
- Error rate and error types
- Dead-letter queue depth
- Producer queue depth

### Alerting Rules

Deploy alert rules from `docs/kafka/alert-rules.yaml`:

```yaml
- name: kafka_migration_alerts
  rules:
    - alert: KafkaProducerHighLatency
      expr: kafka_produce_latency_p99 > 50  # ms
      for: 5m

    - alert: KafkaProducerHighErrorRate
      expr: rate(kafka_produce_errors_total[5m]) > 0.01
      for: 5m

    - alert: KafkaProducerQueueBacklog
      expr: kafka_producer_queue_depth > 10000
      for: 2m
```

---

## Timeline Recommendation

### For Production Deployments

**Strategy B (Dual-Write) Recommended**:

```
Day 1: Deploy Phase 2 alongside legacy (morning)
       Validate data quality (all day)

Day 2: Begin consumer migration (low-traffic consumers)
       Monitor metrics (all day)

Day 3: Continue consumer migration (medium-traffic consumers)
       Validate downstream data (all day)

Day 4: Final consumer migration (high-traffic consumers)
       Full validation (all day)

Day 5: Disable legacy backend
       Archive per-symbol topics
       Monitor for 1 week
```

### For Development/Test Deployments

**Strategy A (Big Bang) Acceptable**:

```
Just switch over to Phase 2
Test consumer changes locally
Deploy with minimal validation
```

---

## Common Issues and Troubleshooting

### Issue: "Topic strategy mismatch"

**Symptom**: Consumers can't find consolidated topics

**Solution**:
```yaml
topic:
  strategy: consolidated   # Must match consumer expectations
  prefix: cryptofeed
```

### Issue: "Message count mismatch"

**Symptom**: Phase 2 topics have fewer messages than legacy

**Cause**: Dual-write started before all feeds switched

**Solution**:
1. Verify all feeds produce to both backends
2. Wait for consumer lag to settle
3. Re-count messages

### Issue: "High latency after migration"

**Symptom**: p99 latency increases from 5ms to 20ms

**Cause**: Partition strategy creating uneven load

**Solution**:
```yaml
partition:
  strategy: round_robin    # Distribute load evenly
  # or
  strategy: exchange       # If exchange-based ordering acceptable
```

### Issue: "Dead-letter queue filling up"

**Symptom**: DLQ messages increasing

**Cause**: Broker unavailability or serialization errors

**Solution**:
1. Check broker health
2. Verify protobuf schema compatibility
3. Review error details in DLQ

---

## Support and Further Resources

- **Migration CLI Tool**: `python tools/migrate-kafka-config.py`
  - Translates legacy configs automatically
  - Validates Phase 2 configs
  - Provides dry-run preview

- **Configuration Examples**: See `docs/kafka/config-translation-examples.md`
  - 10 real-world examples (simple to production)
  - Performance tuning scenarios
  - High-availability setups

- **Operational Runbook**: See `docs/kafka/operational-runbook.md`
  - Incident response procedures
  - Topic management procedures
  - Consumer group management

- **API Documentation**: See `cryptofeed/kafka_callback.py`
  - KafkaCallback implementation
  - KafkaConfig model documentation
  - Partition strategy details

---

## Migration Checklist

### Pre-Migration (1-2 days before)
- [ ] Read this entire guide
- [ ] Review Phase 2 architecture
- [ ] Choose migration strategy (A, B, or C)
- [ ] Prepare rollback procedures
- [ ] Notify team of planned migration

### Migration Day
- [ ] Deploy Phase 2 configuration
- [ ] Start data validation
- [ ] Monitor metrics (latency, errors, count)
- [ ] Begin consumer migration (if strategy B/C)
- [ ] Document any issues

### Post-Migration (1-4 weeks)
- [ ] Monitor for 1-2 weeks
- [ ] Complete consumer migration
- [ ] Disable legacy backend
- [ ] Archive per-symbol topics (keep 30 days for replay)
- [ ] Update documentation
- [ ] Share lessons learned with team

---

## Conclusion

Phase 2 migration is a significant step toward scalable, observable Kafka operations. With proper planning and validation (Strategy B), the migration can be completed safely in 3-5 days with near-zero risk.

**Key Takeaway**: Use the migration CLI tool and dual-write strategy for production deployments. You can validate everything before committing to the new implementation.

For questions, refer to this guide, the configuration examples, or the operational runbook.
