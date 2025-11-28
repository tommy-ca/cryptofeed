# Kafka Backend Troubleshooting Guide

**Comprehensive troubleshooting reference for Kafka backend migration and operation.**

---

## Table of Contents

1. [Common Migration Issues](#common-migration-issues)
2. [Configuration Errors](#configuration-errors)
3. [Connectivity Problems](#connectivity-problems)
4. [Performance Issues](#performance-issues)
5. [Deprecation Warnings](#deprecation-warnings)
6. [Health Check Failures](#health-check-failures)
7. [Production Incident Response](#production-incident-response)

---

## Common Migration Issues

### Issue 1: Missing Bootstrap Servers

**Symptom:**
```
ValueError: legacy_config missing required key 'bootstrap_servers'
```

**Cause:** Configuration doesn't specify Kafka broker addresses.

**Solution:**
```python
# ❌ WRONG
legacy_config = {
    "topic_prefix": "trades"
}

# ✅ CORRECT
legacy_config = {
    "bootstrap_servers": ["kafka:9092"],
    "topic_prefix": "trades"
}
```

**Prevention:** Always validate configuration has `bootstrap_servers` before translation.

---

### Issue 2: Invalid Topic Strategy

**Symptom:**
```
pydantic.ValidationError: Invalid topic strategy: invalid_strategy.
Must be 'consolidated' or 'per_symbol'
```

**Cause:** Topic strategy not in supported values.

**Solution:**
```python
# ❌ WRONG
from cryptofeed.backends.kafka.callback import KafkaTopicConfig

config = KafkaTopicConfig(strategy='invalid_strategy')

# ✅ CORRECT
config = KafkaTopicConfig(strategy='consolidated')  # or 'per_symbol'
```

**Supported Values:** `consolidated`, `per_symbol`

---

### Issue 3: Invalid Partition Strategy

**Symptom:**
```
pydantic.ValidationError: Invalid partition strategy: invalid_partitioner.
Must be one of: composite, exchange, round_robin, symbol
```

**Cause:** Partition strategy not in supported values.

**Solution:**
```python
# ❌ WRONG
from cryptofeed.backends.kafka.callback import KafkaPartitionConfig

config = KafkaPartitionConfig(strategy='invalid_partitioner')

# ✅ CORRECT
config = KafkaPartitionConfig(strategy='composite')
# Valid: composite, symbol, exchange, round_robin
```

**Decision Guide:**
- `composite`: Recommended (exchange-symbol pairs)
- `symbol`: Cross-exchange analysis
- `exchange`: Per-exchange monitoring
- `round_robin`: Maximum parallelism, no ordering

---

### Issue 4: Negative Partitions Per Topic

**Symptom:**
```
pydantic.ValidationError: partitions_per_topic must be > 0
```

**Cause:** Invalid partition count.

**Solution:**
```python
# ❌ WRONG
from cryptofeed.backends.kafka.callback import KafkaTopicConfig

config = KafkaTopicConfig(partitions_per_topic=-1)

# ✅ CORRECT
config = KafkaTopicConfig(partitions_per_topic=3)  # Must be > 0
```

**Recommendations:**
- Minimum: 3 (for redundancy)
- Recommended: 12 (balanced parallelism)
- High throughput: 24-48 (consumer group size × 2)

---

### Issue 5: Negative Replication Factor

**Symptom:**
```
pydantic.ValidationError: replication_factor must be > 0
```

**Cause:** Invalid replication factor.

**Solution:**
```python
# ❌ WRONG
from cryptofeed.backends.kafka.callback import KafkaTopicConfig

config = KafkaTopicConfig(replication_factor=0)

# ✅ CORRECT
config = KafkaTopicConfig(replication_factor=3)  # Must be > 0
```

**Recommendations:**
- Production: 3 (standard HA)
- Development: 1 (single broker)
- Critical data: 5 (increased durability)

**Constraint:** Replication factor ≤ broker count

---

### Issue 6: Unmapped Configuration Options

**Symptom:**
```
⚠️ Unmapped legacy options: custom_retry_logic, internal_buffer_size
```

**Cause:** Legacy config has custom options not in modern schema.

**Solution:**

**Step 1: Identify unmapped options**
```python
from cryptofeed.backends.kafka.migration import translate_legacy_config

legacy = {
    "bootstrap_servers": ["kafka:9092"],
    "custom_retry_logic": True,
    "internal_buffer_size": 10000,
}

result = translate_legacy_config(legacy)
print("Unmapped:", result.unmapped_options)
# {'custom_retry_logic': True, 'internal_buffer_size': 10000}
```

**Step 2: Manual migration strategies**

| Unmapped Option | Modern Equivalent | Action |
|-----------------|-------------------|--------|
| `custom_retry_logic` | `retries`, `retry_backoff_ms` | Map to modern retry config |
| `internal_buffer_size` | `batch_size`, `linger_ms` | Adjust batching params |
| `custom_partition_fn` | `partition.strategy` | Use built-in strategy or subclass |
| `value_serializer` | Use `KafkaProtobufCallback` | Switch to protobuf |
| Other | N/A | Review necessity, implement if needed |

**Step 3: Document custom behavior**
```python
# Document unmapped options for team review
for key, value in result.unmapped_options.items():
    print(f"Manual review required: {key} = {value}")
    # Add to migration notes, update documentation
```

**Prevention:** Review all legacy configuration options before migration.

---

## Configuration Errors

### Issue 7: Configuration Type Mismatch

**Symptom:**
```
TypeError: 'str' object is not iterable (bootstrap_servers)
```

**Cause:** `bootstrap_servers` must be a list, not string.

**Solution:**
```python
# ❌ WRONG
config = KafkaConfig(bootstrap_servers="kafka:9092")

# ✅ CORRECT
config = KafkaConfig(bootstrap_servers=["kafka:9092"])

# Multiple brokers
config = KafkaConfig(bootstrap_servers=["kafka1:9092", "kafka2:9092"])
```

---

### Issue 8: Invalid Acks Value

**Symptom:**
```
pydantic.ValidationError: acks must be '0', '1', or 'all'
```

**Cause:** Invalid acknowledgment level.

**Solution:**
```python
# ❌ WRONG
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    acks=2  # Invalid
)

# ✅ CORRECT
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    acks="all"  # Valid: '0', '1', 'all'
)
```

**Acks Semantics:**
- `'0'`: Fire-and-forget (fastest, least reliable)
- `'1'`: Leader acknowledgment (balanced)
- `'all'`: All replicas acknowledgment (slowest, most reliable)

**Recommendation:** Use `'all'` with `idempotence=True` for exactly-once.

---

### Issue 9: Invalid Compression Type

**Symptom:**
```
pydantic.ValidationError: compression_type must be one of:
none, gzip, snappy, lz4, zstd
```

**Cause:** Unsupported compression algorithm.

**Solution:**
```python
# ❌ WRONG
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    compression_type="bzip2"  # Unsupported
)

# ✅ CORRECT
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    compression_type="snappy"  # Valid
)
```

**Compression Comparison:**

| Algorithm | Speed | Ratio | CPU | Use Case |
|-----------|-------|-------|-----|----------|
| `none` | Fastest | 1.0x | Lowest | Low latency, high bandwidth |
| `snappy` | Fast | ~2x | Low | **Recommended** - balanced |
| `lz4` | Very fast | ~2x | Low | High throughput |
| `gzip` | Medium | ~3x | Medium | High compression, slower OK |
| `zstd` | Medium | ~3.5x | Medium | Best ratio, modern clusters |

**Recommendation:** `snappy` (default) for balanced performance.

---

## Connectivity Problems

### Issue 10: Kafka Broker Unreachable

**Symptom:**
```
KafkaConnectionError: Unable to bootstrap from [('kafka', 9092)]
```

**Cause:** Broker address incorrect or broker down.

**Solution:**

**Step 1: Verify broker address**
```bash
# Test connectivity
telnet kafka 9092
# Or
nc -zv kafka 9092

# Check DNS resolution
nslookup kafka
```

**Step 2: Check broker status**
```bash
# Kafka server logs
docker logs kafka-broker

# Broker health
kafka-broker-api-versions.sh --bootstrap-server kafka:9092
```

**Step 3: Update configuration**
```python
# Use correct broker addresses
config = KafkaConfig(
    bootstrap_servers=["correct-kafka-host:9092"]
)

# Multiple brokers for failover
config = KafkaConfig(
    bootstrap_servers=[
        "kafka-1:9092",
        "kafka-2:9092",
        "kafka-3:9092"
    ]
)
```

**Prevention:** Always test connectivity with health checks before deployment.

### Issue 10a: Redpanda Test Cluster Port In Use

**Symptom:**
```
Error response from daemon: ... Bind for 0.0.0.0:19092 failed: port is already allocated
```

**Cause:** Another process or container is already listening on the host port used by the Redpanda test cluster (default 19092).

**Solution:**

**Step 1: Check which process is using the port**
```bash
# On Linux
sudo lsof -i :19092 || sudo ss -lntp | grep 19092
```

**Step 2: Either stop the conflicting process or run Redpanda on a different host port**
```bash
# Example: run tests on host port 29092
export REDPANDA_HOST_PORT=29092
export REDPANDA_HOST_BOOTSTRAP=localhost:29092
export REDPANDA_COMPOSE_FILE=docker/infra/base.yml

docker compose -f docker/infra/base.yml up -d
```

**Step 3: Run Kafka integration tests**
```bash
REDPANDA_HOST_PORT=29092 \
REDPANDA_HOST_BOOTSTRAP=localhost:29092 \
REDPANDA_COMPOSE_FILE=docker/infra/base.yml \
python -m pytest tests/integration/kafka/test_kafka_protobuf_e2e.py -q
```

**Prevention:** Use the `REDPANDA_HOST_PORT` and `REDPANDA_HOST_BOOTSTRAP` environment variables to avoid host port conflicts when running tests locally.

---

### Issue 11: Health Check Timeout

**Symptom:**
```
Health check failed: Request timed out (timeout_ms=3000)
```

**Cause:** Network latency or broker overload.

**Solution:**

**Step 1: Increase timeout**
```python
from cryptofeed.backends.kafka.health import KafkaHealthCheck

# Default timeout: 3000ms
status = KafkaHealthCheck.check_modern(
    config,
    timeout_ms=10000  # Increase to 10 seconds
)
```

**Step 2: Check network latency**
```bash
# Ping broker
ping kafka

# Check latency
time nc -zv kafka 9092
```

**Step 3: Monitor broker load**
```bash
# Kafka metrics
kafka-run-class.sh kafka.tools.JmxTool \
  --object-name kafka.server:type=BrokerTopicMetrics,name=MessagesInPerSec
```

**Prevention:** Set appropriate timeout based on network characteristics.

---

### Issue 12: Authentication Failure

**Symptom:**
```
KafkaAuthorizationFailedError: Not authorized to access topics
```

**Cause:** Missing or incorrect credentials.

**Solution:**

**Step 1: Add SASL configuration**
```python
from cryptofeed.backends.kafka.callback import KafkaConfig

config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    # Add SASL configuration (if using confluent-kafka)
    # Note: Current implementation may need extension for SASL
)

# For aiokafka (legacy), use:
legacy_config = {
    'bootstrap_servers': ['kafka:9092'],
    'security_protocol': 'SASL_SSL',
    'sasl_mechanism': 'PLAIN',
    'sasl_plain_username': 'user',
    'sasl_plain_password': 'password',
}
```

**Step 2: Verify credentials**
```bash
# Test with console consumer
kafka-console-consumer.sh \
  --bootstrap-server kafka:9092 \
  --topic test-topic \
  --consumer.config client.properties
```

**Prevention:** Store credentials securely (environment variables, secrets manager).

---

## Performance Issues

### Issue 13: High Producer Latency

**Symptom:** Message production takes >100ms consistently.

**Cause:** Suboptimal batching/compression configuration.

**Solution:**

**Step 1: Tune batching parameters**
```python
# ❌ SLOW (default safe values)
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    batch_size=16384,   # 16KB (small)
    linger_ms=10        # 10ms (short wait)
)

# ✅ OPTIMIZED (higher throughput)
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    batch_size=131072,  # 128KB (larger batches)
    linger_ms=50        # 50ms (more batching)
)
```

**Step 2: Optimize compression**
```python
# Fast compression
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    compression_type="lz4"  # Faster than snappy
)
```

**Step 3: Monitor metrics**
```python
# Enable metrics collection
from cryptofeed.backends.kafka.metrics import PrometheusMetricsExporter

metrics = PrometheusMetricsExporter(port=9090)
# Monitor: kafka_producer_latency_seconds
```

**Performance Tuning Guide:**

| Latency Target | batch_size | linger_ms | compression | acks |
|----------------|-----------|-----------|-------------|------|
| <10ms (low latency) | 8192 | 0 | none/lz4 | 1 |
| <50ms (balanced) | 16384 | 10 | snappy | all |
| <200ms (high throughput) | 131072 | 100 | snappy/zstd | all |

---

### Issue 14: Consumer Lag Accumulation

**Symptom:** Consumer lag increasing despite active consumers.

**Cause:** Producer throughput exceeds consumer capacity.

**Solution:**

**Step 1: Check topic partitions**
```bash
# List topic partitions
kafka-topics.sh --describe \
  --bootstrap-server kafka:9092 \
  --topic cryptofeed.trade
```

**Step 2: Increase partition count**
```python
# More partitions = more parallelism
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    topic=KafkaTopicConfig(
        partitions_per_topic=24  # Increase from 3
    )
)
```

**Step 3: Optimize partition strategy**
```python
# Use round-robin for maximum consumer parallelism
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    partition=KafkaPartitionConfig(strategy="round_robin")
)
```

**Consumer Scaling Guide:**
- Partitions = Consumer group max parallelism
- Add consumers until consumer_count = partition_count
- Further consumers idle

---

### Issue 15: Message Loss

**Symptom:** Messages not appearing in Kafka topics.

**Cause:** Low `acks` level or broker failures.

**Solution:**

**Step 1: Enable exactly-once semantics**
```python
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    acks="all",          # Wait for all replicas
    idempotence=True,    # Enable idempotence
    retries=3            # Retry on failure
)
```

**Step 2: Increase replication**
```python
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    topic=KafkaTopicConfig(
        replication_factor=3  # At least 3 for HA
    )
)
```

**Step 3: Monitor delivery**
```python
# Check producer metrics
# Monitor: kafka_producer_errors_total
```

**Prevention:** Always use `acks="all"` + `idempotence=True` for production.

---

## Deprecation Warnings

### Issue 16: Legacy Import Warnings

**Symptom:**
```
DeprecationWarning: Import path 'cryptofeed.kafka_callback' is deprecated.
Import from cryptofeed.backends.kafka.callback instead.
```

**Cause:** Using deprecated import path.

**Solution:**
```python
# ❌ OLD
from cryptofeed.kafka_callback import KafkaCallback

# ✅ NEW
from cryptofeed.backends.kafka.callback import KafkaCallback
```

**Suppression (temporary only):**
```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# NOT RECOMMENDED - Fix the imports instead
```

---

### Issue 17: Legacy Class Warnings

**Symptom:**
```
DeprecationWarning: TradeKafka is deprecated and will be removed.
Use KafkaCallback instead.
```

**Cause:** Using deprecated legacy backend classes.

**Solution:**
```python
# ❌ OLD
from cryptofeed.backends.kafka import TradeKafka

callback = TradeKafka(bootstrap_servers=['kafka:9092'])

# ✅ NEW
from cryptofeed.backends.kafka.callback import KafkaCallback, KafkaConfig

config = KafkaConfig(bootstrap_servers=['kafka:9092'])
callback = KafkaCallback(config=config, key=None)
```

---

### Issue 18: Suppressing Warnings in Tests

**Symptom:** Test suite flooded with deprecation warnings.

**Solution:**

**Option 1: pytest configuration**
```ini
# pytest.ini
[pytest]
filterwarnings =
    ignore::DeprecationWarning:cryptofeed.backends.kafka
```

**Option 2: Per-test suppression**
```python
import pytest

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_legacy_feature():
    from cryptofeed.backends.kafka import TradeKafka
    # Test legacy behavior without warnings
```

**Option 3: Context manager**
```python
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from cryptofeed.kafka_callback import KafkaCallback
```

**Recommendation:** Fix the imports rather than suppressing warnings.

---

## Health Check Failures

### Issue 19: Health Check Returns False Positive

**Symptom:** Health check passes but messages not producing.

**Cause:** Health check only validates connectivity, not full pipeline.

**Solution:**

**Step 1: End-to-end test**
```python
from cryptofeed.backends.kafka.callback import KafkaCallback, KafkaConfig
from cryptofeed.backends.kafka.health import KafkaHealthCheck

# 1. Health check
config = KafkaConfig(bootstrap_servers=['kafka:9092'])
status = KafkaHealthCheck.check_modern(config)

if not status.ok:
    raise ConnectionError(f"Health check failed: {status.error}")

# 2. Create callback
callback = KafkaCallback(config=config, key=None)

# 3. Test message production (requires async context)
# See integration tests for examples
```

**Step 2: Monitor actual topics**
```bash
# Watch for messages
kafka-console-consumer.sh \
  --bootstrap-server kafka:9092 \
  --topic cryptofeed.trade \
  --from-beginning
```

---

### Issue 20: Periodic Health Checks Failing Intermittently

**Symptom:** Health checks fail occasionally with timeouts.

**Cause:** Network congestion or broker load spikes.

**Solution:**

**Step 1: Increase check interval**
```python
from cryptofeed.backends.kafka.health import start_periodic_health_checks

# ❌ TOO FREQUENT (overloads broker)
await start_periodic_health_checks(
    interval_sec=5,  # Every 5 seconds
    check_fn=lambda: KafkaHealthCheck.check_modern(config)
)

# ✅ REASONABLE INTERVAL
await start_periodic_health_checks(
    interval_sec=60,  # Every 60 seconds
    check_fn=lambda: KafkaHealthCheck.check_modern(config)
)
```

**Step 2: Increase timeout**
```python
def check_fn():
    return KafkaHealthCheck.check_modern(
        config,
        timeout_ms=10000  # 10 seconds
    )

await start_periodic_health_checks(
    interval_sec=60,
    check_fn=check_fn,
    alert_threshold_ms=5000.0  # Alert if >5s
)
```

---

## Production Incident Response

### Incident 1: Total Kafka Outage

**Detection:** All health checks failing, no messages producing.

**Response:**

**Step 1: Immediate assessment**
```bash
# Check broker status
systemctl status kafka

# Check broker logs
journalctl -u kafka -f

# Check disk space
df -h /var/lib/kafka
```

**Step 2: Rollback if Kafka unavailable**
```python
# Temporarily disable Kafka callbacks
# Remove Kafka from FeedHandler callbacks
# OR switch to alternative backend (Redis, Arctic)
```

**Step 3: Escalate**
- Page Kafka infrastructure team
- Update status page
- Communicate ETA to stakeholders

**Step 4: Recovery validation**
```python
# Test connectivity
status = KafkaHealthCheck.check_modern(config)
assert status.ok, f"Still unhealthy: {status.error}"

# Test message production
# Run smoke tests
```

---

### Incident 2: Message Duplication

**Detection:** Consumers seeing duplicate messages.

**Response:**

**Step 1: Verify idempotence**
```python
# Check configuration
config = KafkaConfig(bootstrap_servers=['kafka:9092'])
assert config.idempotence is True, "Idempotence not enabled!"
```

**Step 2: Check producer state**
```python
# Producer may be recreated, losing transactional state
# Ensure producer lifecycle management
```

**Step 3: Consumer deduplication**
```python
# Implement consumer-side deduplication
# Use message headers (timestamp, sequence) as dedup key
```

**Prevention:** Always enable `idempotence=True` in production.

---

### Incident 3: High Consumer Lag

**Detection:** Consumer lag >10,000 messages.

**Response:**

**Step 1: Scale consumers**
```bash
# Add consumer instances (up to partition count)
# Ensure consumer group has capacity
```

**Step 2: Optimize partitioning**
```python
# Increase partition count (requires topic recreation or partition addition)
config = KafkaConfig(
    bootstrap_servers=['kafka:9092'],
    topic=KafkaTopicConfig(partitions_per_topic=24)
)
```

**Step 3: Optimize consumer**
```python
# Consumer-side optimizations
# - Batch processing
# - Async processing
# - Message filtering
```

**Step 4: Backpressure**
```python
# Reduce producer rate temporarily
config = KafkaConfig(
    bootstrap_servers=['kafka:9092'],
    linger_ms=100,  # Batch more aggressively
    batch_size=131072  # Larger batches
)
```

---

## Summary Checklist

### Pre-Migration
- [ ] Read migration guide completely
- [ ] Test translation in staging
- [ ] Validate configuration equivalence
- [ ] Run health checks
- [ ] Document unmapped options
- [ ] Plan rollback procedures

### During Migration
- [ ] Monitor deprecation warnings
- [ ] Track usage statistics
- [ ] Run parallel legacy/modern
- [ ] Validate message equivalence
- [ ] Monitor performance metrics
- [ ] Have rollback plan ready

### Post-Migration
- [ ] Verify zero legacy usage
- [ ] Decommission legacy topics
- [ ] Update documentation
- [ ] Archive legacy configurations
- [ ] Remove legacy code
- [ ] Update runbooks

---

## Support Resources

**Documentation:**
- Migration Guide: `docs/kafka/MIGRATION_GUIDE.md`
- API Reference: `docs/kafka/API_REFERENCE.md`
- Best Practices: `docs/kafka/BEST_PRACTICES.md`

**Code:**
- Migration module: `cryptofeed/backends/kafka/migration.py`
- Health checks: `cryptofeed/backends/kafka/health.py`
- Maintenance: `cryptofeed/backends/kafka/maintenance.py`

**Tests:**
- Documentation examples: `tests/unit/kafka/test_documentation_examples.py`
- Integration tests: `tests/integration/kafka/`
- Regression tests: `tests/unit/kafka/test_regression_stability.py`

**Community:**
- GitHub Issues: https://github.com/bmoscon/cryptofeed/issues
- Documentation: https://docs.cryptofeed.ai

---

**Troubleshooting Guide Version:** 1.0.0
**Last Updated:** 2025-11-26
**Status:** Production Ready
