# Kafka Backend Best Practices

**Production-ready patterns and recommendations for Kafka backend usage.**

---

## Table of Contents

1. [Configuration Best Practices](#configuration-best-practices)
2. [Topic Strategy Selection](#topic-strategy-selection)
3. [Partition Strategy Selection](#partition-strategy-selection)
4. [Performance Optimization](#performance-optimization)
5. [Reliability and Fault Tolerance](#reliability-and-fault-tolerance)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Security Best Practices](#security-best-practices)
8. [Migration Best Practices](#migration-best-practices)

---

## Configuration Best Practices

### ✅ Use Exactly-Once Semantics

**Recommendation:** Always enable idempotence for production workloads.

```python
from cryptofeed.backends.kafka.callback import KafkaConfig

config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    acks="all",           # ✅ Wait for all replicas
    idempotence=True,     # ✅ Enable exactly-once
    retries=3             # ✅ Retry on failure
)
```

**Why:**
- Prevents message duplication on network failures
- No performance penalty (producer caching)
- Essential for financial/trading data

**Anti-pattern:**
```python
# ❌ DON'T DO THIS (at-least-once, duplicates possible)
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    acks="1",             # ❌ Only leader
    idempotence=False     # ❌ No deduplication
)
```

---

### ✅ Use Structured Configuration

**Recommendation:** Store configuration in version-controlled YAML files.

```yaml
# config/production_kafka.yaml
bootstrap_servers:
  - kafka-1.prod.company.com:9092
  - kafka-2.prod.company.com:9092
  - kafka-3.prod.company.com:9092

topic:
  strategy: consolidated
  prefix: production
  partitions_per_topic: 12
  replication_factor: 3

partition:
  strategy: composite

acks: all
idempotence: true
compression_type: snappy
retries: 3
retry_backoff_ms: 100
batch_size: 16384
linger_ms: 10
```

```python
import yaml
from cryptofeed.backends.kafka.callback import KafkaConfig

with open('config/production_kafka.yaml') as f:
    config_dict = yaml.safe_load(f)

config = KafkaConfig(**config_dict)
```

**Why:**
- Version control tracks configuration changes
- Easy to review and audit
- Environment-specific configurations
- No hardcoded credentials

---

### ✅ Use Environment Variables for Secrets

**Recommendation:** Never commit credentials; use environment variables.

```python
import os
from cryptofeed.backends.kafka.callback import KafkaConfig

config = KafkaConfig(
    bootstrap_servers=os.getenv('KAFKA_BROKERS', 'localhost:9092').split(','),
    # Add SASL/SSL configuration from environment
    # (Note: Current implementation may need extension)
)
```

**Environment:**
```bash
export KAFKA_BROKERS="kafka-1:9092,kafka-2:9092,kafka-3:9092"
export KAFKA_USERNAME="cryptofeed-producer"
export KAFKA_PASSWORD="***"
```

**Why:**
- Secrets not in version control
- Different credentials per environment
- Follows 12-factor app principles

---

## Topic Strategy Selection

### ✅ Use Consolidated Topics (Modern Default)

**Recommendation:** Prefer consolidated topics unless you have specific per-symbol requirements.

```python
from cryptofeed.backends.kafka.callback import KafkaConfig, KafkaTopicConfig

config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    topic=KafkaTopicConfig(
        strategy="consolidated",  # ✅ RECOMMENDED
        prefix="cryptofeed"
    )
)
```

**Topic Count:**
- **Consolidated:** ~10-20 topics (trade, book, ticker, funding, ...)
- **Per-symbol:** ~1,000-10,000 topics (BTC-USD, ETH-USD, ...)

**Benefits:**
- Fewer topics to manage
- Simpler consumer implementation
- Better Kafka cluster performance
- Easier to add/remove symbols

**When to use per-symbol:**
- Per-symbol retention policies (e.g., keep BTC data longer)
- Symbol-specific access control
- Legacy system compatibility

---

### ✅ Choose Appropriate Partition Count

**Recommendation:** Match partitions to expected consumer parallelism.

```python
# Production (high throughput)
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    topic=KafkaTopicConfig(
        strategy="consolidated",
        partitions_per_topic=12  # ✅ Sufficient parallelism
    )
)

# Development (minimal resources)
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    topic=KafkaTopicConfig(
        strategy="consolidated",
        partitions_per_topic=3   # ✅ Minimal for HA
    )
)
```

**Decision Matrix:**

| Throughput | Consumers | Partitions | Notes |
|------------|-----------|------------|-------|
| Low (<1k msg/s) | 1-3 | 3 | Minimal overhead |
| Medium (<10k msg/s) | 3-6 | 6-12 | Balanced |
| High (<100k msg/s) | 6-12 | 12-24 | High parallelism |
| Very High (>100k msg/s) | 12-24 | 24-48 | Maximum throughput |

**Formula:**
```
partitions = consumer_group_size × 2  (for growth headroom)
```

---

## Partition Strategy Selection

### ✅ Use Composite Strategy (Default)

**Recommendation:** Composite provides best balance for most use cases.

```python
from cryptofeed.backends.kafka.callback import KafkaConfig, KafkaPartitionConfig

config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    partition=KafkaPartitionConfig(strategy="composite")  # ✅ RECOMMENDED
)
```

**Partition Key:** `{exchange}-{symbol}` (e.g., `binance-BTC-USDT`)

**Benefits:**
- Messages for same exchange-symbol are ordered
- Good load distribution across partitions
- Supports parallel processing per exchange-symbol pair

---

### ✅ Match Strategy to Use Case

**Decision Matrix:**

| Use Case | Strategy | Partition Key | Ordering |
|----------|----------|---------------|----------|
| **General (recommended)** | `composite` | `exchange-symbol` | Per pair |
| **Cross-exchange arbitrage** | `symbol` | `symbol` | Per symbol (all exchanges) |
| **Exchange-specific monitoring** | `exchange` | `exchange` | Per exchange (all symbols) |
| **Maximum throughput, no ordering** | `round_robin` | None | No guarantees |

**Example: Cross-exchange arbitrage**

```python
# Analyze BTC-USDT across all exchanges
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    partition=KafkaPartitionConfig(strategy="symbol")  # All BTC-USDT to same partition
)
```

**Example: Exchange-specific processing**

```python
# Monitor Binance separately from other exchanges
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    partition=KafkaPartitionConfig(strategy="exchange")  # All Binance to same partition
)
```

---

## Performance Optimization

### ✅ Tune Batching for Latency vs Throughput

**Recommendation:** Balance latency and throughput based on requirements.

**Low Latency (<10ms):**
```python
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    batch_size=8192,      # ✅ Small batches
    linger_ms=0,          # ✅ Send immediately
    compression_type="lz4"  # ✅ Fast compression
)
```

**High Throughput (latency <200ms acceptable):**
```python
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    batch_size=131072,    # ✅ Large batches (128KB)
    linger_ms=100,        # ✅ Wait for more messages
    compression_type="snappy"  # ✅ Good compression + speed
)
```

**Balanced (recommended):**
```python
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    batch_size=16384,     # ✅ Default (16KB)
    linger_ms=10,         # ✅ Short wait
    compression_type="snappy"  # ✅ Balanced
)
```

---

### ✅ Choose Appropriate Compression

**Recommendation:** Use `snappy` for balanced performance.

**Compression Benchmark:**

| Algorithm | Latency | Compression Ratio | CPU Usage | Network Savings |
|-----------|---------|-------------------|-----------|-----------------|
| `none` | **Fastest** | 1.0x | Lowest | 0% |
| `lz4` | Very fast | ~2.0x | Low | ~50% |
| `snappy` | **Fast** | ~2.0x | **Low** | **~50%** ✅ |
| `gzip` | Medium | ~3.0x | Medium | ~67% |
| `zstd` | Medium | ~3.5x | Medium | ~71% |

**Decision Guide:**
- **Latency-critical:** `lz4` or `none`
- **Balanced (recommended):** `snappy` ✅
- **Bandwidth-limited:** `gzip` or `zstd`

```python
# Recommended
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    compression_type="snappy"  # ✅ Balanced
)
```

---

### ✅ Monitor and Tune Retries

**Recommendation:** Use exponential backoff with bounded retries.

```python
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    retries=3,                # ✅ Limited retries
    retry_backoff_ms=100      # ✅ Exponential backoff
)
```

**Retry Strategy:**
- Retry 1: 100ms
- Retry 2: 200ms (2×)
- Retry 3: 400ms (2×)
- After 3 retries: Fail permanently

**Why 3 retries:**
- Handles transient network issues
- Prevents infinite retry loops
- Fails fast for permanent errors

---

## Reliability and Fault Tolerance

### ✅ Use Replication Factor ≥ 3

**Recommendation:** Always use replication factor of 3+ for production.

```python
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    topic=KafkaTopicConfig(
        replication_factor=3  # ✅ High availability
    )
)
```

**Replication Guidelines:**

| Environment | Replication Factor | Broker Count | Fault Tolerance |
|-------------|-------------------|--------------|-----------------|
| Development | 1 | 1 | None |
| Staging | 2 | 2 | 1 broker failure |
| Production | **3** ✅ | ≥3 | 2 broker failures |
| Critical | 5 | ≥5 | 4 broker failures |

**Constraint:** `replication_factor ≤ broker_count`

---

### ✅ Use Multiple Broker Addresses

**Recommendation:** Specify multiple brokers for failover.

```python
config = KafkaConfig(
    bootstrap_servers=[
        "kafka-1.prod:9092",
        "kafka-2.prod:9092",
        "kafka-3.prod:9092"
    ]  # ✅ Multiple brokers for failover
)
```

**Why:**
- Client connects to available broker
- Automatic failover on broker failure
- Load distribution across brokers

**Anti-pattern:**
```python
# ❌ Single point of failure
config = KafkaConfig(
    bootstrap_servers=["kafka-1.prod:9092"]
)
```

---

### ✅ Implement Health Checks

**Recommendation:** Run periodic health checks in production.

```python
import asyncio
from cryptofeed.backends.kafka.health import (
    start_periodic_health_checks,
    KafkaHealthCheck
)

async def monitor_kafka():
    def check():
        return KafkaHealthCheck.check_modern(config)

    def alert(status):
        if not status.ok:
            # Send to PagerDuty, Slack, etc.
            send_alert(f"Kafka unhealthy: {status.error}")

    await start_periodic_health_checks(
        interval_sec=60,          # ✅ Every minute
        check_fn=check,
        alert_fn=alert,
        alert_threshold_ms=500.0  # ✅ Alert if >500ms
    )

asyncio.run(monitor_kafka())
```

**Health Check Frequency:**
- Production: 30-60 seconds
- Development: 5 minutes
- Critical systems: 10 seconds

---

## Monitoring and Observability

### ✅ Track Key Metrics

**Recommendation:** Monitor these critical metrics:

**Producer Metrics:**
- `kafka_producer_messages_total`: Total messages sent
- `kafka_producer_errors_total`: Total errors
- `kafka_producer_latency_seconds`: Send latency (p50, p95, p99)
- `kafka_producer_batch_size_bytes`: Batch size distribution

**Health Metrics:**
- `kafka_health_check_ok`: Health check success rate
- `kafka_health_check_latency_ms`: Connection latency
- `kafka_health_check_errors_total`: Health check failures

**Example Prometheus Queries:**

```promql
# Error rate
rate(kafka_producer_errors_total[5m])

# 95th percentile latency
histogram_quantile(0.95, kafka_producer_latency_seconds)

# Health check success rate
avg_over_time(kafka_health_check_ok[5m])
```

---

### ✅ Set Up Alerting

**Recommendation:** Alert on critical conditions.

**Alert Rules:**

```yaml
# Prometheus alert rules
groups:
  - name: kafka_backend
    interval: 30s
    rules:
      # Health check failing
      - alert: KafkaHealthCheckFailing
        expr: kafka_health_check_ok == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Kafka health check failing"

      # High error rate
      - alert: KafkaHighErrorRate
        expr: rate(kafka_producer_errors_total[5m]) > 0.01
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Kafka error rate >1%"

      # High latency
      - alert: KafkaHighLatency
        expr: histogram_quantile(0.95, kafka_producer_latency_seconds) > 0.1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Kafka p95 latency >100ms"
```

---

### ✅ Use Structured Logging

**Recommendation:** Log key events with context.

```python
import logging

LOG = logging.getLogger("cryptofeed.kafka")

# Log configuration on startup
LOG.info(
    "Kafka backend initialized",
    extra={
        "bootstrap_servers": config.bootstrap_servers,
        "topic_strategy": config.topic.strategy,
        "partition_strategy": config.partition.strategy,
        "acks": config.acks,
        "compression": config.compression_type
    }
)

# Log errors with context
LOG.error(
    "Kafka send failed",
    extra={
        "exchange": "binance",
        "symbol": "BTC-USDT",
        "error": str(error)
    },
    exc_info=True
)
```

---

## Security Best Practices

### ✅ Never Commit Credentials

**Recommendation:** Use environment variables or secrets manager.

```python
import os

config = KafkaConfig(
    bootstrap_servers=os.getenv('KAFKA_BROKERS').split(','),
    # Credentials from environment or secrets manager
)
```

**Secrets Manager Example:**

```python
import boto3

def get_kafka_credentials():
    client = boto3.client('secretsmanager')
    secret = client.get_secret_value(SecretId='kafka-credentials')
    return json.loads(secret['SecretString'])

credentials = get_kafka_credentials()
# Use credentials['username'], credentials['password']
```

---

### ✅ Use TLS for Production

**Recommendation:** Always encrypt in-flight data.

```python
# Note: Current implementation may need extension for TLS
# Example of desired configuration:

config = KafkaConfig(
    bootstrap_servers=["kafka:9093"],  # TLS port
    # security_protocol="SSL",
    # ssl_ca_location="/path/to/ca.pem",
    # ssl_cert_location="/path/to/cert.pem",
    # ssl_key_location="/path/to/key.pem"
)
```

---

### ✅ Implement Access Control

**Recommendation:** Use Kafka ACLs for topic access.

```bash
# Grant produce permission
kafka-acls.sh --bootstrap-server kafka:9092 \
  --add \
  --allow-principal User:cryptofeed-producer \
  --operation Write \
  --topic 'cryptofeed.*'
```

---

## Migration Best Practices

### ✅ Use Blue-Green Deployment

**Recommendation:** Run legacy and modern in parallel during migration.

```python
# Week 1-2: Dual write
from cryptofeed.backends.kafka.callback import KafkaCallback as ModernKafka
from cryptofeed.backends.kafka import KafkaCallback as LegacyKafka

callbacks = [
    LegacyKafka(**legacy_config),   # Existing
    ModernKafka(config=modern_config)  # New (parallel)
]

# Week 3-4: Validate equivalence
# Compare messages from both backends

# Week 5: Switch traffic
callbacks = [ModernKafka(config=modern_config)]  # Modern only

# Week 6+: Decommission legacy
```

---

### ✅ Validate Configuration Before Deployment

**Recommendation:** Always run validation in CI/CD pipeline.

```python
from cryptofeed.backends.kafka.migration import validate_migration
from cryptofeed.backends.kafka.health import KafkaHealthCheck

# Step 1: Validate translation
report = validate_migration(legacy_config, modern_config)
assert report.is_equivalent, f"Migration invalid: {report.differences}"

# Step 2: Health check
status = KafkaHealthCheck.check_modern(modern_config)
assert status.ok, f"Health check failed: {status.error}"

# Step 3: Deploy
```

---

### ✅ Monitor Migration Progress

**Recommendation:** Track legacy usage to zero.

```python
from cryptofeed.backends.kafka.maintenance import get_deprecation_warning_system

system = get_deprecation_warning_system()

# Daily report
report = system.get_usage_report()
total_legacy_usage = sum(info['count'] for info in report.values())

if total_legacy_usage == 0:
    print("✅ Ready to remove legacy backend")
else:
    print(f"⚠️ Legacy usage: {total_legacy_usage} calls")
    print("Components:", list(report.keys()))
```

---

## Summary Checklist

### Configuration
- [ ] Enable `acks="all"` + `idempotence=True`
- [ ] Use structured YAML configuration
- [ ] Store secrets in environment/secrets manager
- [ ] Specify multiple broker addresses
- [ ] Set replication factor ≥ 3

### Topic Strategy
- [ ] Use consolidated topics (unless per-symbol needed)
- [ ] Set partitions = consumer_group_size × 2
- [ ] Use composite partition strategy (default)

### Performance
- [ ] Tune batching for latency requirements
- [ ] Use snappy compression (balanced)
- [ ] Set retries=3 with exponential backoff

### Reliability
- [ ] Implement health checks (60s interval)
- [ ] Set up monitoring and alerting
- [ ] Use structured logging

### Security
- [ ] Never commit credentials
- [ ] Use TLS in production
- [ ] Implement Kafka ACLs

### Migration
- [ ] Use blue-green deployment
- [ ] Validate configuration before deployment
- [ ] Monitor legacy usage to zero
- [ ] Track deprecation warnings

---

**Best Practices Version:** 1.0.0
**Last Updated:** 2025-11-26
**Status:** Production Ready
