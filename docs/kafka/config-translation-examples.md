# Kafka Configuration Translation Examples

This document provides 10 real-world configuration examples with before/after translations from legacy to Phase 2 format.

---

## Example 1: Simple Single-Broker Setup

**Use Case**: Development environment, single broker

### Legacy Configuration
```yaml
bootstrap_servers:
  - kafka:9092
```

### Translated Phase 2 Configuration
```yaml
bootstrap_servers:
  - kafka:9092

topic:
  strategy: per_symbol
  prefix: cryptofeed
  partitions_per_topic: 3
  replication_factor: 3

partition:
  strategy: composite

acks: '1'
idempotence: false
retries: 3
retry_backoff_ms: 100
batch_size: 16384
linger_ms: 10
compression_type: snappy
```

### Translation Notes
- Broker address copied as-is
- All new Phase 2 fields use defaults
- Per-symbol strategy maintains backward compatibility
- Suitable for development/testing only

---

## Example 2: Multi-Exchange Setup with Custom Prefix

**Use Case**: Production staging environment

### Legacy Configuration
```yaml
bootstrap_servers:
  - kafka1:9092
  - kafka2:9092
  - kafka3:9092
topic_prefix: staging
acks: '1'
compression_type: snappy
```

### Translated Phase 2 Configuration
```yaml
bootstrap_servers:
  - kafka1:9092
  - kafka2:9092
  - kafka3:9092

topic:
  strategy: per_symbol              # For staging compatibility
  prefix: staging                   # Preserved from legacy
  partitions_per_topic: 3
  replication_factor: 3

partition:
  strategy: composite

acks: '1'
idempotence: false
retries: 3
retry_backoff_ms: 100
batch_size: 16384
linger_ms: 10
compression_type: snappy
```

### Translation Notes
- Custom prefix `staging` preserved (legacy `topic_prefix` → Phase 2 `topic.prefix`)
- acks value copied directly
- Compression type copied directly
- Broker list expanded (3 brokers)

---

## Example 3: High-Throughput Setup

**Use Case**: Production with high message volume

### Legacy Configuration
```yaml
bootstrap_servers:
  - kafka-prod-1.internal:9092
  - kafka-prod-2.internal:9092
  - kafka-prod-3.internal:9092
acks: '1'
retries: 5
batch_size: 65536
linger_ms: 100
compression_type: lz4
```

### Translated Phase 2 Configuration
```yaml
bootstrap_servers:
  - kafka-prod-1.internal:9092
  - kafka-prod-2.internal:9092
  - kafka-prod-3.internal:9092

topic:
  strategy: consolidated            # Switch to consolidated for better scaling
  prefix: cryptofeed
  partitions_per_topic: 12          # Increased for high throughput
  replication_factor: 3

partition:
  strategy: composite               # Maintain ordering per exchange-symbol

acks: '1'
idempotence: false                 # Keep false for maximum throughput
retries: 5
retry_backoff_ms: 100
batch_size: 65536
linger_ms: 100
compression_type: lz4
```

### Translation Notes
- **Strategy Change**: Per-symbol → consolidated (reduces topic count from ~1000 to ~20)
- **Partitions**: Increased from 3 to 12 for better parallelism
- **Batch Settings**: Preserved for high throughput (larger batch size, higher linger)
- **Compression**: LZ4 preserved (good for throughput)

---

## Example 4: Strict Delivery Guarantee Setup

**Use Case**: Financial compliance, must not lose data

### Legacy Configuration
```yaml
bootstrap_servers:
  - kafka-secure-1:9092
  - kafka-secure-2:9092
  - kafka-secure-3:9092
acks: all
retries: 999
retry_backoff_ms: 1000
compression_type: none
```

### Translated Phase 2 Configuration
```yaml
bootstrap_servers:
  - kafka-secure-1:9092
  - kafka-secure-2:9092
  - kafka-secure-3:9092

topic:
  strategy: consolidated
  prefix: cryptofeed
  partitions_per_topic: 6
  replication_factor: 3            # Maintain 3x replication

partition:
  strategy: composite              # Ensure ordering for compliance

acks: all                          # Ensures all replicas acknowledge
idempotence: true                  # Auto-enabled by Phase 2 (acks: all)
retries: 999                       # Preserve unlimited retries
retry_backoff_ms: 1000             # Slower retry for reliability
batch_size: 16384                  # Default (no optimization)
linger_ms: 10                      # Default (no optimization)
compression_type: none             # No compression for compliance
```

### Translation Notes
- **acks: all** → `idempotence: true` auto-enabled
- **Retries**: Unlimited (999) preserved for financial compliance
- **Compression**: None preserved (important for compliance)
- **Replication**: 3x replication standard for compliance

---

## Example 5: Low-Latency Trading Setup

**Use Case**: Algorithmic trading, minimize latency

### Legacy Configuration
```yaml
bootstrap_servers:
  - kafka-latency-optimized-1:9092
  - kafka-latency-optimized-2:9092
acks: '1'
batch_size: 1024
linger_ms: 0
compression_type: none
```

### Translated Phase 2 Configuration
```yaml
bootstrap_servers:
  - kafka-latency-optimized-1:9092
  - kafka-latency-optimized-2:9092

topic:
  strategy: consolidated
  prefix: cryptofeed
  partitions_per_topic: 8          # More partitions for parallelism
  replication_factor: 2             # Reduced from 3 (lower latency)

partition:
  strategy: round_robin             # Distribute load, minimize latency

acks: '1'
idempotence: false                 # Disable for speed
retries: 1                         # Minimal retry for latency
retry_backoff_ms: 10               # Fast retry
batch_size: 1024                   # Small batches for low latency
linger_ms: 0                       # No wait, immediate send
compression_type: none
```

### Translation Notes
- **Partition Strategy**: Changed to `round_robin` (no ordering for max speed)
- **Batch Size**: Small (1024 bytes) for immediate sends
- **Linger Time**: Zero (no waiting, sent immediately)
- **Replication**: Reduced to 2x (acceptable for trading, lower latency)
- **Retries**: Minimal (1) to fail fast on errors

---

## Example 6: Multi-Tenant Setup with Custom Topic Prefix

**Use Case**: Service provider with multiple customers

### Legacy Configuration
```yaml
bootstrap_servers:
  - kafka:9092
topic_prefix: customer_acme        # Customer-specific prefix
acks: '1'
```

### Translated Phase 2 Configuration
```yaml
bootstrap_servers:
  - kafka:9092

topic:
  strategy: consolidated            # Use consolidated for scaling
  prefix: customer_acme             # Preserve customer isolation
  partitions_per_topic: 3
  replication_factor: 3

partition:
  strategy: composite

acks: '1'
idempotence: false
retries: 3
retry_backoff_ms: 100
batch_size: 16384
linger_ms: 10
compression_type: snappy
```

### Translation Notes
- **Prefix**: `customer_acme` preserved (enables tenant isolation)
- **Topics Generated**:
  - `customer_acme.trades`
  - `customer_acme.orderbook`
  - `customer_acme.ticker`
- Perfect for multi-tenant architectures

---

## Example 7: Data Warehouse Ingest Setup

**Use Case**: Parallel ingest to data warehouse, optimized for batch processing

### Legacy Configuration
```yaml
bootstrap_servers:
  - kafka-bulk-1:9092
  - kafka-bulk-2:9092
  - kafka-bulk-3:9092
  - kafka-bulk-4:9092
acks: '0'
batch_size: 131072                 # 128KB for bulk transfer
linger_ms: 500                      # Wait to fill batches
compression_type: zstd             # Best compression for DW
retries: 3
```

### Translated Phase 2 Configuration
```yaml
bootstrap_servers:
  - kafka-bulk-1:9092
  - kafka-bulk-2:9092
  - kafka-bulk-3:9092
  - kafka-bulk-4:9092

topic:
  strategy: consolidated
  prefix: cryptofeed
  partitions_per_topic: 16          # High parallelism for DW ingest
  replication_factor: 2             # Reduced (DW can replay from source)

partition:
  strategy: round_robin             # Maximum parallelism for DW

acks: '0'
idempotence: false
retries: 3
retry_backoff_ms: 100
batch_size: 131072                  # Preserved (128KB)
linger_ms: 500                      # Preserved (batch accumulation)
compression_type: zstd
```

### Translation Notes
- **acks: 0**: Fire-and-forget (acceptable, DW can replay)
- **Large Batches**: 131KB for efficient bulk transfer
- **High Linger**: 500ms to accumulate batch data
- **Compression**: zstd (best ratio for Parquet/Iceberg)
- **Partitions**: 16 for parallel DW ingest

---

## Example 8: Consolidated Topics with Symbol Partitioning

**Use Case**: Cross-exchange analysis (data aggregation)

### Legacy Configuration
```yaml
bootstrap_servers:
  - kafka-analytics:9092
topic_prefix: analytics
acks: '1'
```

### Translated Phase 2 Configuration
```yaml
bootstrap_servers:
  - kafka-analytics:9092

topic:
  strategy: consolidated            # All data in per-data-type topics
  prefix: analytics                 # Custom prefix for analytics
  partitions_per_topic: 6
  replication_factor: 3

partition:
  strategy: symbol                  # Key by symbol (not exchange)
                                     # Enables: same symbol from all exchanges → same partition

acks: '1'
idempotence: false
retries: 3
retry_backoff_ms: 100
batch_size: 16384
linger_ms: 10
compression_type: snappy
```

### Translation Notes
- **Partition Strategy**: `symbol` (not composite)
- **Topics**: `analytics.trades`, `analytics.orderbook`, `analytics.ticker`
- **Use Case**: Correlate BTC-USD across binance, coinbase, kraken
- **Key Distribution**: All "BTC-USD" messages → same partition

---

## Example 9: Per-Symbol Topics with Exchange Partitioning

**Use Case**: Legacy system needing exchange-based ordering

### Legacy Configuration
```yaml
bootstrap_servers:
  - kafka:9092
topic_prefix: legacy_feeds
acks: '1'
compression_type: snappy
```

### Translated Phase 2 Configuration
```yaml
bootstrap_servers:
  - kafka:9092

topic:
  strategy: per_symbol              # Maintain per-symbol topic names
  prefix: legacy_feeds              # Preserve prefix
  partitions_per_topic: 3
  replication_factor: 3

partition:
  strategy: exchange                # Key by exchange only
                                     # All symbols from binance → same partition

acks: '1'
idempotence: false
retries: 3
retry_backoff_ms: 100
batch_size: 16384
linger_ms: 10
compression_type: snappy
```

### Translation Notes
- **Topics**: `legacy_feeds.trades.binance.btc-usd`, etc.
- **Partition Strategy**: `exchange` (not composite)
- **Ordering**: All binance trades have order within partition
- **Use Case**: Exchange-based analytics (single exchange analysis)

---

## Example 10: Production HA Setup with Everything Configured

**Use Case**: High-availability production (recommended)

### Legacy Configuration
```yaml
bootstrap_servers:
  - kafka-prod-1.us-east-1a:9092
  - kafka-prod-2.us-east-1b:9092
  - kafka-prod-3.us-east-1c:9092
topic_prefix: production
acks: '1'
idempotence: false          # Legacy didn't have this
retries: 5
retry_backoff_ms: 200
batch_size: 32768
linger_ms: 20
compression_type: snappy
```

### Translated Phase 2 Configuration
```yaml
bootstrap_servers:
  - kafka-prod-1.us-east-1a:9092
  - kafka-prod-2.us-east-1b:9092
  - kafka-prod-3.us-east-1c:9092

topic:
  strategy: consolidated            # Modern best practice
  prefix: production
  partitions_per_topic: 9           # 3 per availability zone
  replication_factor: 3             # Multi-zone replication

partition:
  strategy: composite               # Exchange-symbol ordering for accuracy

acks: all                           # Upgrade for HA (all replicas acknowledge)
idempotence: true                  # Auto-enabled (acks: all)
retries: 5
retry_backoff_ms: 200
batch_size: 32768
linger_ms: 20
compression_type: snappy
```

### Translation Notes
- **acks Upgrade**: Changed from '1' to 'all' for HA compliance
- **Idempotence**: Auto-enabled with acks: all
- **Partitions**: 9 (3 per AZ) for balanced load
- **Replication**: 3x across availability zones
- **All Settings Preserved**: Batch size, compression, retry logic
- **Topics Generated**:
  - `production.trades` (9 partitions, 3x replication)
  - `production.orderbook` (9 partitions, 3x replication)
  - `production.ticker` (9 partitions, 3x replication)

---

## Using the Migration CLI Tool

### Automatic Translation

```bash
# Translate legacy config
python tools/migrate-kafka-config.py translate \
  --input legacy_kafka.yaml \
  --output phase2_kafka.yaml

# Preview changes without saving
python tools/migrate-kafka-config.py translate \
  --input legacy_kafka.yaml \
  --dry-run

# Validate Phase 2 config
python tools/migrate-kafka-config.py validate \
  --config phase2_kafka.yaml

# Test Kafka connectivity
python tools/migrate-kafka-config.py validate \
  --config phase2_kafka.yaml \
  --test-kafka
```

### Custom Adjustments

After translation, customize Phase 2 config for your use case:

```yaml
# Example: Transform Example 3 for consolidated topics
topic:
  strategy: consolidated          # Change from per_symbol
  prefix: cryptofeed
  partitions_per_topic: 12        # Increase for throughput

partition:
  strategy: symbol                # Change from composite if needed
```

---

## Migration Decision Tree

**Which strategy should I choose?**

1. **Keeping per-symbol topics for backward compatibility?**
   - Yes → Use `topic.strategy: per_symbol` (like Example 9)
   - No → Use `topic.strategy: consolidated` (like Example 10)

2. **What partitioning do I need?**
   - Cross-exchange analysis (correlate same symbol) → `partition.strategy: symbol` (Example 8)
   - Exchange-level analysis → `partition.strategy: exchange` (Example 9)
   - Per-exchange-symbol ordering (default) → `partition.strategy: composite` (Most examples)
   - Maximum throughput/parallelism → `partition.strategy: round_robin` (Example 5)

3. **What are my latency requirements?**
   - Low latency (<5ms) → Use settings from Example 5
   - Standard (5-20ms) → Use settings from Example 10
   - High throughput → Use settings from Example 3

4. **What are my reliability requirements?**
   - Must not lose data (compliance) → Use settings from Example 4
   - Standard HA → Use settings from Example 10
   - Can tolerate loss (analytics) → Use settings from Example 7

---

## Configuration Validation Checklist

After translation, verify:

- [ ] `bootstrap_servers` lists all Kafka brokers
- [ ] `topic.strategy` matches your use case (consolidated vs per_symbol)
- [ ] `topic.prefix` is appropriate for your environment
- [ ] `partition.strategy` matches your ordering needs
- [ ] `acks` value is appropriate for reliability needs
- [ ] `batch_size` and `linger_ms` match throughput goals
- [ ] `compression_type` is appropriate (snappy for balanced, lz4 for throughput, none for latency)
- [ ] `retries` count matches failure tolerance
- [ ] All broker addresses include port numbers

Run validation:
```bash
python tools/migrate-kafka-config.py validate --config phase2.yaml
```

---

## Conclusion

These 10 examples cover the most common production scenarios. Start with the example closest to your use case, then adjust settings based on your specific requirements using the decision tree above.

For detailed guidance on each setting, see `cryptofeed/kafka_callback.py` documentation.
