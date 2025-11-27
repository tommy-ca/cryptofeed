# Kafka Backend Migration Guide

**Complete guide for migrating from legacy Kafka backend to modern implementation.**

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start Migration](#quick-start-migration)
3. [Migration Scenarios](#migration-scenarios)
4. [Configuration Translation](#configuration-translation)
5. [Advanced Migration Patterns](#advanced-migration-patterns)
6. [Validation and Testing](#validation-and-testing)
7. [Rollback Procedures](#rollback-procedures)
8. [Best Practices](#best-practices)

---

## Overview

### What Changed

The Kafka backend has evolved from a legacy `BackendQueue`-based implementation to a modern, feature-rich system:

**Legacy Backend** (`cryptofeed.backends.kafka`)
- JSON-only serialization
- `aiokafka` producer
- Flat configuration dictionary
- Limited topic/partition control
- No health monitoring

**Modern Backend** (`cryptofeed.backends.kafka.*`)
- Protobuf + JSON serialization
- `confluent-kafka` producer
- Structured Pydantic configuration
- Flexible topic strategies (consolidated/per-symbol)
- Advanced partitioning (composite/symbol/exchange/round-robin)
- Built-in health checks and monitoring
- Exactly-once semantics with idempotence

### Migration Timeline

| Phase | Status | Timeline |
|-------|--------|----------|
| Deprecation warnings active | ✅ Current | Since v0.1.0 |
| Legacy backend frozen (critical fixes only) | ✅ Current | Ongoing |
| Compatibility shim functional | ✅ Current | Until Q2 2026 |
| Planned legacy removal | ⏳ Future | Q2 2026 (90 days after zero usage) |

### Prerequisites

- Python 3.8+
- `confluent-kafka-python` installed
- Existing Kafka cluster (v2.0+)
- Access to current configuration files

---

## Quick Start Migration

### 3-Step Migration

**Step 1: Update Imports**

```python
# BEFORE (Legacy)
from cryptofeed.kafka_callback import KafkaCallback

# AFTER (Modern)
from cryptofeed.backends.kafka.callback import KafkaCallback
```

**Step 2: Translate Configuration**

```bash
# CLI Tool
python -m cryptofeed.tools.kafka_config_migrate \
  --input legacy_kafka.yaml \
  --output modern_kafka.yaml \
  --pretty
```

**Step 3: Update Callback Usage**

```python
# BEFORE (Legacy - dict config)
callback = KafkaCallback(
    bootstrap_servers=['kafka:9092'],
    topic='trades',
    acks=1
)

# AFTER (Modern - KafkaConfig object)
from cryptofeed.backends.kafka.callback import KafkaCallback, KafkaConfig

config = KafkaConfig(
    bootstrap_servers=['kafka:9092'],
    topic={'prefix': 'trades', 'strategy': 'consolidated'},
    acks='all'
)
callback = KafkaCallback(config=config, key=None, snapshot_interval=1000)
```

---

## Migration Scenarios

### Scenario 1: Basic Legacy Import Migration

**Problem:** Using deprecated import path emits warnings.

**Solution:**

```python
# Test before migration (expect warnings)
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    from cryptofeed.kafka_callback import KafkaCallback as LegacyKafka
    # DeprecationWarning emitted

# After migration (no warnings)
from cryptofeed.backends.kafka.callback import KafkaCallback
```

**Why This Works:** Modern import path bypasses compatibility shim.

---

### Scenario 2: Configuration Translation with Unmapped Options

**Problem:** Legacy config has custom options not in modern schema.

**Solution:**

```python
from cryptofeed.backends.kafka.migration import translate_legacy_config

# Legacy config with custom options
legacy = {
    "bootstrap_servers": ["kafka1:9092", "kafka2:9092"],
    "topic_prefix": "production",
    "partition_strategy": "composite",
    "acks": "all",
    "custom_retry_logic": True,      # Unmapped
    "internal_buffer_size": 10000,   # Unmapped
}

# Translate
result = translate_legacy_config(legacy)
modern_config = result.modern_config

# Check unmapped options
if result.unmapped_options:
    print("Unmapped options:", result.unmapped_options)
    # Output: {'custom_retry_logic': True, 'internal_buffer_size': 10000}
    # Handle these manually in modern implementation

# Use modern config
assert modern_config.bootstrap_servers == ["kafka1:9092", "kafka2:9092"]
assert modern_config.topic.prefix == "production"
assert modern_config.partition.strategy == "composite"
```

**Why This Works:** Translation captures unmapped options for manual review.

---

### Scenario 3: Per-Symbol vs Consolidated Topics

**Problem:** Need to choose between topic strategies.

**Solution:**

```python
from cryptofeed.backends.kafka.callback import KafkaConfig, KafkaTopicConfig

# Option A: Consolidated topics (RECOMMENDED - fewer topics)
# Creates: cryptofeed.trade, cryptofeed.book, cryptofeed.ticker
consolidated_config = KafkaConfig(
    bootstrap_servers=['kafka:9092'],
    topic=KafkaTopicConfig(
        strategy='consolidated',  # Default
        prefix='cryptofeed',
        partitions_per_topic=12,
        replication_factor=3
    )
)

# Option B: Per-symbol topics (Legacy behavior - many topics)
# Creates: trades.BTC-USD, trades.ETH-USD, trades.SOL-USD, ...
per_symbol_config = KafkaConfig(
    bootstrap_servers=['kafka:9092'],
    topic=KafkaTopicConfig(
        strategy='per_symbol',
        prefix='trades',
        partitions_per_topic=3,
        replication_factor=3
    )
)
```

**Decision Matrix:**

| Strategy | Topic Count | Use Case | Migration Impact |
|----------|-------------|----------|------------------|
| `consolidated` | O(10-20) | Modern deployments, simplified management | **Recommended** - fewer topics |
| `per_symbol` | O(1000-10000) | Legacy parity, per-symbol retention | Match legacy behavior |

**Why This Works:** Consolidated reduces operational overhead while maintaining functionality.

---

### Scenario 4: Partition Strategy Selection

**Problem:** Choose optimal partitioning for your use case.

**Solution:**

```python
from cryptofeed.backends.kafka.callback import KafkaConfig, KafkaPartitionConfig

# Option 1: Composite (RECOMMENDED - best balance)
# Routes by exchange-symbol pair (e.g., "binance-BTC-USDT")
composite = KafkaConfig(
    bootstrap_servers=['kafka:9092'],
    partition=KafkaPartitionConfig(strategy='composite')  # Default
)

# Option 2: Symbol (cross-exchange analysis)
# Routes by symbol only (e.g., "BTC-USDT")
# All BTC-USDT messages (any exchange) go to same partition
symbol = KafkaConfig(
    bootstrap_servers=['kafka:9092'],
    partition=KafkaPartitionConfig(strategy='symbol')
)

# Option 3: Exchange (per-exchange processing)
# Routes by exchange only (e.g., "binance")
# All binance messages go to same partition
exchange = KafkaConfig(
    bootstrap_servers=['kafka:9092'],
    partition=KafkaPartitionConfig(strategy='exchange')
)

# Option 4: Round-robin (maximum parallelism, no ordering)
# Random partition assignment for maximum throughput
round_robin = KafkaConfig(
    bootstrap_servers=['kafka:9092'],
    partition=KafkaPartitionConfig(strategy='round_robin')
)
```

**Decision Matrix:**

| Strategy | Ordering | Parallelism | Use Case |
|----------|----------|-------------|----------|
| `composite` | Exchange-symbol pairs ordered | High | **Recommended** - balanced |
| `symbol` | Per-symbol ordered | Medium | Cross-exchange arbitrage analysis |
| `exchange` | Per-exchange ordered | Medium | Exchange-specific monitoring |
| `round_robin` | No guarantees | Maximum | High-throughput, order-independent |

**Why This Works:** Different strategies optimize for different consumer patterns.

---

### Scenario 5: Health Check Integration

**Problem:** Monitor Kafka connectivity during migration.

**Solution:**

```python
from cryptofeed.backends.kafka.health import KafkaHealthCheck
from cryptofeed.backends.kafka.callback import KafkaConfig

# Create modern config
config = KafkaConfig(bootstrap_servers=['kafka1:9092', 'kafka2:9092'])

# Check health
status = KafkaHealthCheck.check_modern(config)

# Inspect status
print(f"Implementation: {status.implementation}")  # 'modern'
print(f"OK: {status.ok}")                          # True/False
print(f"Latency: {status.latency_ms:.2f}ms")      # Connection time
if not status.ok:
    print(f"Error: {status.error}")                # Error message
print(f"Details: {status.details}")                # Bootstrap servers

# Use in monitoring
if status.ok:
    print("✅ Kafka connection healthy")
else:
    print(f"❌ Kafka connection failed: {status.error}")
    # Trigger alerts, rollback, etc.
```

**Periodic Health Monitoring:**

```python
import asyncio
from cryptofeed.backends.kafka.health import start_periodic_health_checks

async def monitor_kafka():
    """Run periodic health checks."""

    def check_fn():
        return KafkaHealthCheck.check_modern(config)

    def alert_fn(status):
        if not status.ok:
            print(f"🚨 ALERT: Kafka unhealthy - {status.error}")
            # Send to PagerDuty, Slack, etc.

    # Start periodic checks (every 30 seconds)
    task = await start_periodic_health_checks(
        interval_sec=30,
        check_fn=check_fn,
        alert_fn=alert_fn,
        alert_threshold_ms=500.0  # Alert if latency > 500ms
    )

    # Keep running
    await task

# Run in production
asyncio.run(monitor_kafka())
```

**Why This Works:** Proactive monitoring detects connectivity issues before production impact.

---

### Scenario 6: Validation of Migration Equivalence

**Problem:** Ensure translated config behaves identically to legacy.

**Solution:**

```python
from cryptofeed.backends.kafka.migration import validate_migration
from cryptofeed.backends.kafka.callback import (
    KafkaConfig, KafkaTopicConfig, KafkaPartitionConfig
)

# Legacy config
legacy = {
    "bootstrap_servers": ["kafka:9092"],
    "topic_prefix": "crypto",
    "partition_strategy": "symbol",
    "acks": "all",
    "compression_type": "snappy",
}

# Expected modern equivalent
expected = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    topic=KafkaTopicConfig(
        strategy="per_symbol",  # Legacy default
        prefix="crypto"
    ),
    partition=KafkaPartitionConfig(strategy="symbol"),
    acks="all",
    compression_type="snappy"
)

# Validate
report = validate_migration(legacy, expected)

# Check equivalence
if report.is_equivalent:
    print("✅ Migration is functionally equivalent")
else:
    print("❌ Migration has differences:")
    for field, legacy_val, modern_val in report.differences:
        print(f"  {field}: {legacy_val} → {modern_val}")

# Check warnings
for warning in report.warnings:
    print(f"⚠️ {warning}")

# Check unmapped options
if report.unmapped_options:
    print(f"⚠️ Unmapped options: {report.unmapped_options}")
```

**Why This Works:** Automated validation prevents configuration drift during migration.

---

### Scenario 7: Deprecation Warning Tracking

**Problem:** Understand legacy usage patterns for migration planning.

**Solution:**

```python
from cryptofeed.backends.kafka.maintenance import get_deprecation_warning_system

# Get singleton instance
system = get_deprecation_warning_system()

# Track usage (automatically happens during legacy usage)
system.track_usage("TradeKafka", {
    "exchange": "binance",
    "symbol": "BTC-USDT"
})

# Get usage statistics
stats = system.get_usage_stats()
print(f"Total legacy usage: {sum(stats.values())} calls")
print(f"Components used: {list(stats.keys())}")

# Get detailed report
report = system.get_usage_report()
for component, info in report.items():
    print(f"\n{component}:")
    print(f"  Count: {info['count']}")
    print(f"  Last seen: {info['last_timestamp']}")
    print(f"  Last context: {info['last_context']}")

# Export for analytics
system.emit_usage_report()  # Logs to feedhandler logger

# Reset (for testing)
system.reset_usage_stats()
```

**Production Analytics:**

```python
import time

# Track over time window
start = time.time()
# ... application runs ...
end = time.time()

report = system.get_usage_report()

# Calculate deprecation velocity
total_calls = sum(info['count'] for info in report.values())
runtime_hours = (end - start) / 3600
calls_per_hour = total_calls / runtime_hours

print(f"Legacy usage: {calls_per_hour:.1f} calls/hour")

# Decision: Safe to remove legacy if calls_per_hour == 0 for 90 days
```

**Why This Works:** Data-driven migration planning based on actual usage.

---

### Scenario 8: CLI Migration Tool Usage

**Problem:** Migrate large configuration files efficiently.

**Solution:**

```bash
# Basic migration
python -m cryptofeed.tools.kafka_config_migrate \
  --input legacy_config.yaml \
  --output modern_config.yaml \
  --pretty

# Dry run (preview changes)
python -m cryptofeed.tools.kafka_config_migrate \
  --input legacy_config.yaml \
  --dry-run

# Validate without writing
python -m cryptofeed.tools.kafka_config_migrate \
  --input legacy_config.yaml \
  --validate-only

# Process multiple files
for file in configs/legacy_*.yaml; do
  output="configs/modern_$(basename $file)"
  python -m cryptofeed.tools.kafka_config_migrate \
    --input "$file" \
    --output "$output" \
    --pretty
done
```

**Programmatic API:**

```python
from cryptofeed.backends.kafka.migration import translate_legacy_config
import yaml

# Read legacy config
with open('legacy_config.yaml', 'r') as f:
    legacy = yaml.safe_load(f)

# Translate
result = translate_legacy_config(legacy)

# Write modern config
modern_dict = {
    'bootstrap_servers': result.modern_config.bootstrap_servers,
    'topic': result.modern_config.topic.model_dump(),
    'partition': result.modern_config.partition.model_dump(),
    'acks': result.modern_config.acks,
    # ... other fields
}

with open('modern_config.yaml', 'w') as f:
    yaml.safe_dump(modern_dict, f, default_flow_style=False)

# Log warnings
for warning in result.warnings:
    print(f"⚠️ {warning}")
```

**Why This Works:** Automation reduces manual errors in configuration migration.

---

### Scenario 9: Gradual Rollout (Blue-Green Migration)

**Problem:** Migrate production without downtime.

**Solution:**

**Phase 1: Parallel Operation (Week 1-2)**

```python
from cryptofeed.backends.kafka.callback import KafkaCallback as ModernKafka
from cryptofeed.backends.kafka import KafkaCallback as LegacyKafka

# Run both backends in parallel
legacy_config = {
    'bootstrap_servers': ['kafka:9092'],
    'topic': 'trades_legacy',
}

modern_config = KafkaConfig(
    bootstrap_servers=['kafka:9092'],
    topic=KafkaTopicConfig(prefix='trades_modern')
)

# Dual callbacks
callbacks = [
    LegacyKafka(**legacy_config),   # Keep existing
    ModernKafka(config=modern_config)  # Add modern
]

# FeedHandler uses both
f = FeedHandler()
f.add_feed(
    Binance,
    channels=[TRADES],
    symbols=['BTC-USDT'],
    callbacks={TRADES: callbacks}  # Dual write
)
```

**Phase 2: Monitor Comparison (Week 3-4)**

```python
# Validate modern output matches legacy
# (Run consumers on both topics, compare messages)
```

**Phase 3: Traffic Switch (Week 5)**

```python
# Remove legacy callback
callbacks = [
    ModernKafka(config=modern_config)  # Modern only
]
```

**Phase 4: Cleanup (Week 6)**

```python
# Decommission legacy topics
# Remove legacy code
```

**Why This Works:** Risk-free migration with instant rollback capability.

---

### Scenario 10: Complete Production Migration Workflow

**Problem:** End-to-end production migration.

**Solution:**

```python
"""
Complete migration workflow for production environment.
"""

from cryptofeed.backends.kafka.migration import (
    translate_legacy_config,
    validate_migration
)
from cryptofeed.backends.kafka.health import KafkaHealthCheck
from cryptofeed.backends.kafka.callback import KafkaCallback
import yaml
import logging

LOG = logging.getLogger(__name__)


def migrate_production_config():
    """Step-by-step production migration."""

    # Step 1: Load legacy configuration
    LOG.info("Loading legacy configuration...")
    with open('production_legacy.yaml', 'r') as f:
        legacy_config = yaml.safe_load(f)

    # Step 2: Translate to modern format
    LOG.info("Translating to modern format...")
    result = translate_legacy_config(legacy_config)
    modern_config = result.modern_config

    # Step 3: Check for unmapped options
    if result.unmapped_options:
        LOG.warning(f"Unmapped options detected: {result.unmapped_options}")
        LOG.warning("Review these options and migrate manually if needed")

    # Step 4: Validate translation equivalence
    LOG.info("Validating migration equivalence...")
    validation = validate_migration(legacy_config, modern_config)

    if not validation.is_equivalent:
        LOG.error("Migration validation failed!")
        for field, legacy_val, modern_val in validation.differences:
            LOG.error(f"  {field}: {legacy_val} → {modern_val}")
        raise ValueError("Migration validation failed - review differences")

    LOG.info("✅ Migration validation passed")

    # Step 5: Health check (pre-migration)
    LOG.info("Running pre-migration health check...")
    status = KafkaHealthCheck.check_modern(modern_config)

    if not status.ok:
        LOG.error(f"Pre-migration health check failed: {status.error}")
        raise ConnectionError(f"Cannot connect to Kafka: {status.error}")

    LOG.info(f"✅ Health check passed ({status.latency_ms:.2f}ms latency)")

    # Step 6: Save modern configuration
    LOG.info("Saving modern configuration...")
    modern_dict = {
        'bootstrap_servers': modern_config.bootstrap_servers,
        'topic': modern_config.topic.model_dump(),
        'partition': modern_config.partition.model_dump(),
        'acks': modern_config.acks,
        'idempotence': modern_config.idempotence,
        'retries': modern_config.retries,
        'retry_backoff_ms': modern_config.retry_backoff_ms,
        'batch_size': modern_config.batch_size,
        'linger_ms': modern_config.linger_ms,
        'compression_type': modern_config.compression_type,
    }

    with open('production_modern.yaml', 'w') as f:
        yaml.safe_dump(modern_dict, f, default_flow_style=False)

    LOG.info("✅ Modern configuration saved to production_modern.yaml")

    # Step 7: Create test callback
    LOG.info("Testing modern callback instantiation...")
    try:
        callback = KafkaCallback(config=modern_config, key=None)
        LOG.info("✅ Modern callback created successfully")
    except Exception as e:
        LOG.error(f"Failed to create modern callback: {e}")
        raise

    # Step 8: Migration summary
    LOG.info("\n" + "="*60)
    LOG.info("MIGRATION SUMMARY")
    LOG.info("="*60)
    LOG.info(f"Bootstrap servers: {modern_config.bootstrap_servers}")
    LOG.info(f"Topic strategy: {modern_config.topic.strategy}")
    LOG.info(f"Topic prefix: {modern_config.topic.prefix}")
    LOG.info(f"Partition strategy: {modern_config.partition.strategy}")
    LOG.info(f"Acks: {modern_config.acks}")
    LOG.info(f"Compression: {modern_config.compression_type}")
    LOG.info(f"Idempotence: {modern_config.idempotence}")
    LOG.info("="*60)

    if result.unmapped_options:
        LOG.warning("\n⚠️  UNMAPPED OPTIONS REQUIRE MANUAL REVIEW:")
        for key, value in result.unmapped_options.items():
            LOG.warning(f"  {key}: {value}")

    LOG.info("\n✅ Migration completed successfully!")
    LOG.info("Next steps:")
    LOG.info("  1. Review production_modern.yaml")
    LOG.info("  2. Test in staging environment")
    LOG.info("  3. Run blue-green deployment")
    LOG.info("  4. Monitor health metrics")
    LOG.info("  5. Decommission legacy after validation")

    return modern_config


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    migrate_production_config()
```

**Why This Works:** Comprehensive validation at each step prevents production issues.

---

## Configuration Translation

### Legacy to Modern Key Mapping

| Legacy Key | Modern Path | Default | Notes |
|------------|-------------|---------|-------|
| `bootstrap_servers` | `bootstrap_servers` | Required | No change |
| `acks` | `acks` | `"all"` | String format |
| `idempotence` | `idempotence` | `True` | Enables exactly-once |
| `retries` | `retries` | `3` | Retry count |
| `retry_backoff_ms` | `retry_backoff_ms` | `100` | Backoff time |
| `batch_size` | `batch_size` | `16384` | Bytes |
| `linger_ms` | `linger_ms` | `10` | Latency vs throughput |
| `compression_type` | `compression_type` | `"snappy"` | Algorithm |
| `topic_prefix` | `topic.prefix` | `"cryptofeed"` | Topic naming |
| `topic_strategy` | `topic.strategy` | `"consolidated"` | Modern default |
| `partitions_per_topic` | `topic.partitions_per_topic` | `3` | Partition count |
| `replication_factor` | `topic.replication_factor` | `3` | Replication |
| `partition_strategy` | `partition.strategy` | `"composite"` | Partitioning |

### Default Behavior Changes

**Legacy Defaults:**
- Topic strategy: `per_symbol` (creates O(1000s) topics)
- Partition strategy: `composite`
- Acks: `1` (at-least-once)
- Compression: `none`

**Modern Defaults:**
- Topic strategy: `consolidated` (creates O(10s) topics)
- Partition strategy: `composite`
- Acks: `all` (exactly-once with idempotence)
- Compression: `snappy`

**Migration Impact:** Review defaults and explicitly set legacy behavior if needed for compatibility.

---

## Advanced Migration Patterns

### Pattern 1: Custom Serialization Migration

**Legacy:**
```python
from cryptofeed.backends.kafka import KafkaCallback

def custom_serializer(data):
    return msgpack.packb(data)

legacy_callback = KafkaCallback(
    bootstrap_servers=['kafka:9092'],
    value_serializer=custom_serializer
)
```

**Modern:**
```python
from cryptofeed.backends.kafka.callback import KafkaCallback, KafkaConfig
from cryptofeed.backends.kafka.protobuf_callback import KafkaProtobufCallback

# Option 1: Use built-in protobuf serialization
config = KafkaConfig(bootstrap_servers=['kafka:9092'])
callback = KafkaProtobufCallback(config=config)

# Option 2: Custom serialization (requires subclassing)
# See docs/kafka/custom-serialization.md
```

### Pattern 2: Multi-Cluster Migration

**Problem:** Migrate across multiple Kafka clusters.

**Solution:**
```python
# Production cluster
prod_config = KafkaConfig(
    bootstrap_servers=['prod-kafka-1:9092', 'prod-kafka-2:9092'],
    topic=KafkaTopicConfig(prefix='production')
)

# Staging cluster
staging_config = KafkaConfig(
    bootstrap_servers=['staging-kafka:9092'],
    topic=KafkaTopicConfig(prefix='staging')
)

# Parallel testing
prod_callback = KafkaCallback(config=prod_config)
staging_callback = KafkaCallback(config=staging_config)

# Use both for validation
callbacks = [staging_callback, prod_callback]
```

### Pattern 3: Configuration from Environment

**Modern with environment variables:**

```python
import os
from cryptofeed.backends.kafka.callback import KafkaConfig, KafkaTopicConfig

config = KafkaConfig(
    bootstrap_servers=os.getenv('KAFKA_BROKERS', 'localhost:9092').split(','),
    topic=KafkaTopicConfig(
        strategy=os.getenv('KAFKA_TOPIC_STRATEGY', 'consolidated'),
        prefix=os.getenv('KAFKA_TOPIC_PREFIX', 'cryptofeed')
    ),
    acks=os.getenv('KAFKA_ACKS', 'all'),
    compression_type=os.getenv('KAFKA_COMPRESSION', 'snappy')
)
```

---

## Validation and Testing

### Pre-Migration Validation

```python
from cryptofeed.backends.kafka.migration import validate_migration, translate_legacy_config
from cryptofeed.backends.kafka.health import KafkaHealthCheck

# 1. Translation validation
legacy = {...}  # Your legacy config
result = translate_legacy_config(legacy)

# 2. Equivalence check
validation = validate_migration(legacy, result.modern_config)
assert validation.is_equivalent, f"Differences: {validation.differences}"

# 3. Health check
status = KafkaHealthCheck.check_modern(result.modern_config)
assert status.ok, f"Health check failed: {status.error}"

# 4. Unmapped options review
if result.unmapped_options:
    print(f"Review unmapped options: {result.unmapped_options}")
```

### Post-Migration Testing

```python
import pytest
from cryptofeed.backends.kafka.callback import KafkaCallback, KafkaConfig

def test_modern_callback_creation():
    """Verify modern callback can be created."""
    config = KafkaConfig(bootstrap_servers=['kafka:9092'])
    callback = KafkaCallback(config=config, key=None)
    assert callback is not None

def test_configuration_values():
    """Verify configuration matches expectations."""
    config = KafkaConfig(
        bootstrap_servers=['kafka:9092'],
        acks='all',
        compression_type='snappy'
    )
    assert config.acks == 'all'
    assert config.compression_type == 'snappy'
    assert config.idempotence is True
```

---

## Rollback Procedures

### Immediate Rollback

If migration issues arise:

**Step 1: Revert Imports**
```python
# Change back to legacy import
from cryptofeed.backends.kafka import KafkaCallback  # Legacy path
```

**Step 2: Restore Legacy Configuration**
```python
# Use legacy config format
legacy_config = {
    'bootstrap_servers': ['kafka:9092'],
    'topic': 'trades',
    'acks': 1
}
callback = KafkaCallback(**legacy_config)
```

**Step 3: Monitor**
- Check Kafka consumer lag
- Verify message production rate
- Monitor error logs

### Partial Rollback (Blue-Green)

Keep both backends running:

```python
from cryptofeed.backends.kafka.callback import KafkaCallback as ModernKafka
from cryptofeed.backends.kafka import KafkaCallback as LegacyKafka

# Dual write during rollback evaluation
callbacks = [
    LegacyKafka(**legacy_config),      # Primary (rolled back)
    ModernKafka(config=modern_config)  # Secondary (monitoring)
]
```

---

## Best Practices

### ✅ DO

1. **Test in staging first**
   - Always validate migration in non-production environment
   - Run parallel legacy/modern for 1-2 weeks

2. **Use validation tools**
   - Run `validate_migration()` before production
   - Monitor health checks continuously

3. **Plan rollback procedures**
   - Document rollback steps before migration
   - Keep legacy configuration accessible

4. **Monitor migration progress**
   - Track deprecation warnings with analytics
   - Use usage tracking to identify remaining legacy usage

5. **Leverage consolidated topics**
   - Modern default reduces operational overhead
   - Easier topic management and monitoring

6. **Enable idempotence**
   - Exactly-once semantics prevent duplicates
   - Safer in network failures

### ❌ DON'T

1. **Don't migrate without validation**
   - Always run `validate_migration()` first
   - Verify health checks pass

2. **Don't ignore unmapped options**
   - Review all unmapped configuration keys
   - Manually migrate custom settings

3. **Don't skip parallel testing**
   - Blue-green deployment reduces risk
   - Parallel operation validates equivalence

4. **Don't remove legacy immediately**
   - Keep legacy code until zero usage confirmed
   - Monitor for 90 days minimum

5. **Don't ignore deprecation warnings**
   - Warnings indicate migration urgency
   - Track usage patterns for planning

6. **Don't change multiple things at once**
   - Migrate configuration first
   - Test thoroughly before adding new features

---

## Summary

**Key Takeaways:**

1. **3-step migration:** Update imports → Translate config → Update usage
2. **Validation is critical:** Use `validate_migration()` before production
3. **Health checks required:** Monitor connectivity throughout migration
4. **Blue-green recommended:** Parallel operation reduces risk
5. **Consolidated topics preferred:** Reduces operational complexity
6. **Idempotence enabled:** Exactly-once semantics by default
7. **Track deprecation warnings:** Data-driven migration timeline

**Next Steps:**

1. Review this guide completely
2. Run migration in staging environment
3. Validate configuration equivalence
4. Monitor health checks
5. Execute blue-green deployment
6. Track legacy usage to zero
7. Decommission legacy backend (Q2 2026)

**Support:**

- API Reference: `docs/kafka/API_REFERENCE.md`
- Troubleshooting: `docs/kafka/TROUBLESHOOTING.md`
- Best Practices: `docs/kafka/BEST_PRACTICES.md`
- Examples: `examples/kafka/migration_examples.py`

---

**Migration Guide Version:** 1.0.0
**Last Updated:** 2025-11-26
**Status:** Production Ready
