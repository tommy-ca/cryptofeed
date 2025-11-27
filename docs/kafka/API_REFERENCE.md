# Kafka Backend API Reference

**Complete API reference for modern Kafka backend implementation.**

---

## Table of Contents

1. [Configuration Models](#configuration-models)
2. [Callback Classes](#callback-classes)
3. [Migration Utilities](#migration-utilities)
4. [Health Check System](#health-check-system)
5. [Maintenance Utilities](#maintenance-utilities)
6. [Legacy Backend (Deprecated)](#legacy-backend-deprecated)

---

## Configuration Models

### KafkaConfig

**Module:** `cryptofeed.backends.kafka.callback`

**Description:** Main configuration model for modern Kafka backend.

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `bootstrap_servers` | `list[str]` | Required | Kafka broker addresses |
| `topic` | `KafkaTopicConfig` | See below | Topic configuration |
| `partition` | `KafkaPartitionConfig` | See below | Partition configuration |
| `acks` | `str` | `"all"` | Acknowledgment level (`'0'`, `'1'`, `'all'`) |
| `idempotence` | `bool` | `True` | Enable idempotent producer |
| `retries` | `int` | `3` | Retry count on failure |
| `retry_backoff_ms` | `int` | `100` | Backoff time between retries (ms) |
| `batch_size` | `int` | `16384` | Maximum batch size (bytes) |
| `linger_ms` | `int` | `10` | Time to wait before sending batch (ms) |
| `compression_type` | `str` | `"snappy"` | Compression algorithm |

**Example:**

```python
from cryptofeed.backends.kafka.callback import KafkaConfig

config = KafkaConfig(
    bootstrap_servers=["kafka1:9092", "kafka2:9092"],
    acks="all",
    compression_type="snappy"
)
```

**Validation Rules:**
- `bootstrap_servers`: Must be non-empty list
- `acks`: Must be `'0'`, `'1'`, or `'all'`
- `compression_type`: Must be `'none'`, `'gzip'`, `'snappy'`, `'lz4'`, or `'zstd'`
- `batch_size`: Must be > 0
- `retries`: Must be ≥ 0

---

### KafkaTopicConfig

**Module:** `cryptofeed.backends.kafka.callback`

**Description:** Configuration for Kafka topic management.

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | `str` | `"consolidated"` | Topic naming strategy |
| `prefix` | `str` | `"cryptofeed"` | Topic name prefix |
| `partitions_per_topic` | `int` | `3` | Number of partitions per topic |
| `replication_factor` | `int` | `3` | Replication factor |

**Example:**

```python
from cryptofeed.backends.kafka.callback import KafkaTopicConfig

# Consolidated topics (modern default)
consolidated = KafkaTopicConfig(
    strategy="consolidated",
    prefix="production",
    partitions_per_topic=12,
    replication_factor=3
)

# Per-symbol topics (legacy compatibility)
per_symbol = KafkaTopicConfig(
    strategy="per_symbol",
    prefix="trades",
    partitions_per_topic=3,
    replication_factor=3
)
```

**Validation Rules:**
- `strategy`: Must be `'consolidated'` or `'per_symbol'`
- `prefix`: Whitespace-only defaults to `'cryptofeed'`
- `partitions_per_topic`: Must be > 0
- `replication_factor`: Must be > 0

**Topic Naming:**
- **Consolidated:** `{prefix}.{data_type}` (e.g., `cryptofeed.trade`)
- **Per-symbol:** `{prefix}.{symbol}` (e.g., `trades.BTC-USD`)

---

### KafkaPartitionConfig

**Module:** `cryptofeed.backends.kafka.callback`

**Description:** Configuration for partition key strategies.

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | `str` | `"composite"` | Partitioner strategy |

**Example:**

```python
from cryptofeed.backends.kafka.callback import KafkaPartitionConfig

# Composite (recommended)
composite = KafkaPartitionConfig(strategy="composite")

# Symbol-based
symbol = KafkaPartitionConfig(strategy="symbol")

# Exchange-based
exchange = KafkaPartitionConfig(strategy="exchange")

# Round-robin
round_robin = KafkaPartitionConfig(strategy="round_robin")
```

**Validation Rules:**
- `strategy`: Must be `'composite'`, `'symbol'`, `'exchange'`, or `'round_robin'`

**Strategy Details:**

| Strategy | Partition Key | Ordering Guarantee | Use Case |
|----------|---------------|-------------------|----------|
| `composite` | `{exchange}-{symbol}` | Per exchange-symbol pair | **Recommended** - balanced |
| `symbol` | `{symbol}` | Per symbol (all exchanges) | Cross-exchange analysis |
| `exchange` | `{exchange}` | Per exchange (all symbols) | Exchange-specific processing |
| `round_robin` | `None` | No ordering | Maximum parallelism |

---

## Callback Classes

### KafkaCallback

**Module:** `cryptofeed.backends.kafka.callback`

**Description:** Main callback class for Kafka message production (JSON serialization).

**Constructor:**

```python
KafkaCallback(
    config: KafkaConfig,
    key: str | None = None,
    snapshot_interval: int = 1000,
    numeric_type: type = float,
    none_to: Any = None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `KafkaConfig` | Required | Kafka configuration |
| `key` | `str \| None` | `None` | Callback identifier |
| `snapshot_interval` | `int` | `1000` | Snapshot interval for books |
| `numeric_type` | `type` | `float` | Numeric serialization type |
| `none_to` | `Any` | `None` | None value replacement |

**Example:**

```python
from cryptofeed.backends.kafka.callback import KafkaCallback, KafkaConfig
from cryptofeed.feedhandler import FeedHandler
from cryptofeed.exchanges import Binance
from cryptofeed.defines import TRADES

# Create config
config = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    acks="all",
    compression_type="snappy"
)

# Create callback
callback = KafkaCallback(config=config, key=None, snapshot_interval=1000)

# Use with FeedHandler
f = FeedHandler()
f.add_feed(
    Binance,
    channels=[TRADES],
    symbols=['BTC-USDT'],
    callbacks={TRADES: callback}
)

# Start
f.run()
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `start()` | `async def start() -> None` | Initialize producer |
| `stop()` | `async def stop() -> None` | Shutdown producer |
| `__call__()` | `async def __call__(dtype, timestamp: float)` | Process data |

**Inherited from BackendCallback:**
- Handles serialization and queueing
- Supports snapshot intervals for order books
- Thread-safe message queueing

---

### KafkaProtobufCallback

**Module:** `cryptofeed.backends.kafka.protobuf_callback`

**Description:** Callback class for Kafka message production with protobuf serialization.

**Constructor:**

```python
KafkaProtobufCallback(
    config: KafkaConfig,
    key: str | None = None,
    snapshot_interval: int = 1000
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `KafkaConfig` | Required | Kafka configuration |
| `key` | `str \| None` | `None` | Callback identifier |
| `snapshot_interval` | `int` | `1000` | Snapshot interval for books |

**Example:**

```python
from cryptofeed.backends.kafka.protobuf_callback import KafkaProtobufCallback
from cryptofeed.backends.kafka.callback import KafkaConfig
from cryptofeed.feedhandler import FeedHandler
from cryptofeed.exchanges import Binance
from cryptofeed.defines import TRADES

# Create config
config = KafkaConfig(bootstrap_servers=["kafka:9092"])

# Create protobuf callback
callback = KafkaProtobufCallback(config=config, key=None)

# Use with FeedHandler
f = FeedHandler()
f.add_feed(
    Binance,
    channels=[TRADES],
    symbols=['BTC-USDT'],
    callbacks={TRADES: callback}
)

f.run()
```

**Differences from KafkaCallback:**
- Uses protobuf serialization (binary)
- ~63% smaller message size
- ~2.1µs serialization latency
- Requires protobuf schema definitions

---

## Migration Utilities

### translate_legacy_config()

**Module:** `cryptofeed.backends.kafka.migration`

**Description:** Translate legacy configuration to modern `KafkaConfig`.

**Signature:**

```python
def translate_legacy_config(
    legacy_config: Dict[str, Any]
) -> MigrationResult
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `legacy_config` | `Dict[str, Any]` | Legacy configuration dictionary |

**Returns:** `MigrationResult`

**Raises:**
- `ValueError`: If `bootstrap_servers` missing or invalid

**Example:**

```python
from cryptofeed.backends.kafka.migration import translate_legacy_config

legacy = {
    "bootstrap_servers": ["kafka:9092"],
    "topic_prefix": "trades",
    "partition_strategy": "composite",
    "acks": "all"
}

result = translate_legacy_config(legacy)

# Access modern config
modern = result.modern_config
print(modern.bootstrap_servers)  # ["kafka:9092"]
print(modern.topic.prefix)       # "trades"

# Check unmapped options
if result.unmapped_options:
    print("Unmapped:", result.unmapped_options)

# Check warnings
for warning in result.warnings:
    print("Warning:", warning)
```

---

### validate_migration()

**Module:** `cryptofeed.backends.kafka.migration`

**Description:** Validate migration by comparing legacy config to expected modern config.

**Signature:**

```python
def validate_migration(
    legacy_config: Dict[str, Any],
    expected_modern: KafkaConfig | None = None
) -> MigrationValidationReport
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `legacy_config` | `Dict[str, Any]` | Legacy configuration dictionary |
| `expected_modern` | `KafkaConfig \| None` | Expected modern config (optional) |

**Returns:** `MigrationValidationReport`

**Example:**

```python
from cryptofeed.backends.kafka.migration import validate_migration
from cryptofeed.backends.kafka.callback import KafkaConfig, KafkaTopicConfig

legacy = {
    "bootstrap_servers": ["kafka:9092"],
    "topic_prefix": "crypto",
    "partition_strategy": "symbol"
}

# Define expected modern config
expected = KafkaConfig(
    bootstrap_servers=["kafka:9092"],
    topic=KafkaTopicConfig(prefix="crypto"),
    partition={"strategy": "symbol"}
)

# Validate
report = validate_migration(legacy, expected)

if report.is_equivalent:
    print("✅ Migration equivalent")
else:
    print("❌ Differences found:")
    for field, legacy_val, modern_val in report.differences:
        print(f"  {field}: {legacy_val} → {modern_val}")
```

---

### MigrationResult

**Module:** `cryptofeed.backends.kafka.migration`

**Description:** Result of configuration translation.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `modern_config` | `KafkaConfig` | Translated modern configuration |
| `unmapped_options` | `Dict[str, Any]` | Options without modern equivalents |
| `warnings` | `List[str]` | Warning messages |

**Example:**

```python
result = translate_legacy_config(legacy_config)

# Access fields
assert isinstance(result.modern_config, KafkaConfig)
assert isinstance(result.unmapped_options, dict)
assert isinstance(result.warnings, list)
```

---

### MigrationValidationReport

**Module:** `cryptofeed.backends.kafka.migration`

**Description:** Validation report for migration comparison.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `is_equivalent` | `bool` | True if configs are functionally equivalent |
| `differences` | `List[Tuple[str, Any, Any]]` | List of (field, legacy_val, modern_val) |
| `unmapped_options` | `Dict[str, Any]` | Unmapped legacy options |
| `warnings` | `List[str]` | Warning messages |

**Example:**

```python
report = validate_migration(legacy, expected)

if not report.is_equivalent:
    for field, old, new in report.differences:
        print(f"{field}: {old} → {new}")
```

---

## Health Check System

### KafkaHealthCheck

**Module:** `cryptofeed.backends.kafka.health`

**Description:** Connectivity validation for Kafka backends.

**Methods:**

#### check_modern()

**Signature:**

```python
@staticmethod
def check_modern(
    config: KafkaConfig,
    *,
    producer_factory: Callable[[Dict[str, Any]], Producer] | None = None,
    timeout_ms: int = 3000
) -> KafkaHealthStatus
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `KafkaConfig` | Required | Modern configuration |
| `producer_factory` | `Callable \| None` | `None` | Custom producer factory (testing) |
| `timeout_ms` | `int` | `3000` | Connection timeout (ms) |

**Returns:** `KafkaHealthStatus`

**Example:**

```python
from cryptofeed.backends.kafka.health import KafkaHealthCheck
from cryptofeed.backends.kafka.callback import KafkaConfig

config = KafkaConfig(bootstrap_servers=["kafka:9092"])
status = KafkaHealthCheck.check_modern(config)

if status.ok:
    print(f"✅ Healthy ({status.latency_ms:.2f}ms)")
else:
    print(f"❌ Unhealthy: {status.error}")
```

---

#### check_legacy()

**Signature:**

```python
@staticmethod
def check_legacy(
    legacy_config: Dict[str, Any],
    *,
    producer_factory: Callable[[Dict[str, Any]], Producer] | None = None,
    timeout_ms: int = 3000
) -> KafkaHealthStatus
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `legacy_config` | `Dict[str, Any]` | Required | Legacy configuration dict |
| `producer_factory` | `Callable \| None` | `None` | Custom producer factory (testing) |
| `timeout_ms` | `int` | `3000` | Connection timeout (ms) |

**Returns:** `KafkaHealthStatus`

**Example:**

```python
legacy = {"bootstrap_servers": ["kafka:9092"]}
status = KafkaHealthCheck.check_legacy(legacy)
```

---

### KafkaHealthStatus

**Module:** `cryptofeed.backends.kafka.health`

**Description:** Health check result.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `implementation` | `str` | `'modern'` or `'legacy'` |
| `ok` | `bool` | True if healthy |
| `latency_ms` | `float` | Connection latency (ms) |
| `error` | `str \| None` | Error message if unhealthy |
| `details` | `Dict[str, Any] \| None` | Additional details |

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `as_dict()` | `def as_dict() -> Dict[str, Any]` | Convert to dictionary |

**Example:**

```python
status = KafkaHealthCheck.check_modern(config)

print(status.implementation)  # 'modern'
print(status.ok)             # True/False
print(status.latency_ms)     # 15.3
print(status.error)          # None or error message
print(status.details)        # {'bootstrap': ['kafka:9092']}

# Convert to dict
data = status.as_dict()
# {'implementation': 'modern', 'ok': True, ...}
```

---

### start_periodic_health_checks()

**Module:** `cryptofeed.backends.kafka.health`

**Description:** Start periodic health monitoring loop.

**Signature:**

```python
async def start_periodic_health_checks(
    interval_sec: float,
    check_fn: Callable[[], KafkaHealthStatus],
    on_result: Callable[[KafkaHealthStatus], Any] | None = None,
    *,
    max_runs: int | None = None,
    alert_fn: Callable[[KafkaHealthStatus], Any] | None = None,
    alert_threshold_ms: float = 500.0
) -> asyncio.Task
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `interval_sec` | `float` | Required | Interval between checks (seconds) |
| `check_fn` | `Callable` | Required | Function returning `KafkaHealthStatus` |
| `on_result` | `Callable \| None` | `None` | Callback for each result |
| `max_runs` | `int \| None` | `None` | Max iterations (None = infinite) |
| `alert_fn` | `Callable \| None` | `None` | Alert callback |
| `alert_threshold_ms` | `float` | `500.0` | Alert if latency > threshold (ms) |

**Returns:** `asyncio.Task`

**Example:**

```python
import asyncio
from cryptofeed.backends.kafka.health import (
    start_periodic_health_checks,
    KafkaHealthCheck
)

async def monitor():
    def check():
        return KafkaHealthCheck.check_modern(config)

    def alert(status):
        if not status.ok:
            print(f"🚨 ALERT: {status.error}")

    task = await start_periodic_health_checks(
        interval_sec=60,
        check_fn=check,
        alert_fn=alert,
        alert_threshold_ms=500.0
    )

    await task

asyncio.run(monitor())
```

---

## Maintenance Utilities

### DeprecationWarningSystem

**Module:** `cryptofeed.backends.kafka.maintenance`

**Description:** Centralized deprecation warning management (singleton).

**Constructor:**

```python
DeprecationWarningSystem()
```

**Methods:**

#### emit_class_warning()

**Signature:**

```python
def emit_class_warning(
    self,
    class_name: str,
    replacement: str,
    stacklevel: int
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `class_name` | `str` | Deprecated class name |
| `replacement` | `str` | Replacement class/path |
| `stacklevel` | `int` | Stack level for warning |

**Example:**

```python
from cryptofeed.backends.kafka.maintenance import DeprecationWarningSystem

system = DeprecationWarningSystem()
system.emit_class_warning(
    "TradeKafka",
    "cryptofeed.backends.kafka.callback.KafkaCallback",
    stacklevel=2
)
```

---

#### track_usage()

**Signature:**

```python
def track_usage(
    self,
    component: str,
    context: Dict[str, Any]
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `component` | `str` | Component name |
| `context` | `Dict[str, Any]` | Additional context |

**Example:**

```python
system.track_usage("TradeKafka", {
    "exchange": "binance",
    "symbol": "BTC-USDT"
})
```

---

#### get_usage_stats()

**Signature:**

```python
def get_usage_stats(self) -> Dict[str, int]
```

**Returns:** Dictionary mapping component names to usage counts.

**Example:**

```python
stats = system.get_usage_stats()
# {'TradeKafka': 42, 'BookKafka': 13}
```

---

#### get_usage_report()

**Signature:**

```python
def get_usage_report(self) -> Dict[str, Dict[str, Any]]
```

**Returns:** Detailed usage report with counts, timestamps, and context.

**Example:**

```python
report = system.get_usage_report()
# {
#   'TradeKafka': {
#     'count': 42,
#     'last_timestamp': 1234567890.0,
#     'last_context': {'exchange': 'binance', ...}
#   }
# }
```

---

### Convenience Functions

#### emit_class_deprecation_warning()

**Module:** `cryptofeed.backends.kafka.maintenance`

**Signature:**

```python
def emit_class_deprecation_warning(
    class_name: str,
    replacement: str
) -> None
```

**Example:**

```python
from cryptofeed.backends.kafka.maintenance import emit_class_deprecation_warning

emit_class_deprecation_warning(
    "TradeKafka",
    "cryptofeed.backends.kafka.callback.KafkaCallback"
)
```

---

#### emit_import_deprecation_warning()

**Module:** `cryptofeed.backends.kafka.maintenance`

**Signature:**

```python
def emit_import_deprecation_warning(
    old_path: str,
    new_path: str
) -> None
```

**Example:**

```python
from cryptofeed.backends.kafka.maintenance import emit_import_deprecation_warning

emit_import_deprecation_warning(
    "cryptofeed.kafka_callback",
    "cryptofeed.backends.kafka.callback"
)
```

---

## Legacy Backend (Deprecated)

### KafkaCallback (Legacy)

**Module:** `cryptofeed.backends.kafka` (deprecated)

**Description:** Legacy Kafka backend using `aiokafka` (DEPRECATED).

⚠️ **DEPRECATION WARNING:** This class is deprecated and will be removed in Q2 2026. Use `cryptofeed.backends.kafka.callback.KafkaCallback` instead.

**Constructor:**

```python
KafkaCallback(
    key=None,
    serialization_format=None,
    numeric_type=float,
    none_to=None,
    **kwargs
)
```

**Example:**

```python
# ⚠️ DEPRECATED - DO NOT USE
from cryptofeed.backends.kafka import KafkaCallback

callback = KafkaCallback(
    bootstrap_servers=['kafka:9092'],
    topic='trades',
    acks=1
)

# ✅ Use modern implementation instead
from cryptofeed.backends.kafka.callback import KafkaCallback, KafkaConfig

config = KafkaConfig(bootstrap_servers=['kafka:9092'])
callback = KafkaCallback(config=config, key=None)
```

---

## Summary

**Configuration:**
- `KafkaConfig`: Main configuration model
- `KafkaTopicConfig`: Topic management
- `KafkaPartitionConfig`: Partition strategies

**Callbacks:**
- `KafkaCallback`: JSON serialization
- `KafkaProtobufCallback`: Protobuf serialization

**Migration:**
- `translate_legacy_config()`: Configuration translation
- `validate_migration()`: Equivalence validation
- `MigrationResult`: Translation result
- `MigrationValidationReport`: Validation report

**Health Checks:**
- `KafkaHealthCheck`: Connectivity validation
- `KafkaHealthStatus`: Health check result
- `start_periodic_health_checks()`: Periodic monitoring

**Maintenance:**
- `DeprecationWarningSystem`: Warning management
- `emit_class_deprecation_warning()`: Class warnings
- `emit_import_deprecation_warning()`: Import warnings

---

**API Reference Version:** 1.0.0
**Last Updated:** 2025-11-26
**Status:** Production Ready
