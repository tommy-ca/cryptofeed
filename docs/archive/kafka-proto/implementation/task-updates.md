# Protobuf Callback Serialization - Task Update Plan

## Executive Summary

This document provides the detailed plan for updating `.kiro/specs/protobuf-callback-serialization/tasks.md` to align with the enhanced requirements (requirements.md updated 2025-10-31).

**Status**: Ready to apply changes
**Requirements Version**: Enhanced (92 acceptance criteria, up from 60)
**Current Tasks**: 10 tasks, 24-33 days
**Updated Tasks**: 14 tasks, 28-37 days

---

## Changes Overview

### Tasks to Add (5 new)
1. **Task 1.0**: Exception Classes (pre-foundation)
2. **Task 1.3.0**: Config Loading
3. **Task 1.3.1**: Kafka Topic Routing
4. **Task 1.3.2**: Redis/ZMQ Support (optional)
5. **Task 1.6.1**: Wrapper Adapter Layer
6. **Task 1.11**: User Documentation

### Tasks to Split (1)
- **Task 1.3** → Split into **1.3.0, 1.3.1, 1.3.2, 1.3.3**

### Tasks to Enhance (1)
- **Task 1.9**: Add baseline metrics, profiling, regression tracking

### Tasks Unchanged (7)
- 1.1, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.10

---

## Detailed Task Changes

### ✨ NEW: Task 1.0 - Define Exception Classes

**Insert Before**: Task 1.1 (Prerequisites phase)

**Content**:
```markdown
## Phase 0: Prerequisites (NEW)

### Task 1.0: Define Custom Exception Classes

**Estimate**: XS (Extra Small) - 0.5 day
**Dependencies**: None
**Blocks**: Task 1.5 (ProtobufSerializer uses these exceptions)

**Objective**: Create custom exception hierarchy for serialization errors with clear, actionable error messages.

**Files to Create**:
- `cryptofeed/exceptions.py` - Custom exception classes
- `tests/unit/test_exceptions.py` - Exception tests

**Acceptance Criteria** (from R4.5):

```gherkin
GIVEN CryptofeedSerializationException base class
WHEN SerializationError is raised
THEN it inherits from CryptofeedSerializationException

GIVEN SerializationError with data type context
WHEN exception message is generated
THEN message includes: "{TypeName} missing to_proto() method. Ensure all data types implement to_proto()."

GIVEN ProtobufEncodeError with protobuf failure
WHEN exception is raised
THEN message includes underlying protobuf error and data type context

GIVEN exception with original cause
WHEN raised with "from" clause
THEN exception chain is preserved (no suppressed exceptions)

GIVEN ProtobufEncodeError
WHEN error message is generated
THEN message includes protobuf schema name and version

GIVEN invalid serialization format "invalid"
WHEN ValueError is raised
THEN message is: "Invalid serialization format 'invalid'. Valid formats: json, protobuf"
```

**Implementation**:

```python
# cryptofeed/exceptions.py

class CryptofeedSerializationException(Exception):
    """Base class for all serialization-related exceptions."""
    pass


class SerializationError(CryptofeedSerializationException):
    """Raised when serialization fails due to missing methods or invalid data."""
    
    def __init__(self, message: str, data_type: str = None):
        self.data_type = data_type
        if data_type:
            message = f"{data_type}: {message}"
        super().__init__(message)


class ProtobufEncodeError(CryptofeedSerializationException):
    """Raised when protobuf encoding fails."""
    
    def __init__(self, message: str, data_type: str = None, schema_name: str = None, schema_version: str = None):
        self.data_type = data_type
        self.schema_name = schema_name
        self.schema_version = schema_version
        
        details = []
        if data_type:
            details.append(f"data_type={data_type}")
        if schema_name:
            details.append(f"schema={schema_name}")
        if schema_version:
            details.append(f"version={schema_version}")
        
        if details:
            message = f"{message} ({', '.join(details)})"
        
        super().__init__(message)
```

**Test Specifications** (TDD):

```python
# tests/unit/test_exceptions.py

def test_cryptofeed_serialization_exception_base():
    """Base exception can be caught."""
    with pytest.raises(CryptofeedSerializationException):
        raise SerializationError("test")

def test_serialization_error_with_data_type():
    """SerializationError includes data type in message."""
    error = SerializationError("missing to_proto()", data_type="Trade")
    assert "Trade" in str(error)
    assert "missing to_proto()" in str(error)

def test_protobuf_encode_error_with_context():
    """ProtobufEncodeError includes schema context."""
    error = ProtobufEncodeError(
        "encoding failed",
        data_type="Trade",
        schema_name="trade_pb2.Trade",
        schema_version="v0.1.0"
    )
    assert "Trade" in str(error)
    assert "trade_pb2.Trade" in str(error)
    assert "v0.1.0" in str(error)

def test_exception_chain_preserved():
    """Exception chains are preserved with 'from' clause."""
    original = ValueError("original error")
    try:
        try:
            raise original
        except ValueError as e:
            raise SerializationError("wrapped") from e
    except SerializationError as e:
        assert e.__cause__ is original
        assert e.__suppress_context__ is False
```

**Success Verification**:
```bash
pytest tests/unit/test_exceptions.py -v --cov=cryptofeed.exceptions
# Expected: 4/4 tests passing, 100% coverage
```

**Engineering Principles**:
- ✅ **Single Responsibility**: Each exception class handles one error category
- ✅ **Open/Closed**: Base class extensible for future exception types
- ✅ **Liskov Substitution**: All exceptions substitutable for base class
- ✅ **Interface Segregation**: Minimal exception interface
- ✅ **KISS**: Simple exception hierarchy, clear messages
- ✅ **TDD**: Tests written first, implementation follows

---
```

### ✂️ SPLIT: Task 1.3 → 1.3.0, 1.3.1, 1.3.2, 1.3.3

**Current Task 1.3**: "Integrate BackendCallback with Serialization Support" (2-3 days, monolithic)

**Split Into**:

#### **Task 1.3.0: Configuration Loading and Validation**

**Estimate**: S (Small) - 1 day
**Dependencies**: Task 1.0 (exception classes)
**Blocks**: Task 1.3.3

**Objective**: Implement configuration parsing for `serialization_format` parameter with YAML and environment variable support.

**Files to Create/Modify**:
- `cryptofeed/config.py` - Config parser (if not exists)
- Modify backend factory methods to accept `serialization_format`
- `tests/unit/test_config_serialization.py` - Config tests

**Acceptance Criteria** (from R3.7-3.10):

```gherkin
GIVEN YAML config with serialization_format: protobuf
WHEN config is parsed
THEN parser validates format and accepts 'protobuf' or 'json'

GIVEN YAML config with serialization_format: invalid
WHEN config is parsed
THEN parser raises ValueError with message: "Invalid serialization format 'invalid'. Valid formats: json, protobuf"

GIVEN environment variable CRYPTOFEED_CALLBACK_FORMAT=PROTOBUF
WHEN config is loaded
THEN value is normalized to lowercase 'protobuf'

GIVEN environment variable and YAML both specify format
WHEN config is loaded
THEN environment variable takes precedence

GIVEN programmatic API with serialization_format='Protobuf'
WHEN callback is instantiated
THEN value is normalized to 'protobuf'
```

**Implementation Pattern**:
```python
# cryptofeed/config.py or backends/backend.py

VALID_SERIALIZATION_FORMATS = {'json', 'protobuf'}

def validate_serialization_format(format_str: str) -> str:
    """Validate and normalize serialization format.
    
    Args:
        format_str: Format string (case-insensitive)
        
    Returns:
        Normalized format string (lowercase)
        
    Raises:
        ValueError: If format is invalid
    """
    if format_str is None:
        return 'json'  # Default
    
    normalized = format_str.lower()
    if normalized not in VALID_SERIALIZATION_FORMATS:
        raise ValueError(
            f"Invalid serialization format '{format_str}'. "
            f"Valid formats: {', '.join(sorted(VALID_SERIALIZATION_FORMATS))}"
        )
    
    return normalized

def get_serialization_format_from_env() -> str | None:
    """Get serialization format from environment variable."""
    env_value = os.getenv('CRYPTOFEED_CALLBACK_FORMAT')
    if env_value:
        return validate_serialization_format(env_value)
    return None
```

**Test Specifications**:
```python
def test_config_valid_formats():
    """Valid formats are accepted."""
    assert validate_serialization_format('json') == 'json'
    assert validate_serialization_format('protobuf') == 'protobuf'

def test_config_case_insensitive():
    """Format is case-insensitive."""
    assert validate_serialization_format('Protobuf') == 'protobuf'
    assert validate_serialization_format('PROTOBUF') == 'protobuf'
    assert validate_serialization_format('Json') == 'json'

def test_config_invalid_format():
    """Invalid format raises ValueError."""
    with pytest.raises(ValueError, match="Invalid serialization format"):
        validate_serialization_format('avro')

def test_env_var_precedence():
    """Environment variable overrides YAML."""
    os.environ['CRYPTOFEED_CALLBACK_FORMAT'] = 'protobuf'
    # ... test config loading logic
    assert loaded_format == 'protobuf'
```

---

#### **Task 1.3.1: Kafka Topic Routing Implementation**

**Estimate**: S (Small) - 1 day
**Dependencies**: Task 1.3.0
**Blocks**: Task 1.10 (E2E testing)

**Objective**: Implement Kafka-specific topic routing and partition key strategy for protobuf format.

**Files to Modify**:
- `cryptofeed/backends/kafka.py` - Override `topic()` and `partition_key()` methods

**Acceptance Criteria** (from R4.10-4.14):

```gherkin
GIVEN KafkaCallback with serialization_format="protobuf"
WHEN topic() method is called with Trade data
THEN returns "cryptofeed.market.trades.{exchange}"

GIVEN KafkaCallback with serialization_format="protobuf"
WHEN partition_key() method is called with Trade data
THEN returns normalized symbol as UTF-8 bytes (e.g., b'BTC-USD')

GIVEN KafkaCallback with serialization_format="json"
WHEN topic() method is called
THEN returns legacy topic format "trades-{exchange}-{symbol}" (backward compat)

GIVEN multiple exchanges (coinbase, binance) with protobuf
WHEN messages are produced
THEN each exchange writes to separate topic

GIVEN OrderBook data with protobuf format
WHEN topic() is called
THEN returns "cryptofeed.market.orderbook.{exchange}"
```

**Implementation**:
```python
# cryptofeed/backends/kafka.py

class KafkaCallback(BackendQueue):
    def __init__(self, serialization_format='json', **kwargs):
        self.serialization_format = validate_serialization_format(serialization_format)
        # ... existing init
    
    def topic(self, data: dict) -> str:
        """Generate topic name based on serialization format."""
        if self.serialization_format == 'protobuf':
            # Hierarchical: cryptofeed.market.{data_type}.{exchange}
            data_type = self.default_key  # e.g., 'trades', 'orderbook'
            exchange = data['exchange']
            return f"cryptofeed.market.{data_type}.{exchange}"
        else:
            # Legacy: {data_type}-{exchange}-{symbol}
            return f"{self.key}-{data['exchange']}-{data['symbol']}"
    
    def partition_key(self, data: dict) -> bytes | None:
        """Generate partition key for consistent symbol routing."""
        if self.serialization_format == 'protobuf':
            # Partition by symbol for ordered processing
            symbol = data['symbol']
            return symbol.encode('utf-8')
        else:
            # Legacy behavior (None = round-robin)
            return None
```

**Test Specifications**:
```python
def test_kafka_protobuf_topic_routing():
    """Protobuf uses hierarchical topic naming."""
    callback = TradeKafka(serialization_format='protobuf')
    data = {'exchange': 'coinbase', 'symbol': 'BTC-USD'}
    assert callback.topic(data) == 'cryptofeed.market.trades.coinbase'

def test_kafka_json_topic_routing():
    """JSON uses legacy topic naming."""
    callback = TradeKafka(serialization_format='json')
    data = {'exchange': 'coinbase', 'symbol': 'BTC-USD'}
    assert callback.topic(data) == 'trades-coinbase-BTC-USD'

def test_kafka_partition_key_protobuf():
    """Protobuf uses symbol-based partitioning."""
    callback = TradeKafka(serialization_format='protobuf')
    data = {'symbol': 'BTC-USD'}
    assert callback.partition_key(data) == b'BTC-USD'

def test_kafka_partition_key_json():
    """JSON uses legacy partitioning (None)."""
    callback = TradeKafka(serialization_format='json')
    data = {'symbol': 'BTC-USD'}
    assert callback.partition_key(data) is None
```

---

#### **Task 1.3.2: Redis/ZMQ Protobuf Support (OPTIONAL)**

**Estimate**: S (Small) - 1 day (OPTIONAL - can defer to v2)
**Dependencies**: Task 1.3.1
**Blocks**: None

**Objective**: Extend Redis and ZMQ callbacks to support binary protobuf payloads.

**Scope**: In requirements but LOW priority (Kafka is primary)

**Recommendation**: DEFER to v2 unless Redis/ZMQ are critical use cases

---

#### **Task 1.3.3: BackendCallback Format Selection**

**Estimate**: M (Medium) - 1-2 days
**Dependencies**: Tasks 1.0, 1.1, 1.2, 1.3.0
**Blocks**: Tasks 1.4-1.8

**Objective**: Integrate serializer selection into BackendCallback base class.

**Files to Modify**:
- `cryptofeed/backends/backend.py` - Add `serialization_format` parameter

**Acceptance Criteria** (from R2):

```gherkin
GIVEN BackendCallback with no serialization_format parameter
WHEN callback is instantiated
THEN defaults to 'json' format

GIVEN BackendCallback with serialization_format='protobuf'
WHEN data is written
THEN ProtobufSerializer is used

GIVEN BackendCallback with serialization_format='json'
WHEN data is written
THEN JSONSerializer is used

GIVEN both JSON and Protobuf callbacks in same FeedHandler
WHEN messages are processed
THEN both operate independently without interference
```

**Implementation**:
```python
# cryptofeed/backends/backend.py

class BackendCallback:
    def __init__(self, serialization_format='json', numeric_type=float, none_to=None, **kwargs):
        self.serialization_format = validate_serialization_format(serialization_format)
        self.serializer = self._get_serializer()
        self.numeric_type = numeric_type
        self.none_to = none_to
    
    def _get_serializer(self) -> Serializer:
        """Factory method for serializer selection."""
        if self.serialization_format == 'protobuf':
            from cryptofeed.serializers.protobuf import ProtobufSerializer
            return ProtobufSerializer()
        elif self.serialization_format == 'json':
            from cryptofeed.serializers.json import JSONSerializer
            return JSONSerializer()
        else:
            # Should never reach here due to validation
            raise ValueError(f"Unsupported format: {self.serialization_format}")
    
    async def __call__(self, dtype, receipt_timestamp: float):
        """Serialize and write data."""
        # Existing to_dict() logic for JSON compatibility
        data = dtype.to_dict(numeric_type=self.numeric_type, none_to=self.none_to)
        if not dtype.timestamp:
            data['timestamp'] = receipt_timestamp
        data['receipt_timestamp'] = receipt_timestamp
        
        # Serialize using selected serializer
        serialized = self.serializer.serialize(data)
        await self.write(serialized)
```

---

### ✨ NEW: Task 1.6.1 - Wrapper Adapter Layer

**Insert After**: Task 1.6 (before Task 1.7)

**Content**:
```markdown
### Task 1.6.1: Implement Wrapper Adapter Layer

**Estimate**: S (Small) - 1 day
**Dependencies**: Task 1.6 (Trade/OrderBook wrappers implemented)
**Blocks**: Task 1.9 (Performance benchmarking needs adapter)

**Objective**: Create adapter function that wraps C extension objects in Python wrapper classes for protobuf serialization.

**Files to Create**:
- `cryptofeed/proto_adapters/adapter.py` - Main adapter function
- `tests/unit/proto_adapters/test_adapter.py` - Adapter tests

**Acceptance Criteria** (from R7.5):

```gherkin
GIVEN C extension Trade object
WHEN wrap_for_serialization(trade) is called
THEN returns TradeWrapper instance with to_proto() method

GIVEN C extension OrderBook object
WHEN wrap_for_serialization(book) is called
THEN returns OrderBookWrapper instance

GIVEN unsupported data type (e.g., str)
WHEN wrap_for_serialization(obj) is called
THEN raises SerializationError: "No protobuf wrapper available for {TypeName}"

GIVEN wrapper adapter processes 10,000 messages
WHEN overhead is measured
THEN adds <100 microseconds per message vs direct serialization

GIVEN adapter with no internal state
WHEN multiple messages processed
THEN each invocation is independent (stateless)
```

**Implementation**:

```python
# cryptofeed/proto_adapters/adapter.py

from cryptofeed.types import Trade, OrderBook, Ticker, Candle, Funding, Liquidation
# ... import other types

from .trade import TradeWrapper
from .orderbook import OrderBookWrapper
from .ticker import TickerWrapper
# ... import other wrappers

from cryptofeed.exceptions import SerializationError

# Type registry for adapter routing
_WRAPPER_REGISTRY = {
    Trade: TradeWrapper,
    OrderBook: OrderBookWrapper,
    Ticker: TickerWrapper,
    Candle: CandleWrapper,
    Funding: FundingWrapper,
    Liquidation: LiquidationWrapper,
    # ... add remaining types
}

def wrap_for_serialization(obj):
    """Wrap C extension object in Python wrapper for protobuf serialization.
    
    Args:
        obj: C extension data type instance
        
    Returns:
        Wrapper instance with to_proto() method
        
    Raises:
        SerializationError: If no wrapper available for type
    """
    obj_type = type(obj)
    wrapper_class = _WRAPPER_REGISTRY.get(obj_type)
    
    if wrapper_class is None:
        raise SerializationError(
            f"No protobuf wrapper available for {obj_type.__name__}. "
            f"Supported types: {', '.join(t.__name__ for t in _WRAPPER_REGISTRY.keys())}"
        )
    
    return wrapper_class(obj)


def register_wrapper(data_type, wrapper_class):
    """Register a new wrapper class for a data type.
    
    Enables extension without modifying adapter.py.
    """
    _WRAPPER_REGISTRY[data_type] = wrapper_class
```

**Test Specifications**:

```python
def test_adapter_trade_wrapper():
    """Adapter wraps Trade in TradeWrapper."""
    trade = Trade(...)  # C extension object
    wrapped = wrap_for_serialization(trade)
    assert isinstance(wrapped, TradeWrapper)
    assert hasattr(wrapped, 'to_proto')

def test_adapter_unsupported_type():
    """Adapter raises error for unsupported types."""
    with pytest.raises(SerializationError, match="No protobuf wrapper available"):
        wrap_for_serialization("not a data type")

def test_adapter_stateless():
    """Adapter has no internal state."""
    trade1 = Trade(...)
    trade2 = Trade(...)
    
    wrapped1 = wrap_for_serialization(trade1)
    wrapped2 = wrap_for_serialization(trade2)
    
    # Each wrapping is independent
    assert wrapped1 is not wrapped2
    assert wrapped1._trade is trade1
    assert wrapped2._trade is trade2

def test_adapter_performance_overhead():
    """Adapter adds <100µs overhead per message."""
    trade = Trade(...)
    
    # Measure wrapping overhead
    start = time.perf_counter()
    for _ in range(10000):
        wrapped = wrap_for_serialization(trade)
    elapsed = time.perf_counter() - start
    
    avg_overhead_us = (elapsed / 10000) * 1_000_000
    assert avg_overhead_us < 100, f"Overhead {avg_overhead_us:.2f}µs exceeds 100µs"
```

**Integration with BackendCallback**:

```python
# Update Task 1.3.3 implementation to use adapter

async def __call__(self, dtype, receipt_timestamp: float):
    """Serialize and write data."""
    if self.serialization_format == 'protobuf':
        # Wrap C extension object
        from cryptofeed.proto_adapters.adapter import wrap_for_serialization
        dtype = wrap_for_serialization(dtype)
    
    # Serialize
    serialized = self.serializer.serialize(dtype)
    await self.write(serialized)
```

**Success Verification**:
```bash
pytest tests/unit/proto_adapters/test_adapter.py -v --cov=cryptofeed.proto_adapters.adapter
# Expected: 4/4 tests passing, 100% coverage
```

---
```

### 🔧 ENHANCE: Task 1.9 - Performance Benchmarking

**Current**: Generic benchmarking (2-3 days)

**Enhanced** (3-4 days):

**Add to Acceptance Criteria**:

From R8:
- Baseline dataset definition (10k Trade, 1k OrderBook, mixed workload)
- Percentile targets (p50, p95, p99) for each data type
- Throughput requirements (≥10k msg/s Trade, ≥5k msg/s mixed)
- Memory stability (< 5% growth after 1M messages)
- Profiling requirements (cProfile hot paths)
- Performance baseline documentation

**Add to Implementation**:

```python
# tests/benchmarks/test_serialization_performance.py

import cProfile
import pstats
from memory_profiler import profile

# Baseline datasets
TRADE_WORKLOAD = [create_trade(...) for _ in range(10000)]
ORDERBOOK_WORKLOAD = [create_orderbook(...) for _ in range(1000)]
MIXED_WORKLOAD = (
    [create_trade(...) for _ in range(7000)] +
    [create_orderbook(...) for _ in range(2000)] +
    [create_ticker(...) for _ in range(1000)]
)

def test_latency_percentiles():
    """Measure p50/p95/p99 latency for Trade serialization."""
    serializer = ProtobufSerializer()
    latencies = []
    
    for trade in TRADE_WORKLOAD:
        start = time.perf_counter()
        serializer.serialize(trade)
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    assert p50 < 0.3, f"p50 {p50:.3f}ms exceeds 0.3ms"
    assert p95 < 0.6, f"p95 {p95:.3f}ms exceeds 0.6ms"
    assert p99 < 1.0, f"p99 {p99:.3f}ms exceeds 1.0ms"

def test_throughput_trade_workload():
    """Measure throughput for Trade workload."""
    serializer = ProtobufSerializer()
    
    start = time.perf_counter()
    for trade in TRADE_WORKLOAD:
        serializer.serialize(trade)
    elapsed = time.perf_counter() - start
    
    throughput = len(TRADE_WORKLOAD) / elapsed
    assert throughput >= 10000, f"Throughput {throughput:.0f} msg/s < 10k"

@profile
def test_memory_stability():
    """Verify no memory leaks after 1M messages."""
    serializer = ProtobufSerializer()
    
    # Measure baseline
    baseline_memory = get_memory_usage()
    
    # Serialize 1M messages
    for _ in range(100):
        for trade in TRADE_WORKLOAD:  # 10k * 100 = 1M
            serializer.serialize(trade)
    
    # Measure final
    final_memory = get_memory_usage()
    growth_pct = ((final_memory - baseline_memory) / baseline_memory) * 100
    
    assert growth_pct < 5, f"Memory growth {growth_pct:.1f}% exceeds 5%"

def test_profile_hot_paths():
    """Profile serialization to identify hot paths."""
    serializer = ProtobufSerializer()
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    for trade in TRADE_WORKLOAD:
        serializer.serialize(trade)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime')
    
    # Identify functions consuming >5% of time
    # Document in performance baseline
```

**Add to Deliverables**:
- `docs/performance-baseline.md` with metrics and profiling results
- CI integration for regression detection

---

### ✨ NEW: Task 1.11 - User Documentation and Examples

**Insert After**: Task 1.10

**Content**:
```markdown
### Task 1.11: User Documentation and Integration Examples

**Estimate**: M (Medium) - 2-3 days
**Dependencies**: Tasks 1.8 (all wrappers), 1.10 (E2E testing)
**Can Parallelize With**: Task 1.10

**Objective**: Create comprehensive user documentation and integration examples for protobuf serialization.

**Files to Create**:
- `docs/protobuf-serialization-user-guide.md` - User guide
- `docs/consumer-integration-guide.md` - Consumer examples
- `examples/kafka_protobuf_producer.py` - Producer example
- `examples/kafka_protobuf_consumer.py` - Consumer example
- `examples/dual_format_comparison.py` - JSON + Protobuf side-by-side

**Acceptance Criteria** (from R9):

```gherkin
GIVEN user guide documentation
WHEN operator wants to configure protobuf
THEN guide provides complete YAML configuration example

GIVEN user guide documentation
WHEN developer wants programmatic API
THEN guide provides complete Python code example

GIVEN user guide documentation
WHEN operator reads topic naming section
THEN pattern "cryptofeed.market.{data_type}.{exchange}" is clearly documented

GIVEN consumer integration guide
WHEN consumer wants to deserialize protobuf
THEN guide provides Python snippet with protobuf bindings

GIVEN migration guide section
WHEN operator wants to migrate from JSON
THEN guide demonstrates running both formats in parallel

GIVEN troubleshooting section
WHEN operator encounters serialization error
THEN guide lists common issues and resolutions
```

**Documentation Structure**:

**1. User Guide** (`docs/protobuf-serialization-user-guide.md`):
- Introduction and benefits (50-60% size reduction)
- Configuration (YAML + env vars + programmatic)
- Kafka topic naming conventions
- Migration guide (JSON → Protobuf)
- Troubleshooting (common errors, solutions)

**2. Consumer Integration Guide** (`docs/consumer-integration-guide.md`):
- Python consumer example (Kafka + protobuf deserialization)
- Flink consumer reference (Java)
- DuckDB consumer reference (SQL)
- Schema registry integration (Buf/Confluent)

**3. Code Examples**:
- Producer: Complete example with protobuf format
- Consumer: Deserialization and processing
- Dual format: JSON + Protobuf comparison

**Example Content Snippets**:

```yaml
# User Guide - YAML Configuration Example
backends:
  kafka_protobuf:
    type: kafka
    serialization_format: protobuf  # Enable protobuf
    bootstrap_servers: localhost:9092
    
feeds:
  binance_trades:
    exchange: binance
    symbols: [BTC-USDT, ETH-USDT]
    channels: [trades]
    callbacks:
      trades: [kafka_protobuf]
```

```python
# Consumer Integration Guide - Python Consumer
from kafka import KafkaConsumer
from cryptofeed_protobuf.normalized.v1 import trade_pb2

consumer = KafkaConsumer(
    'cryptofeed.market.trades.coinbase',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: trade_pb2.Trade().ParseFromString(m)
)

for msg in consumer:
    trade = msg.value
    print(f"{trade.symbol}: {trade.price} @ {trade.timestamp_us}")
```

**Success Verification**:
- Documentation complete and reviewed
- All code examples run successfully
- Troubleshooting guide covers common errors
- Consumer examples tested with real Kafka topics

---
```

---

## Updated Task Summary

| ID | Phase | Task | Est. | Days | Dependencies | Can Parallelize |
|----|-------|------|------|------|--------------|-----------------|
| 1.0 | Prerequisites | Exception Classes | XS | 0.5 | None | N/A |
| 1.1 | Foundation | Serializer ABC | S | 1-2 | 1.0 | No |
| 1.2 | Foundation | JSONSerializer | S | 1-2 | 1.1 | No |
| 1.3.0 | Foundation | Config Loading | S | 1 | 1.0 | No |
| 1.3.1 | Foundation | Kafka Routing | S | 1 | 1.3.0 | No |
| 1.3.2 | Foundation | Redis/ZMQ (optional) | S | 1 | 1.3.1 | Yes (defer to v2) |
| 1.3.3 | Foundation | BackendCallback Integration | M | 1-2 | 1.0-1.3.0 | No |
| 1.4 | Data Integration | Protobuf Bindings | S | 1 | 1.3.3 | No |
| 1.5 | Data Integration | ProtobufSerializer | M | 2-3 | 1.4 | No |
| 1.6 | Data Integration | Trade + OrderBook | M | 3-4 | 1.5 | No |
| 1.6.1 | Data Integration | Wrapper Adapter | S | 1 | 1.6 | No |
| 1.7 | Data Integration | Ticker + Candle + Funding | M | 2-3 | 1.6.1 | Yes (with 1.8) |
| 1.8 | Data Integration | 9 Remaining Types | L | 4-5 | 1.6.1 | Yes (with 1.7) |
| 1.9 | Production | Performance Baseline | M | 3-4 | 1.8 | No |
| 1.10 | Production | Kafka E2E | L | 3-4 | 1.9 | Yes (with 1.11) |
| 1.11 | Production | Documentation | M | 2-3 | 1.8, 1.10 | Yes (with 1.10) |

**New Total**: 14 tasks (was 10), 28-37 days (was 24-33), 3-4 weeks optimized

**Critical Path**: 1.0 → 1.1 → 1.2 → 1.3.0 → 1.3.1 → 1.3.3 → 1.4 → 1.5 → 1.6 → 1.6.1 → 1.7/1.8 → 1.9 → 1.10/1.11

---

## Requirements Coverage Matrix (Updated)

| Requirement | Covered by Task | Status | Gap Resolved |
|-------------|----------------|--------|--------------|
| R1: to_proto() methods | 1.6, 1.7, 1.8 | ✅ Complete | N/A |
| R2: Format selection | 1.3.3 | ✅ Complete | N/A |
| R3: Config & backward compat | 1.2, 1.3.0, 1.3.3 | ✅ Complete | ✅ Config loading explicit |
| R4: Kafka routing | 1.3.1 | ✅ Complete | ✅ Dedicated task added |
| R4.5: Exception handling | 1.0 | ✅ Complete | ✅ Prerequisites task added |
| R5: Schema alignment | 1.4 | ✅ Complete | N/A |
| R6: Type safety | 1.5, 1.6-1.8 | ✅ Complete | N/A |
| R7: Testing coverage | All tasks | ✅ Complete | N/A |
| R7.5: Wrapper adapter | 1.6.1 | ✅ Complete | ✅ Adapter task added |
| R8: Performance baseline | 1.9 | ✅ Complete | ✅ Enhanced with metrics |
| R9: Documentation | 1.11 | ✅ Complete | ✅ Documentation task added |

**Coverage**: 11/11 complete (100%) ✅

---

## Application Steps

1. ✅ Review this plan
2. ⏭️ Insert Task 1.0 before Phase 1
3. ⏭️ Split Task 1.3 into 1.3.0, 1.3.1, 1.3.2, 1.3.3
4. ⏭️ Insert Task 1.6.1 after Task 1.6
5. ⏭️ Enhance Task 1.9 with baseline metrics
6. ⏭️ Add Task 1.11 after Task 1.10
7. ⏭️ Update task summary table
8. ⏭️ Update phase summaries
9. ⏭️ Re-sign off tasks

---

**Ready to Apply**: ✅ All changes specified, requirements 100% covered
