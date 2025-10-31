# Protobuf Callback Serialization - Implementation Tasks (Spec 1)

## Overview

9 implementation tasks for adding protobuf serialization support to cryptofeed's backend callback system. Tasks are organized into 3 phases with sequential dependencies, enabling parallel execution where possible.

**Total Effort**: ~2 weeks (14-16 days)
**Team Size**: 1 engineer (can be parallelized across 2-3 engineers for Phase 2)
**Dependencies**:
- Spec 0 (normalized-data-schema-crypto) must be merged first
- ProtobufSerializer relies on generated proto bindings from Spec 0
- All tasks build sequentially through Phase 1, then can parallelize in Phase 2

---

## Phase Summary

| Phase | Tasks | Effort | Purpose |
|-------|-------|--------|---------|
| 1: Foundation | 1.1-1.3 | 4-5 days | Core serialization abstraction and BackendCallback integration |
| 2: Data Integration | 1.4-1.7 | 12-14 days | to_proto() methods for all 20 data types (parallelize by type) |
| 3: Production | 1.8-1.9 | 6-8 days | Performance optimization, integration testing, hardening |

---

## Phase 1: Foundation (Tasks 1.1-1.3)

All Foundation tasks must complete before Phase 2 begins. No parallelization in Phase 1.

### Task 1.1: Implement Serializer Abstract Base Class

**Estimate**: S (Small) - 1-2 days
**Dependencies**: None
**Blocks**: 1.2, 1.3, all Phase 2 tasks

**Objective**: Create the core serialization abstraction that all serializers implement.

**Files to Create**:
- `cryptofeed/serializers/__init__.py`
- `cryptofeed/serializers/base.py`

**Acceptance Criteria**:

```gherkin
GIVEN the Serializer abstract base class is defined
WHEN a developer attempts to instantiate it directly
THEN it raises TypeError

GIVEN ProtobufSerializer and JSONSerializer subclasses
WHEN they inherit from Serializer
THEN they override both serialize() and content_type() methods

GIVEN serialize() abstract method signature
WHEN subclass implements it
THEN accepts Any type object, returns bytes

GIVEN content_type() abstract method
WHEN called
THEN returns MIME type string (e.g., 'application/x-protobuf')

GIVEN import "from cryptofeed.serializers import Serializer"
WHEN executed
THEN succeeds without errors
```

**Test Specifications (TDD - Write Tests First)**:

```python
# tests/unit/serializers/test_serializer_base.py

def test_serializer_cannot_be_instantiated():
    """Serializer ABC raises TypeError on direct instantiation."""
    with pytest.raises(TypeError):
        Serializer()

def test_serializer_has_abstract_methods():
    """Serializer defines serialize and content_type as abstract."""
    assert hasattr(Serializer, 'serialize')
    assert hasattr(Serializer, 'content_type')
    assert Serializer.serialize.__isabstractmethod__
    assert Serializer.content_type.__isabstractmethod__

def test_incomplete_implementation_fails():
    """Incomplete subclass implementation raises TypeError."""
    class IncompleteSerializer(Serializer):
        def serialize(self, obj):
            return b'test'

    with pytest.raises(TypeError):
        IncompleteSerializer()

def test_complete_implementation_succeeds():
    """Complete subclass implementation instantiates successfully."""
    class ConcreteSerializer(Serializer):
        def serialize(self, obj):
            return b'test'
        def content_type(self):
            return 'text/plain'

    s = ConcreteSerializer()
    assert s.serialize("anything") == b'test'
    assert s.content_type() == 'text/plain'
```

**Engineering Principles**:
- ✅ **Single Responsibility**: Defines contract only, no implementation
- ✅ **Open/Closed**: Open for extension (subclasses), closed for modification
- ✅ **Liskov Substitution**: All serializers substitutable for Serializer
- ✅ **Interface Segregation**: Minimal 2-method interface
- ✅ **Dependency Inversion**: Code depends on Serializer, not concrete classes
- ✅ **KISS**: No complexity, just abstract methods
- ✅ **DRY**: Single contract definition
- ✅ **YAGNI**: No v2 features
- ✅ **TDD**: Tests written first

**Success Verification**:
```bash
pytest tests/unit/serializers/test_serializer_base.py -v --cov=cryptofeed.serializers.base
# Expected: 4/4 tests passing, 100% coverage
```

---

### Task 1.2: Implement JSONSerializer

**Estimate**: S (Small) - 1-2 days
**Dependencies**: Task 1.1 (Serializer ABC)
**Blocks**: 1.3, all Phase 2 tasks

**Objective**: Create JSONSerializer as default backward-compatible format.

**Files to Create**:
- `cryptofeed/serializers/json.py`

**Acceptance Criteria**:

```gherkin
GIVEN Trade object with symbol='BTC-USD', price=Decimal('50000.12345678')
WHEN JSONSerializer.serialize() is called
THEN returns valid JSON bytes with Decimal precision preserved

GIVEN Trade price Decimal('123.456789012345')
WHEN serialized to JSON and deserialized
THEN full precision maintained (no IEEE 754 rounding)

GIVEN Trade timestamp as float seconds
WHEN serialized to JSON
THEN timestamp remains float (not converted to microseconds)

GIVEN JSONSerializer.content_type() called
THEN returns 'application/json'

GIVEN invalid Trade object (missing required fields)
WHEN serialize() is called
THEN raises AttributeError or TypeError with clear message

GIVEN large OrderBook with 1000 bids/asks
WHEN serialized to JSON
THEN produces valid JSON parseable by json.loads()
```

**Test Specifications (TDD)**:

```python
# tests/unit/serializers/test_json_serializer.py

def test_json_serializer_trade_basic():
    """JSONSerializer correctly serializes Trade objects."""
    trade = Trade(
        symbol='BTC-USD',
        price=Decimal('50000.12345678'),
        amount=Decimal('1.5'),
        timestamp=1700000000.123,
        side='buy',
        exchange='coinbase'
    )

    serializer = JSONSerializer()
    result = serializer.serialize(trade)

    assert isinstance(result, bytes)
    obj = json.loads(result)
    assert obj['symbol'] == 'BTC-USD'
    assert obj['price'] == '50000.12345678'

def test_json_serializer_decimal_precision():
    """Decimal precision preserved in serialization."""
    trade = Trade(
        symbol='BTC-USD',
        price=Decimal('123.456789012345'),
        amount=Decimal('1.0'),
        timestamp=1700000000.0,
        side='buy',
        exchange='test'
    )

    serializer = JSONSerializer()
    result = serializer.serialize(trade)
    obj = json.loads(result)

    assert obj['price'] == '123.456789012345'

def test_json_serializer_content_type():
    """JSONSerializer returns correct MIME type."""
    serializer = JSONSerializer()
    assert serializer.content_type() == 'application/json'

def test_json_serializer_error_handling():
    """JSONSerializer raises error for invalid objects."""
    serializer = JSONSerializer()

    with pytest.raises((AttributeError, TypeError)):
        serializer.serialize("not a trade")
```

**Engineering Principles**:
- ✅ **Single Responsibility**: Handles only JSON format
- ✅ **Open/Closed**: Extensible with schema validation
- ✅ **Liskov Substitution**: Substitutable for Serializer
- ✅ **Interface Segregation**: Implements exact Serializer contract
- ✅ **Dependency Inversion**: Depends on Serializer abstraction
- ✅ **KISS**: Simple to_dict() → json.dumps() pipeline
- ✅ **DRY**: Reuses existing to_dict() methods
- ✅ **YAGNI**: No compression, streaming, or schema validation
- ✅ **TDD**: All tests written first

**Success Verification**:
```bash
pytest tests/unit/serializers/test_json_serializer.py -v --cov=cryptofeed.serializers.json
# Expected: 4/4 tests passing, 100% coverage
```

---

### Task 1.3: Integrate BackendCallback with Serialization Support

**Estimate**: M (Medium) - 2-3 days
**Dependencies**: Tasks 1.1, 1.2
**Blocks**: All Phase 2 tasks

**Objective**: Extend BackendCallback to accept serialization format and use appropriate serializer.

**Files to Modify**:
- `cryptofeed/callback.py` - Extend BackendCallback
- `cryptofeed/serializers/__init__.py` - Export ProtobufSerializer placeholder

**Acceptance Criteria**:

```gherkin
GIVEN BackendCallback initialized with serialization_format='json'
WHEN Trade message is received
THEN JSON-serialized bytes are passed to backend

GIVEN BackendCallback initialized with serialization_format='protobuf'
WHEN Trade message is received
THEN ProtobufSerializer is used (will be implemented in Phase 2)

GIVEN BackendCallback with no serialization_format parameter
WHEN Trade message is received
THEN defaults to 'json' (backward compatible)

GIVEN invalid serialization_format='unknown'
WHEN BackendCallback is initialized
THEN raises ValueError with clear message

GIVEN all 20 data types (Trade, OrderBook, Ticker, etc.)
WHEN processed by BackendCallback with serialization
THEN each is correctly serialized by appropriate serializer

GIVEN exception during serialization
WHEN raised by serialize()
THEN BackendCallback logs error and continues (graceful degradation)
```

**Test Specifications (TDD)**:

```python
# tests/integration/test_callback_serialization.py

def test_callback_json_serialization():
    """BackendCallback routes to JSONSerializer by default."""
    callback = BackendCallback(backend_name='test')

    trade = Trade(
        symbol='BTC-USD',
        price=Decimal('50000'),
        amount=Decimal('1'),
        timestamp=1700000000.0,
        side='buy',
        exchange='coinbase'
    )

    written_data = []
    callback.backend.write = lambda x: written_data.append(x)
    callback.trade(trade)

    assert len(written_data) == 1
    result = written_data[0]
    assert isinstance(result, bytes)
    obj = json.loads(result)
    assert obj['symbol'] == 'BTC-USD'

def test_callback_default_format():
    """BackendCallback defaults to JSON format."""
    callback = BackendCallback(backend_name='test')
    assert callback.serialization_format == 'json'

def test_callback_invalid_format():
    """BackendCallback raises error for unknown format."""
    with pytest.raises(ValueError, match='unknown'):
        BackendCallback(
            backend_name='test',
            serialization_format='unknown'
        )

def test_callback_error_handling():
    """BackendCallback handles serialization errors gracefully."""
    callback = BackendCallback(
        backend_name='test',
        serialization_format='json'
    )

    # Mock serializer that fails
    callback.serializer.serialize = lambda obj: (_ for _ in ()).throw(ValueError("fail"))

    trade = Trade(...)

    # Should not raise, should log error
    callback.trade(trade)
```

**Engineering Principles**:
- ✅ **Single Responsibility**: Callback coordinates, Serializer serializes
- ✅ **Open/Closed**: Open to new formats without modifying callback
- ✅ **Liskov Substitution**: Any Serializer subclass works
- ✅ **Interface Segregation**: Uses only serialize() + content_type()
- ✅ **Dependency Inversion**: Depends on Serializer, not concrete classes
- ✅ **KISS**: Simple factory pattern for serializer selection
- ✅ **DRY**: All message types use same pipeline
- ✅ **YAGNI**: No middleware, decorators, or plugins
- ✅ **TDD**: Tests written first, integration verified

**Success Verification**:
```bash
pytest tests/integration/test_callback_serialization.py -v
# Expected: 4/4 tests passing
```

---

## Phase 2: Data Integration (Tasks 1.4-1.7)

**Parallelization Note**: Tasks 1.4-1.7 can be executed in parallel by different engineers after Task 1.3 completes. Each task implements to_proto() for different data types using established patterns.

### Task 1.4: Implement to_proto() for Trade and OrderBook

**Estimate**: M (Medium) - 3-4 days
**Dependencies**: Task 1.3, Spec 0 merged
**Can Parallelize With**: Tasks 1.5, 1.6, 1.7

**Objective**: Implement protobuf serialization for Trade and OrderBook (2 of 20 data types).

**Files to Create/Modify**:
- `cryptofeed/trades.py` - Add Trade.to_proto()
- `cryptofeed/orderbook.py` - Add OrderBook.to_proto()

**Acceptance Criteria**:

```gherkin
GIVEN Trade object with all fields populated
WHEN to_proto() is called
THEN returns cryptofeed.schema.v1.Trade protobuf message

GIVEN Trade price as Decimal('50000.123456789')
WHEN serialized to protobuf
THEN price stored as string '50000.123456789' (precision preserved)

GIVEN Trade timestamp as float seconds (e.g., 1700000000.123)
WHEN serialized to protobuf
THEN timestamp converted to int64 microseconds (1700000000123000)

GIVEN Trade with minimum required fields
WHEN to_proto() is called
THEN succeeds and produces valid protobuf message

GIVEN protobuf Trade message serialized with SerializeToString()
WHEN deserialized with Trade.FromString()
THEN all fields match original values (round-trip verified)

GIVEN OrderBook with 100 bid/ask levels
WHEN to_proto() is called
THEN returns OrderBook protobuf with all levels preserved

GIVEN OrderBook Level objects
WHEN to_proto() converts them
THEN each Level → OrderBook.Level protobuf message

GIVEN ProtobufSerializer with Trade
WHEN serialize() calls trade.to_proto().SerializeToString()
THEN produces parseable binary protobuf bytes
```

**Test Specifications (TDD)**:

```python
# tests/unit/serializers/test_trade_proto.py

def test_trade_to_proto_basic():
    """Trade.to_proto() returns valid protobuf message."""
    trade = Trade(
        symbol='BTC-USD',
        price=Decimal('50000'),
        amount=Decimal('1.5'),
        timestamp=1700000000.123,
        side='buy',
        exchange='coinbase'
    )

    proto = trade.to_proto()

    assert proto.symbol == 'BTC-USD'
    assert proto.price == '50000'
    assert proto.amount == '1.5'
    assert proto.side == 'buy'
    assert proto.exchange == 'coinbase'

def test_trade_decimal_precision():
    """Decimal precision preserved in protobuf serialization."""
    trade = Trade(
        symbol='BTC-USD',
        price=Decimal('123.456789012345'),
        amount=Decimal('1.0'),
        timestamp=1700000000.0,
        side='buy',
        exchange='test'
    )

    proto = trade.to_proto()

    assert isinstance(proto.price, str)
    assert proto.price == '123.456789012345'

def test_trade_timestamp_conversion():
    """Trade timestamp float → int64 microseconds."""
    trade = Trade(
        symbol='BTC-USD',
        price=Decimal('50000'),
        amount=Decimal('1'),
        timestamp=1700000000.123,  # → 1700000000123000 microseconds
        side='buy',
        exchange='test'
    )

    proto = trade.to_proto()

    assert isinstance(proto.timestamp_us, int)
    assert proto.timestamp_us == 1700000000123000

def test_trade_roundtrip():
    """Trade serialization roundtrip: to_proto → bytes → FromString."""
    from cryptofeed.schema.v1.trade_pb2 import Trade as TradeProto

    original = Trade(
        symbol='BTC-USD',
        price=Decimal('50000.123456'),
        amount=Decimal('1.5'),
        timestamp=1700000000.123,
        side='buy',
        exchange='coinbase'
    )

    # Serialize
    proto = original.to_proto()
    bytes_data = proto.SerializeToString()

    # Deserialize
    restored_proto = TradeProto.FromString(bytes_data)

    # Verify
    assert restored_proto.symbol == original.symbol
    assert restored_proto.price == str(original.price)
    assert restored_proto.timestamp_us == int(original.timestamp * 1_000_000)
```

**Implementation Pattern**:

```python
def to_proto(self) -> TradeProto:
    """Convert to protobuf representation.

    Converts Decimal precision to string and timestamp to microseconds.

    Returns:
        TradeProto: Serializable protobuf message
    """
    return TradeProto(
        symbol=self.symbol,
        price=str(self.price),
        amount=str(self.amount),
        timestamp_us=int(self.timestamp * 1_000_000),
        side=self.side,
        exchange=self.exchange
    )
```

**Success Verification**:
```bash
pytest tests/unit/serializers/test_trade_proto.py tests/unit/serializers/test_orderbook_proto.py -v
# Expected: 8+/8+ tests passing, 100% coverage
```

---

### Task 1.5: Implement to_proto() for Ticker, Candle, FundingRate

**Estimate**: M (Medium) - 2-3 days
**Dependencies**: Task 1.4
**Can Parallelize With**: Task 1.6

**Objective**: Implement protobuf serialization for 3 additional data types.

**Files to Create/Modify**:
- `cryptofeed/ticker.py` - Add Ticker.to_proto()
- `cryptofeed/candle.py` - Add Candle.to_proto()
- `cryptofeed/funding.py` - Add FundingRate.to_proto()

**Acceptance Criteria**: (Same pattern as Task 1.4 - Decimal→string, float seconds→int64 microseconds)

**Test Specifications**: (Follow Task 1.4 pattern - basic, precision, timestamp, roundtrip)

**Implementation Pattern**: (Reuse helpers from Task 1.4)

**Success Verification**:
```bash
pytest tests/unit/serializers/test_{ticker,candle,funding}_proto.py -v
# Expected: All tests passing
```

---

### Task 1.6: Implement to_proto() for Liquidation, Index, OrderInfo

**Estimate**: M (Medium) - 2-3 days
**Dependencies**: Task 1.5
**Can Parallelize With**: Task 1.7

**Objective**: Implement protobuf serialization for 3 less-common data types.

**Files to Create/Modify**:
- `cryptofeed/liquidation.py` - Add Liquidation.to_proto()
- `cryptofeed/index.py` - Add Index.to_proto()
- `cryptofeed/orderinfo.py` - Add OrderInfo.to_proto()

**Acceptance Criteria**: (Same as Tasks 1.4-1.5)

**Test Specifications**: (Same pattern)

---

### Task 1.7: Implement to_proto() for Remaining 10 User Data Types

**Estimate**: L (Large) - 3-4 days
**Dependencies**: Task 1.6

**Objective**: Complete remaining data types: Fill, Balance, Position, MarginInfo, UserTrade, UserFunding, TransactionLog, UserLiquidation, UserCandle, UserOrderBook, UserTicker.

**Files to Create/Modify**:
- 10+ individual files in `cryptofeed/user_*.py`

**Acceptance Criteria**:
- All 10 types implement to_proto()
- All round-trip tests passing
- 100% code coverage for serialization methods
- No precision loss in Decimal or timestamp conversions

**Success Verification**:
```bash
pytest tests/unit/serializers/ -v --cov=cryptofeed.*.to_proto
# Expected: All 20 data types passing tests
```

---

## Phase 3: Production Readiness (Tasks 1.8-1.9)

### Task 1.8: Performance Benchmarking and Optimization

**Estimate**: M (Medium) - 2-3 days
**Dependencies**: Tasks 1.4-1.7
**Blocks**: 1.9

**Objective**: Benchmark serialization against targets, optimize hot paths, establish baseline metrics.

**Files to Create**:
- `tests/benchmarks/test_serialization_perf.py`
- `docs/SERIALIZATION_PERFORMANCE.md`

**Acceptance Criteria**:

```gherkin
GIVEN ProtobufSerializer with Trade
WHEN serialize() called 10,000 times
THEN p99 latency < 1ms per message

GIVEN JSONSerializer with Trade
WHEN serialize() called 10,000 times
THEN p99 latency < 2ms per message

GIVEN protobuf Trade message
WHEN serialized
THEN size < 50% of JSON representation

GIVEN mixed workload at 10,000 msg/s
WHEN serialized for extended period
THEN memory usage stable (no leaks)

GIVEN all optimizations complete
WHEN benchmarks run
THEN targets achieved or documented
```

**Test Specifications**:

```python
# tests/benchmarks/test_serialization_perf.py

def test_protobuf_trade_latency(benchmark):
    """Benchmark protobuf Trade serialization."""
    trade = Trade(...)
    serializer = ProtobufSerializer()
    result = benchmark(serializer.serialize, trade)
    assert isinstance(result, bytes)

def test_protobuf_vs_json_size():
    """Measure protobuf size vs JSON."""
    trade = Trade(...)
    proto_size = len(ProtobufSerializer().serialize(trade))
    json_size = len(JSONSerializer().serialize(trade))
    ratio = proto_size / json_size
    assert ratio < 0.5, f"Protobuf {ratio*100:.1f}% of JSON"

@pytest.mark.performance
def test_throughput_10k_messages():
    """Test 10,000 messages/sec throughput."""
    serializer = ProtobufSerializer()
    trades = [Trade(...) for _ in range(10000)]

    start = time.time()
    for trade in trades:
        serializer.serialize(trade)
    elapsed = time.time() - start

    throughput = 10000 / elapsed
    assert throughput > 10000, f"Only {throughput:.0f} msg/s"
```

**Success Verification**:
```bash
pytest tests/benchmarks/test_serialization_perf.py -v
# Expected: All targets met or documented with rationale
```

---

### Task 1.9: Integration Testing with Real Kafka Topics

**Estimate**: L (Large) - 3-4 days
**Dependencies**: Task 1.8

**Objective**: End-to-end testing with real Kafka infrastructure and consumer integration.

**Files to Create**:
- `tests/integration/test_kafka_serialization_e2e.py`
- `docker-compose.test.yml`
- `docs/KAFKA_INTEGRATION_TESTING.md`

**Acceptance Criteria**:

```gherkin
GIVEN Kafka cluster running (via docker-compose)
WHEN BackendCallback publishes Trade to cryptofeed.trades.test.btc-usd
THEN message appears in Kafka with correct protobuf serialization

GIVEN Kafka consumer reading protobuf messages
WHEN deserialized via Trade.FromString()
THEN all fields match original Trade object

GIVEN 1000 Trade messages published
WHEN consumed and deserialized
THEN no message loss or corruption (exact roundtrip)

GIVEN multiple data types (Trade, OrderBook, Ticker)
WHEN published simultaneously to different topics
THEN all consumed and deserialized correctly

GIVEN Kafka broker failure
WHEN producer reconnects
THEN resumes without message loss (idempotent config)

GIVEN consumer integration guide
WHEN consumer runs against real cryptofeed topics
THEN successfully consumes and deserializes protobuf
```

**Test Specifications**:

```python
# tests/integration/test_kafka_serialization_e2e.py

@pytest.mark.integration
@pytest.mark.kafka
def test_kafka_trade_roundtrip():
    """Trade serialization roundtrip through Kafka."""
    producer = BackendCallback(
        backend_name='kafka',
        serialization_format='protobuf',
        bootstrap_servers=['kafka:9092']
    )

    original_trade = Trade(...)

    # Publish
    producer.trade(original_trade)

    # Consume
    consumer = KafkaConsumer(
        'cryptofeed.trades.test.btc-usd',
        bootstrap_servers=['kafka:9092'],
        value_deserializer=lambda m: Trade.FromString(m)
    )

    msg = next(consumer)
    restored = msg.value

    # Verify roundtrip
    assert restored.symbol == original_trade.symbol
    assert restored.price == str(original_trade.price)

@pytest.mark.integration
@pytest.mark.kafka
def test_kafka_no_message_loss():
    """1000 messages without loss."""
    producer = BackendCallback(...)
    consumer = KafkaConsumer(...)

    trades = [Trade(...) for _ in range(1000)]

    for trade in trades:
        producer.trade(trade)

    consumed = []
    for _ in range(1000):
        msg = consumer.poll(timeout_ms=5000)
        if msg:
            consumed.append(msg.value)

    assert len(consumed) == 1000
```

**Docker Compose Setup**:
```yaml
version: '3.8'
services:
  kafka:
    image: confluentinc/cp-kafka:7.5.0
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: 'true'
    ports:
      - "9092:9092"
    depends_on:
      - zookeeper

  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
    ports:
      - "2181:2181"
```

**Success Verification**:
```bash
docker-compose -f docker-compose.test.yml up -d
pytest tests/integration/test_kafka_serialization_e2e.py -v -m kafka
# Expected: All E2E tests passing
```

---

## Task Summary Table

| ID | Phase | Task | Est. | Days | Status |
|----|-------|------|------|------|--------|
| 1.1 | Foundation | Serializer ABC | S | 1-2 | Ready |
| 1.2 | Foundation | JSONSerializer | S | 1-2 | Ready |
| 1.3 | Foundation | BackendCallback Integration | M | 2-3 | Ready |
| 1.4 | Data Integration | Trade + OrderBook | M | 3-4 | Ready |
| 1.5 | Data Integration | Ticker + Candle + FundingRate | M | 2-3 | Ready (can parallelize) |
| 1.6 | Data Integration | Liquidation + Index + OrderInfo | M | 2-3 | Ready (can parallelize) |
| 1.7 | Data Integration | 10 User Data Types | L | 3-4 | Ready (can parallelize) |
| 1.8 | Production | Performance Benchmarking | M | 2-3 | Ready |
| 1.9 | Production | Kafka Integration E2E | L | 3-4 | Ready |

**Total Estimated Effort**: 21-29 days
**Critical Path**: 1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6 → 1.7 → 1.8 → 1.9 (9 sequential)
**Optimized Timeline**: ~2 weeks (Phase 1: 4-5 days, Phase 2: 10-12 days with parallelization, Phase 3: 6-8 days)

---

## Engineering Excellence Checklist

All tasks must satisfy:
- ✅ **Test-First (TDD)**: Write tests before code
- ✅ **100% Coverage**: New code coverage ≥90%
- ✅ **No Mocks**: Use real objects and serializers
- ✅ **Conventional Commits**: feat:, fix:, test:, docs: prefixes
- ✅ **SOLID Principles**: Applied systematically
- ✅ **Type Annotations**: On all public methods
- ✅ **Docstrings**: Classes and methods documented
- ✅ **Error Handling**: Clear messages and logging
- ✅ **Integration Tests**: Verify real-world usage
- ✅ **Performance Targets**: Documented and tracked

---

## Sign-Off

This specification is complete and ready for implementation. Begin with Task 1.1 and proceed sequentially through Phase 1. After Task 1.3 completes, Tasks 1.4-1.7 can parallelize.

