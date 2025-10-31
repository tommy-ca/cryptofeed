# Market Data Kafka Producer - Implementation Tasks (Spec 3)

## Overview

12 implementation tasks for adding high-performance Kafka producer capability to cryptofeed. Tasks are organized into 3 phases with sequential dependencies, designed to be executed by 1-2 engineers over 2-3 weeks.

**Total Effort**: ~3 weeks (21-24 days)
**Team Size**: 1-2 engineers (can parallelize Phase 2 tasks)
**Dependencies**:
- Spec 1 (protobuf-callback-serialization) must be merged first
- Spec 0 (normalized-data-schema-crypto) already merged
- External: Kafka cluster (3+ brokers) available for testing

---

## Phase Summary

| Phase | Tasks | Effort | Purpose |
|-------|-------|--------|---------|
| 1: Foundation | 3.1-3.3 | 5-7 days | KafkaCallback base class, topic management, partitioning strategies |
| 2: Message Processing | 3.4-3.7 | 8-10 days | Message pipeline, serialization integration, error handling, DLQ |
| 3: Production Hardening | 3.8-3.12 | 8-10 days | Monitoring, health checks, configuration, testing, documentation |

---

## Phase 1: Foundation (Tasks 3.1-3.3)

All Foundation tasks must complete before Phase 2 begins. No parallelization in Phase 1.

### Task 3.1: Implement KafkaCallback Base Class

**Estimate**: M (Medium) - 2-3 days
**Dependencies**: Spec 1 merged (ProtobufSerializer available)
**Blocks**: All Phase 2 tasks

**Objective**: Create the core KafkaCallback class extending BackendCallback with Kafka producer integration.

**Files to Create**:
- `cryptofeed/kafka_callback.py` - KafkaCallback implementation
- `cryptofeed/kafka_producer.py` - Kafka producer wrapper (handles connection pooling, retries)

**Acceptance Criteria**:

```gherkin
GIVEN KafkaCallback initialized with bootstrap_servers=['kafka:9092']
WHEN Trade message is received via callback.trade(trade)
THEN KafkaCallback serializes via Spec 1 and queues for Kafka delivery

GIVEN KafkaCallback with acks='all' and enable_idempotence=True
WHEN producer sends message
THEN Kafka broker ensures exactly-once delivery (idempotent + acks=all)

GIVEN KafkaCallback with invalid bootstrap_servers
WHEN initialized
THEN raises ConnectionError with clear message about broker unavailability

GIVEN multiple KafkaCallback instances (for different backends)
WHEN running simultaneously
THEN each manages independent Kafka connection (no resource contention)

GIVEN KafkaCallback.is_connected() method
WHEN producer is connected to Kafka
THEN returns True

GIVEN KafkaCallback.is_connected()
WHEN producer is disconnected
THEN returns False

GIVEN KafkaCallback integration with BackendCallback
WHEN FeedHandler routes callbacks
THEN KafkaCallback correctly receives Trade, OrderBook, Ticker, etc.
```

**Test Specifications (TDD)**:

```python
# tests/unit/kafka/test_kafka_callback_base.py

def test_kafka_callback_initialization():
    """KafkaCallback initializes with valid configuration."""
    callback = KafkaCallback(
        bootstrap_servers=['kafka:9092'],
        acks='all',
        enable_idempotence=True
    )

    assert callback.bootstrap_servers == ['kafka:9092']
    assert callback.acks == 'all'
    assert callback.enable_idempotence == True

def test_kafka_callback_invalid_bootstrap_servers():
    """KafkaCallback raises error for unreachable brokers."""
    with pytest.raises(ConnectionError, match='broker'):
        callback = KafkaCallback(
            bootstrap_servers=['invalid:9999'],
            connection_timeout_ms=1000  # Quick timeout
        )

def test_kafka_callback_is_connected():
    """KafkaCallback.is_connected() reflects actual connection state."""
    callback = KafkaCallback(
        bootstrap_servers=['kafka:9092']
    )

    assert callback.is_connected() == True

def test_kafka_callback_message_queueing():
    """KafkaCallback queues messages for async delivery."""
    callback = KafkaCallback(
        bootstrap_servers=['kafka:9092']
    )

    trade = Trade(...)
    queued = callback._queue_message('trade', trade)

    assert queued == True
    assert callback.queue_size() > 0

@pytest.mark.integration
@pytest.mark.kafka
def test_kafka_callback_with_real_broker():
    """KafkaCallback connects and exchanges metadata with real Kafka."""
    callback = KafkaCallback(
        bootstrap_servers=['kafka:9092'],
        connection_timeout_ms=5000
    )

    # Verify can fetch metadata
    assert callback.get_topics() is not None
    assert callback.get_broker_count() >= 3  # At least 3 brokers

def test_kafka_callback_inherits_backend_callback():
    """KafkaCallback is subclass of BackendCallback."""
    callback = KafkaCallback(bootstrap_servers=['kafka:9092'])
    assert isinstance(callback, BackendCallback)

def test_kafka_callback_supports_all_data_types():
    """KafkaCallback has callbacks for all 20 data types."""
    callback = KafkaCallback(bootstrap_servers=['kafka:9092'])

    # Check all callback methods exist
    assert hasattr(callback, 'trade')
    assert hasattr(callback, 'orderbook')
    assert hasattr(callback, 'ticker')
    assert hasattr(callback, 'candle')
    # ... (all 20 data types)
```

**Implementation Pattern**:

```python
from cryptofeed.callback import BackendCallback
from confluent_kafka import Producer

class KafkaCallback(BackendCallback):
    """Kafka producer backend for cryptofeed."""

    def __init__(self, bootstrap_servers: List[str],
                 acks: str = 'all',
                 enable_idempotence: bool = True,
                 **kwargs):
        """
        Initialize Kafka producer callback.

        Args:
            bootstrap_servers: List of Kafka brokers
            acks: Delivery guarantee (0, 1, all)
            enable_idempotence: Enable idempotent producer
        """
        super().__init__('kafka', **kwargs)

        self.bootstrap_servers = bootstrap_servers
        self.acks = acks

        # Initialize producer
        config = {
            'bootstrap.servers': ','.join(bootstrap_servers),
            'acks': acks,
            'enable.idempotence': enable_idempotence,
            'retries': 3,
        }

        self.producer = Producer(config)
        self._message_queue = asyncio.Queue()

    def is_connected(self) -> bool:
        """Check if producer is connected to Kafka."""
        try:
            metadata = self.producer.list_topics(timeout=5)
            return metadata is not None
        except Exception:
            return False

    def trade(self, trade: Trade) -> None:
        """Handle Trade message."""
        self._route_message('trades', trade)

    def _route_message(self, data_type: str, obj: Any) -> None:
        """Route message for Kafka delivery."""
        # Serialize via Spec 1
        serialized = ProtobufSerializer().serialize(obj)

        # Queue for async delivery
        self._queue_message(data_type, obj, serialized)
```

**Engineering Principles**:
- ✅ **Single Responsibility**: Routes messages to Kafka only
- ✅ **Open/Closed**: Extends BackendCallback without modification
- ✅ **Liskov Substitution**: Substitutable for BackendCallback
- ✅ **Interface Segregation**: Minimal interface requirements
- ✅ **Dependency Inversion**: Depends on Kafka producer abstraction
- ✅ **KISS**: Simple producer initialization + message routing
- ✅ **DRY**: Reuses BackendCallback infrastructure
- ✅ **YAGNI**: No compression, schema registry, or clustering yet
- ✅ **TDD**: All tests written first

**Success Verification**:
```bash
pytest tests/unit/kafka/test_kafka_callback_base.py -v --cov=cryptofeed.kafka_callback
# Expected: 8/8 tests passing, 100% coverage
```

---

### Task 3.2: Implement Topic Management

**Estimate**: M (Medium) - 2 days
**Dependencies**: Task 3.1 (KafkaCallback base)
**Blocks**: Phase 2 tasks

**Objective**: Implement topic naming, creation, and caching logic for all data types.

**Files to Create/Modify**:
- `cryptofeed/kafka_callback.py` - Add TopicManager class
- `cryptofeed/kafka_topic.py` - Topic utilities

**Acceptance Criteria**:

```gherkin
GIVEN Trade from Coinbase with symbol BTC-USD
WHEN _generate_topic_name() is called
THEN returns 'cryptofeed.trades.coinbase.btc-usd'

GIVEN OrderBook from Binance with symbol ETH-USDT
WHEN _generate_topic_name() is called
THEN returns 'cryptofeed.orderbook.binance.eth-usdt'

GIVEN topic 'cryptofeed.trades.coinbase.btc-usd'
WHEN _ensure_topic_exists() is called
THEN topic is created in Kafka (idempotent - no error if exists)

GIVEN auto_create_topics=False
WHEN message for non-existent topic arrives
THEN raises TopicNotFoundError with clear message

GIVEN 1000 messages to same topic
WHEN first message arrives
THEN topic created once, no duplicate creation attempts

GIVEN topic created with 3 partitions
WHEN metadata is fetched
THEN partition_count returns 3

GIVEN multiple data types (Trade, OrderBook, Ticker)
WHEN all routed to Kafka
THEN each creates topic named correctly: cryptofeed.{type}.{exchange}.{symbol}
```

**Test Specifications (TDD)**:

```python
# tests/unit/kafka/test_topic_manager.py

def test_topic_name_generation_trade():
    """Topic name generated correctly for Trade."""
    manager = TopicManager()

    topic = manager.generate_topic_name(
        data_type='Trade',
        exchange='coinbase',
        symbol='BTC-USD'
    )

    assert topic == 'cryptofeed.trades.coinbase.btc-usd'

def test_topic_name_generation_orderbook():
    """Topic name generated correctly for OrderBook."""
    manager = TopicManager()

    topic = manager.generate_topic_name(
        data_type='OrderBook',
        exchange='binance',
        symbol='ETH-USDT'
    )

    assert topic == 'cryptofeed.orderbook.binance.eth-usdt'

def test_topic_name_normalization():
    """Topic names are normalized (lowercase, dashes)."""
    manager = TopicManager()

    topic = manager.generate_topic_name(
        data_type='Trade',
        exchange='COINBASE',
        symbol='BTC_USD'
    )

    # Should normalize to lowercase and replace _ with -
    assert topic == 'cryptofeed.trades.coinbase.btc-usd'

@pytest.mark.integration
@pytest.mark.kafka
def test_topic_creation():
    """Topic is created in Kafka."""
    manager = TopicManager(admin_client=admin_client)

    topic = 'cryptofeed.test.coinbase.btc-usd'
    manager.ensure_topic_exists(topic)

    # Verify topic exists
    topics = admin_client.list_topics(timeout=10)
    assert topic in topics

@pytest.mark.integration
@pytest.mark.kafka
def test_topic_creation_idempotent():
    """Topic creation is idempotent (no error if exists)."""
    manager = TopicManager(admin_client=admin_client)

    topic = 'cryptofeed.test.coinbase.btc-usd'

    # Create twice
    manager.ensure_topic_exists(topic)
    manager.ensure_topic_exists(topic)  # Should not raise

def test_topic_caching():
    """Topic names are cached to avoid repeated generation."""
    manager = TopicManager()

    topic1 = manager.generate_topic_name('Trade', 'coinbase', 'BTC-USD')
    topic2 = manager.generate_topic_name('Trade', 'coinbase', 'BTC-USD')

    assert topic1 == topic2
    # Should use cache (no recomputation)
    assert manager._cache_hits > 0
```

**Implementation Pattern**:

```python
class TopicManager:
    def __init__(self, admin_client=None):
        self.admin_client = admin_client
        self._topic_cache = {}
        self._created_topics = set()

    def generate_topic_name(self, data_type: str,
                           exchange: str,
                           symbol: str) -> str:
        """
        Generate topic name from data type, exchange, symbol.

        Returns: cryptofeed.{data_type}.{exchange}.{symbol}
        """
        # Normalize
        data_type_normalized = data_type.lower().replace('_', '-')
        exchange_normalized = exchange.lower()
        symbol_normalized = symbol.upper().replace('_', '-').lower()

        topic = f"cryptofeed.{data_type_normalized}.{exchange_normalized}.{symbol_normalized}"

        return topic

    def ensure_topic_exists(self, topic: str,
                           num_partitions: int = 3,
                           replication_factor: int = 3) -> None:
        """
        Create topic if not exists. Idempotent.
        """
        if topic in self._created_topics:
            return  # Already created

        # Check if exists
        if self.admin_client:
            topics = self.admin_client.list_topics(timeout=10)
            if topic in topics:
                self._created_topics.add(topic)
                return

            # Create topic
            self.admin_client.create_topics([...])
            self._created_topics.add(topic)
```

**Success Verification**:
```bash
pytest tests/unit/kafka/test_topic_manager.py -v
# Expected: All tests passing
```

---

### Task 3.3: Implement Partitioning Strategies

**Estimate**: M (Medium) - 2 days
**Dependencies**: Task 3.2 (Topic Management)
**Blocks**: Phase 2

**Objective**: Implement symbol-based, round-robin, and exchange-based partitioning strategies.

**Files to Create**:
- `cryptofeed/kafka_partitioner.py` - Partitioner implementations

**Acceptance Criteria**:

```gherkin
GIVEN SymbolPartitioner with symbol='BTC-USD'
WHEN get_partition_key() is called
THEN returns b'BTC-USD' (bytes)

GIVEN same symbol 'BTC-USD' called 100 times
WHEN partition keys generated
THEN all 100 keys are identical (consistent hashing)

GIVEN SymbolPartitioner with 3 partitions
WHEN routing symbol 'BTC-USD' 100 times
THEN always maps to same partition (e.g., partition 1)

GIVEN RoundRobinPartitioner with 3 partitions
WHEN get_partition() called 10 times
THEN returns [0, 1, 2, 0, 1, 2, 0, 1, 2, 0] (round robin)

GIVEN ExchangePartitioner with exchange='binance'
WHEN get_partition_key() called
THEN returns b'binance'

GIVEN 'binance' and 'binance' (same exchange)
WHEN both routed
THEN both map to same partition (ordering per exchange)

GIVEN partitioner_strategy='symbol'
WHEN KafkaCallback initialized with this strategy
THEN SymbolPartitioner is used for all messages
```

**Test Specifications (TDD)**:

```python
# tests/unit/kafka/test_partitioner.py

def test_symbol_partitioner_consistency():
    """Same symbol always generates same partition key."""
    partitioner = SymbolPartitioner()

    key1 = partitioner.get_partition_key('BTC-USD')
    key2 = partitioner.get_partition_key('BTC-USD')
    key3 = partitioner.get_partition_key('BTC-USD')

    assert key1 == key2 == key3

def test_symbol_partitioner_different_symbols():
    """Different symbols generate different partition keys."""
    partitioner = SymbolPartitioner()

    key_btc = partitioner.get_partition_key('BTC-USD')
    key_eth = partitioner.get_partition_key('ETH-USD')

    assert key_btc != key_eth

def test_symbol_partitioner_normalization():
    """Symbol normalized before partition key generation."""
    partitioner = SymbolPartitioner()

    key1 = partitioner.get_partition_key('BTC-USD')
    key2 = partitioner.get_partition_key('btc-usd')
    key3 = partitioner.get_partition_key('btc_usd')

    # All should normalize to same key
    assert key1 == key2 == key3

def test_round_robin_partitioner():
    """RoundRobinPartitioner cycles through partitions."""
    partitioner = RoundRobinPartitioner()

    partitions = [partitioner.get_partition(3) for _ in range(9)]

    assert partitions == [0, 1, 2, 0, 1, 2, 0, 1, 2]

def test_exchange_partitioner_consistency():
    """Same exchange always maps to same partition."""
    partitioner = ExchangePartitioner()

    key1 = partitioner.get_partition_key('binance')
    key2 = partitioner.get_partition_key('binance')

    assert key1 == key2

def test_partitioner_strategy_selection():
    """KafkaCallback selects correct partitioner based on strategy."""
    callback = KafkaCallback(
        bootstrap_servers=['kafka:9092'],
        partitioner_strategy='symbol'
    )

    assert isinstance(callback.partitioner, SymbolPartitioner)

    callback2 = KafkaCallback(
        bootstrap_servers=['kafka:9092'],
        partitioner_strategy='round_robin'
    )

    assert isinstance(callback2.partitioner, RoundRobinPartitioner)
```

**Success Verification**:
```bash
pytest tests/unit/kafka/test_partitioner.py -v --cov=cryptofeed.kafka_partitioner
# Expected: All tests passing
```

---

## Phase 2: Message Processing (Tasks 3.4-3.7)

**Parallelization Note**: Tasks 3.4-3.7 can be executed in parallel by different engineers after Task 3.3 completes.

### Task 3.4: Implement Message Serialization & Enrichment Pipeline

**Estimate**: M (Medium) - 3 days
**Dependencies**: Task 3.3, Spec 1 (ProtobufSerializer)
**Can Parallelize With**: Tasks 3.5, 3.6, 3.7

**Objective**: Implement message serialization via Spec 1 and header enrichment.

**Acceptance Criteria**:

```gherkin
GIVEN Trade object from Coinbase
WHEN serialized via ProtobufSerializer (Spec 1)
THEN returns binary protobuf bytes

GIVEN message enriched with headers
THEN includes: schema_version=v1, timestamp_generated, exchange, data_type

GIVEN serialized message
WHEN size measured
THEN protobuf < 50% of JSON size

GIVEN serialization pipeline with Trade, OrderBook, Ticker
WHEN all 3 types processed
THEN each correctly serialized to appropriate protobuf message type

GIVEN serialization error (e.g., missing required field)
WHEN serialize() called
THEN raises SerializationError with context

GIVEN 10,000 messages serialized
WHEN latency measured (p99)
THEN p99 latency < 2ms per message
```

**Test Specifications**: Follow pattern from Tasks 3.1-3.3

---

### Task 3.5: Implement Error Handling & Delivery Guarantees

**Estimate**: M (Medium) - 2-3 days
**Dependencies**: Task 3.4
**Can Parallelize With**: Tasks 3.6, 3.7

**Objective**: Implement exactly-once semantics, retries, and error classification.

**Acceptance Criteria**:

```gherkin
GIVEN producer with acks='all' and enable_idempotence=True
WHEN message sent to Kafka
THEN broker deduplicates by (producer_id, sequence_number)

GIVEN Kafka broker temporarily unavailable
WHEN message send attempted
THEN exponential backoff retry: 100ms, 200ms, 400ms

GIVEN serialization error (e.g., invalid Decimal)
WHEN error classified
THEN ErrorType.UNRECOVERABLE (don't retry)

GIVEN broker network error
WHEN error classified
THEN ErrorType.RECOVERABLE (retry with backoff)

GIVEN max retries exhausted
WHEN message still failing
THEN send to dead-letter-queue (DLQ) topic

GIVEN 1000 messages sent with broker failure on message 500
WHEN broker recovers
THEN all 1000 messages eventually delivered (no loss)
```

**Test Specifications**: Error injection, retry verification, DLQ tests

---

### Task 3.6: Implement Dead Letter Queue (DLQ)

**Estimate**: S (Small) - 2 days
**Dependencies**: Task 3.5
**Can Parallelize With**: Task 3.7

**Objective**: Implement DLQ topic for failed messages with error context.

**Acceptance Criteria**:

```gherkin
GIVEN message fails after max retries
WHEN sent to DLQ
THEN DLQ topic created: cryptofeed.dlq.{original_topic}

GIVEN DLQ message
THEN contains: original_message, error, timestamp, retry_count

GIVEN 100 messages, 5 unrecoverable errors
WHEN messages processed
THEN 95 in Kafka, 5 in DLQ

GIVEN DLQ message consumed
WHEN deserialized
THEN original message recoverable via base64 decode
```

---

### Task 3.7: Implement Configuration Support (YAML + Python API)

**Estimate**: M (Medium) - 2 days
**Dependencies**: Task 3.3 (all components ready)
**Can Parallelize With**: Phase 3 start

**Objective**: Support both YAML config files and Python API configuration.

**Files to Create/Modify**:
- `cryptofeed/config.py` - Add Kafka config parsing
- `docs/examples/kafka_config.yaml` - Example config

**Acceptance Criteria**:

```gherkin
GIVEN YAML config with kafka section
WHEN loaded via load_config()
THEN KafkaCallback initialized with correct parameters

GIVEN config with bootstrap_servers=['kafka1:9092', 'kafka2:9092']
WHEN KafkaCallback initialized
THEN producer connects to both brokers

GIVEN Python API: KafkaCallback(bootstrap_servers=[...], acks='all')
WHEN instantiated
THEN exactly-once semantics enabled

GIVEN environment variable: KAFKA_BOOTSTRAP_SERVERS='kafka:9092'
WHEN loaded
THEN overrides YAML config value

GIVEN invalid config (e.g., acks='invalid')
WHEN loaded
THEN raises ConfigError with clear message
```

**Success Verification**:
```bash
pytest tests/unit/kafka/test_configuration.py -v
# Expected: All config tests passing
```

---

## Phase 3: Production Hardening (Tasks 3.8-3.12)

### Task 3.8: Implement Prometheus Metrics

**Estimate**: M (Medium) - 2 days
**Dependencies**: Phase 2 complete

**Objective**: Add comprehensive Prometheus metrics for monitoring.

**Files to Create**:
- `cryptofeed/kafka_metrics.py` - Metrics definitions and recording

**Acceptance Criteria**:

```gherkin
GIVEN KafkaCallback with metrics_enabled=True
WHEN messages sent
THEN Prometheus counter incremented: cryptofeed_kafka_messages_sent_total

GIVEN message sent
WHEN latency measured
THEN Prometheus histogram recorded: cryptofeed_kafka_produce_latency_seconds

GIVEN 1000 messages at 100 msg/s
WHEN throughput measured
THEN metrics show correct rate

GIVEN /metrics endpoint
WHEN accessed
THEN returns Prometheus-formatted metrics (text/plain)

GIVEN metrics with labels (exchange, data_type)
WHEN queried
THEN can filter by label (e.g., data_type='Trade')
```

**Test Specifications**: Metric recording verification, endpoint tests

---

### Task 3.9: Implement Structured Logging & Health Check

**Estimate**: S (Small) - 1-2 days
**Dependencies**: Phase 2 complete

**Objective**: JSON-formatted logging and /metrics/kafka health endpoint.

**Acceptance Criteria**:

```gherkin
GIVEN INFO log event
WHEN logged
THEN JSON format: {"event": "...", "timestamp": "...", "data": {...}}

GIVEN /metrics/kafka endpoint
WHEN accessed
THEN returns: {"status": "healthy", "brokers_available": 3, ...}

GIVEN broker failure
WHEN /metrics/kafka accessed
THEN status='unhealthy', brokers_available < brokers_total

GIVEN tail -f logs
WHEN monitoring
THEN JSON logs parseable via jq
```

---

### Task 3.10: Integration Testing with Real Kafka

**Estimate**: L (Large) - 4-5 days
**Dependencies**: Tasks 3.8-3.9

**Objective**: End-to-end testing with docker-compose Kafka cluster.

**Files to Create**:
- `tests/integration/test_kafka_e2e.py` - End-to-end tests
- `docker-compose.kafka.yml` - Kafka test environment

**Acceptance Criteria**:

```gherkin
GIVEN Kafka cluster via docker-compose (3 brokers)
WHEN started
THEN all brokers healthy, cluster stable

GIVEN KafkaCallback connected to Kafka
WHEN Trade message sent
THEN message appears in cryptofeed.trades.* topic

GIVEN 1000 Trade messages from Coinbase
WHEN published and consumed
THEN exactly 1000 messages in topic (no loss, no duplication)

GIVEN OrderBook with 100 bid/ask levels
WHEN serialized to Kafka
THEN Flink/DuckDB consumer can deserialize correctly

GIVEN broker failure (kill 1 of 3)
WHEN KafkaCallback continues producing
THEN no message loss, brief latency increase

GIVEN all brokers fail
WHEN brokers recover
THEN KafkaCallback auto-reconnects (no manual restart needed)
```

---

### Task 3.11: Performance Benchmarking

**Estimate**: M (Medium) - 2-3 days
**Dependencies**: Task 3.10

**Objective**: Benchmark against targets: 10K msg/s, p99 < 10ms, 50% size reduction.

**Files to Create**:
- `tests/benchmarks/test_kafka_perf.py` - Performance benchmarks
- `docs/KAFKA_PERFORMANCE.md` - Performance analysis report

**Acceptance Criteria**:

```gherkin
GIVEN Trade message (250 bytes)
WHEN serialized to protobuf
THEN latency p99 < 2ms

GIVEN 10,000 Trade messages/sec
WHEN produced to Kafka
THEN sustained latency p99 < 10ms (no memory leaks)

GIVEN protobuf payload size
WHEN compared to JSON
THEN protobuf < 50% of JSON size

GIVEN OrderBook (1000 bytes JSON)
WHEN serialized to protobuf
THEN size reduction 50-60%

GIVEN throughput test
WHEN 50,000 msg/sec attempted
THEN either succeeds or documented limitation reached
```

---

### Task 3.12: Documentation & Consumer Integration Examples

**Estimate**: M (Medium) - 2-3 days
**Dependencies**: Task 3.11

**Objective**: Complete documentation and consumer reference implementations.

**Files to Create/Modify**:
- `docs/KAFKA_PRODUCER_GUIDE.md` - User guide
- `docs/KAFKA_ARCHITECTURE.md` - Architecture doc
- `examples/kafka_producer_example.py` - Python example
- `examples/kafka_consumer_flink.py` - Flink consumer example (reference)
- `examples/kafka_consumer_duckdb.py` - DuckDB consumer example (reference)

**Acceptance Criteria**:

```gherkin
GIVEN kafka_producer_example.py
WHEN executed
THEN connects to Kafka, publishes Trade messages to correct topics

GIVEN consumer integration guide
WHEN user reads it
THEN understands: Flink, Spark, DuckDB, Python consumer patterns

GIVEN troubleshooting guide
WHEN user encounters error
THEN finds diagnostic steps and resolution

GIVEN architecture documentation
WHEN user reviews it
THEN understands: topic naming, partitioning, message flow, monitoring

GIVEN example configurations
WHEN user copies them
THEN can customize and run with minimal changes
```

---

## Task Summary Table

| ID | Phase | Task | Est. | Status |
|----|-------|------|------|--------|
| 3.1 | Foundation | KafkaCallback Base Class | M | Ready |
| 3.2 | Foundation | Topic Management | M | Ready |
| 3.3 | Foundation | Partitioning Strategies | M | Ready |
| 3.4 | Message Processing | Serialization & Enrichment | M | Ready (can parallelize) |
| 3.5 | Message Processing | Error Handling | M | Ready (can parallelize) |
| 3.6 | Message Processing | Dead Letter Queue | S | Ready (can parallelize) |
| 3.7 | Message Processing | Configuration | M | Ready (can parallelize) |
| 3.8 | Production | Prometheus Metrics | M | Ready |
| 3.9 | Production | Logging & Health Check | S | Ready |
| 3.10 | Production | Integration Testing | L | Ready |
| 3.11 | Production | Performance Benchmarking | M | Ready |
| 3.12 | Production | Documentation | M | Ready |

**Total Estimated Effort**: 31-38 days
**Critical Path**: 3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6 → 3.7 → 3.8 → 3.9 → 3.10 → 3.11 → 3.12
**Optimized Timeline**: ~3 weeks (Foundation: 6-7 days, Message Processing: 8-10 days with parallelization, Production: 8-10 days)

---

## Engineering Excellence Checklist

All tasks must satisfy:
- ✅ **Test-First (TDD)**: Write tests before code
- ✅ **100% Coverage**: New code coverage ≥90%
- ✅ **No Mocks**: Use real Kafka cluster (docker-compose)
- ✅ **Conventional Commits**: feat:, fix:, test:, docs: prefixes
- ✅ **SOLID Principles**: Applied systematically
- ✅ **Type Annotations**: On all public methods
- ✅ **Docstrings**: Classes and methods documented
- ✅ **Error Handling**: Clear messages and logging
- ✅ **Integration Tests**: Real Kafka broker verification
- ✅ **Performance Targets**: Documented and tracked
- ✅ **Configuration Support**: YAML + Python API
- ✅ **Production Ready**: Metrics, logging, health checks

---

## Sign-Off

This specification is complete and ready for implementation. Begin with Task 3.1 and proceed sequentially through Phase 1. After Task 3.3 completes, Tasks 3.4-3.7 can parallelize. Tasks 3.8+ proceed in sequence.

**Next Steps**:
1. Schedule 1-2 engineers for 3-week implementation
2. Set up docker-compose Kafka cluster for testing
3. Begin Phase 1 with Task 3.1
4. Daily standup to track progress and unblock issues
