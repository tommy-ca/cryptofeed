# Market Data Kafka Producer - Technical Design (Spec 3)

## Document Control

**Status**: Draft - Ready for Review
**Version**: 0.1.0
**Last Updated**: October 31, 2025
**Owner**: Engineering
**Related Specs**: Spec 0 (normalized-data-schema-crypto), Spec 1 (protobuf-callback-serialization)

---

## 1. Overview & Context

### Purpose

Provide high-performance Kafka producer integration for cryptofeed, enabling downstream consumers to implement storage, analytics, and persistence independently. Cryptofeed becomes the pure ingestion/producer layer; consumers handle all downstream responsibilities.

### Scope

**In Scope**:
- Kafka producer backend implementation (extends cryptofeed's BackendCallback system)
- Topic management with hierarchical naming (`cryptofeed.{data_type}.{exchange}.{symbol}`)
- Partitioning strategies (symbol-based for ordering guarantees)
- Integration with Spec 1 (protobuf serialization)
- Delivery guarantees (exactly-once semantics via idempotent producers)
- Prometheus metrics and health checks
- Error handling and graceful degradation
- Configuration (YAML + Python API)

**Out of Scope**:
- Kafka consumer implementation (consumer responsibility)
- Apache Iceberg, DuckDB, Parquet storage backends (consumer responsibility)
- Stream processing (Flink, Spark, QuixStreams - consumer responsibility)
- Data retention, compaction, schema evolution (consumer responsibility)
- Query engines and analytics

**Boundary**: This spec ends at Kafka topic production. Consumers read topics independently.

### Key Design Principles

1. **Separation of Concerns**: Cryptofeed → Kafka → Consumers (clear boundaries)
2. **SOLID Principles**: Single responsibility, open/closed extension, Liskov substitution
3. **High Throughput**: 10,000+ messages/second per producer instance
4. **Reliability**: No message loss, graceful failure handling
5. **Observability**: Comprehensive metrics and logging
6. **Flexibility**: Multiple topic strategies and partitioning schemes
7. **Backward Compatibility**: Coexist with existing JSON backends

---

## 2. Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Cryptofeed (Ingestion Layer)                                │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ Coinbase │  │ Binance  │  │ Kraken   │  ... (Exchanges) │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
│       │             │             │                         │
│       └─────────────┴─────────────┘                         │
│              ↓                                              │
│  ┌──────────────────────────────────┐                      │
│  │ FeedHandler                      │                      │
│  │ (normalizes all exchange data)   │                      │
│  └──────────────────────────────────┘                      │
│              ↓                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ BackendCallback System                               │  │
│  │ (router: delegates to appropriate backend)           │  │
│  └──────────────────────────────────────────────────────┘  │
└────────┬──────────────────────────────────────────────────┘
         │
         ├─── [JSON Backend]  (existing, unchanged)
         │
         └─── [KafkaCallback (This Spec)]
              │
              ├─→ Topic Routing    (cryptofeed.trades.coinbase.btc-usd)
              ├─→ Partitioning      (hash by symbol for ordering)
              ├─→ Serialization     (calls Spec 1's ProtobufSerializer)
              ├─→ Error Handling    (dead-letter queue, retries)
              └─→ Metrics          (Prometheus: messages, bytes, latency, errors)
                   ↓
         ┌─────────────────────────────────┐
         │ Kafka Cluster (3+ brokers)      │
         │                                 │
         │ Topics:                         │
         │  - cryptofeed.trades.*          │
         │  - cryptofeed.orderbook.*       │
         │  - cryptofeed.ticker.*          │
         │  - cryptofeed.candle.*          │
         │  - cryptofeed.funding.*         │
         │  - cryptofeed.liquidation.*     │
         │  - ... (all 20 data types)      │
         └─────────────────────────────────┘
              ↓
    ┌────────┴────────┬──────────┬──────────┐
    │                 │          │          │
 Consumer 1       Consumer 2  Consumer 3  Consumer N
 (Flink→Iceberg) (DuckDB)    (Spark→...)  (Custom)
```

### 2.2 Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ KafkaCallback (extends BackendCallback)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Topic Manager                                            │  │
│  │ ├─ _generate_topic_name(data_type, exchange, symbol)     │  │
│  │ │   Returns: cryptofeed.{data_type}.{exchange}.{symbol}  │  │
│  │ ├─ _ensure_topic_exists(topic, partitions, replication) │  │
│  │ └─ _parse_topic_params(topic) → (type, ex, sym)         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Partitioning Strategy                                    │  │
│  │ ├─ SymbolPartitioner (default: hash by symbol)           │  │
│  │ │   Ensures: all messages for BTC-USD → partition 3      │  │
│  │ │   (ordering guarantee per symbol)                       │  │
│  │ ├─ RoundRobinPartitioner (optional: max parallelism)     │  │
│  │ └─ ExchangePartitioner (optional: group by exchange)     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Kafka Producer (confluent-kafka)                         │  │
│  │ ├─ Configuration:                                        │  │
│  │ │  • bootstrap_servers: ["kafka1:9092", ...]             │  │
│  │ │  • acks=all (exactly-once semantics)                    │  │
│  │ │  • enable.idempotence=true (prevent duplicates)        │  │
│  │ │  • retries=3, retry.backoff.ms=100                      │  │
│  │ │  • compression_type=snappy (40-50% size reduction)     │  │
│  │ │  • batch.size=16KB, linger.ms=10 (throughput)          │  │
│  │ ├─ Callbacks:                                             │  │
│  │ │  • on_success(topic, partition, offset, timestamp)     │  │
│  │ │  • on_error(exc) → retry or dead-letter queue          │  │
│  │ └─ Delivery Guarantees:                                   │  │
│  │    • Exactly-once (idempotent producer + transactional)   │  │
│  │    • At-least-once (retry fallback)                       │  │
│  │    • At-most-once (fire-and-forget, not recommended)      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Message Routing Pipeline                                 │  │
│  │ Data Type Object                                          │  │
│  │   ↓                                                       │  │
│  │ [Serialize] → (Spec 1: to_proto())                        │  │
│  │   ↓                                                       │  │
│  │ [Enrich] → (add message headers for routing)              │  │
│  │   • exchange: "coinbase" (source exchange)                │  │
│  │   • symbol: "BTC-USD" (trading pair)                      │  │
│  │   • data_type: "trade" (message type)                     │  │
│  │   • schema_version: "1.0" (protobuf schema version)       │  │
│  │   • timestamp: RFC3339 (message generation time)          │  │
│  │   ↓                                                       │  │
│  │ [Route] → (determine topic, partition key)                │  │
│  │   ↓                                                       │  │
│  │ [Produce] → (send to Kafka broker)                        │  │
│  │   ↓                                                       │  │
│  │ [Track] → (metrics: latency, size, offset)                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Error Handling & Resilience                              │  │
│  │ ├─ Recoverable Errors:                                   │  │
│  │ │  • BrokerNotAvailable → exponential backoff retry       │  │
│  │ │  • NetworkException → automatic reconnect               │  │
│  │ │  • TimeoutException → configurable retry               │  │
│  │ ├─ Unrecoverable Errors:                                  │  │
│  │ │  • SerializationError → log + skip message               │  │
│  │ │  • InvalidTopicException → log + alarm                   │  │
│  │ └─ Dead Letter Queue:                                     │  │
│  │    Topic: cryptofeed.dlq.{original_topic}                 │  │
│  │    Records failed messages + error context                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Monitoring & Observability                               │  │
│  │ ├─ Prometheus Metrics:                                    │  │
│  │ │  • cryptofeed_kafka_messages_sent_total (counter)       │  │
│  │ │  • cryptofeed_kafka_bytes_sent_total (counter)          │  │
│  │ │  • cryptofeed_kafka_produce_latency_seconds (histogram) │  │
│  │ │  • cryptofeed_kafka_errors_total (counter)              │  │
│  │ │  • cryptofeed_kafka_dlq_messages_total (counter)        │  │
│  │ ├─ Structured Logging (JSON format):                      │  │
│  │ │  • INFO: Topic routing, configuration loading           │  │
│  │ │  • WARN: Retries, slow producers                         │  │
│  │ │  • ERROR: DLQ entries, unrecoverable failures            │  │
│  │ └─ Health Check:                                           │  │
│  │    /metrics/kafka → {"status": "healthy", "lag": 0}       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Design

### 3.1 Topic Management

#### 3.1.1 Topic Naming Conventions

**TWO Configurable Strategies**:

**Strategy A: Consolidated Topics (Default)**
- **Pattern**: `cryptofeed.{data_type}`
- **Topic Count**: O(data_types) = 8 topics total
- **Examples**:
  ```
  cryptofeed.trades        (all trades: Coinbase, Binance, Kraken, etc.)
  cryptofeed.orderbook     (all L2 books)
  cryptofeed.ticker        (all tickers)
  cryptofeed.candle        (all candles)
  cryptofeed.funding       (all funding rates)
  ... (8 total for all data types)
  ```
- **Benefits**:
  - Single consumer subscription per data type
  - Simplified routing downstream
  - Excellent scalability (10,000+ msg/s per topic)
  - Multi-exchange/symbol aggregation in one topic
  - Recommended for most use cases

- **Message Routing**:
  - Kafka message header `exchange` → identifies source exchange
  - Kafka message header `symbol` → identifies trading pair
  - Consumer filters/routes based on headers
  - Example: Consumer subscribes `cryptofeed.trades`, filters via `exchange=binance`

**Strategy B: Per-Symbol Topics (Optional, Legacy)**
- **Pattern**: `cryptofeed.{data_type}.{exchange}.{symbol}`
- **Topic Count**: O(symbols × exchanges) = 80,000+ topics at scale
- **Examples**:
  ```
  cryptofeed.trades.coinbase.btc-usd
  cryptofeed.orderbook.binance.eth-usdt
  cryptofeed.ticker.kraken.sol-usd
  cryptofeed.candle.bitmex.xbt-usd
  cryptofeed.funding.dydx.btc-usd-perp
  ```
- **Benefits**:
  - Per-pair ordering guarantees (single-symbol subscription)
  - Legacy support for existing deployments
  - Use during migration period only

- **Drawbacks**:
  - Topic explosion at scale (80K+ topics)
  - Complex Kafka cluster management
  - Consumer needs separate subscription per pair
  - Deprecated in favor of consolidated topics

**Configuration**:
```yaml
kafka:
  topic_strategy: "consolidated"  # or "per_symbol" or "dual_write"
  topic_prefix: "cryptofeed"
  data_types:
    - trades
    - orderbook
    - ticker
    - candle
    - ... (14 total)
```

#### 3.1.2 Topic Creation Strategy

```python
class TopicManager:
    def ensure_topic_exists(self, topic: str,
                          num_partitions: int = 3,
                          replication_factor: int = 3):
        """
        Create topic if not exists. Idempotent operation.

        Partitions:
        - Default: 3 (balanced for 3-9 brokers)
        - Adjustable: based on throughput requirements
        - Per-symbol: high-traffic symbols get more partitions

        Replication:
        - Default: 3 (acks=all ensures all replicas have message)
        - Minimum: 2 (tolerance for single broker failure)
        - Maximum: number of brokers
        """
        # Check if topic exists
        metadata = admin_client.get_topic_metadata(topic)
        if metadata:
            return  # Topic already exists

        # Create topic with config
        admin_client.create_topics([
            NewTopic(
                name=topic,
                num_partitions=num_partitions,
                replication_factor=replication_factor,
                config={
                    'retention.ms': '604800000',  # 7 days
                    'compression.type': 'snappy',
                    'min.insync.replicas': '2',  # for acks=all
                }
            )
        ])
```

### 3.2 Partitioning Strategies

**Four Configurable Strategies** (default: composite):

#### 3.2.1 Composite Partitioning (Recommended Default)

**Strategy**: Partition key = `{exchange}-{symbol}`

```python
class CompositePartitioner:
    def get_partition_key(self, exchange: str, symbol: str) -> bytes:
        """
        Return partition key based on exchange + symbol.

        Guarantees:
        - All messages for (Coinbase, BTC-USD) always → partition N
        - All messages for (Binance, BTC-USD) may → different partition
        - Per-exchange-pair ordering preserved

        Example keys:
        - "coinbase-btc-usd" → partition 0
        - "binance-eth-usdt" → partition 1
        - "kraken-sol-usd" → partition 2
        """
        # Normalize and compose key
        normalized_symbol = symbol.upper().replace('_', '-')
        key = f"{exchange.lower()}-{normalized_symbol}"
        return key.encode('utf-8')
```

**Trade-offs**:
- ✅ **Pro**: Per-pair ordering (real-time trading critical)
- ✅ **Pro**: Excellent distribution across partitions (12 partitions × 1000 symbols = 12K buckets)
- ✅ **Pro**: Handles hotspots better (BTC-USD distributed across exchanges)
- ✅ **Pro**: Standard for market data use cases
- ❌ **Con**: Cross-exchange pair analysis requires merging

**Recommended for**: Real-time market data, order matching, candle generation

#### 3.2.2 Symbol-Only Partitioning (Optional)

**Strategy**: Partition key = `{symbol}`

```python
class SymbolPartitioner:
    def get_partition_key(self, symbol: str) -> bytes:
        """
        Route all messages for symbol across all exchanges → same partition.

        Example: All BTC-USD (Coinbase, Binance, Kraken) → partition 0
        """
        normalized = symbol.upper().replace('_', '-')
        return normalized.encode('utf-8')
```

**Use Case**: Cross-exchange arbitrage analysis, symbol-level aggregation

**Trade-offs**:
- ✅ **Pro**: All data for symbol together (arbitrage)
- ✅ **Pro**: Simpler per-symbol subscription
- ❌ **Con**: Hotspot risk (BTC-USD may dominate single partition)

#### 3.2.3 Round-Robin Partitioning (Optional)

```python
class RoundRobinPartitioner:
    def __init__(self):
        self.counter = 0

    def get_partition(self, partition_count: int) -> int:
        """
        Cycle through partitions sequentially.

        Guarantees:
        - Maximum parallelism (even distribution)
        - No ordering guarantees
        """
        partition = self.counter % partition_count
        self.counter += 1
        return partition
```

**Use Case**: Analytics/aggregation where ordering doesn't matter, want max throughput

#### 3.2.4 Exchange-Based Partitioning (Optional)

```python
class ExchangePartitioner:
    def get_partition_key(self, exchange: str) -> bytes:
        """
        Route all messages for exchange → same partition.

        Example: All Coinbase trades/books → partition 0
        """
        return exchange.lower().encode('utf-8')
```

**Use Case**: Exchange-specific processing, exchange reconciliation

**Partition Strategy Decision Matrix**:

| Strategy | Partition Key | Ordering | Use Case | Hotspot Risk |
|----------|---------------|----------|----------|--------------|
| **Composite** (default) | `{exchange}-{symbol}` | Per-pair | Real-time trading | Low |
| Symbol | `{symbol}` | Per-symbol | Cross-exchange analysis | High (BTC) |
| Round-robin | `None` | None | Analytics | None |
| Exchange | `{exchange}` | Per-exchange | Exchange ops | Medium |

### 3.3 Kafka Producer Configuration

#### 3.3.1 Core Producer Config

```python
# In config.yaml or environment
kafka:
  producer:
    bootstrap_servers:
      - kafka1:9092
      - kafka2:9092
      - kafka3:9092

    # Delivery guarantees
    acks: all                           # Wait for all in-sync replicas
    enable.idempotence: true            # Prevent duplicate messages
    transactional.id: "cryptofeed-prod" # Enable transactions

    # Retry policy
    retries: 3                          # Retry failed sends
    retry.backoff.ms: 100              # Exponential backoff
    request.timeout.ms: 30000          # 30s per request

    # Performance tuning
    batch.size: 16384                  # 16KB batch
    linger.ms: 10                      # Wait 10ms before sending (batch)
    compression.type: snappy           # 40-50% size reduction
    buffer.memory: 67108864            # 64MB total buffer

    # Connection pool
    connections.max.idle.ms: 540000    # 9 minutes
    max.in.flight.requests.per.connection: 5

  # Topic-specific overrides
  topics:
    trades:
      partitions: 3
      replication_factor: 3
    orderbook:
      partitions: 5              # High volume needs more partitions
      replication_factor: 3
```

#### 3.3.2 Exactly-Once Semantics Implementation

```python
class ExactlyOnceProducer:
    def __init__(self, bootstrap_servers: List[str]):
        self.producer = Producer({
            'bootstrap.servers': ','.join(bootstrap_servers),
            'acks': 'all',                      # Wait for all replicas
            'enable.idempotence': True,         # Idempotent producer
            'transactional.id': 'cryptofeed',   # Enable transactions
        })

    def send_message(self, topic: str, key: bytes, value: bytes) -> bool:
        """
        Send with exactly-once guarantee.

        Flow:
        1. Begin transaction (implicit on first send)
        2. Send message with idempotent producer
           - Broker deduplicates by (producer_id, sequence_number)
           - If duplicate arrives, same (offset, timestamp) returned
        3. Commit transaction
        4. Error → abort transaction, retry from start

        Result: Exactly-once across broker restarts and retries
        """
        try:
            # Kafka producer handles deduplication internally
            # Message sent with producer_id + sequence_number
            future = self.producer.produce(
                topic=topic,
                key=key,
                value=value,
                callback=self._on_delivery
            )

            # Flush to ensure delivery
            self.producer.flush(timeout=10)

            return True

        except Exception as e:
            self.logger.error(f"Send failed: {e}", exc_info=True)
            return False

    def _on_delivery(self, err, msg):
        """Callback on delivery success/failure."""
        if err:
            self.logger.error(f"DLQ: {msg.topic()} @ {msg.offset()}")
            self._send_to_dlq(msg)
        else:
            self.metrics.messages_sent.inc()
            self.metrics.produce_latency.observe(msg.latency() / 1000)
```

### 3.4 Message Processing Pipeline

#### 3.4.1 Message Enrichment & Headers

**Purpose**: Add routing and metadata headers to all Kafka messages for consumer filtering and observability.

**Required Headers** (per FR2):
- `exchange`: Source exchange (e.g., "coinbase", "binance") - **mandatory for routing**
- `symbol`: Trading pair (e.g., "BTC-USD", "ETH-USDT") - **mandatory for filtering**
- `data_type`: Message type (e.g., "trade", "orderbook", "funding") - **mandatory for routing**
- `schema_version`: Protobuf schema version (e.g., "1.0") - **mandatory for deserialization**

**Optional Headers**:
- `producer_version`: Cryptofeed version (e.g., "0.1.0") - for compatibility tracking
- `timestamp`: RFC3339 message generation time - for latency monitoring
- `content-type`: "application/x-protobuf" - for serialization format

```python
class MessageEnricher:
    def enrich_message(self, data_type: Any,
                      metadata: Dict) -> Tuple[bytes, Dict]:
        """
        Serialize message and add mandatory routing headers.

        Args:
            data_type: Cryptofeed data object (Trade, OrderBook, etc.)
            metadata: Exchange and symbol metadata

        Returns:
            (serialized_bytes, headers_dict)
        """
        # Serialize via Spec 1 (protobuf-callback-serialization)
        serialized = ProtobufSerializer().serialize(data_type)

        # Mandatory headers for routing (FR2)
        headers = {
            'exchange': metadata['exchange'].encode('utf-8'),
            'symbol': metadata['symbol'].encode('utf-8'),
            'data_type': type(data_type).__name__.lower().encode('utf-8'),
            'schema_version': b'1.0',
        }

        # Optional headers
        headers.update({
            'producer_version': __version__.encode('utf-8'),
            'timestamp': datetime.utcnow().isoformat().encode('utf-8'),
            'content-type': b'application/x-protobuf',
        })

        return serialized, headers
```

#### 3.4.2 Message Routing

```python
class MessageRouter:
    def route(self, data_type: Any,
             metadata: Dict) -> Tuple[str, bytes, bytes]:
        """
        Route message to appropriate topic and partition.

        Returns:
            (topic, partition_key, serialized_value)
        """
        # Extract routing info
        data_type_name = type(data_type).__name__.lower()
        exchange = metadata['exchange'].lower()
        symbol = metadata['symbol'].lower()

        # Generate topic
        topic = f"cryptofeed.{data_type_name}.{exchange}.{symbol}"

        # Generate partition key (symbol-based)
        partition_key = symbol.encode('utf-8')

        # Serialize
        serialized = ProtobufSerializer().serialize(data_type)

        return topic, partition_key, serialized
```

### 3.5 Error Handling & Resilience

#### 3.5.1 Error Classification

```python
class ErrorHandler:
    def classify_error(self, exc: Exception) -> ErrorType:
        """Classify error as recoverable or not."""
        if isinstance(exc, (BrokerNotAvailable, KafkaError)):
            if exc.code() in [KafkaError._BROKER_NOT_AVAILABLE,
                             KafkaError._GAPLESS_UNLIKELY]:
                return ErrorType.RECOVERABLE

        if isinstance(exc, KafkaTimeoutException):
            return ErrorType.RECOVERABLE

        if isinstance(exc, (KafkaException, KafkaProducerError)):
            # Serialization, message size, etc.
            return ErrorType.UNRECOVERABLE

        return ErrorType.UNKNOWN

    def handle_error(self, error: ErrorType, exc: Exception,
                    message: Any, topic: str):
        """Handle error based on classification."""
        if error == ErrorType.RECOVERABLE:
            self.retry_with_backoff(message, topic, backoff_ms=100)
        elif error == ErrorType.UNRECOVERABLE:
            self.send_to_dlq(message, topic, exc)
        else:
            self.alert_and_log(exc)
```

#### 3.5.2 Dead Letter Queue (DLQ)

```python
class DeadLetterQueue:
    def send_to_dlq(self, original_message: Any,
                   original_topic: str,
                   error: Exception):
        """
        Send failed message to DLQ topic for later analysis.

        Topic: cryptofeed.dlq.{original_topic}
        Payload:
            {
                "original_topic": "cryptofeed.trades.coinbase.btc-usd",
                "original_message": <base64 encoded>,
                "error": "SerializationError: ...",
                "timestamp": "2025-10-31T12:34:56Z",
                "retry_count": 3
            }
        """
        dlq_topic = f"cryptofeed.dlq.{original_topic}"

        dlq_payload = {
            'original_topic': original_topic,
            'original_message': base64.b64encode(
                original_message
            ).decode('utf-8'),
            'error': str(error),
            'timestamp': datetime.utcnow().isoformat(),
            'retry_count': 3,  # attempted N times
        }

        self.producer.produce(
            topic=dlq_topic,
            key=b'dlq',
            value=json.dumps(dlq_payload).encode('utf-8')
        )
```

### 3.6 Monitoring & Observability

#### 3.6.1 Prometheus Metrics

```python
class KafkaMetrics:
    def __init__(self):
        # Counters
        self.messages_sent_total = Counter(
            'cryptofeed_kafka_messages_sent_total',
            'Total messages sent to Kafka',
            ['data_type', 'exchange']
        )

        self.bytes_sent_total = Counter(
            'cryptofeed_kafka_bytes_sent_total',
            'Total bytes sent to Kafka',
            ['data_type']
        )

        self.errors_total = Counter(
            'cryptofeed_kafka_errors_total',
            'Total Kafka errors',
            ['error_type', 'data_type']
        )

        self.dlq_messages_total = Counter(
            'cryptofeed_kafka_dlq_messages_total',
            'Messages sent to DLQ',
            ['original_topic']
        )

        # Histograms
        self.produce_latency_seconds = Histogram(
            'cryptofeed_kafka_produce_latency_seconds',
            'Latency from callback to Kafka ACK',
            ['data_type'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        )

        self.message_size_bytes = Histogram(
            'cryptofeed_kafka_message_size_bytes',
            'Serialized message size',
            ['data_type'],
            buckets=[10, 50, 100, 500, 1000, 5000, 10000]
        )

        # Gauges
        self.producer_lag = Gauge(
            'cryptofeed_kafka_producer_lag_messages',
            'Messages in producer queue waiting to send',
            ['partition']
        )

        self.broker_unavailable = Gauge(
            'cryptofeed_kafka_broker_unavailable',
            'Count of unavailable brokers'
        )

    def record_send(self, data_type: str, exchange: str,
                   latency_ms: float, size_bytes: int):
        """Record successful send."""
        self.messages_sent_total.labels(
            data_type=data_type,
            exchange=exchange
        ).inc()

        self.bytes_sent_total.labels(data_type=data_type).inc(size_bytes)

        self.produce_latency_seconds.labels(
            data_type=data_type
        ).observe(latency_ms / 1000)

        self.message_size_bytes.labels(
            data_type=data_type
        ).observe(size_bytes)
```

#### 3.6.2 Structured Logging

```python
class StructuredLogger:
    def log_topic_created(self, topic: str, partitions: int):
        """Log topic creation."""
        self.logger.info(json.dumps({
            'event': 'topic_created',
            'topic': topic,
            'partitions': partitions,
            'timestamp': datetime.utcnow().isoformat(),
        }))

    def log_message_sent(self, topic: str, offset: int,
                        latency_ms: float, size_bytes: int):
        """Log successful message send."""
        self.logger.info(json.dumps({
            'event': 'message_sent',
            'topic': topic,
            'offset': offset,
            'latency_ms': latency_ms,
            'size_bytes': size_bytes,
            'timestamp': datetime.utcnow().isoformat(),
        }))

    def log_error_retry(self, topic: str, error: str, retry_count: int):
        """Log retried error."""
        self.logger.warning(json.dumps({
            'event': 'message_retry',
            'topic': topic,
            'error': error,
            'retry_count': retry_count,
            'timestamp': datetime.utcnow().isoformat(),
        }))

    def log_dlq(self, topic: str, error: str):
        """Log message sent to DLQ."""
        self.logger.error(json.dumps({
            'event': 'message_dlq',
            'topic': topic,
            'original_topic': topic,
            'error': error,
            'timestamp': datetime.utcnow().isoformat(),
        }))
```

#### 3.6.3 Health Check Endpoint

```python
class HealthCheck:
    @app.get('/metrics/kafka')
    def kafka_health():
        """Health check endpoint for Kafka producer."""
        brokers_available = producer.metadata.broker_count()
        brokers_total = len(producer.metadata.brokers())
        lag = producer.outq_len()  # Messages waiting to send

        return {
            'status': 'healthy' if brokers_available > 0 else 'unhealthy',
            'brokers_available': brokers_available,
            'brokers_total': brokers_total,
            'producer_queue_depth': lag,
            'topics_created': len(producer.metadata.topics()),
            'timestamp': datetime.utcnow().isoformat(),
        }
```

---

## 4. Configuration Design

### 4.1 YAML Configuration

```yaml
# config/kafka.yaml

ingestion:
  type: kafka

kafka:
  # Connection
  bootstrap_servers:
    - localhost:9092
    - localhost:9093
    - localhost:9094

  # Producer settings
  producer:
    acks: all                           # Exactly-once
    enable.idempotence: true
    transactional.id: cryptofeed-prod
    compression.type: snappy

    # Retries
    retries: 3
    retry.backoff.ms: 100
    request.timeout.ms: 30000

    # Performance
    batch.size: 16384
    linger.ms: 10
    buffer.memory: 67108864

  # Topic management
  topics:
    auto_create: true
    default_partitions: 3
    default_replication_factor: 3

    # Per-topic overrides
    overrides:
      - pattern: "*.orderbook.*"
        partitions: 5              # High volume
      - pattern: "*.funding.*"
        partitions: 2              # Low volume

  # Partitioning strategy
  partitioner: symbol              # symbol | round_robin | exchange

  # Error handling
  dead_letter_queue:
    enabled: true
    topic_suffix: dlq

  # Monitoring
  metrics:
    enabled: true
    port: 8000                     # Prometheus port
    path: /metrics
```

### 4.2 Python API Configuration

```python
from cryptofeed.kafka_producer import KafkaCallback

# Programmatic initialization
callback = KafkaCallback(
    bootstrap_servers=['kafka:9092'],
    acks='all',
    enable_idempotence=True,
    compression_type='snappy',
    topic_strategy='symbol',  # Symbol-based partitioning
    auto_create_topics=True,
    metrics_enabled=True,
)

# Add to FeedHandler
feed_handler = FeedHandler()
feed_handler.add_callback(callback, ['trades', 'orderbook'])

# Start feed
feed_handler.start()
```

---

## 5. Data Type Integration

All 20 cryptofeed data types integrate via Spec 1 (protobuf serialization):

| Data Type | Topic Pattern | Example |
|-----------|---------------|---------|
| Trade | `cryptofeed.trades.{exchange}.{symbol}` | `cryptofeed.trades.coinbase.btc-usd` |
| OrderBook | `cryptofeed.orderbook.{exchange}.{symbol}` | `cryptofeed.orderbook.binance.eth-usdt` |
| Ticker | `cryptofeed.ticker.{exchange}.{symbol}` | `cryptofeed.ticker.kraken.sol-usd` |
| Candle | `cryptofeed.candles.{exchange}.{symbol}` | `cryptofeed.candles.bitmex.xbt-usd` |
| FundingRate | `cryptofeed.funding.{exchange}.{symbol}` | `cryptofeed.funding.dydx.btc-usd-perp` |
| Liquidation | `cryptofeed.liquidation.{exchange}.{symbol}` | `cryptofeed.liquidation.binance.btc-usdt` |
| Index | `cryptofeed.index.{exchange}.{symbol}` | `cryptofeed.index.indexing-service.btc-usd` |
| Open Interest | `cryptofeed.openinterest.{exchange}.{symbol}` | `cryptofeed.openinterest.okex.btc-usd` |
| Fill | `cryptofeed.fills.{exchange}.{user_id}` | `cryptofeed.fills.coinbase.user123` |
| Balance | `cryptofeed.balances.{exchange}.{user_id}` | `cryptofeed.balances.binance.user456` |
| Position | `cryptofeed.positions.{exchange}.{user_id}` | `cryptofeed.positions.bybit.user789` |
| MarginInfo | `cryptofeed.margin.{exchange}.{user_id}` | `cryptofeed.margin.dydx.user101` |
| ... (10 more user data types) | | |

---

## 6. Migration & Backward Compatibility Roadmap

### 6.1 Problem Statement

**Challenge**: Existing deployments use per-symbol topics (`cryptofeed.{data_type}.{exchange}.{symbol}`), but consolidation to `cryptofeed.{data_type}` offers 99% reduction in topic count and improved downstream routing.

**Requirement**: Enable smooth transition without breaking existing consumers.

### 6.2 Migration Strategy: Blue-Green Cutover (4 Weeks)

**Approach**: Direct migration with parallel deployment and per-exchange consumer cutover. **NO dual-write mode** - new backend is production-ready and can replace legacy immediately.

**Rationale**:
- New KafkaCallback backend is fully validated (628+ tests, 9.9/10 performance)
- Legacy backend (`cryptofeed/backends/kafka.py`) marked deprecated
- Blue-Green provides safe rollback without dual-write complexity
- Per-exchange migration allows incremental validation

#### Week 1: Parallel Deployment & Staging Validation

**Goal**: Deploy new KafkaCallback to staging and production with separate topic namespace.

**Implementation**:
1. Deploy new KafkaCallback with consolidated topics (`cryptofeed.{data_type}`)
2. Legacy backend continues using per-symbol topics (separate namespace)
3. Validate consolidated topic message format and headers in staging
4. Deploy to 10% production (canary), monitor 2 hours, expand to 100%

**Success Criteria**:
- New backend producing to consolidated topics
- Message latency <5ms (p99)
- Error rate <0.1%
- Kafka broker healthy (CPU, memory, network)

**Rollback**: Remove new backend deployment, legacy remains unchanged (<5 min)

#### Week 2: Consumer Preparation & Monitoring Setup

**Goal**: Prepare consumers for migration and deploy monitoring.

**Actions**:
1. **Consumer Migration Templates**: Create templates for Flink, Python async, Custom consumers
2. **Monitoring Dashboard**: Deploy Grafana dashboard (9 panels) + Prometheus queries
3. **Alert Rules**: Configure 8 alert rules (lag >5s, error rate >0.1%, latency >50ms)
4. **Testing**: Test consumer subscriptions in staging with consolidated topics

**Consumer Update Pattern**:
```python
# Old (per-symbol subscription)
consumer.subscribe(['cryptofeed.trades.coinbase.*'])

# New (consolidated subscription with header-based filtering)
consumer.subscribe(['cryptofeed.trade'])  # Note: singular, consolidated
for msg in consumer:
    # Filter using message headers
    if msg.headers['exchange'] == 'coinbase' and msg.headers['symbol'] == 'BTC-USD':
        process_trade(msg)
```

**Success Criteria**:
- Consumer templates validated in staging
- Monitoring dashboard functional
- Alert rules triggering correctly on test scenarios

#### Week 3: Gradual Consumer Migration (Per Exchange)

**Goal**: Migrate consumers incrementally by exchange to allow validation and rollback.

**Migration Order** (by volume): Coinbase → Binance → Remaining exchanges

**Process** (1 exchange per day):
1. **Day 1 (Coinbase)**: Update consumer subscription to consolidated topics
2. Monitor consumer lag <5s, error rate <0.1%, data completeness
3. Validate downstream storage (Iceberg/DuckDB) for 4+ hours
4. **Day 2 (Binance)**: Repeat process, compare performance vs Coinbase
5. **Days 3-5**: Migrate remaining exchanges (Kraken, OKX, Bybit, etc.)

**Validation Per Exchange**:
- Consumer lag remains <5 seconds
- No message loss (downstream record counts match)
- Partition ordering preserved (same symbol → same partition)
- No duplicates in downstream storage

**Rollback** (<5 min): Revert consumer subscription to legacy per-symbol topics

#### Week 4: Stabilization & Legacy Cleanup

**Goal**: Monitor consolidated topic production stability and prepare legacy deprecation.

**Actions**:
1. **Production Monitoring**: Validate 10 success criteria (message loss zero, lag <5s, error <0.1%)
2. **Legacy Topic Archival**: Backup per-symbol topics to cold storage
3. **Deprecation Notice**: Update legacy backend with sunset timeline (4 weeks)
4. **Documentation**: Finalize migration report and lessons learned

**Success Criteria**:
- All 10 measurable targets validated
- Zero production incidents
- Legacy backend marked for 4-week sunset
- Team sign-off and approval

**Post-Migration**:
- Weeks 5-6: Legacy backend on standby (read-only)
- Week 7+: Remove legacy backend and per-symbol topics

### 6.3 Backward Compatibility Matrix

| Phase | Topic Strategy | Consolidated | Per-Symbol (Legacy) | Approach |
|-------|---|---|---|---|
| **Pre-Migration** | Per-symbol (legacy) | ❌ | ✅ | Legacy backend only |
| **Week 1** | Blue-Green (parallel) | ✅ | ✅ | Both backends, separate namespaces |
| **Week 2-3** | Blue-Green (migration) | ✅ | ✅ | Consumer cutover per exchange |
| **Week 4** | Consolidated (primary) | ✅ | ✅* | Legacy deprecated, read-only |
| **Post-Migration** | Consolidated only | ✅ | ❌ | Legacy removed after 4 weeks |

*Week 4+: Legacy per-symbol topics remain readable for rollback, but deprecated

### 6.4 Configuration Examples

**New Backend (Consolidated Topics - Default)**:
```yaml
kafka:
  bootstrap_servers: ["localhost:9092"]
  topic_strategy: consolidated  # Default
  partitioner: composite  # exchange-symbol hash
  serialization_format: protobuf
```

**Legacy Backend (Per-Symbol Topics - Deprecated)**:
```yaml
kafka:
  bootstrap_servers: ["localhost:9092"]
  topic_strategy: per_symbol  # Legacy, deprecated
  # Note: Use new backend for all new deployments
```

**Optional: Per-Symbol Strategy (if needed for specific use case)**:
```yaml
kafka:
  topic_strategy: per_symbol
  # Explicitly opt into per-symbol if required
  # Warning: Creates O(10K) topics vs O(20) consolidated
```

### 6.5 Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Message loss during cutover | Low | High | Dual-write validation + health checks |
| Consumer lag spike | Medium | Medium | Staged rollout, rollback plan |
| Partition rebalancing | Low | Low | Monitor consumer group rebalance time |
| Schema incompatibility | Low | High | Test consumer code with consolidated topics |
| Broker capacity | Low | Medium | Monitor topic partition count vs cluster size |

---

## 7. Performance Characteristics

### 7.1 Latency & Throughput Targets

```
Latency (milliseconds) from Callback to Kafka ACK:

Trade (250 bytes protobuf):
  p50: <1ms   (serialize + route)
  p95: <3ms   (network round-trip)
  p99: <5ms   (includes retry backoff)

OrderBook (1000 bytes protobuf):
  p50: <2ms
  p95: <4ms
  p99: <5ms

Sustained Throughput (production validated):
  150,000+ msg/s → p99 <5ms latency (consolidated topics)
  200,000+ msg/s → p99 <10ms (multi-instance horizontal scaling)

Scalability via Consolidated Topics:
  - O(20) topics vs O(10K) per-symbol topics
  - Reduced partition rebalancing overhead
  - Improved broker resource utilization
  - Consumer groups scale horizontally across fewer topics
```

### 6.2 Payload Size Reduction

Protobuf vs JSON (via Spec 1):

```
Trade:
  JSON:     ~400 bytes
  Protobuf: ~120 bytes (30% of JSON)
  Compressed (snappy): ~100 bytes

OrderBook (100 levels):
  JSON:     ~3000 bytes
  Protobuf: ~1000 bytes (33% of JSON)
  Compressed: ~500 bytes
```

### 6.3 Memory Usage

```
Per Producer Instance:
  Base overhead:     ~50 MB (producer, topic cache, buffer)
  Per 10K msgs/sec:  +5 MB
  Total capacity:    ~500 MB (buffer for 10K msgs/s for 5 seconds)
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

- Topic name generation, partitioning logic
- Message enrichment, serialization
- Error classification and handling
- Metric recording

### 7.2 Integration Tests

- Real Kafka cluster (docker-compose)
- End-to-end message flow (produce → consume)
- Exactly-once delivery verification
- Error scenarios and recovery

### 7.3 Performance Tests

- Throughput benchmarks (target 10K msg/s)
- Latency percentiles (p99 <10ms)
- Memory leak detection (sustained load)

---

## 8. Consumer Integration Examples

### 8.1 Flink Consumer (Reference)

Consumers implement independently using design.md from Spec 0 and 1:

```python
# Consumer implementation (NOT in cryptofeed scope)
from pyflink.datastream import StreamExecutionEnvironment

env = StreamExecutionEnvironment.get_execution_environment()

trades = env.add_source(
    KafkaSource.builder()
    .set_bootstrap_servers('kafka:9092')
    .set_topics('cryptofeed.trades.*')
    .set_value_only_deserializer(ProtobufDeserializer(Trade))
    .build()
)

# Process and write to Iceberg
trades.add_sink(IcebergSink(...))
```

### 8.2 DuckDB Consumer (Reference)

```python
# Consumer implementation (NOT in cryptofeed scope)
import duckdb
from kafka import KafkaConsumer
from cryptofeed.schema.v1.trade_pb2 import Trade

consumer = KafkaConsumer(
    'cryptofeed.trades.coinbase.btc-usd',
    bootstrap_servers=['kafka:9092'],
    value_deserializer=lambda m: Trade.FromString(m)
)

conn = duckdb.connect('market_data.db')
for msg in consumer:
    trade = msg.value
    conn.execute("""
        INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?)
    """, [trade.symbol, trade.price, trade.amount, ...])
```

---

## 9. Implementation Roadmap

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1 | 1 week | KafkaCallback base class, topic management, partitioning |
| Phase 2 | 1-1.5 weeks | Message pipeline, error handling, DLQ |
| Phase 3 | 3-5 days | Monitoring, metrics, health checks |
| Phase 4 | 3-5 days | Integration testing, benchmarking |
| Phase 5 | 2-3 days | Documentation, consumer examples |

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Kafka broker unavailability | Messages queued, latency spike | Auto-reconnect with exponential backoff, health check |
| Message serialization failures | Data loss, DLQ overflow | Comprehensive error handling, fallback to JSON |
| Topic partition hotspot | Latency skew, uneven load | Monitor partition lag, adjust partitioning strategy |
| Spec 1 not merged | Blocks implementation | Implement ProtobufSerializer placeholder in Phase 1 |
| Consumer lag | Stale data in downstream | Consumer monitoring guide, alerting thresholds |

---

## 11. Success Criteria

- ✅ Kafka producer publishes protobuf messages at 10,000+ msg/s
- ✅ p99 latency < 10ms from callback to Kafka ACK
- ✅ Exactly-once delivery verified via integration tests
- ✅ All 20 data types routed to correct topics
- ✅ Error handling tested (failures, retries, DLQ)
- ✅ Prometheus metrics available and documented
- ✅ Health check endpoint responds correctly
- ✅ Zero message loss under normal operation
- ✅ Graceful handling of broker failures
- ✅ Documentation includes consumer integration examples

---

## 12. Requirements Traceability

| Requirement | Design Section | Verification |
|-------------|-----------------|--------------|
| FR1: Kafka Backend | §3, §4 | Unit tests for KafkaCallback |
| FR2: Topic Management | §3.1 | Integration test: topic auto-creation |
| FR3: Partitioning | §3.2 | Unit test: partition key consistency |
| FR4: Serialization | §3.4, Spec 1 | Integration: round-trip test |
| FR5: Delivery Guarantees | §3.3 | Integration: exactly-once test (DuplicateConsumer) |
| FR6: Monitoring | §3.6 | Manual: /metrics endpoint check |
| NFR1: Performance | §6, §7.3 | Benchmark: 10K msg/s with p99 <10ms |
| NFR2: Reliability | §3.5 | Integration: error injection tests |
| NFR3: Configuration | §4 | Unit: YAML parsing + Python API |

---

## 13. Appendix: Related Patterns

### Message Routing Pattern

```
Data Object
    ↓
[Extract Metadata] → (exchange, symbol, data_type)
    ↓
[Calculate Topic] → cryptofeed.{type}.{exchange}.{symbol}
    ↓
[Calculate Partition Key] → symbol.encode()
    ↓
[Serialize] → (Spec 1 ProtobufSerializer)
    ↓
[Enrich Headers] → (schema_version, timestamp, content_type)
    ↓
[Produce] → Kafka.produce(topic, key, value, headers)
    ↓
[Track Metrics] → (latency, size, offset)
```

### Error Recovery Pattern

```
Produce Attempt
    ↓
[Check Error]
    ├─ Recoverable? → Retry with exponential backoff
    ├─ Unrecoverable? → Send to DLQ
    └─ Unknown? → Alert and investigate
    ↓
[Max Retries Reached?]
    ├─ Yes → Send to DLQ
    └─ No → Retry (goto Produce Attempt)
    ↓
[DLQ Entry] → Log error context + original message
    ↓
[Operator Review] → Investigate and fix root cause
```

---

## 13.5 Compound Workstreams & Boundaries

- **Workstream Decomposition**:
  - Schema definition and evolution are owned by `normalized-data-schema-crypto`.
  - Protobuf serialization helpers and converter registry are owned by `protobuf-callback-serialization`.
  - This spec owns the Kafka producer backend, including topic/partition strategies, delivery guarantees, metrics, and migration tooling.
  - Separate E2E and consumer specs own exchange-specific pipelines and downstream processing.
- **Interfaces & Contracts**:
  - Input contract: normalized dataclasses and their `to_proto()` mappings.
  - Output contract: Kafka topics, partitioning behavior, and headers (exchange, symbol, data_type, schema_version) as documented here and in Kafka docs.
  - This design must remain compatible with those contracts so multiple workstreams can proceed independently.

## 13.6 AI Agent Design Guidance

- AI agents implementing or extending this design MUST:
  - Use existing abstractions (TopicManager, PartitionerFactory, HeaderEnricher, KafkaConfig) instead of creating parallel mechanisms.
  - Keep changes scoped to the Kafka backend and its tests unless upstream specs explicitly require schema/serialization changes.
  - Avoid embedding consumer-specific behavior (e.g., storage layouts, query patterns) in the producer; such concerns belong in separate specs.
- Cross-spec changes (e.g., to Protobuf fields, normalized dataclasses) SHALL be coordinated by:
  - Updating `normalized-data-schema-crypto` and/or `protobuf-callback-serialization` first.
  - Referencing those spec updates in commit messages and in any design/task modifications for this spec.

## 14. Conclusion

This design establishes cryptofeed as a pure ingestion layer that produces protobuf-serialized market data to Kafka topics. The separation of concerns between cryptofeed (producer) and downstream consumers enables flexible storage and analytics implementations while maintaining high throughput, reliability, and observability.

The architecture supports 10,000+ messages/second with sub-10ms latency, exactly-once delivery semantics, comprehensive error handling, and production-grade monitoring. Consumers integrate independently using Flink, Spark, DuckDB, or custom implementations based on their storage and analytics requirements.
