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
│  │ [Enrich] → (add headers: schema_version, timestamp_gen)   │  │
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

#### 3.1.1 Topic Naming Convention

**Pattern**: `cryptofeed.{data_type}.{exchange}.{symbol}`

**Examples**:
```
cryptofeed.trades.coinbase.btc-usd
cryptofeed.orderbook.binance.eth-usdt
cryptofeed.ticker.kraken.sol-usd
cryptofeed.candle.bitmex.xbt-usd
cryptofeed.funding.dydx.btc-usd-perp
cryptofeed.liquidation.liquidations.btc-usd
```

**Benefits**:
- Clear hierarchical structure (data type → exchange → symbol)
- Easy subscription patterns: `cryptofeed.trades.*`, `cryptofeed.*.binance.*`
- Supports both topic-per-symbol and topic-per-exchange strategies
- Schema evolution per data type

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

#### 3.2.1 Symbol-Based Partitioning (Default)

**Strategy**: Hash symbol → consistent partition

```python
class SymbolPartitioner:
    def get_partition_key(self, data_type: str,
                         exchange: str,
                         symbol: str) -> bytes:
        """
        Return partition key based on symbol.

        Guarantees:
        - All messages for BTC-USD always → partition N (same broker)
        - Ordering guaranteed within symbol
        - Load balanced across symbols
        """
        # Normalize symbol
        normalized = symbol.upper().replace('_', '-')

        # Return bytes for Kafka
        return normalized.encode('utf-8')

    def route(self, partition_count: int, key: bytes) -> int:
        """
        Route message to partition using consistent hash.

        Hash(key) % num_partitions ensures same key always
        maps to same partition (order preservation).
        """
        # Kafka uses this internally
        return hash(key) % partition_count
```

**Trade-offs**:
- ✅ **Pro**: Preserves order per symbol (FIFO)
- ✅ **Pro**: Load balanced across symbols
- ✅ **Pro**: Consumers can shard by symbol
- ❌ **Con**: Hotspot if one symbol dominates (e.g., BTC-USD)

#### 3.2.2 Round-Robin Partitioning (Optional)

```python
class RoundRobinPartitioner:
    def __init__(self):
        self.counter = 0

    def get_partition(self, partition_count: int) -> int:
        """
        Cycle through partitions sequentially.

        Guarantees:
        - Maximum parallelism (messages distributed evenly)
        - No ordering guarantees (different symbols may interleave)
        """
        partition = self.counter % partition_count
        self.counter += 1
        return partition
```

**Use Case**: Analytics where ordering doesn't matter, want max throughput.

#### 3.2.3 Exchange-Based Partitioning (Optional)

```python
class ExchangePartitioner:
    def get_partition_key(self, exchange: str) -> bytes:
        """
        Route all messages for exchange → same partition.

        Guarantees:
        - All Coinbase trades/books → partition N
        - All Binance → partition M
        - Ordering per exchange (useful for exchange reconciliation)
        """
        return exchange.lower().encode('utf-8')
```

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

#### 3.4.1 Message Enrichment

```python
class MessageEnricher:
    def enrich_message(self, data_type: Any,
                      metadata: Dict) -> Tuple[bytes, Dict]:
        """
        Serialize message and add metadata headers.

        Headers added:
        - schema_version: v1 (for consumer validation)
        - producer_version: 0.1.0 (for compatibility)
        - timestamp_generated: ISO8601 (when produced)
        - exchange: coinbase (from data)
        - data_type: Trade (message type)
        """
        # Serialize via Spec 1
        serialized = ProtobufSerializer().serialize(data_type)

        headers = {
            'schema_version': b'v1',
            'producer_version': b'0.1.0',
            'timestamp_generated': str(datetime.utcnow().isoformat()).encode(),
            'exchange': metadata.get('exchange', b'unknown'),
            'data_type': metadata.get('data_type', b'unknown'),
            'content_type': b'application/x-protobuf',
        }

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

## 6. Performance Characteristics

### 6.1 Latency Targets

```
Latency (milliseconds) from Callback to Kafka ACK:

Trade (250 bytes):
  p50: 0.5ms  (serialize + route)
  p95: 2ms    (network round-trip)
  p99: 5ms    (includes retry backoff)

OrderBook (1000 bytes):
  p50: 2ms
  p95: 5ms
  p99: 10ms

Sustained Throughput (p99 latency):
  10,000 msg/s → <10ms latency
  50,000 msg/s → <50ms latency (multi-instance needed)
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

## 14. Conclusion

This design establishes cryptofeed as a pure ingestion layer that produces protobuf-serialized market data to Kafka topics. The separation of concerns between cryptofeed (producer) and downstream consumers enables flexible storage and analytics implementations while maintaining high throughput, reliability, and observability.

The architecture supports 10,000+ messages/second with sub-10ms latency, exactly-once delivery semantics, comprehensive error handling, and production-grade monitoring. Consumers integrate independently using Flink, Spark, DuckDB, or custom implementations based on their storage and analytics requirements.
