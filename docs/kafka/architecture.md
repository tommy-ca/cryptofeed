# Kafka Backend Architecture

## System Overview

The Kafka backend implements a modular, high-performance architecture for streaming market data to Apache Kafka. It provides both JSON and Protocol Buffer serialization with configurable topic and partitioning strategies.

## Core Architecture

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    Cryptofeed Core                           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Exchange Connectors                        │ │
│  │  (Binance, Coinbase, Kraken, etc.)                      │ │
│  └─────────────────┬───────────────────────────────────────┘ │
└───────────────────┼───────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                Kafka Backend Layer                          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │         KafkaBackendBase (Shared Infrastructure)       │ │
│  │  ┌─────────────────────────────────────────────────────┐ │
│  │  │ Queue Management │ Serialization │ Header Enrichment│ │
│  │  │ Producer Lifecycle │ Error Handling │ Metrics       │ │
│  │  └─────────────────────────────────────────────────────┘ │
│  └─────────────────┬───────────────────────────────────────┘ │
└───────────────────┼───────────────────────────────────────────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│ KafkaCallback   │ │KafkaProtobufCb  │
│ (JSON+Protobuf) │ │  (Protobuf)     │
└─────────────────┘ └─────────────────┘
```

### Data Flow

```
Exchange Data → Normalization → Queue → Serialization → Header Enrichment → Kafka Producer → Topic
```

## Key Components

### KafkaBackendBase

**Purpose**: Shared infrastructure for all Kafka callbacks

**Responsibilities**:
- Message queue management with async processing
- Producer lifecycle (startup/shutdown)
- Error handling and retry logic
- Metrics collection
- Configuration validation

**Key Methods**:
- `_process_message()`: Main message processing loop
- `_serialize_message()`: Abstract serialization hook
- `_enrich_headers()`: Abstract header enrichment hook
- `write()`: Public write interface

### KafkaCallback

**Purpose**: Unified callback supporting both JSON and Protocol Buffer serialization

**Features**:
- Runtime serialization format selection
- Automatic format detection in headers
- Backward compatibility with legacy configurations

**Configuration**:
```python
KafkaCallback(
    bootstrap_servers=['kafka:9092'],
    topic='market-data',
    serialization_format='protobuf'  # or 'json'
)
```

### KafkaProtobufCallback

**Purpose**: Protocol Buffer-only callback for maximum performance

**Features**:
- Fixed protobuf serialization
- Schema validation on every message
- Optimized for high-throughput scenarios

**Configuration**:
```python
KafkaProtobufCallback(
    bootstrap_servers=['kafka:9092'],
    topic='market-data'
)
```

## Topic Management

### TopicManager

**Purpose**: Handles topic naming and creation logic

**Strategies**:

#### Consolidated Strategy
```
Input: {'strategy': 'consolidated'}
Output: Single topic for all data
Example: "market-data"
```

#### Per-Symbol Strategy
```
Input: {'strategy': 'per_symbol'}
Output: Topic per trading symbol
Example: BTC-USDT → "BTC-USDT"
```

### Topic Naming Convention

```
<prefix>-<base_name>
```

- **prefix**: Optional topic prefix (e.g., "prod", "dev")
- **base_name**: Strategy-specific name

## Partitioning System

### Partitioner Classes

```
Partitioner (Abstract Base)
├── SymbolPartitioner      → symbol-based keys
├── CompositePartitioner   → exchange:symbol keys
├── ExchangePartitioner    → exchange-based keys
└── RoundRobinPartitioner  → no keys (Kafka default)
```

### Partition Key Generation

#### Composite Strategy (Recommended)
```
Key Format: f"{exchange}:{symbol}"
Example: "binance:BTC-USDT"
Benefits: Even distribution, related data co-location
```

#### Symbol Strategy
```
Key Format: symbol
Example: "BTC-USDT"
Benefits: All exchanges for symbol on same partition
```

#### Exchange Strategy
```
Key Format: exchange
Example: "binance"
Benefits: Exchange-specific processing
```

## Serialization Layer

### Serialization Architecture

```
Message → Converter → Protobuf Message → Validation → Bytes
```

### Protocol Buffer Integration

#### Schema Organization
```
gen/python/cryptofeed/normalized/v1/
├── events_pb2.py       # Base event definitions
├── trade_pb2.py        # Trade messages
├── ticker_pb2.py       # Ticker messages
├── candle_pb2.py       # OHLCV candles
└── ...                 # All supported data types
```

#### Converter Registry
```python
# converters.py
CONVERTERS = {
    'trade': convert_trade,
    'ticker': convert_ticker,
    'candle': convert_candle,
    # ...
}
```

#### Validation Layer
```python
# validation.py
class SchemaValidator:
    def validate(self, message: Message) -> None:
        # Check required fields
        # Validate enum values
        # Ensure schema compatibility
```

## Header Enrichment

### Standard Headers

All messages include metadata headers for routing and debugging:

| Header | Description | Example |
|--------|-------------|---------|
| `cf.exchange` | Exchange name | `b"binance"` |
| `cf.symbol` | Trading symbol | `b"BTC-USDT"` |
| `cf.data_type` | Message type | `b"trade"` |
| `cf.serialization_format` | Format used | `b"protobuf"` |
| `cf.schema_version` | Schema version | `b"v1"` |
| `cf.timestamp` | Unix timestamp | `b"1640995200.123"` |

### HeaderEnricher Class

**Purpose**: Centralized header management

**Features**:
- Automatic header generation
- Configurable header sets
- Type-safe header values

## Error Handling

### Error Types

- **Connection Errors**: Kafka connectivity issues
- **Serialization Errors**: Message format problems
- **Validation Errors**: Schema compliance failures
- **Producer Errors**: Delivery failures

### Error Recovery

- **Retry Logic**: Configurable retry attempts with backoff
- **Circuit Breaker**: Optional failure threshold detection
- **Graceful Degradation**: Continue processing other messages
- **Structured Logging**: Detailed error context

## Metrics and Monitoring

### Metrics Architecture

```
MetricsExporter
├── QueueMetrics      → Depth, processing rates
├── ProducerMetrics   → Delivery success/failure
├── SerializationMetrics → Performance, error counts
└── PartitionMetrics  → Distribution statistics
```

### Prometheus Integration

```python
# Example metrics
kafka_messages_total{status="sent"} 15432
kafka_serialization_duration_seconds{quantile="0.95"} 0.0021
kafka_queue_depth 23
```

## Configuration System

### Configuration Flow

```
User Config → Validation → Normalization → Component Injection
```

### KafkaConfig Class

**Purpose**: Type-safe configuration management

**Features**:
- Runtime validation
- Environment variable support
- YAML configuration loading
- Backward compatibility

## Performance Optimizations

### Throughput Optimizations

1. **Batch Processing**: Configurable batch sizes and linger times
2. **Async Processing**: Non-blocking message queuing
3. **Connection Pooling**: Efficient Kafka producer reuse
4. **Compression**: Optional gzip/lz4 compression

### Memory Optimizations

1. **Object Reuse**: Message object pooling
2. **Streaming Serialization**: Minimal memory allocation
3. **Queue Bounded**: Configurable queue depths
4. **GC Tuning**: Optimized garbage collection

### Latency Optimizations

1. **Zero-Copy**: Direct buffer operations where possible
2. **Header Caching**: Pre-computed header values
3. **Fast Path**: Optimized code paths for common cases
4. **Async I/O**: Non-blocking network operations

## Testing Architecture

### Test Layers

```
Unit Tests → Integration Tests → E2E Tests
     │             │              │
   Components  → Services → Full Pipeline
```

### Mock Strategy

- **Kafka Mock**: aiokafka mock for unit tests
- **Real Kafka**: Docker-based integration tests
- **Performance Tests**: Benchmarking with real workloads

## Deployment Patterns

### Development Deployment

```
┌─────────────────┐    ┌─────────────────┐
│   Cryptofeed    │───▶│   Kafka (Local) │
│   Producer      │    │   Single Node   │
└─────────────────┘    └─────────────────┘
```

### Production Deployment

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Cryptofeed    │───▶│ Kafka Cluster   │───▶│   Consumers     │
│   Producers     │    │ 3+ Nodes        │    │   (Flink/Spark) │
│   (Multiple)    │    │ HA Configuration│    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### High Availability

- **Producer Redundancy**: Multiple producer instances
- **Kafka Clustering**: Multi-broker setup
- **Consumer Groups**: Automatic failover
- **Monitoring**: Comprehensive observability

## Security Considerations

### Network Security

- **TLS Encryption**: Optional SSL/TLS transport
- **Authentication**: SASL/SCRAM or Kerberos
- **Authorization**: ACL-based topic access control

### Data Security

- **Message Encryption**: Optional end-to-end encryption
- **Audit Logging**: Comprehensive access logging
- **Data Validation**: Schema-based input validation

## Future Extensions

### Planned Features

- **Exactly-Once Semantics**: Idempotent producers
- **Schema Registry Integration**: Confluent Schema Registry
- **Advanced Routing**: Rule-based message routing
- **Multi-Cluster Support**: Cross-datacenter replication

### Extension Points

- **Custom Serializers**: Pluggable serialization formats
- **Custom Partitioners**: User-defined partitioning logic
- **Custom Validators**: Domain-specific validation rules
- **Custom Metrics**: Application-specific monitoring</content>
<parameter name="filePath">docs/kafka/architecture.md