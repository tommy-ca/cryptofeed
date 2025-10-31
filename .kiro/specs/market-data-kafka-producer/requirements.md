# Market Data Kafka Producer - Requirements

## Overview

Provide high-performance Kafka producer integration for cryptofeed, serializing normalized market data (from Spec 0) into protobuf messages (from Spec 1) and publishing to Kafka topics.

**Scope**: Ingestion layer only. Storage integration (Iceberg, Parquet, DuckDB) delegated to downstream consumers.

## Goals

1. **High Throughput**: Handle 10,000+ messages/second per feed
2. **Reliability**: Exactly-once semantics with idempotent producers
3. **Observability**: Comprehensive metrics and logging
4. **Flexibility**: Support multiple topic strategies and partitioning schemes

## Functional Requirements

### FR1: Kafka Backend Implementation
- Extend `BackendCallback` with `KafkaCallback` class
- Integrate with `confluent-kafka-python` for producer API
- Support both sync and async message publishing
- Configurable batch size, linger time, compression

### FR2: Topic Management
- Topic naming convention: `cryptofeed.{data_type}.{exchange}.{symbol}`
  - Example: `cryptofeed.trades.coinbase.btc-usd`
- Auto-create topics with configurable partition count
- Support topic prefix/namespace for multi-tenant deployments
- Allow custom topic routing via configuration

### FR3: Partitioning Strategies
- Default: Hash by symbol (ensures order per symbol)
- Optional: Round-robin for maximum parallelism
- Optional: Key by exchange (group by exchange)
- Configurable via YAML

### FR4: Serialization Integration
- Use `to_proto()` methods from Spec 1
- Serialize all 20 data types (Trade, L2Book, Ticker, etc.)
- Include schema version in message headers
- Support schema registry (Confluent or Buf)

### FR5: Delivery Guarantees
- Exactly-once semantics via idempotent producer
- Configurable acks (0, 1, all)
- Retry logic with exponential backoff
- Dead letter queue for failed messages

### FR6: Monitoring & Observability
- Prometheus metrics:
  - `cryptofeed_kafka_messages_sent_total`
  - `cryptofeed_kafka_bytes_sent_total`
  - `cryptofeed_kafka_produce_latency_seconds`
  - `cryptofeed_kafka_errors_total`
- Structured logging (JSON format)
- Health check endpoint

## Non-Functional Requirements

### NFR1: Performance
- Target: 10,000 messages/second per producer instance
- Latency: p99 < 100ms from callback to Kafka ACK
- Memory: < 512MB per producer instance

### NFR2: Reliability
- Handle Kafka broker failures gracefully
- Automatic reconnection with backoff
- No message loss under normal operation

### NFR3: Configuration
- YAML-based configuration
- Environment variable overrides
- Hot reload for non-critical settings

## Scope Boundaries

### IN-SCOPE
- Kafka producer implementation (BackendCallback extension)
- Topic management and partitioning
- Protobuf serialization integration
- Delivery guarantees and error handling
- Metrics and monitoring

### OUT-OF-SCOPE (Delegated to Consumers)
- Kafka consumer implementation
- Apache Iceberg integration
- DuckDB/Parquet storage backends
- Stream processing (Flink, Spark, QuixStreams)
- Data retention and compaction policies
- Query engines and analytics

**Boundary**: This spec ends at Kafka topic production. Consumers read topics and implement storage/analytics independently.

## Integration Examples

### Example 1: Flink → Iceberg
Consumer implements Flink job reading `cryptofeed.trades.*` topics and writing to Iceberg tables with schema evolution.

### Example 2: DuckDB Direct
Consumer implements Python script reading Kafka topics and inserting into DuckDB tables via `INSERT INTO ... SELECT`.

### Example 3: Spark Streaming
Consumer implements Spark Structured Streaming job aggregating trades into OHLCV candles and writing to Parquet.

## Success Criteria

1. Kafka producer publishes protobuf messages at 10,000 msg/s
2. Exactly-once delivery verified via integration tests
3. Metrics available in Prometheus format
4. Documentation includes consumer integration examples
5. Zero message loss under failover scenarios

## Dependencies

- **Spec 0** (normalized-data-schema-crypto): Provides .proto schemas
- **Spec 1** (protobuf-callback-serialization): Provides `to_proto()` methods
- **External**: Kafka cluster (3+ brokers recommended)
- **External**: Schema registry (Confluent or Buf)

## Timeline

- **Design Phase**: 3-5 days
- **Implementation**: 2-3 weeks
- **Testing**: 1 week
- **Total**: 4-5 weeks

## Open Questions

1. Should we support Kafka Streams for stateful processing? (Likely NO - delegate to consumers)
2. Should we provide reference consumer implementations? (YES - as examples in docs)
3. Should we support Avro in addition to protobuf? (DEFER - protobuf only for now)
