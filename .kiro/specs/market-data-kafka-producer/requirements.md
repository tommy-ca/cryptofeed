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
**Two topic strategies** (configurable):

**Default: Consolidated Topics** (O(data_types) = 8 topics)
- Topic naming: `cryptofeed.{data_type}`
  - Examples: `cryptofeed.trades`, `cryptofeed.orderbook`, `cryptofeed.ticker`
- Advantages: Single consumer subscription per data type, simplified downstream routing
- Target: 10,000+ msg/s per topic (multi-exchange, multi-symbol aggregation)

**Optional: Per-Symbol Topics** (O(symbols × exchanges) = 80,000+ topics)
- Topic naming: `cryptofeed.{data_type}.{exchange}.{symbol}`
  - Examples: `cryptofeed.trades.coinbase.btc-usd`, `cryptofeed.orderbook.binance.eth-usdt`
- Advantages: Per-pair ordering guarantees, single-symbol consumer subscriptions
- Legacy support: For transition period during migration

**Common**:
- Auto-create topics with configurable partition count (default: 12 partitions per topic)
- Support topic prefix/namespace for multi-tenant deployments
- Allow custom topic routing via configuration
- Include routing metadata in message headers (exchange, symbol, data_type, schema_version)

### FR3: Partitioning Strategies
**Four partitioning strategies** (configurable via YAML, default: composite):

1. **Composite (Recommended Default)**: Partition key = `{exchange}-{symbol}`
   - Ensures per-exchange-pair ordering guarantees
   - Distributes across partitions when symbols per exchange > partition count
   - Example: `coinbase-btc-usd`, `binance-eth-usdt`

2. **Symbol-Only**: Partition key = `{symbol}`
   - Groups same symbol across all exchanges in one partition
   - Cross-exchange order preservation (useful for arbitrage scenarios)

3. **Exchange-Only**: Partition key = `{exchange}`
   - Groups all symbols for one exchange in same partition
   - Useful when consumer processes exchange-specific logic

4. **Round-Robin**: Partition key = `None`
   - Maximum parallelism, no ordering guarantees
   - When order irrelevant (e.g., aggregation windows)

**Strategy Matrix** (when to use):
| Strategy | Use Case | Ordering | Partition Distribution |
|----------|----------|----------|------------------------|
| Composite | Real-time trading (DEFAULT) | Per-pair | Excellent |
| Symbol | Cross-exchange analysis | Per-symbol | Good |
| Exchange | Exchange-specific processing | Per-exchange | Fair |
| Round-robin | Aggregate analytics | None | Perfect (load balance) |

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
  - `cryptofeed_kafka_messages_sent_total` (by data_type, exchange, topic_strategy)
  - `cryptofeed_kafka_bytes_sent_total` (by data_type, exchange)
  - `cryptofeed_kafka_produce_latency_seconds` (p50, p95, p99)
  - `cryptofeed_kafka_errors_total` (by error_type, exchange)
  - Topic health metrics (partition lag, replication status)
- Structured logging (JSON format with correlation IDs)
- Health check endpoint (Kafka broker connectivity, producer status)

### FR7: Migration & Backward Compatibility
**Problem**: Existing deployments use per-symbol topics; moving to consolidated topics requires coordination.

**Solution**: Support dual-write transition period with gradual consumer migration.

**Phase 1: Dual-Write (Weeks 1-2)**
- Configure KafkaCallback to publish to **both** topic strategies simultaneously
- Message published to consolidated `cryptofeed.{data_type}` AND per-symbol `cryptofeed.{data_type}.{exchange}.{symbol}`
- No consumer changes required; existing consumers on per-symbol topics continue unchanged
- New consumers can subscribe to consolidated topics

**Phase 2: Consumer Migration (Weeks 3-8)**
- Existing consumers migrate subscriptions from per-symbol to consolidated topics
- Validation suite confirms dual-write message ordering equivalence
- Rollback plan: disable dual-write, revert to per-symbol only

**Phase 3: Cutover (Weeks 9-10)**
- Disable per-symbol topic publishing (consolidated topics only)
- Health monitoring for consumer processing latency and lag
- Alert if any consumer lag increases >5 seconds

**Phase 4: Cleanup (Weeks 11-12)**
- Delete per-symbol topics and Kafka cleanup code
- Archive legacy configuration examples
- Document migration lessons learned

**Backward Compatibility**:
- Configuration flag: `topic_strategy: [consolidated | per_symbol | dual_write]`
- Default for new deployments: `consolidated`
- Default for upgrades: `dual_write` (automatic, no code changes)
- Removal timeline: 4-5 weeks from initial dual-write deployment

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
