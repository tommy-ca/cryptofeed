# Requirements: Protobuf Callback Serialization (Spec 1)

## Introduction

This specification establishes the foundation for protobuf-native data serialization in cryptofeed backend callbacks, enabling efficient binary encoding of normalized market data. Protobuf serialization reduces payload sizes by 50-60% compared to JSON while maintaining full type safety and backward compatibility. This foundation layer directly enables the QuixStreams stream processing layer (Spec 2) and the lakehouse backend adapter (Spec 3).

**Scope**: Binary serialization for 20 cryptofeed data types using protobuf format. Spec 1 produces serialized messages for consumption by downstream systems (Kafka producers, storage backends).

---

## Scope Boundaries

### IN-SCOPE (Spec 1 Responsibilities)
- Add `to_proto()` methods to all 20 cryptofeed data types
- Extend BackendCallback to support protobuf serialization format
- Kafka backend integration with protobuf messages
- Redis/ZMQ backend support for protobuf payloads
- Schema registry integration (publish schemas to Buf/Confluent)
- Serialization performance testing and benchmarks
- Backward-compatible schema evolution

### OUT-OF-SCOPE (Delegated to Downstream Consumers)
- Storage layer implementation (Apache Iceberg, DuckDB, Parquet)
- Lakehouse table management and schema evolution
- Query engine integration (Flink, Spark, Trino, DuckDB)
- Stream processing and real-time aggregations (QuixStreams, Kafka Streams)
- Data retention policies and compaction strategies
- Analytics pipelines and cross-exchange analytics
- Consumer implementations (sinks, integrations)

**Architectural Boundary**: Spec 1 produces protobuf-serialized messages to Kafka topics (or other backends). Storage, analytics, and consumer integrations are downstream responsibilities handled by Spec 3 (market-data-kafka-producer) and external consumers.

---

## Requirements

### Requirement 1: Protobuf Conversion Methods on Market Data Types

**Objective**: As a cryptofeed extension developer, I want each normalized market data type to provide a `to_proto()` method that converts Python objects to protobuf messages, so that serialization to Kafka is type-safe and efficient.

#### Acceptance Criteria

1. **WHEN** a `Trade` object is created with exchange, symbol, side, price, amount, and timestamp fields **THEN** the `Trade` object SHALL have a callable `to_proto()` method that returns a `cryptofeed.normalized.v1.trade_pb2.Trade` message
2. **WHEN** a `Trade.to_proto()` method is invoked **THEN** the protobuf message SHALL preserve all fields (exchange, symbol, side, trade_id, price, amount, timestamp, type) with numeric precision maintained via string encoding for Decimal fields
3. **WHEN** an `OrderBook` object with bids/asks structure is created **THEN** the `OrderBook` object SHALL have a callable `to_proto()` method that returns a `cryptofeed.normalized.v1.order_book_pb2.OrderBook` message
4. **WHEN** an `OrderBook.to_proto()` method is invoked **THEN** the protobuf message SHALL encode all price levels with timestamp, symbol, exchange, and book snapshot metadata
5. **WHEN** a `Candle` object with OHLCV data is created **THEN** the `Candle` object SHALL have a callable `to_proto()` method that returns a `cryptofeed.normalized.v1.candle_pb2.Candle` message
6. **WHEN** a `Candle.to_proto()` method is invoked **THEN** the protobuf message SHALL preserve open, high, low, close, volume, trades count, interval, and timestamp fields
7. **WHEN** a `Ticker` object with top-of-book data is created **THEN** the `Ticker` object SHALL have a callable `to_proto()` method that returns a `cryptofeed.normalized.v1.ticker_pb2.Ticker` message
8. **WHEN** a `Ticker.to_proto()` method is invoked **THEN** the protobuf message SHALL encode bid, ask, bid_size, ask_size, timestamp, and symbol
9. **WHEN** a `FundingRate` object is created **THEN** the `FundingRate` object SHALL have a callable `to_proto()` method that returns a `cryptofeed.normalized.v1.funding_pb2.FundingRate` message
10. **WHEN** a `FundingRate.to_proto()` method is invoked **THEN** the protobuf message SHALL encode symbol, rate, next_funding_time, and timestamp
11. **WHEN** a `Liquidation` object is created **THEN** the `Liquidation` object SHALL have a callable `to_proto()` method that returns a `cryptofeed.normalized.v1.liquidation_pb2.Liquidation` message
12. **WHEN** a `Liquidation.to_proto()` method is invoked **THEN** the protobuf message SHALL encode symbol, side, quantity, price, timestamp, and exchange

---

### Requirement 2: Backend Callback Serialization Format Support

**Objective**: As a cryptofeed backend operator, I want to configure the serialization format (JSON or Protobuf) at the callback level, so that I can leverage efficient binary formats for Kafka topic ingestion without changing existing feed configuration.

#### Acceptance Criteria

1. **WHEN** a callback is instantiated without explicit serialization format **THEN** the callback SHALL default to JSON serialization for backward compatibility
2. **WHEN** a `KafkaCallback` is instantiated with `serialization_format="protobuf"` parameter **THEN** the callback SHALL serialize all received market data types to protobuf binary format
3. **WHEN** a `KafkaCallback` is instantiated with `serialization_format="json"` parameter **THEN** the callback SHALL serialize all received market data types to JSON text format
4. **WHEN** a callback receives a data object and the serialization format is protobuf **THEN** the callback SHALL invoke `data.to_proto()` to obtain a protobuf message
5. **WHEN** a callback receives a data object and the serialization format is protobuf **THEN** the callback SHALL call `.SerializeToString()` on the protobuf message to obtain binary bytes
6. **WHEN** a callback receives a data object and the serialization format is JSON **THEN** the callback SHALL invoke existing `to_dict()` or dictionary conversion to obtain JSON-serializable structure
7. **WHEN** both JSON and Protobuf formats are available **THEN** both serialization paths SHALL operate independently without interference or state coupling

---

### Requirement 3: Backward Compatibility and Configuration

**Objective**: As a cryptofeed operator, I want existing JSON-based backends to continue functioning without modification, so that I can migrate to Protobuf incrementally without breaking production systems.

#### Acceptance Criteria

1. **WHEN** an existing backend configuration uses JSON serialization **THEN** the backend SHALL continue to function exactly as it did before this specification
2. **WHEN** a new backend configuration specifies protobuf format **THEN** the new backend SHALL operate alongside JSON backends in the same FeedHandler instance
3. **WHEN** configuration is provided via YAML with key `serialization_format: protobuf` **THEN** the callback factory SHALL instantiate the callback with protobuf format enabled
4. **WHEN** configuration is provided via YAML with key `serialization_format: json` or omitted **THEN** the callback factory SHALL instantiate the callback with JSON format (default)
5. **WHEN** configuration is provided via environment variable `CRYPTOFEED_CALLBACK_FORMAT=protobuf` **THEN** the callback SHALL override default JSON format with protobuf format
6. **WHEN** both YAML and environment variable are specified **THEN** the environment variable SHALL take precedence

---

### Requirement 4: Kafka Topic Routing and Partition Strategy

**Objective**: As a lakehouse engineer, I want Kafka topics to be organized by market data type and exchange, so that stream processors can subscribe to specific data channels (e.g., all trades, all orderbook snapshots) without topic explosion.

#### Acceptance Criteria

1. **WHEN** a Trade callback is configured with `serialization_format="protobuf"` and exchange "coinbase" **THEN** the message SHALL be written to topic `cryptofeed.market.trades.coinbase`
2. **WHEN** a Trade callback is configured with `serialization_format="protobuf"` and exchange "binance" **THEN** the message SHALL be written to topic `cryptofeed.market.trades.binance`
3. **WHEN** an OrderBook callback is configured with `serialization_format="protobuf"` **THEN** the message SHALL be written to topic `cryptofeed.market.orderbook.{exchange}` (e.g., `cryptofeed.market.orderbook.coinbase`)
4. **WHEN** a Candle callback is configured with `serialization_format="protobuf"` **THEN** the message SHALL be written to topic `cryptofeed.market.candles.{exchange}` (e.g., `cryptofeed.market.candles.coinbase`)
5. **WHEN** a Ticker callback is configured with `serialization_format="protobuf"` **THEN** the message SHALL be written to topic `cryptofeed.market.ticker.{exchange}` (e.g., `cryptofeed.market.ticker.coinbase`)
6. **WHEN** a FundingRate callback is configured with `serialization_format="protobuf"` **THEN** the message SHALL be written to topic `cryptofeed.market.funding.{exchange}` (e.g., `cryptofeed.market.funding.binance`)
7. **WHEN** a Liquidation callback is configured with `serialization_format="protobuf"` **THEN** the message SHALL be written to topic `cryptofeed.market.liquidation.{exchange}` (e.g., `cryptofeed.market.liquidation.binance`)
8. **WHEN** a message is written to a Kafka topic **THEN** the partition key SHALL be the normalized trading symbol (e.g., `BTC-USD`, `BTCUSDT`) to ensure all events for a symbol are routed to the same partition
9. **WHEN** a message is written to a Kafka topic with protobuf serialization **THEN** the Kafka message value SHALL be the binary protobuf bytes (not wrapped in JSON)

---

### Requirement 5: Protobuf Schema Alignment

**Objective**: As a schema maintainer, I want the cryptofeed data types to map precisely to normalized-data-schema-crypto protobuf definitions, so that downstream Kafka consumers and lakehouse systems can depend on a single schema source.

#### Acceptance Criteria

1. **WHEN** any market data type is serialized to protobuf **THEN** the resulting message SHALL conform to the schema defined in `normalized-data-schema-crypto` v0.1.0 (release date Oct 20, 2025)
2. **WHEN** a Trade is serialized **THEN** the protobuf message SHALL comply with `proto/cryptofeed/normalized/v1/trade.proto` including all required fields (exchange, symbol, side, price, amount, timestamp)
3. **WHEN** an OrderBook is serialized **THEN** the protobuf message SHALL comply with `proto/cryptofeed/normalized/v1/order_book.proto` including bids/asks structure and snapshot metadata
4. **WHEN** a Candle is serialized **THEN** the protobuf message SHALL comply with `proto/cryptofeed/normalized/v1/candle.proto` including OHLCV fields and timestamp
5. **WHEN** a Ticker is serialized **THEN** the protobuf message SHALL comply with `proto/cryptofeed/normalized/v1/ticker.proto` including bid/ask and size fields
6. **WHEN** a FundingRate is serialized **THEN** the protobuf message SHALL comply with `proto/cryptofeed/normalized/v1/funding.proto` including rate and next_funding_time fields
7. **WHEN** a Liquidation is serialized **THEN** the protobuf message SHALL comply with `proto/cryptofeed/normalized/v1/liquidation.proto` including symbol, side, quantity, price fields
8. **WHEN** protobuf schemas in `normalized-data-schema-crypto` are updated **THEN** cryptofeed serialization logic SHALL be updated within one release cycle to maintain alignment

---

### Requirement 6: Type Safety and Validation

**Objective**: As a data engineer, I want protobuf serialization to be type-safe and statically verifiable, so that I can catch serialization errors at development time rather than at runtime.

#### Acceptance Criteria

1. **WHEN** a `to_proto()` method is implemented on a data type **THEN** the method signature SHALL include a return type hint (e.g., `-> trade_pb2.Trade`)
2. **WHEN** protobuf serialization code is analyzed with mypy **THEN** mypy SHALL report zero type errors
3. **WHEN** a Decimal field is serialized to protobuf **THEN** the serialization logic SHALL convert the Decimal to a string representation to preserve precision beyond float64 limits
4. **WHEN** a timestamp field is serialized to protobuf **THEN** the timestamp SHALL be converted from float seconds to int64 microseconds to match protobuf schema precision
5. **WHEN** a string field (e.g., symbol, exchange) is serialized to protobuf **THEN** the string SHALL be validated for non-null and non-empty before serialization
6. **WHEN** an enum field (e.g., side: buy/sell) is serialized to protobuf **THEN** the value SHALL be converted to the corresponding protobuf enum type (e.g., `TradeSide.BUY`)

---

### Requirement 7: Testing Coverage

**Objective**: As a QA engineer, I want comprehensive test coverage for protobuf serialization, so that I can confidently deploy this change to production without regression risks.

#### Acceptance Criteria

1. **WHEN** the protobuf serialization module is tested **THEN** test coverage SHALL be ≥ 95% for all serialization code
2. **WHEN** a Trade is serialized to protobuf and then deserialized **THEN** the resulting object fields SHALL match the original Trade object fields (round-trip equality)
3. **WHEN** an OrderBook with 10+ price levels is serialized to protobuf **THEN** all price levels SHALL be preserved without loss or reordering
4. **WHEN** a Candle with high precision OHLCV values (e.g., 1.123456789) is serialized **THEN** the protobuf message SHALL preserve all decimal places via string encoding
5. **WHEN** a callback receives 1000 messages and serialization format is protobuf **THEN** all 1000 messages SHALL be serialized without errors or memory leaks
6. **WHEN** protobuf serialization fails due to invalid data **THEN** the error SHALL be logged with context (data type, exchange, symbol) and not crash the feed
7. **WHEN** both JSON and Protobuf formats are tested with identical input data **THEN** size comparison metrics SHALL show Protobuf at 40-60% of JSON size

---

### Requirement 8: Performance Characteristics

**Objective**: As a platform engineer, I want protobuf serialization to meet performance requirements for high-throughput market data ingestion, so that serialization overhead does not become a bottleneck.

#### Acceptance Criteria

1. **WHEN** a Trade message is serialized to protobuf **THEN** serialization latency SHALL be < 1 millisecond
2. **WHEN** an OrderBook message with 50 price levels is serialized to protobuf **THEN** serialization latency SHALL be < 2 milliseconds
3. **WHEN** a sequence of 10,000 market messages is serialized **THEN** average serialization latency per message SHALL be < 500 microseconds
4. **WHEN** protobuf serialization is compared to JSON serialization on identical workloads **THEN** protobuf SHALL perform serialization in ≤ 150% of JSON time
5. **WHEN** memory profiling is performed on protobuf serialization **THEN** peak memory usage SHALL not exceed JSON serialization memory usage
6. **WHEN** Kafka writes compressed with lz4 **THEN** protobuf-serialized messages SHALL result in 50-60% smaller compressed payloads than JSON

---

### Requirement 9: Documentation and Integration Examples

**Objective**: As a cryptofeed operator, I want clear documentation and example configurations, so that I can quickly enable protobuf serialization in my Kafka backends without trial and error.

#### Acceptance Criteria

1. **WHEN** an operator wants to enable protobuf serialization for Kafka **THEN** documentation SHALL provide a complete YAML configuration example with all required fields
2. **WHEN** an operator wants to enable protobuf serialization via Python API **THEN** documentation SHALL provide a complete code example showing callback instantiation with `serialization_format="protobuf"`
3. **WHEN** an operator wants to understand Kafka topic naming conventions **THEN** documentation SHALL clearly specify the topic pattern (e.g., `cryptofeed.market.trades.{exchange}`) and partition key strategy
4. **WHEN** an operator wants to consume protobuf-serialized messages from Kafka **THEN** documentation SHALL provide a Python snippet showing how to deserialize using protobuf bindings
5. **WHEN** an operator wants to migrate from JSON to Protobuf incrementally **THEN** documentation SHALL demonstrate running both formats in parallel in the same FeedHandler
6. **WHEN** an operator encounters a serialization error **THEN** documentation SHALL include a troubleshooting guide with common issues and resolutions

---

## Scope Boundaries

### In Scope
- Protobuf serialization for 6 market data types: Trade, OrderBook, Candle, Ticker, FundingRate, Liquidation
- Kafka topic routing with hierarchical naming (data_type + exchange)
- MVP: Coinbase and Binance SPOT products
- Full backward compatibility with existing JSON backends
- Type-safe conversion using protobuf Python bindings

### Out of Scope
- User/account data types (Balance, Position, Fill, Order, Transaction, OrderInfo) — Phase 2
- PERPETUAL products — Phase 2
- Apache Iceberg or Parquet storage — Spec 3
- QuixStreams stream processors — Spec 2
- Alternative serialization formats (MessagePack, CBOR, etc.)

---

## Dependencies

**Upstream (Blocking)**:
- `normalized-data-schema-crypto` v0.1.0 — provides .proto schemas and generated Python bindings ✅ RELEASED

**Downstream (Depends on this spec)**:
- `quixstreams-integration` (Spec 2) — requires Kafka topics with protobuf-serialized messages
- `lakehouse-backend-adapter` (Spec 3) — requires protobuf-serialized Kafka streams for ingestion

---

## Success Criteria

1. ✅ All 6 market data types implement `to_proto()` method
2. ✅ `BackendCallback` supports both JSON and Protobuf formats with configuration
3. ✅ Kafka topic routing implemented with `cryptofeed.market.{data_type}.{exchange}` pattern
4. ✅ Configuration via YAML and Python API fully documented with examples
5. ✅ 95%+ test coverage for serialization layer
6. ✅ Performance benchmarks show < 2ms serialization latency
7. ✅ Size metrics show 40-60% reduction vs JSON
8. ✅ Zero breaking changes to existing JSON-based backends
9. ✅ End-to-end integration test with Kafka for all 6 data types
10. ✅ Production deployment guide documented

---

## Timeline & Resources

**Duration**: 1-2 weeks (Weeks 1-2 of 10-week plan)

**Tasks**: Implementation tasks will be generated in separate specification document

**Ownership**: Development team with data engineering oversight for Kafka topology validation

---

*This specification narrows the scope from the initial design to focus exclusively on market data types (6 types) for Phase 1, deferring user/account data and PERPETUAL products to Phase 2.*
