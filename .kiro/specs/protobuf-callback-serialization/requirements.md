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
7. **WHEN** configuration specifies an invalid serialization format (not 'json' or 'protobuf') **THEN** the system SHALL raise a `ValueError` with a clear error message listing valid formats
8. **WHEN** YAML configuration is parsed **THEN** the parser SHALL validate the `serialization_format` field and reject invalid values before callback instantiation
9. **WHEN** environment variable `CRYPTOFEED_CALLBACK_FORMAT` is set **THEN** the value SHALL be case-insensitive (e.g., 'Protobuf', 'PROTOBUF', 'protobuf' all accepted)
10. **WHEN** programmatic API is used with `serialization_format` parameter **THEN** the parameter SHALL accept string values 'json' or 'protobuf' and raise `ValueError` for invalid inputs

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
10. **WHEN** KafkaCallback is instantiated with `serialization_format="protobuf"` **THEN** the `topic()` method SHALL override the default topic naming to use hierarchical pattern `cryptofeed.market.{data_type}.{exchange}`
11. **WHEN** KafkaCallback is instantiated with `serialization_format="protobuf"` **THEN** the `partition_key()` method SHALL return the normalized symbol as bytes (encoded as UTF-8)
12. **WHEN** KafkaCallback produces a protobuf message **THEN** the message SHALL be sent with the configured `value_serializer` or default to binary bytes (not JSON-wrapped)
13. **WHEN** multiple exchanges are configured for the same data type **THEN** each exchange SHALL write to its own topic (e.g., `cryptofeed.market.trades.coinbase`, `cryptofeed.market.trades.binance`)
14. **WHEN** JSON format is used **THEN** the existing topic naming strategy SHALL be preserved for backward compatibility (e.g., `trades-coinbase-BTC-USD`)

---

### Requirement 4.5: Exception Handling and Error Management

**Objective**: As a cryptofeed operator, I want clear, actionable error messages when serialization fails, so that I can quickly diagnose and resolve issues without deep debugging.

#### Acceptance Criteria

1. **WHEN** a data type is missing the `to_proto()` method **THEN** the system SHALL raise `SerializationError` with message: `"{TypeName} missing to_proto() method. Ensure all data types implement to_proto()."`
2. **WHEN** protobuf encoding fails due to invalid data **THEN** the system SHALL raise `ProtobufEncodeError` with the underlying protobuf error message and data type context
3. **WHEN** a serializer encounters an unexpected exception **THEN** the system SHALL wrap it in `SerializationError` and include the original exception as the cause (via `from` clause)
4. **WHEN** an invalid serialization format is specified **THEN** the system SHALL raise `ValueError` with message: `"Invalid serialization format '{format}'. Valid formats: json, protobuf"`
5. **WHEN** a `to_proto()` method returns a non-protobuf object **THEN** the system SHALL raise `ProtobufEncodeError` with message: `"to_proto() returned {type}, expected protobuf Message"`
6. **WHEN** serialization fails in a BackendCallback **THEN** the error SHALL be logged with context (data type, exchange, symbol, timestamp) and SHALL NOT crash the feed
7. **WHEN** a SerializationError occurs **THEN** the error message SHALL include the data type name and a hint about checking the `to_proto()` implementation
8. **WHEN** custom exceptions are defined **THEN** they SHALL inherit from a common `CryptofeedSerializationException` base class for easy catching
9. **WHEN** exceptions are raised **THEN** they SHALL preserve the full exception chain for debugging (no suppressed exceptions)
10. **WHEN** a ProtobufEncodeError occurs **THEN** the error message SHALL include the protobuf schema name and version for schema compatibility debugging

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

### Requirement 7.5: C Extension Wrapper Adapter

**Objective**: As a cryptofeed developer, I want a clean adapter layer between C extension data types and protobuf wrappers, so that serialization logic is decoupled from the core C extension implementation.

#### Acceptance Criteria

1. **WHEN** a C extension data type (e.g., `cryptofeed.types.Trade`) is passed to serialization **THEN** an adapter function SHALL wrap it in a Python wrapper class with `to_proto()` method
2. **WHEN** the wrapper adapter is invoked **THEN** it SHALL detect the data type and route to the appropriate wrapper class (e.g., `TradeWrapper`, `OrderBookWrapper`)
3. **WHEN** a wrapper class is instantiated **THEN** it SHALL store a reference to the underlying C extension object without copying data
4. **WHEN** a wrapper's `to_proto()` method is called **THEN** it SHALL access fields from the C extension object and construct a protobuf message
5. **WHEN** an unsupported data type is passed to the adapter **THEN** it SHALL raise `SerializationError` with message: `"No protobuf wrapper available for {TypeName}"`
6. **WHEN** multiple data types are serialized in sequence **THEN** the adapter SHALL maintain zero internal state between invocations (stateless design)
7. **WHEN** a wrapper adapter is used **THEN** it SHALL add <100 microseconds overhead per message compared to direct protobuf serialization
8. **WHEN** all 14 data types are implemented **THEN** the adapter SHALL support all types via a single `wrap_for_serialization(obj)` function
9. **WHEN** a new data type is added in the future **THEN** adding wrapper support SHALL require only creating a new wrapper class and registering it in the adapter
10. **WHEN** wrapper classes are tested **THEN** they SHALL have 100% test coverage including edge cases (null fields, empty collections, boundary values)

---

### Requirement 8: Performance Characteristics

**Objective**: As a platform engineer, I want protobuf serialization to meet performance requirements for high-throughput market data ingestion, so that serialization overhead does not become a bottleneck.

#### Acceptance Criteria

**Baseline Dataset Definition**:
- **Trade Workload**: 10,000 Trade messages (BTC-USD, typical payload ~200 bytes JSON, ~80 bytes protobuf)
- **OrderBook Workload**: 1,000 OrderBook messages (50 price levels, typical payload ~5KB JSON, ~2KB protobuf)
- **Mixed Workload**: 7,000 Trade + 2,000 OrderBook + 1,000 Ticker messages
- **Measurement Method**: Single-threaded execution, warm JIT, average of 5 runs, discard outliers

**Latency Requirements**:
1. **WHEN** a Trade message is serialized to protobuf **THEN** p50 latency SHALL be < 0.3ms, p95 < 0.6ms, p99 < 1ms
2. **WHEN** an OrderBook message with 50 price levels is serialized to protobuf **THEN** p50 latency SHALL be < 1ms, p95 < 1.5ms, p99 < 2ms
3. **WHEN** a sequence of 10,000 market messages is serialized **THEN** average serialization latency per message SHALL be < 500 microseconds
4. **WHEN** protobuf serialization is compared to JSON serialization on identical workloads **THEN** protobuf SHALL perform serialization in ≤ 150% of JSON time (acceptable tradeoff for size reduction)

**Throughput Requirements**:
5. **WHEN** serializing the baseline Trade workload **THEN** throughput SHALL be ≥ 10,000 messages/second on a single core
6. **WHEN** serializing the baseline mixed workload **THEN** throughput SHALL be ≥ 5,000 messages/second on a single core

**Memory Requirements**:
7. **WHEN** memory profiling is performed on protobuf serialization **THEN** peak memory usage SHALL not exceed JSON serialization memory usage + 10MB
8. **WHEN** 1 million messages are serialized **THEN** memory usage SHALL remain stable (no memory leaks, <5% growth)
9. **WHEN** wrapper adapter creates wrapper instances **THEN** wrapper object overhead SHALL be < 200 bytes per instance

**Size Reduction Requirements**:
10. **WHEN** Kafka writes compressed with lz4 **THEN** protobuf-serialized messages SHALL result in 50-60% smaller compressed payloads than JSON
11. **WHEN** Kafka writes compressed with zstd **THEN** protobuf-serialized messages SHALL result in 55-65% smaller compressed payloads than JSON
12. **WHEN** uncompressed protobuf messages are measured **THEN** they SHALL be 55-65% smaller than uncompressed JSON
13. **WHEN** size metrics are collected **THEN** the performance report SHALL include percentile distributions (p50, p95, p99) for both JSON and protobuf sizes across uncompressed, lz4-compressed, and zstd-compressed outputs

**Profiling Requirements**:
14. **WHEN** performance benchmarks are run **THEN** cProfile SHALL be used to identify hot paths (functions consuming >5% of total time)
15. **WHEN** profiling results are analyzed **THEN** Decimal-to-string conversion and timestamp conversion SHALL be identified as optimization targets if they exceed 10% of serialization time
16. **WHEN** baseline metrics are established **THEN** they SHALL be documented in `docs/protobuf-performance-baseline.md` for regression tracking

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

1. ✅ All 14 data types have protobuf serialization via wrapper pattern (Trade, OrderBook, Ticker, Candle, Funding, Liquidation, OpenInterest, Index, Balance, Position, Fill, OrderInfo, Transaction, Order)
2. ✅ Custom exception classes defined (`CryptofeedSerializationException`, `SerializationError`, `ProtobufEncodeError`) with clear error messages
3. ✅ `BackendCallback` supports both JSON and Protobuf formats with configuration (YAML + environment variables)
4. ✅ Configuration parser validates `serialization_format` and handles case-insensitive values
5. ✅ Kafka topic routing implemented with `cryptofeed.market.{data_type}.{exchange}` pattern for protobuf format
6. ✅ Kafka partition key set to normalized symbol (UTF-8 bytes) for consistent routing
7. ✅ Wrapper adapter layer implemented for C extension → Python wrapper conversion
8. ✅ Configuration via YAML and Python API fully documented with examples
9. ✅ 95%+ test coverage for serialization layer, 100% for wrapper classes
10. ✅ Performance benchmarks meet baseline targets (p99 <1ms Trade, <2ms OrderBook, ≥10k msg/s throughput)
11. ✅ Size metrics show ≥50% reduction vs JSON (compressed with lz4 and zstd)
12. ✅ Memory usage stable after 1M messages (<5% growth)
13. ✅ Zero breaking changes to existing JSON-based backends (backward compatible)
14. ✅ End-to-end integration test with Kafka for all 14 data types
15. ✅ Performance baseline documented in `docs/protobuf-performance-baseline.md`
16. ✅ User guide and consumer integration examples documented

---

## Timeline & Resources

**Duration**: 1-2 weeks (Weeks 1-2 of 10-week plan)

**Tasks**: Implementation tasks will be generated in separate specification document

**Ownership**: Development team with data engineering oversight for Kafka topology validation

---

*This specification narrows the scope from the initial design to focus exclusively on market data types (6 types) for Phase 1, deferring user/account data and PERPETUAL products to Phase 2.*
