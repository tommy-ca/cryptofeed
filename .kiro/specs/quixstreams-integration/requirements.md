# Requirements Document: QuixStreams Integration

## Project Description (Input)

Stream processing layer for cryptofeed data aggregation and analytics. Consume protobuf-serialized trade and orderbook data from Kafka topics and build real-time aggregations: OHLCV candles (1m, 5m, 1h), Volume-Weighted Average Price (VWAP), volume-weighted metrics with configurable windows. Support cross-exchange analytics: price correlation analysis, bid-ask spread tracking, inter-exchange liquidity monitoring, arb opportunity detection. Python-native Kafka Streams alternative using QuixStreams framework with stateful stream processing (tumbling/sliding windows), consumer groups for multi-worker deployment, and RocksDB state stores for large aggregation windows. Enable downstream lakehouse storage of processed streams with full lineage tracking and backfill capabilities. Production-ready with comprehensive error handling, metrics emission, and configuration-driven analytics expressions.

## Specification Overview

This specification establishes the stream processing layer for real-time cryptofeed analytics, enabling:
- Efficient consumption of protobuf-serialized market data from Kafka topics
- Real-time OHLCV aggregation across configurable time windows
- Volume-weighted metrics (VWAP, weighted spread, weighted volatility)
- Cross-exchange analytics and correlation detection
- Stateful processing with RocksDB-backed state stores
- Multi-worker horizontal scaling via Kafka consumer groups
- Full lineage tracking and downstream lakehouse integration

## Context & Motivation

### Current State
- Protobuf callback serialization foundation (Spec 1: `protobuf-callback-serialization`) enables binary Kafka topics
- Cryptofeed produces 13+ exchange feeds with real-time trade and orderbook updates
- No built-in aggregation layer exists for real-time analytics
- Lakehouse architecture designed but disabled, awaiting processing foundation

### Why Now
- Protobuf foundation enables efficient stream processing without JSON overhead
- QuixStreams provides Python-native Kafka Streams alternative with lower operational overhead than Java
- Market data consumers require real-time OHLCV, VWAP, and cross-exchange metrics
- Enables data lakehouse initialization with pre-aggregated analytical streams

## Requirements

### Functional Requirements

**FR1: Real-Time OHLCV Aggregation**
- Support tumbling windows: 1m, 5m, 1h (configurable)
- Compute Open, High, Low, Close, Volume per window per symbol
- Handle late-arriving trades (configurable grace period)
- Emit completed candles to output Kafka topic

**FR2: Volume-Weighted Metrics**
- Compute VWAP (Volume-Weighted Average Price) per window
- Compute volume-weighted bid-ask spread
- Compute volume-weighted price volatility
- Support configurable weighting schemes (linear time decay, exponential)

**FR3: Cross-Exchange Analytics**
- Price correlation analysis across exchanges (sliding 1h window)
- Bid-ask spread tracking per exchange (aggregated + individual)
- Inter-exchange liquidity monitoring (depth, spreads, update frequency)
- Arbitrage opportunity detection (price deltas, funding rate deltas)

**FR4: Stateful Stream Processing**
- Use RocksDB state stores for aggregation windows
- Support tumbling windows (fixed-size, non-overlapping)
- Support sliding windows (overlapping, configurable slide interval)
- Implement session windows for order book state tracking

**FR5: Consumer Group Deployment**
- Configure Kafka consumer group for horizontal scaling
- Support multiple worker instances processing same topics
- Implement partition-aware state isolation
- Rebalancing and failure recovery

**FR6: Configuration-Driven Analytics**
- YAML configuration for aggregation expressions
- Define custom metrics via expression language (e.g., `vwap`, `spread * volume`)
- Per-exchange, per-symbol metric overrides
- Runtime metric registration without code changes

**FR7: Lineage & Backfill**
- Track input topic, window, symbol, exchange for each output record
- Enable historical reprocessing with configurable start/end times
- Support state store checkpointing for recovery
- Implement idempotent aggregation (restart-safe)

### Technical Requirements

**TR1: QuixStreams Framework**
- Use QuixStreams as primary stream processing library
- Leverage Topics API for declarative topology definition
- Implement custom functions for OHLCV and metric computation
- Use KafkaTopics for state management

**TR2: Protobuf Deserialization**
- Deserialize Trade, OrderBook, and Ticker protobuf messages
- Validate schema versions (handle v0.1.0+)
- Implement error handling for malformed messages
- Logging and metrics for deserialization failures

**TR3: State Management**
- RocksDB state stores for windowed aggregations
- Compaction strategy for long-running aggregations
- Configurable state store retention policy
- State store recovery and restoration

**TR4: Kafka Integration**
- Multi-topic consumption (trade, orderbook, ticker topics)
- Output topics for aggregated metrics (candles, vwap, correlation)
- Consumer group configuration (group.id, auto.offset.reset)
- Rebalancing listener implementation

**TR5: Error Handling & Monitoring**
- Dead letter queue (DLQ) for poison pill messages
- Comprehensive error logging with context
- Metrics emission: throughput, latency, error rates
- Health check endpoints (for orchestration)

**TR6: Output Schema**
- Define protobuf schemas for output metrics
- OHLCV Candle: timestamp, open, high, low, close, volume, vwap
- Metric message: timestamp, symbol, exchange, metric_name, value
- Correlation message: symbol_pair, exchange_pair, correlation, lookback_window

### Non-Functional Requirements

**NFR1: Performance**
- Process 100k+ trades/sec per worker instance
- End-to-end latency <5s for 1m candles from trade ingestion to emission
- RocksDB state store writes <1ms per aggregation update
- Memory footprint <2GB per 1M active symbol/window combinations

**NFR2: Scalability**
- Horizontal scaling: linear throughput increase with consumer group size
- Support 1000+ symbol/exchange pairs without performance degradation
- Window size scaling: 1K-1M trades per window
- State store disk usage: <10GB for 24h rolling window

**NFR3: Reliability**
- Exactly-once semantics for aggregation results
- Graceful shutdown with state checkpoint
- Automatic recovery from broker failures
- No data loss during rebalancing

**NFR4: Maintainability**
- Clear separation between topology definition and computation logic
- Testable aggregation functions with isolated unit tests
- Configuration-as-code for metric definitions
- Comprehensive logging for debugging

## Dependencies & Related Specifications

- **Upstream (Blocking)**: `protobuf-callback-serialization` (Spec 1) - Provides protobuf-serialized Kafka topics
- **Upstream (Optional)**: `normalized-data-schema-crypto` v0.1.0 - Provides protobuf schemas
- **Downstream**: `lakehouse-backend-adapter` (Spec 3) - Consumes aggregated streams for historical storage
- **External**: QuixStreams (Python Kafka Streams library) - Will be added as dependency

## Success Criteria

1. ✅ OHLCV candles computed correctly for 1m, 5m, 1h windows
2. ✅ VWAP computed with configurable weighting schemes
3. ✅ Cross-exchange correlation analysis functioning end-to-end
4. ✅ Horizontal scaling to 3+ worker instances verified
5. ✅ 100% exactly-once semantics for aggregations (no duplicates/data loss)
6. ✅ Configuration-driven metric expressions working for custom analytics
7. ✅ Dead letter queue captures and logs all malformed messages
8. ✅ Backfill capability tested: reprocess 24h historical data in <30 minutes
9. ✅ Performance benchmarks: <5s end-to-end latency for 1m candles
10. ✅ Production integration guide with deployment examples

## Next Steps

1. **Requirements Approval**: Review and approve this requirements document
2. **Design Phase**: Create design.md specifying stream topology, state management, and output schemas
3. **Tasks Generation**: Generate implementation tasks and timeline
4. **Implementation**: Execute tasks using TDD methodology (must complete Spec 1 first)

---

**Phase**: Initialized
**Timeline**: 2-3 weeks (after Spec 1 completion)
**Owner**: Development Team
**Blocked By**: `protobuf-callback-serialization` (Spec 1) - awaiting spec completion and Kafka topics
