# CRYPTOFEED DATA FLOW ARCHITECTURE - EXPLORATION SUMMARY

**Document**: `CRYPTOFEED_ARCHITECTURE_EXPLORATION.md` (1,528 lines)
**Scope**: Complete data pipeline from exchanges through Kafka publishing
**Completed**: November 13, 2025

---

## QUICK REFERENCE

### Data Flow Path
```
Exchanges (30+ APIs)
  ↓ REST/WebSocket (raw JSON/binary)
Exchange Adapters (Native + CCXT + Backpack)
  ↓ Normalized objects
Data Types (20 Cython classes: Trade, OrderBook, Ticker, etc.)
  ↓ Decimal + timestamp precision
Protobuf Serialization (14 converters in protobuf_helpers.py)
  ↓ 63% payload reduction vs JSON
KafkaCallback Producer (1,754 LOC, topic mgmt, partitioning, errors)
  ↓ Consolidated topics (8) or per-symbol (80K+)
Kafka Topics (protobuf + headers + routing metadata)
  ↓ Exactly-once delivery, composite partitioning
Consumer Implementations (Flink, Spark, DuckDB, custom)
```

### Key Statistics
| Metric | Value |
|--------|-------|
| **Implementation Size** | 30,653 LOC (cryptofeed) + 1,754 (Kafka) + 671 (protobuf) |
| **Test Coverage** | 124 test files, 14,913 LOC in Kafka tests, 493+ tests |
| **Performance** | 10,000+ msg/s, p99 <10ms latency, 63% size reduction |
| **Data Types** | 20 message types, 14 protobuf converters |
| **Exchanges** | 30+ native adapters + CCXT generic + Backpack native |
| **Specifications** | 3 active specs (kafka-producer, protobuf, schema) all PRODUCTION READY |

---

## DOCUMENT STRUCTURE

### Phase 1: Specification Layer Analysis
- Specification files and status
- Data flow design from design.md
- Layer boundaries and contracts
- Scope definitions (in-scope vs out-of-scope)

**Key Insights:**
- Market-Data-Kafka-Producer: All 18 tasks complete, 493+ tests passing
- Protobuf serialization: 671 LOC, 144+ tests, 26µs latency per Trade
- Normalized schema: Buf-managed, 20 .proto files, canonical sources

### Phase 2: Exchange Adapter Layer
- 30+ native exchange implementations
- CCXT generic adapter (200+ exchanges)
- Backpack native integration (ED25519 auth)
- REST API methods and WebSocket channels
- Rate limiting and proxy support

**Key Insights:**
- Each exchange extends `Feed` base class
- Symbol normalization: `BTC_USD` → `BTC-USD`
- REST: symbol mapping, trade history, order books
- WebSocket: real-time trades, L2 updates, ticker, funding rates

### Phase 3: Normalization Layer
- 20 data type definitions in `types.pyx` (Cython)
- Trade, Ticker, OrderBook (L2/L3), Candle, Funding, Liquidation, OpenInterest, Index
- Balance, Position, Fill, OrderInfo, Transaction, MarginInfo
- Precision handling (Decimal type)
- Symbol and timestamp standardization

**Key Insights:**
- Decimal for all numeric fields (preserve precision)
- Float seconds converted to int64 microseconds in protobuf
- Enums for side/type (buy/sell, market/limit)
- Raw data preserved for audit trails

### Phase 4: Protobuf Serialization Layer
- 14 converter functions in `protobuf_helpers.py`
- 20 .proto message definitions
- Field mappings: Decimal→string, float→int64, enums
- Converter registry with dynamic lookup
- Performance: Trade ≈26µs, OrderBook ≈320µs

**Key Insights:**
- Consolidated converters (no separate serializers/)
- Backward compatible with JSON (format flag)
- Headers enrich messages: schema_version, exchange, symbol, data_type
- Snappy compression: 63% payload reduction

### Phase 5: Kafka Producer Layer
- KafkaCallback architecture (1,754 LOC)
- Topic management strategies: consolidated (8 topics) vs per-symbol (80K+)
- 4 partition strategies: composite (default), symbol, exchange, round-robin
- Exactly-once delivery (idempotent producer + broker dedup)
- Error handling and Dead Letter Queue
- Prometheus metrics and health checks

**Key Insights:**
- TopicManager: Topic naming, creation, parsing
- PartitionerFactory: Strategy selection pattern
- Composite partitioning: `{exchange}-{symbol}` key for per-pair ordering
- DLQ: unrecoverable errors sent to `cryptofeed.dlq.{topic}`
- Metrics: messages_sent, bytes_sent, latency, errors, dlq

### Phase 6: Configuration & Integration
- Pydantic models: KafkaTopicConfig, KafkaPartitionConfig, KafkaProducerConfig
- YAML configuration loading with validation
- Python API for programmatic setup
- Consumer integration examples (Flink, DuckDB)
- Best practices for producers and consumers

**Key Insights:**
- Configuration via YAML or Python dict
- Validation ensures data type, range, enum constraints
- Nested model structure supports per-topic overrides
- Consolidated topics recommended (simpler, scalable)

### Phase 7: Testing Strategy
- 124 test files, 14,913 LOC in Kafka tests
- Unit tests: Configuration, topic naming, partitioning, headers
- Integration tests: Real Kafka cluster, end-to-end flow, exactly-once
- Performance tests: Throughput, latency percentiles, memory
- Quality gates: 80%+ coverage, ruff clean, mypy strict, p99 <10ms

**Key Insights:**
- Test coverage across all layers
- Real Kafka cluster for integration (not mocked)
- Exactly-once delivery verified
- Error scenarios tested (broker unavailable, serialization)

### Phase 8: Architecture Patterns & Design
- Factory pattern: PartitionerFactory
- Strategy pattern: Partition strategies
- Observer pattern: Callback system
- Builder pattern: Configuration
- SOLID principles adherence
- Module boundaries and dependencies

**Key Insights:**
- New strategies can be added without modifying KafkaCallback
- All Partitioner subclasses substitutable
- Dependencies injected via constructor
- Clean layer separation

---

## KEY COMPONENTS & FILES

### Exchange Adapters
| File | LOC | Purpose |
|------|-----|---------|
| `exchange.py` | 400 | Base Exchange class |
| `feed.py` | 500 | Base Feed class (async mgmt) |
| `exchanges/*.py` | 500+ | 30+ native adapters (Binance, Coinbase, etc.) |
| `exchanges/ccxt/adapters/` | 1,000+ | CCXT generic (200+ exchanges) |
| `exchanges/backpack/` | 1,500+ | Backpack native (ED25519) |

### Normalization & Serialization
| File | LOC | Purpose |
|------|-----|---------|
| `types.pyx` | 35,700 | 20 data type definitions (Cython) |
| `symbols.py` | 200 | Symbol normalization |
| `defines.py` | 200 | Constants (TRADES, L2_BOOK, etc.) |
| `backends/protobuf_helpers.py` | 671 | 14 converters, registry |
| `proto/cryptofeed/normalized/v1/` | 500+ | 20 .proto message definitions |

### Kafka Producer
| File | LOC | Purpose |
|------|-----|---------|
| `kafka_callback.py` | 1,754 | KafkaCallback (topic mgmt, partitioning) |
| `backends/kafka.py` | 355 | Legacy backend (deprecated) |
| `backends/kafka_dlq.py` | TBD | DLQ helper |
| `backends/kafka_schema.py` | TBD | Schema registry integration |
| `backends/kafka_circuit_breaker.py` | TBD | Resilience |

### Testing
| Directory | Files | LOC | Purpose |
|-----------|-------|-----|---------|
| `tests/unit/kafka/` | 24 | 8,000 | Unit tests (config, logic, errors) |
| `tests/integration/kafka/` | TBD | 3,000 | Integration tests (real Kafka) |
| `tests/performance/` | TBD | 2,000 | Benchmarks (throughput, latency) |
| `tests/proto_integration/` | TBD | 1,000 | Protobuf serialization |

---

## CRITICAL DESIGN DECISIONS

### 1. Consolidated Topics (Default)
- **Decision**: 8 topics (`cryptofeed.trades`, etc.) vs 80K+ per-symbol
- **Rationale**: Simpler, scalable, header-based filtering
- **Trade-off**: Requires consumer header parsing vs automatic filtering
- **Status**: Recommended for all new deployments

### 2. Composite Partitioning (Default)
- **Decision**: `{exchange}-{symbol}` key vs symbol-only
- **Rationale**: Per-pair ordering for trading + distribution across partitions
- **Trade-off**: Per-pair (good for trading), not ideal for cross-exchange analysis
- **Status**: Recommended for real-time trading use cases

### 3. Exactly-Once Semantics
- **Decision**: Idempotent producer + broker deduplication
- **Rationale**: Protect against retries causing duplicates
- **Trade-off**: Slightly higher latency (wait for all replicas)
- **Status**: Enabled by default, required for trading

### 4. Protobuf Over JSON
- **Decision**: Binary serialization by default
- **Rationale**: 63% payload reduction, type-safe, version-aware
- **Trade-off**: Requires deserialization in consumers
- **Status**: Recommended, JSON still supported via config flag

### 5. DLQ for Unrecoverable Errors
- **Decision**: Send to separate topic instead of failing silently
- **Rationale**: Enable operator review and root cause analysis
- **Trade-off**: Requires monitoring/alerting on DLQ depth
- **Status**: Enabled by default, topic: `cryptofeed.dlq.{original_topic}`

### 6. 4-Phase Migration Strategy
- **Decision**: Dual-write → Consumer migration → Cutover → Cleanup
- **Rationale**: Zero-downtime transition from per-symbol to consolidated
- **Trade-off**: Extended migration period (12 weeks)
- **Status**: Documented, ready to execute

---

## PERFORMANCE CHARACTERISTICS

### Throughput
- **Target**: 10,000+ msg/s per producer instance
- **Achieved**: Verified in tests
- **Scaling**: Multi-instance deployment (separate Kafka partitions)

### Latency
- **p50**: 0.5-2ms (callback to Kafka ACK)
- **p95**: 2-5ms
- **p99**: <10ms (target)
- **SLA**: p99 latency remains sub-10ms up to 10,000 msg/s

### Payload Size
- **Trade**: JSON ~400 bytes → Protobuf ~120 bytes (30%) → Compressed ~100 bytes
- **OrderBook**: JSON ~3000 bytes → Protobuf ~1000 bytes (33%) → Compressed ~500 bytes
- **Overall**: 63% reduction vs JSON

### Memory
- **Base overhead**: ~50 MB per producer instance
- **Per 10K msg/s**: +5 MB
- **Total capacity**: ~500 MB (buffer for 10K msg/s for 5 seconds)

---

## ERROR HANDLING PHILOSOPHY

### Recoverable Errors (Retry with Exponential Backoff)
- BrokerNotAvailable
- NetworkException
- KafkaTimeoutException
- Actions: Retry (100ms, 200ms, 400ms, ...)

### Unrecoverable Errors (Send to DLQ)
- SerializationError
- InvalidTopicException
- Actions: Log + DLQ entry + alert

### Unknown Errors
- Other exceptions
- Actions: Log + alert + investigate

---

## OBSERVABILITY

### Prometheus Metrics
- `cryptofeed_kafka_messages_sent_total` (counter)
- `cryptofeed_kafka_bytes_sent_total` (counter)
- `cryptofeed_kafka_produce_latency_seconds` (histogram)
- `cryptofeed_kafka_errors_total` (counter)
- `cryptofeed_kafka_dlq_messages_total` (counter)

### Structured Logging
- JSON format with event, timestamp, metadata
- INFO: topic_created, message_sent
- WARN: message_retry, slow_producer
- ERROR: message_dlq, broker_unavailable

### Health Check
- Endpoint: `/metrics/kafka`
- Returns: status, brokers_available, producer_lag
- Response: <10ms

---

## CONSUMER INTEGRATION

### Flink Example
```python
trades = env.add_source(KafkaSource(
    topics=['cryptofeed.trades'],
    deserializer=ProtobufDeserializer(Trade)
))
trades.add_sink(IcebergSink(...))
```

### DuckDB Example
```python
consumer = KafkaConsumer('cryptofeed.trades')
for msg in consumer:
    trade = Trade.FromString(msg.value)
    conn.execute('INSERT INTO trades VALUES (...)', [trade.exchange, ...])
```

---

## PRODUCTION READINESS CHECKLIST

- ✅ All 18 tasks complete
- ✅ 493+ tests passing (unit, integration, performance, proto)
- ✅ Code quality: Codex score 7-8/10
- ✅ Performance targets met: 10K msg/s, p99 <10ms
- ✅ Exactly-once delivery verified
- ✅ Error handling comprehensive
- ✅ Monitoring metrics available
- ✅ Documentation complete
- ✅ Consumer examples provided
- ✅ Migration strategy documented

---

## RECOMMENDATIONS

1. **Start with consolidated topics** (not per-symbol)
2. **Use composite partitioning** (default)
3. **Enable exactly-once semantics** (default)
4. **Monitor DLQ depth** (alert on messages)
5. **Implement consumer lag monitoring** (offset tracking)
6. **Plan for schema evolution** (version in headers)
7. **Test with real Kafka cluster** (not mocked)
8. **Implement circuit breaker** (graceful degradation on broker failure)

---

## NEXT STEPS

1. Read full document: `CRYPTOFEED_ARCHITECTURE_EXPLORATION.md`
2. Review specifications: `.kiro/specs/market-data-kafka-producer/`
3. Examine implementation: `cryptofeed/kafka_callback.py`
4. Run tests: `pytest tests/unit/kafka/ -v`
5. Deploy to staging: Test with real Kafka cluster
6. Plan migration: Dual-write → Consumer migration → Cutover → Cleanup

---

**Document Location**: `/home/tommyk/projects/quant/data-sources/crypto-data/cryptofeed/`
**Files**:
- `CRYPTOFEED_ARCHITECTURE_EXPLORATION.md` (1,528 lines - comprehensive analysis)
- `ARCHITECTURE_EXPLORATION_SUMMARY.md` (this file - quick reference)

