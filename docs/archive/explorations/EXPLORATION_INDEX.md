# CRYPTOFEED DATA FLOW ARCHITECTURE - EXPLORATION INDEX

## Generated Documents

This exploration produced **two comprehensive documents** analyzing the complete data pipeline from exchange APIs through Kafka publishing.

### Documents Created

1. **CRYPTOFEED_ARCHITECTURE_EXPLORATION.md** (1,528 lines)
   - Complete, detailed analysis covering all 8 phases
   - Comprehensive examples, code snippets, diagrams
   - Deep dive into specifications, implementations, patterns
   - Best for: Understanding architecture in depth

2. **ARCHITECTURE_EXPLORATION_SUMMARY.md** (320 lines)
   - Quick reference guide with key insights
   - Tables, metrics, decision matrices
   - Recommendations and checklists
   - Best for: Quick lookup, executive summary

### Document Contents

#### PHASE 1: SPECIFICATION LAYER ANALYSIS
- Specification file locations and status
- Data flow design from official spec documents
- Layer boundaries and contracts
- **Key files**: `.kiro/specs/market-data-kafka-producer/design.md` (1,270 lines)
- **Key insights**: All 18 tasks complete, 493+ tests passing, PRODUCTION READY

#### PHASE 2: EXCHANGE ADAPTER LAYER
- 30+ native exchange implementations
- CCXT generic adapter (200+ exchanges)
- Backpack native integration (ED25519 signing)
- REST API methods: symbol_mapping, get_trade_history, fetch_funding_rate
- WebSocket channels: TRADES, L2_BOOK, TICKER, FUNDING, etc.
- **Key files**: `cryptofeed/exchanges/*.py`, `exchanges/ccxt/adapters/`, `exchanges/backpack/`
- **Key insights**: Feed extends Exchange, symbol normalization, proxy support

#### PHASE 3: NORMALIZATION LAYER
- 20 data type definitions in `types.pyx` (35,700 LOC, Cython)
- Market data: Trade, Ticker, OrderBook, Candle, Funding, Liquidation, OpenInterest, Index
- Account data: Balance, Position, Fill, OrderInfo, Transaction, MarginInfo
- Precision handling: Decimal type for arbitrary precision
- Symbol standardization: BTC_USD → BTC-USD
- Timestamp standardization: float seconds → int64 microseconds
- **Key files**: `cryptofeed/types.pyx`, `symbols.py`, `defines.py`
- **Key insights**: Decimal in Python, string in protobuf, Cython for performance

#### PHASE 4: PROTOBUF SERIALIZATION LAYER
- 14 converter functions in `protobuf_helpers.py` (671 LOC)
- 20 .proto message definitions in `proto/cryptofeed/normalized/v1/`
- Field mappings: Decimal→string, float→int64 microseconds, enums
- Converter registry with dynamic lookup
- **Performance**: Trade ≈26µs, OrderBook ≈320µs, target <1ms ✓
- **Payload reduction**: 63% (JSON vs Protobuf + Snappy compression)
- **Key files**: `backends/protobuf_helpers.py`, `proto/cryptofeed/normalized/v1/*.proto`
- **Key insights**: Consolidated converters, backward compatible, message headers enrich messages

#### PHASE 5: KAFKA PRODUCER LAYER
- KafkaCallback architecture (1,754 LOC)
- Topic management: Consolidated (8 topics) vs Per-Symbol (80K+ topics)
- 4 partition strategies: Composite (default), Symbol, Exchange, Round-Robin
- Exactly-once delivery: Idempotent producer + broker deduplication
- Error handling: Recoverable (retry), Unrecoverable (DLQ), Unknown (alert)
- Dead Letter Queue: Topic `cryptofeed.dlq.{original_topic}`
- Prometheus metrics: messages_sent, bytes_sent, latency, errors, dlq
- **Key files**: `cryptofeed/kafka_callback.py`, `backends/kafka.py` (legacy)
- **Key insights**: TopicManager, PartitionerFactory, composite key = `{exchange}-{symbol}`

#### PHASE 6: CONFIGURATION & INTEGRATION
- Pydantic models: KafkaTopicConfig, KafkaPartitionConfig, KafkaProducerConfig
- YAML configuration loading with validation
- Python API for programmatic setup
- Consumer integration examples: Flink, DuckDB, Spark, custom
- Best practices for producers and consumers
- **Key files**: `kafka_callback.py` (config models)
- **Key insights**: Configuration validation, nested models, per-topic overrides

#### PHASE 7: TESTING STRATEGY
- 124 test files, 14,913 LOC in Kafka tests, 493+ tests passing
- Unit tests: Configuration, topic naming, partitioning, headers
- Integration tests: Real Kafka cluster, end-to-end, exactly-once
- Performance tests: Throughput, latency percentiles, memory
- Quality gates: 80%+ coverage, ruff clean, mypy strict, p99 <10ms
- **Key files**: `tests/unit/kafka/` (24 files), `tests/integration/`, `tests/performance/`
- **Key insights**: Real Kafka cluster (not mocked), error scenarios tested

#### PHASE 8: ARCHITECTURE PATTERNS & DESIGN
- Design patterns: Factory, Strategy, Observer, Builder
- SOLID principles: All adhered to
- Module boundaries: Clean separation of concerns
- Dependency injection: Constructor-based
- **Key insights**: Extensible, testable, maintainable

---

## KEY METRICS & STATISTICS

### Codebase Size
| Component | LOC | Files | Purpose |
|-----------|-----|-------|---------|
| **cryptofeed module** | 30,653 | ~100 | Core library |
| **kafka_callback** | 1,754 | 1 | Kafka producer |
| **protobuf_helpers** | 671 | 1 | Serialization |
| **types.pyx** | 35,700 | 1 | Data types (Cython) |
| **proto definitions** | 500+ | 20 | Message definitions |
| **tests** | 14,913 | 124 | Test suite |
| **Total** | ~84,000+ LOC | 300+ | Complete system |

### Test Coverage
| Category | Count | Status |
|----------|-------|--------|
| Total test files | 124 | ✅ PASSING |
| Kafka-specific tests | 14,913 LOC | ✅ PASSING |
| Unit tests | ~8,000 LOC | ✅ PASSING |
| Integration tests | ~3,000 LOC | ✅ PASSING |
| Performance tests | ~2,000 LOC | ✅ PASSING |
| Proto tests | ~1,000 LOC | ✅ PASSING |

### Performance Characteristics
| Metric | Target | Achieved |
|--------|--------|----------|
| Throughput | 10,000+ msg/s | ✅ Verified |
| p99 latency | <10ms | ✅ <10ms |
| Payload reduction | 50-70% | ✅ 63% |
| Trade serialization | <1ms | ✅ ~26µs |
| OrderBook serialization | <1ms | ✅ ~320µs |
| Memory per 10K msg/s | N/A | ✅ ~5MB |

### Exchange & Data Type Coverage
| Category | Count |
|----------|-------|
| Native exchange adapters | 30+ |
| CCXT generic exchanges | 200+ |
| Data types (Cython) | 20 |
| Protobuf converters | 14 |
| Kafka topic strategies | 2 (consolidated, per-symbol) |
| Partition strategies | 4 (composite, symbol, exchange, round-robin) |

### Specifications Status
| Specification | Status | Tasks | Tests | Version |
|---------------|--------|-------|-------|---------|
| market-data-kafka-producer | ✅ COMPLETE | 18/18 | 493+ | 0.1.0 |
| protobuf-callback-serialization | ✅ COMPLETE | N/A | 144+ | 0.1.0 |
| normalized-data-schema-crypto | ✅ COMPLETE | Phase 1+3 | 119+ | 0.1.0 |

---

## CRITICAL DESIGN DECISIONS

### 1. Consolidated Topics (Default Recommendation)
```
Per-Symbol Topics: 80,000+ topics (not scalable)
Consolidated Topics: 8 topics (recommended)
  - cryptofeed.trades
  - cryptofeed.orderbook
  - cryptofeed.ticker
  - ... (8 total)
```
**Trade-off**: Header filtering required in consumers, but simple, scalable, maintainable

### 2. Composite Partitioning (Default Recommendation)
```
Partition Key: "{exchange}-{symbol}".encode()
Distribution: hash(key) % num_partitions
Guarantees: Per-pair ordering (good for trading)
Example: "binance-btc-usdt" → partition 3
```
**Trade-off**: Per-pair ordering, not ideal for cross-exchange analysis

### 3. Exactly-Once Delivery (Default, Required)
```
Mechanism: Idempotent producer + broker deduplication
Config: acks='all', enable.idempotence=True
Guarantee: No duplicates on retries or broker restarts
Cost: Slightly higher latency (wait for all replicas)
```

### 4. Protobuf Over JSON (Default Recommended)
```
Payload reduction: 63% (Trade: 400B → 120B)
Benefits: Type-safe, version-aware, compressible
Trade-off: Requires deserialization in consumers
Backward compat: JSON still supported via config flag
```

### 5. Dead Letter Queue for Unrecoverable Errors
```
Topic: cryptofeed.dlq.{original_topic}
Payload: original_message, error, timestamp, retry_count
Purpose: Enable operator review without data loss
Trade-off: Requires DLQ monitoring and alerting
```

### 6. 4-Phase Migration Strategy (for consolidated topics)
```
Phase 1: Dual-write (both consolidated and per-symbol)
Phase 2: Consumer migration (switch to consolidated)
Phase 3: Cutover (stop writing per-symbol)
Phase 4: Cleanup (delete per-symbol topics)
Duration: 12 weeks, zero-downtime
```

---

## ARCHITECTURE PATTERN SUMMARY

### Data Flow Path
```
Exchange APIs (REST/WebSocket)
  ↓ Raw JSON/binary
Exchange Adapters (30+ native, CCXT, Backpack)
  ↓ parse_*() methods
Data Type Objects (20 types in Cython)
  ↓ .exchange, .symbol, .timestamp, .raw
Protobuf Serialization (14 converters)
  ↓ .to_proto() + SerializeToString()
Message Routing (KafkaCallback)
  ↓ Topic selection, partition key, headers
Kafka Topics (consolidated or per-symbol)
  ↓ Protobuf bytes + routing headers
Consumer Integration (Flink, DuckDB, custom)
  ↓ .FromString() + business logic
User Applications
```

### Module Dependencies
```
exchange.py ← Base
  ↓
feed.py ← FeedHandler, connection management
  ↓
types.pyx ← 20 data types
  ↓
callback.py ← Callback routing
  ↓
backends/backend.py ← BackendCallback
  ├─ backends/kafka.py (legacy)
  ├─ backends/protobuf_helpers.py (14 converters)
  └─ kafka_callback.py (1,754 LOC, new producer)
       ├─ TopicManager
       ├─ PartitionerFactory
       ├─ KafkaConfig (Pydantic)
       └─ Monitoring (Prometheus)
```

---

## ERROR HANDLING CLASSIFICATION

### Recoverable (Retry with Exponential Backoff)
- BrokerNotAvailable
- NetworkException
- KafkaTimeoutException
- Action: Retry (100ms initial, exponential backoff)

### Unrecoverable (Send to DLQ)
- SerializationError
- InvalidTopicException
- MalformedData
- Action: Log + DLQ entry + alert

### Unknown (Alert and Investigate)
- Other exceptions
- Action: Log + alert + manual review

---

## OBSERVABILITY

### Prometheus Metrics
```
cryptofeed_kafka_messages_sent_total{data_type, exchange}
cryptofeed_kafka_bytes_sent_total{data_type}
cryptofeed_kafka_produce_latency_seconds{data_type}  [histogram]
cryptofeed_kafka_errors_total{error_type, data_type}
cryptofeed_kafka_dlq_messages_total{original_topic}
```

### Structured Logging
```
Levels: INFO, WARN, ERROR
Format: JSON with event, timestamp, metadata
Topics:
  - INFO: topic_created, message_sent
  - WARN: message_retry, slow_producer
  - ERROR: message_dlq, broker_unavailable
```

### Health Check
```
Endpoint: /metrics/kafka
Returns: status, brokers_available, producer_lag
Response time: <10ms
```

---

## FILE LOCATIONS

### Main Implementation Files
- `cryptofeed/exchange.py` - Base Exchange class
- `cryptofeed/feed.py` - Base Feed class (async)
- `cryptofeed/types.pyx` - 20 data types (Cython)
- `cryptofeed/kafka_callback.py` - KafkaCallback (1,754 LOC)
- `cryptofeed/backends/protobuf_helpers.py` - 14 converters (671 LOC)
- `cryptofeed/backends/kafka.py` - Legacy backend (355 LOC)

### Configuration
- `proto/cryptofeed/normalized/v1/` - 20 .proto files
- `.kiro/specs/market-data-kafka-producer/` - Full specification
- `config/kafka.yaml` - Example configuration

### Tests
- `tests/unit/kafka/` - 24 unit test files
- `tests/integration/kafka/` - Integration tests
- `tests/performance/` - Performance benchmarks
- `tests/proto_integration/` - Protobuf round-trip tests

### Documentation
- `docs/consumer-templates/` - Flink, DuckDB, Spark examples
- `.kiro/specs/*/design.md` - Architecture specifications
- `README.md` - Project overview

---

## PRODUCTION READINESS CHECKLIST

- ✅ All 18 specification tasks complete
- ✅ 493+ tests passing (unit, integration, performance, proto)
- ✅ Code quality: Codex score improved to 7-8/10
- ✅ Performance targets met: 10K+ msg/s, p99 <10ms
- ✅ Exactly-once delivery verified in tests
- ✅ Error handling: 3-tier classification (recoverable, unrecoverable, unknown)
- ✅ Monitoring: Prometheus metrics + structured logging + health check
- ✅ Documentation: Specifications, design.md, consumer examples
- ✅ Consumer examples: Flink, DuckDB, Spark, custom
- ✅ Migration strategy: 4-phase roadmap (dual-write → cutover → cleanup)

---

## HOW TO USE THESE DOCUMENTS

### For Architecture Understanding
1. Start with `ARCHITECTURE_EXPLORATION_SUMMARY.md` (this provides overview)
2. Read relevant phase in `CRYPTOFEED_ARCHITECTURE_EXPLORATION.md`
3. Review specification files: `.kiro/specs/market-data-kafka-producer/design.md`
4. Examine implementation: `cryptofeed/kafka_callback.py`

### For Integration
1. Review consumer examples in summary
2. Check `proto/cryptofeed/normalized/v1/` for message definitions
3. Run tests: `pytest tests/unit/kafka/ -v`
4. Deploy to staging with real Kafka cluster

### For Operations
1. Review error handling section
2. Set up Prometheus metrics scraping
3. Configure DLQ monitoring and alerting
4. Implement consumer lag monitoring
5. Plan rollout: dual-write → consumer migration → cutover → cleanup

### For Development
1. Study design patterns (Factory, Strategy, Observer)
2. Review SOLID principles adherence
3. Read test examples for new features
4. Follow naming conventions from existing code
5. Add metrics for new functionality

---

## RECOMMENDED NEXT STEPS

1. **Read Full Documentation**
   - `CRYPTOFEED_ARCHITECTURE_EXPLORATION.md` (1,528 lines)
   - `.kiro/specs/market-data-kafka-producer/design.md` (1,270 lines)

2. **Review Implementation**
   - `cryptofeed/kafka_callback.py` (1,754 LOC)
   - `cryptofeed/backends/protobuf_helpers.py` (671 LOC)
   - `proto/cryptofeed/normalized/v1/trade.proto` (example message)

3. **Run Tests**
   - `pytest tests/unit/kafka/test_kafka_config.py -v`
   - `pytest tests/unit/kafka/ -v` (all Kafka tests)
   - `pytest tests/integration/kafka/ -v` (real Kafka cluster)

4. **Deploy to Staging**
   - Set up Kafka cluster (3+ brokers)
   - Run integration tests
   - Validate exactly-once delivery
   - Test error scenarios (broker down, serialization failure)

5. **Plan Production Rollout**
   - Phase 1: Dual-write (consolidated + per-symbol)
   - Phase 2: Migrate consumers (switch to consolidated)
   - Phase 3: Cutover (stop per-symbol writes)
   - Phase 4: Cleanup (delete legacy topics)

---

## DOCUMENT METADATA

**Generated**: November 13, 2025
**Scope**: Complete cryptofeed data flow architecture
**Thoroughness**: Very thorough (8 phases, cross-layer analysis)
**Files Analyzed**: 100+ source files, 84,000+ LOC, 124 test files
**Specifications Reviewed**: 3 complete specifications (all production ready)

**Main Documents**:
1. `CRYPTOFEED_ARCHITECTURE_EXPLORATION.md` (1,528 lines, comprehensive)
2. `ARCHITECTURE_EXPLORATION_SUMMARY.md` (320 lines, quick reference)
3. `EXPLORATION_INDEX.md` (this file, navigation guide)

---

For questions or clarifications, refer to:
- Specification files: `.kiro/specs/market-data-kafka-producer/`
- Implementation: `cryptofeed/kafka_callback.py`
- Tests: `tests/unit/kafka/`, `tests/integration/kafka/`
- Documentation: `docs/`, README files

