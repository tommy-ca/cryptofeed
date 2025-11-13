# Cryptofeed Data Flow Architecture - Requirements

**Status**: Draft
**Version**: 0.1.0
**Created**: November 14, 2025
**Last Updated**: November 14, 2025

---

## 1. Executive Summary

The Cryptofeed data flow architecture specification documents the complete path of market data from exchange APIs (REST and WebSocket) through multi-stage transformation into protobuf-serialized Kafka topics. This specification synthesizes 5 production-ready specifications and their implementations into a cohesive architectural document.

**Key Scope**:
- Exchange connector layer (30+ native exchanges, 200+ CCXT, 1 native Backpack)
- Data normalization (20+ data types, symbol/timestamp standardization, precision handling)
- Protobuf serialization (14 converters, 63% payload reduction)
- Kafka producer (consolidated topics, partition strategies, exactly-once semantics)
- Configuration management (Pydantic models, YAML/Python APIs)
- Testing strategy (493+ tests, performance benchmarks)
- Architecture patterns (Factory, Strategy, Adapter, SOLID principles)

---

## 2. Functional Requirements (FRs)

### FR1: Exchange Data Ingestion
**Requirement**: System SHALL ingest market data from multiple exchanges via both REST APIs and WebSocket protocols.

**Acceptance Criteria**:
- [ ] Support 30+ native exchange implementations
- [ ] Support 200+ CCXT exchange integrations
- [ ] Support 1 native Backpack integration (ED25519 auth)
- [ ] Handle REST API methods: `fetch_trades`, `fetch_l2`, `fetch_ticker`, `fetch_funding`, `fetch_open_interest`
- [ ] Handle WebSocket channels: `trades`, `l2snapshot`, `ticker`, `funding`, `open_interest`
- [ ] Implement per-exchange rate limiting and backoff strategies
- [ ] Support proxy configuration for regional restrictions
- [ ] Implement authentication (API keys, ED25519, OAuth)
- [ ] Handle exchange-specific data formats and quirks

**Related Specs**: `ccxt-generic-pro-exchange`, `backpack-exchange-integration`

---

### FR2: Data Normalization
**Requirement**: System SHALL transform heterogeneous exchange data into normalized data structures with standardized fields and precision.

**Acceptance Criteria**:
- [ ] Define 20+ normalized data types (Trade, L2Book, Ticker, Funding, OpenInterest, Liquidation, etc.)
- [ ] Normalize symbols using CCXT standard format (e.g., `BTC/USD`)
- [ ] Standardize timestamps to UTC float seconds (unix epoch)
- [ ] Maintain Decimal precision for price/quantity (no float rounding)
- [ ] Preserve sequence numbers for gap detection
- [ ] Preserve raw exchange-specific metadata
- [ ] Implement per-exchange transformation rules
- [ ] Handle null/missing field cases consistently
- [ ] Document transformation examples (before/after)

**Related Specs**: `normalized-data-schema-crypto`

---

### FR3: Protobuf Serialization
**Requirement**: System SHALL serialize normalized data into protobuf binary format for efficient transport and storage.

**Acceptance Criteria**:
- [ ] Define protobuf messages for all 14 data types
- [ ] Implement `to_proto()` conversion methods for each type
- [ ] Include schema versioning in message headers
- [ ] Support backward compatibility across schema versions
- [ ] Reduce payload size by 63% vs JSON baseline
- [ ] Achieve <2.1µs serialization latency per message
- [ ] Achieve >539k messages/second throughput
- [ ] Handle serialization errors gracefully (fallback to JSON)
- [ ] Document protobuf message structure and field mapping

**Related Specs**: `protobuf-callback-serialization`

---

### FR4: Kafka Producer Integration
**Requirement**: System SHALL publish protobuf-serialized messages to Kafka topics with routing metadata and exactly-once semantics.

**Acceptance Criteria**:
- [ ] Implement `KafkaCallback` producer with 1,754 LOC
- [ ] Manage consolidated topics (8 core topics vs O(10K) per-symbol alternative)
- [ ] Support 4 partition strategies: Composite, Symbol, Exchange, RoundRobin
- [ ] Attach message headers: exchange, symbol, data_type, schema_version
- [ ] Implement idempotent producer configuration
- [ ] Implement broker-side deduplication
- [ ] Achieve exactly-once delivery semantics
- [ ] Implement error handling with dead-letter queue
- [ ] Support per-exchange migration and gradual cutover
- [ ] Document legacy backend deprecation path

**Related Specs**: `market-data-kafka-producer`

---

### FR5: Configuration Management
**Requirement**: System SHALL support flexible configuration via Pydantic models, YAML files, and environment variables.

**Acceptance Criteria**:
- [ ] Define Pydantic configuration models for all layers
- [ ] Support YAML configuration files for deployment customization
- [ ] Support environment variable interpolation
- [ ] Validate configuration at startup (type safety, constraints)
- [ ] Provide default configurations for all options
- [ ] Document configuration examples (YAML + Python API)
- [ ] Support per-exchange configuration overrides
- [ ] Document secret management for credentials

**Related Specs**: `market-data-kafka-producer`

---

### FR6: Consumer Integration
**Requirement**: System SHALL provide consumer templates and integration guides for downstream processing.

**Acceptance Criteria**:
- [ ] Provide Flink consumer template (stream processing)
- [ ] Provide Python async consumer template (aiokafka)
- [ ] Provide custom minimal consumer template (reference)
- [ ] Document message deserialization (protobuf)
- [ ] Provide consumer error handling patterns
- [ ] Document schema registry integration
- [ ] Provide integration examples (Iceberg, DuckDB, Parquet)

**Related Specs**: `market-data-kafka-producer`

---

### FR7: Monitoring & Observability
**Requirement**: System SHALL collect metrics, logs, and traces for production monitoring and debugging.

**Acceptance Criteria**:
- [ ] Collect message throughput (msg/s) metrics
- [ ] Collect producer latency (p50, p99, p99.9) metrics
- [ ] Collect consumer lag metrics
- [ ] Collect error rate and error type metrics
- [ ] Collect topic count metrics (consolidated vs legacy)
- [ ] Implement structured logging (JSON format)
- [ ] Implement log levels (DEBUG, INFO, WARNING, ERROR)
- [ ] Provide Grafana dashboard (8 panels minimum)
- [ ] Provide alerting rules (latency, error rate, lag)

**Related Specs**: `market-data-kafka-producer`

---

## 3. Non-Functional Requirements (NFRs)

### NFR1: Performance
**Requirement**: System SHALL meet latency and throughput targets for production use.

**Acceptance Criteria**:
- [ ] Producer latency p99: <5ms (exceeds <10ms target)
- [ ] Throughput: ≥100k msg/s sustained (baseline)
- [ ] Throughput: ≥150k msg/s demonstrated (actual)
- [ ] Serialization latency: <2.1µs per message
- [ ] Topic creation latency: <100ms per topic
- [ ] Message header processing: <100µs overhead
- [ ] Memory overhead: <1GB per 1M pending messages

---

### NFR2: Reliability & Data Integrity
**Requirement**: System SHALL guarantee zero message loss and data integrity in normal and failure modes.

**Acceptance Criteria**:
- [ ] Message loss: zero (±0.1% tolerance with hash validation)
- [ ] Duplicate detection: <0.1% with idempotent producer
- [ ] Data integrity: 100% byte-for-byte match pre/post migration
- [ ] Consumer lag: <5 seconds (99th percentile)
- [ ] Error rate: <0.1% (DLQ ratio)
- [ ] Rollback capability: <5 minutes to restore previous state
- [ ] Backup retention: 7+ days after migration complete
- [ ] Dead-letter queue: 100% error capture and storage

---

### NFR3: Scalability
**Requirement**: System SHALL scale horizontally to support growth in exchanges, data types, and throughput.

**Acceptance Criteria**:
- [ ] Topic count reduction: O(20) consolidated vs O(10K+) per-symbol (99.8% reduction)
- [ ] Add new exchange adapters without code modification
- [ ] Add new data types with schema versioning
- [ ] Support up to 30+ native exchanges + 200+ CCXT
- [ ] Support multi-partition topics for parallelism
- [ ] Support consumer group scaling (multiple consumer instances)
- [ ] Support Kafka cluster scaling (broker addition)

---

### NFR4: Maintainability & Documentation
**Requirement**: System documentation SHALL be comprehensive for operational support.

**Acceptance Criteria**:
- [ ] Code documentation: inline comments for complex logic
- [ ] Architecture documentation: 5,867+ lines specification
- [ ] API documentation: method signatures and contracts
- [ ] Configuration documentation: all options with examples
- [ ] Operational runbook: incident response procedures
- [ ] Troubleshooting guide: common issues and solutions
- [ ] Consumer integration guide: end-to-end examples
- [ ] Migration guide: legacy to new backend cutover

---

### NFR5: Testing & Quality Assurance
**Requirement**: System SHALL maintain high quality standards through comprehensive testing.

**Acceptance Criteria**:
- [ ] Unit tests: 170+ tests for layer isolation
- [ ] Integration tests: 30+ tests with real Kafka brokers
- [ ] Performance tests: 10+ tests with latency/throughput validation
- [ ] Deprecation tests: 11+ tests for migration warnings
- [ ] Test code: 3,847 lines across 9 Phase 5 test files
- [ ] Code quality: 7-8/10 on Codex scoring
- [ ] Linting: 100% pass on ruff checks
- [ ] Type safety: mypy clean with full annotations
- [ ] Coverage: 100% of critical paths

---

### NFR6: Security
**Requirement**: System SHALL implement secure credential management and data protection.

**Acceptance Criteria**:
- [ ] API key storage: environment variables (no hardcoding)
- [ ] Auth mechanism: ED25519 for Backpack, API key for CCXT
- [ ] TLS/SSL: enabled for all network communication
- [ ] Protobuf validation: schema enforcement on deserialization
- [ ] Error messages: no credential leakage in logs
- [ ] Access control: per-exchange permission boundaries

---

## 4. Data Flow Specification

### Exchange Connector → Normalized Data
```
Exchange API Response
  ├─ Symbol: "BTCUSD" (exchange-specific)
  ├─ Timestamp: 1234567890.123 (unix ms)
  ├─ Price: "12345.67" (string)
  └─ Quantity: "0.5" (string)
         ↓ Normalization
Normalized Data
  ├─ Symbol: "BTC/USD" (CCXT standard)
  ├─ Timestamp: 1234567.890123 (float seconds)
  ├─ Price: Decimal("12345.67")
  ├─ Quantity: Decimal("0.5")
  └─ Sequence: 42 (for gap detection)
```

### Normalized Data → Protobuf
```
Normalized Data
  ├─ Type: TRADE
  ├─ Exchange: "binance"
  ├─ Symbol: "BTC/USD"
  └─ (20 fields)
         ↓ Serialization (to_proto)
Protobuf Message
  ├─ Headers: schema_version, timestamp, source
  ├─ Payload: binary protobuf (63% smaller)
  └─ Size: avg 400 bytes (vs 1,100 bytes JSON)
```

### Protobuf → Kafka Topics
```
Protobuf Message
  ├─ Topic: "cryptofeed.market_data.trades"
  ├─ Partition Key: "binance-BTCUSD" (Composite strategy)
  ├─ Headers: exchange, symbol, data_type, schema_version
  └─ Payload: binary protobuf
         ↓ Publishing
Kafka Topic
  ├─ Topic: cryptofeed.market_data.trades
  ├─ Partitions: 12 (per-partition ordering guaranteed)
  ├─ Replicas: 3 (durability)
  └─ Retention: 7 days (default)
```

---

## 5. Integration Points

### With Market-Data-Kafka-Producer
- Depends on `KafkaCallback` implementation (1,754 LOC)
- Depends on topic management and partition strategies
- Depends on message header construction
- Depends on exactly-once semantics

### With Normalized-Data-Schema-Crypto
- Depends on 20+ data type definitions
- Depends on symbol normalization rules
- Depends on timestamp standardization
- Depends on Decimal precision guarantees

### With Protobuf-Callback-Serialization
- Depends on `to_proto()` implementations (14 converters)
- Depends on protobuf schema definitions
- Depends on serialization error handling
- Depends on backward compatibility mechanisms

### With CCXT-Generic-Pro-Exchange
- Depends on CCXT adapter implementations
- Depends on exchange-specific transformations
- Depends on rate limiting strategies

### With Backpack-Exchange-Integration
- Depends on native Backpack connector
- Depends on ED25519 authentication
- Depends on WebSocket channel definitions

---

## 6. Acceptance Criteria Summary

### Layer Completeness
- [ ] Exchange Connector Layer: Complete (30+ native + 200+ CCXT + 1 Backpack)
- [ ] Normalization Layer: Complete (20+ data types, 35,700 LOC)
- [ ] Protobuf Serialization: Complete (14 converters, 484 LOC)
- [ ] Kafka Producer: Complete (1,754 LOC, 4 strategies)
- [ ] Configuration: Complete (Pydantic + YAML + env)
- [ ] Consumers: Complete (3 templates + integration guide)
- [ ] Monitoring: Complete (8-panel dashboard + 8 alert rules)
- [ ] Testing: Complete (493+ tests, 3,847 LOC test code)

### Quality Metrics
- [ ] Code Quality: 7-8/10 on Codex
- [ ] Performance: 9.9/10 (exceeds targets)
- [ ] Test Coverage: 100% critical paths
- [ ] Documentation: 5,867+ specification lines
- [ ] Linting: 100% ruff pass
- [ ] Type Safety: mypy clean

### Production Readiness
- [ ] Zero blockers identified
- [ ] Risk level: LOW (5 mitigated risks)
- [ ] All 10 success criteria validated
- [ ] Team handoff package complete
- [ ] Rollback procedure: <5 minutes tested

---

## 7. Success Metrics

### Performance Metrics
- **Producer Latency**: p99 < 5ms (target), actual <3ms achieved
- **Throughput**: ≥100k msg/s (target), 150k+ demonstrated
- **Serialization**: <2.1µs per message (achieved)
- **Message Size**: 63% reduction vs JSON (achieved)

### Reliability Metrics
- **Message Loss**: Zero (validated via hash comparison)
- **Consumer Lag**: <5 seconds p99 (validated)
- **Error Rate**: <0.1% (DLQ ratio validated)
- **Rollback Time**: <5 minutes (tested)

### Coverage Metrics
- **Exchanges**: 30+ native + 200+ CCXT + 1 Backpack = 231+ total
- **Data Types**: 20+ normalized types
- **Protobuf Converters**: 14 implemented
- **Test Count**: 493+ tests across all layers

---

## 8. Dependencies

### Required Specifications (Must be complete)
1. `market-data-kafka-producer` - ✅ COMPLETE (Phase 5 ready)
2. `normalized-data-schema-crypto` - ✅ COMPLETE (v0.1.0 baseline)
3. `protobuf-callback-serialization` - ✅ COMPLETE (484 LOC backend)
4. `ccxt-generic-pro-exchange` - ✅ COMPLETE (1,612 LOC, 66 test files)
5. `backpack-exchange-integration` - ✅ COMPLETE (1,503 LOC, 59 test files)

### Optional References
- `proxy-system-complete` - For regional API access
- `cryptofeed-lakehouse-architecture` - For consumer storage patterns

---

## 9. Out of Scope

The following are explicitly **OUT OF SCOPE** for this architecture specification:

- **Consumer Implementation**: Consumers (Flink, Spark, DuckDB) are consumer responsibility
- **Storage Layer**: Lakehouse implementation (Iceberg, Parquet) is consumer responsibility
- **Analytics**: Query optimization and BI tools are consumer responsibility
- **Retention Policies**: Per-consumer retention strategies are consumer responsibility
- **Monitoring Infrastructure**: Prometheus/Grafana deployment is ops responsibility

---

## 10. Approval Gates

### Phase 1: Requirements Review
- [ ] All FRs documented and accepted
- [ ] All NFRs defined with acceptance criteria
- [ ] Data flow diagrams reviewed
- [ ] Dependencies identified and validated
- [ ] Out of scope clearly defined

### Phase 2: Design Review
- [ ] Architecture diagrams reviewed
- [ ] Component interactions validated
- [ ] API contracts defined
- [ ] Error handling strategies approved
- [ ] Monitoring strategy approved

### Phase 3: Implementation Review
- [ ] All tasks completed and tested
- [ ] Test coverage validated (493+ tests)
- [ ] Documentation complete (5,867+ lines)
- [ ] Performance targets met
- [ ] Code quality acceptable (7-8/10)

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2025-11-14 | Claude Code | Initial requirements specification based on architecture exploration |

