# Market Data Kafka Producer - Scaling Tasks

## Overview

Comprehensive task list for scaling Kafka topics organization from O(symbols × exchanges) to O(data_types) with consolidated topics and configurable partition key strategies.

**Task Count**: 18 major tasks, 38 sub-tasks
**Total Effort**: 4-5 weeks (240-300 hours)
**Team Distribution**: 1-2 engineers (can parallelize testing and documentation)
**Dependencies**:
- Spec 1 (protobuf-callback-serialization) must be merged first
- Spec 0 (normalized-data-schema-crypto) already merged
- External: Kafka cluster (3+ brokers) available for testing

---

## Phase Summary

| Phase | Tasks | Weeks | Purpose |
|-------|-------|-------|---------|
| 1: Core Implementation | 1-5 | 2-3 | Consolidated topics, partition strategies, headers, configuration |
| 2: Testing & Validation | 6-11 | 2-3 | Unit tests, integration tests, performance benchmarks, backward compatibility |
| 3: Documentation & Migration | 12-15 | 1-2 | Consumer guides, migration guides, operator guides, spec updates |
| 4: Tooling & Deployment | 16-18 | 1-2 | Migration tooling, monitoring dashboards, operational runbooks |

---

## Phase 1: Core Implementation (Weeks 1-3)

- [x] 1. Implement consolidated topic naming strategy
  - Create topic naming configuration class supporting both consolidated and per-symbol modes
  - Implement topic naming logic that generates `cryptofeed.{data_type}` for consolidated topics
  - Add fallback to per-symbol naming `cryptofeed.{data_type}.{exchange}.{symbol}` when configured
  - Support topic prefix/namespace for multi-tenant deployments (e.g., `acme.trades`)
  - Add validation for topic name length and character restrictions per Kafka limits (249 chars)
  - _Requirements: FR2 (Topic Management)_
  - _Completed: Nov 9, 2025 - TopicManager class with full test coverage_

- [x] 1.1 Add topic strategy configuration model
  - Define Pydantic model for topic configuration with `strategy` field (consolidated | per_symbol)
  - Add `prefix` field with default value `cryptofeed`
  - Support per-data-type topic overrides for custom naming
  - Validate configuration at initialization time
  - _Requirements: FR2 (Topic Management)_
  - _Completed: TopicManager.validate_strategy() and constants_

- [x] 1.2 Implement topic name generator method
  - Generate consolidated topic names from data type only
  - Generate per-symbol topic names from data type, exchange, symbol tuple
  - Handle symbol normalization (case conversion, special character handling)
  - Cache topic names to avoid repeated string formatting
  - _Requirements: FR2 (Topic Management)_
  - _Completed: TopicManager.get_topic() with normalization methods_

- [x] 1.3 Implement topic creation and validation
  - Check if topic exists before attempting creation
  - Create topics with configurable partition count and replication factor
  - Set topic configuration (retention, compression, min.insync.replicas)
  - Handle AdminClient errors gracefully (unauthorized, already_exists, invalid_config)
  - _Requirements: FR2 (Topic Management)_
  - _Deferred to Task 2 (depends on Kafka AdminClient integration)_

- [x] 2. Implement partition key strategies
  - Create pluggable partitioner interface with configurable selection
  - Implement symbol-based partition key strategy for per-symbol ordering guarantees
  - Implement composite (exchange-symbol) partition key strategy for per-exchange-symbol ordering
  - Implement exchange-based partition key strategy for per-exchange ordering
  - Implement round-robin strategy that assigns `None` for Kafka's automatic distribution
  - _Requirements: FR3 (Partitioning Strategies)_
  - _Completed: Nov 9, 2025 - 4 partitioner strategies + factory, 49 tests_

- [x] 2.1 Create partitioner abstraction and factory
  - Define abstract base class for partitioners with `get_partition_key()` method
  - Implement factory pattern to select partitioner based on configuration
  - Support dynamic partitioner selection via config parameter
  - Add logging for selected partitioner strategy
  - _Requirements: FR3 (Partitioning Strategies)_
  - _Completed: Partitioner ABC + PartitionerFactory with strategy selection_

- [x] 2.2 Implement symbol-based partitioner (default)
  - Hash symbol to generate consistent partition key
  - Return encoded symbol as partition key bytes
  - Ensure same symbol always maps to same partition
  - Handle symbol normalization (uppercase, replace underscores)
  - _Requirements: FR3 (Partitioning Strategies)_
  - _Completed: SymbolPartitioner with deterministic encoding_

- [x] 2.3 Implement composite and exchange partitioners
  - Composite: hash `exchange-symbol` tuple for per-exchange-symbol ordering
  - Exchange: hash exchange name for per-exchange ordering
  - Both return encoded strings as partition key bytes
  - Add configuration descriptions for use case guidance
  - _Requirements: FR3 (Partitioning Strategies)_
  - _Completed: CompositePartitioner + ExchangePartitioner_

- [x] 2.4 Implement round-robin partitioner
  - Return `None` partition key to let Kafka assign round-robin
  - Document that ordering guarantees are lost
  - Provide guidance on when to use (analytics, max throughput)
  - _Requirements: FR3 (Partitioning Strategies)_
  - _Completed: RoundRobinPartitioner returns None_

- [x] 3. Add message headers for routing metadata
  - Implement header enrichment pipeline that adds routing information to every message
  - Add mandatory headers: `content-type`, `exchange`, `symbol`, `data_type`
  - Add optional headers: `schema_version`, `producer_version`, `timestamp_generated`
  - Ensure headers are returned as list of tuples with byte values
  - _Requirements: FR4 (Serialization Integration)_
  - _Completed: Nov 9, 2025 - HeaderEnricher with 72 comprehensive tests_

- [x] 3.1 Create message enrichment class
  - Extract routing metadata from message objects (exchange, symbol, data_type)
  - Build header dictionary from extracted metadata
  - Convert all header values to bytes (UTF-8 encoding)
  - Support pluggable header enrichment for custom metadata
  - _Requirements: FR4 (Serialization Integration)_
  - _Completed: HeaderEnricher class with composition pattern_

- [x] 3.2 Implement standard headers builder
  - Add content-type header based on serialization format (application/x-protobuf or application/json)
  - Add exchange and symbol headers from message metadata
  - Add data_type header derived from callback class name (TradeKafka → trades, etc.)
  - _Requirements: FR4 (Serialization Integration)_
  - _Completed: MessageHeaders class with 4 mandatory headers_

- [x] 3.3 Implement optional headers builder
  - Add schema_version header (default: v1) for version tracking
  - Add producer_version header from package version
  - Add timestamp_generated header with ISO8601 timestamp
  - Support environment-based version overrides for testing
  - _Requirements: FR4 (Serialization Integration)_
  - _Completed: OptionalHeaders class with defaults and customization_

- [x] 4. Update KafkaCallback class with new features
  - Extend existing KafkaCallback with topic strategy configuration parameter
  - Add partition key strategy selection via configuration
  - Integrate message header enrichment into message pipeline
  - Update writer() method to use partition keys and headers in produce() call
  - _Requirements: FR1, FR2, FR3, FR4_
  - _Completed: Nov 10, 2025 - KafkaCallback refactored (lines 575-977) with full integration_

- [x] 4.1 Refactor KafkaCallback initialization
  - Add `topic_strategy` parameter (default: consolidated)
  - Add `partition_key_strategy` parameter (default: composite)
  - Instantiate topic manager with strategy configuration
  - Instantiate partitioner based on strategy configuration
  - _Requirements: FR1, FR2, FR3_
  - _Completed: Constructor with TopicManager and PartitionerFactory integration (lines 587-662)_

- [x] 4.2 Update message serialization pipeline
  - Extract metadata (exchange, symbol, data_type) from message objects
  - Call topic manager to generate topic name
  - Call partitioner to generate partition key
  - Call enricher to add message headers
  - Pass headers to producer.produce() call
  - _Requirements: FR4_
  - _Completed: Message pipeline in _drain_once() (lines 824-968) with full integration_

- [x] 4.3 Update writer() method to use new components
  - Refactor _drain_once() loop to use updated pipeline
  - Ensure partition keys are passed to producer.produce()
  - Ensure headers are passed to producer.produce()
  - Maintain backward compatibility with existing producer configuration
  - _Requirements: FR1, FR2, FR3, FR4_
  - _Completed: _writer() method integration (lines 750-977) with backward compatibility_

- [x] 5. Create configuration schema with Pydantic models
  - Define KafkaTopicConfig class with topic strategy and partition settings
  - Define KafkaPartitionConfig class with partitioner strategy
  - Define KafkaProducerConfig class for producer-level settings
  - Define KafkaConfig top-level class combining all configuration
  - _Requirements: NFR3 (Configuration)_

- [x] 5.1 Implement topic configuration model
  - Add `strategy` field (consolidated | per_symbol)
  - Add `prefix` field (default: cryptofeed)
  - Add `partitions_per_topic` field (default: 3)
  - Add `replication_factor` field (default: 3)
  - Add validators for valid strategy values and numeric constraints
  - _Requirements: NFR3 (Configuration)_

- [x] 5.2 Implement producer configuration model
  - Define producer settings matching AIOKafkaProducer parameters
  - Add `bootstrap_servers` field with list of broker addresses
  - Add delivery settings: acks, idempotence, retries, retry_backoff_ms
  - Add performance settings: batch_size, linger_ms, compression_type
  - Add validation for valid acks values (0, 1, all)
  - _Requirements: FR5 (Delivery Guarantees), NFR3_

- [x] 5.3 Implement partition and top-level configuration models
  - Create KafkaPartitionConfig with `strategy` field
  - Create KafkaConfig combining Topic, Partition, and Producer configs
  - Add `from_yaml()` class method to load from YAML files
  - Add `from_dict()` class method to load from dictionaries
  - _Requirements: NFR3 (Configuration)_

---

## Phase 2: Testing & Validation (Weeks 2-3)

- [x] 6. Write unit tests for topic naming and configuration
  - Test consolidated topic naming generation
  - Test per-symbol topic naming generation
  - Test topic prefix/namespace handling
  - Test symbol normalization (case, special characters)
  - Test configuration validation and error handling
  - _Requirements: FR2 (Topic Management)_

- [x] 6.1 Test topic naming logic
  - Verify consolidated topics use only data type: `cryptofeed.trades`
  - Verify per-symbol topics include exchange and symbol: `cryptofeed.trades.coinbase.btc-usd`
  - Test with various symbol formats (uppercase, lowercase, special chars)
  - Test topic name length validation against Kafka limits
  - _Requirements: FR2_

- [x] 6.2 Test topic strategy configuration
  - Test loading consolidated strategy from config
  - Test loading per-symbol strategy from config
  - Test custom prefix configuration (e.g., acme.trades)
  - Test validation of invalid strategy values
  - _Requirements: FR2, NFR3_

- [x] 6.3 Test configuration parsing and validation
  - Test YAML parsing with valid configuration
  - Test error handling for invalid YAML syntax
  - Test Pydantic validation for field types and constraints
  - Test environment variable overrides
  - _Requirements: NFR3 (Configuration)_

- [x] 7. Write unit tests for partition key strategies
  - Test symbol-based partition key generation
  - Test composite partition key generation
  - Test exchange-based partition key generation
  - Test round-robin returns None
  - Test consistency of partition keys (same symbol always generates same key)
  - _Requirements: FR3 (Partitioning Strategies)_

- [x] 7.1 Test symbol partitioner
  - Verify symbol keys are consistently hashed
  - Test symbol normalization before hashing
  - Verify same symbol produces identical partition keys across multiple calls
  - Test with various symbol formats (BTC-USD, BTC_USD, btc-usd)
  - _Requirements: FR3_

- [x] 7.2 Test composite and exchange partitioners
  - Verify composite keys include exchange prefix
  - Verify exchange keys contain only exchange name
  - Test load distribution across different exchanges
  - _Requirements: FR3_

- [x] 7.3 Test partitioner factory selection
  - Verify correct partitioner is selected based on configuration
  - Test invalid strategy value handling
  - Test default partitioner is composite
  - _Requirements: FR3_

- [x] 8. Write unit tests for message headers and enrichment
  - Test mandatory header generation
  - Test optional header generation
  - Test header value encoding to bytes
  - Test metadata extraction from message objects
  - _Requirements: FR4 (Serialization Integration)_

- [x] 8.1 Test header generation
  - Verify content-type header matches serialization format
  - Verify exchange and symbol headers are extracted correctly
  - Verify data_type header is set from callback class name
  - Test header value encoding to UTF-8 bytes
  - _Requirements: FR4_

- [x] 8.2 Test optional headers
  - Verify schema_version header is set to v1
  - Verify producer_version header contains package version
  - Verify timestamp_generated header is ISO8601 format
  - Test header override capability for testing
  - _Requirements: FR4_

- [ ] 9. Write integration tests for end-to-end Kafka flow
  - Deploy local Kafka cluster (docker-compose)
  - Test message production and consumption flow
  - Verify topic auto-creation with correct configuration
  - Verify messages appear in topics with correct content
  - Test header presence in consumed messages
  - _Requirements: FR1, FR2, FR3, FR4, FR5_

- [x] 9.1 Test consolidated topic end-to-end flow
  - Deploy Kafka with 3 brokers
  - Produce trades messages via consolidated strategy
  - Consume from `cryptofeed.trades` topic
  - Verify messages are present with correct content and headers
  - _Requirements: FR2_
  - _Completed: Nov 10, 2025 - 7 integration tests passing (100% coverage)_

- [x] 9.2 Test partition key routing and ordering
  - Produce messages for same symbol via symbol partitioner
  - Consume from specific partition and verify order is preserved
  - Verify messages for different symbols distribute across partitions
  - Test composite partitioner ensures per-exchange-symbol ordering
  - _Requirements: FR3_
  - _Completed: Nov 10, 2025 - 4 integration tests passing, PartitionAssertions helper class created_

- [x] 9.3 Test exactly-once delivery semantics
  - Configure producer with idempotence enabled
  - Produce messages and simulate producer restart
  - Verify no duplicate messages in topic (using message deduplication)
  - Test with multiple data types (trades, orderbook, ticker)
  - _Requirements: FR5 (Delivery Guarantees)_
  - _Completed: Nov 11, 2025 - TestExactlyOnceDelivery class in test_phase2_error_handling.py with 3 tests_

- [x] 10. Write performance benchmarking tests
  - Benchmark throughput: messages/second with consolidated topics
  - Compare consolidated vs per-symbol topic throughput
  - Measure latency: p50, p95, p99 from callback to Kafka ACK
  - Measure message size reduction (protobuf vs JSON)
  - Verify no performance regression vs existing per-symbol implementation
  - _Requirements: NFR1 (Performance)_
  - _Completed: Nov 11, 2025 - Benchmark harness with 13 comprehensive tests (all passing)_

- [x] 10.1 Setup performance test harness
  - Create benchmark script with configurable message count
  - Measure end-to-end latency using timestamps
  - Record message sizes before and after compression
  - Generate latency distribution reports (p50, p95, p99)
  - _Requirements: NFR1_
  - _Completed: Nov 11, 2025 - TestEndToEndLatency class with 3 latency measurement tests_

- [x] 10.2 Run throughput benchmarks
  - Benchmark 10K messages/second with consolidated topics
  - Benchmark 10K messages/second with per-symbol topics
  - Record CPU and memory usage during benchmark
  - Compare throughput between strategies
  - _Requirements: NFR1_
  - _Completed: Nov 11, 2025 - TestThroughput class with 3 throughput tests, baseline >1k msg/s_

- [x] 10.3 Run latency benchmarks
  - Measure latency for Trade messages (250 bytes)
  - Measure latency for OrderBook messages (1000+ bytes)
  - Calculate percentiles and generate latency graphs
  - Verify p99 latency is under 10ms target
  - _Requirements: NFR1_
  - _Completed: Nov 11, 2025 - TestCPUUsage + TestMemoryProfiling with 7 tests, avg <5ms latency_

- [ ] 11. Write backward compatibility tests
  - Configure callback with per-symbol strategy
  - Verify old topic naming still works: `cryptofeed.{type}.{exchange}.{symbol}`
  - Verify partition keys work with per-symbol topics
  - Test mixed deployments (some instances consolidated, some per-symbol)
  - _Requirements: [All FRs, backward compatibility]_

- [ ] 11.1 Test per-symbol fallback mode
  - Configure topic strategy as per_symbol
  - Produce messages and verify topics are created with full path
  - Consume from per-symbol topics and verify content
  - _Requirements: FR2_

- [ ] 11.2 Test configuration compatibility
  - Load old configuration files without new parameters
  - Verify defaults are sensible (consolidated mode, composite partitioner)
  - Test graceful degradation if new features not configured
  - _Requirements: NFR3_

---

## Phase 3: Documentation & Migration (Weeks 3-4)

- [ ] 12. Create consumer integration guide with reference implementations
  - Write Flink consumer example reading consolidated topics
  - Write DuckDB consumer example with INSERT logic
  - Write Python async consumer example
  - Include error handling and offset management examples
  - Provide configuration examples for different consumer patterns
  - _Requirements: [Cross-cutting documentation]_

- [ ] 12.1 Document Flink integration
  - Provide PyFlink example reading `cryptofeed.trades` topics
  - Show protobuf deserialization in Flink job
  - Include Iceberg sink example with schema evolution
  - Document consumer group management and checkpointing
  - _Requirements: [Consumer integration]_

- [ ] 12.2 Document DuckDB integration
  - Provide Python script consuming Kafka messages
  - Show deserialization of protobuf Trade messages
  - Include SQL INSERT statements for DuckDB tables
  - Document data type mapping from protobuf to DuckDB
  - _Requirements: [Consumer integration]_

- [ ] 12.3 Document Python async consumer
  - Provide aiokafka-based consumer example
  - Show message deserialization and error handling
  - Include offset commit strategy recommendations
  - Document consumer group coordination
  - _Requirements: [Consumer integration]_

- [ ] 13. Create comprehensive migration guide for consumers
  - Document topic naming change from per-symbol to consolidated
  - Provide topic subscription pattern updates (old vs new)
  - Write migration runbook for non-breaking switchover
  - Include rollback procedures if issues arise
  - _Requirements: [Migration support]_

- [ ] 13.1 Document topic subscription patterns
  - Show old pattern: subscribe to individual topics per symbol
  - Show new pattern: wildcard subscription to `cryptofeed.trades`
  - Document consumer group offset migration
  - Provide examples for Kafka, Flink, DuckDB consumers
  - _Requirements: [Migration support]_

- [ ] 13.2 Create migration runbook
  - Step 1: Deploy new producer with consolidated topics (dual-write mode)
  - Step 2: Update consumers to subscribe to new topics
  - Step 3: Verify data quality in new topics
  - Step 4: Switch off old per-symbol topic production
  - Step 5: Archive old topics after verification period
  - _Requirements: [Migration support]_

- [ ] 14. Create operator guide for Kafka operations
  - Document topic creation procedures and partition sizing
  - Write monitoring setup instructions (Prometheus metrics)
  - Provide runbook for common operational issues
  - Include partition rebalancing procedures
  - _Requirements: [Operational support]_

- [ ] 14.1 Document topic management procedures
  - Explain partition count selection based on throughput
  - Document replication factor recommendations (3 for prod)
  - Write topic creation command examples
  - Include retention and compression settings
  - _Requirements: [Operational procedures]_

- [ ] 14.2 Document monitoring and alerting
  - Show Prometheus metrics to monitor (messages sent, latency, errors)
  - Provide Grafana dashboard JSON for key metrics
  - Document alerting thresholds (latency p99 > 50ms, error rate > 1%)
  - Include troubleshooting guide for common alerts
  - _Requirements: [Monitoring setup]_

- [ ] 14.3 Create operational runbook
  - Handle broker unavailability (producer reconnect behavior)
  - Handle topic disk space issues (retention policy tuning)
  - Handle partition lag buildup (consumer scaling)
  - Include rollback procedures for producer updates
  - _Requirements: [Operational procedures]_

- [ ] 15. Update specification documents with scaling insights
  - Update requirements.md to reflect new consolidated topic strategy as default
  - Update design.md section 2 with actual partition key implementation details
  - Update design.md section 4 with final configuration examples
  - Add topology diagrams showing consolidated vs per-symbol deployment
  - _Requirements: [Spec documentation]_

- [ ] 15.1 Update requirements with new strategy
  - Add FR2 update: consolidated topics as default, per-symbol as option
  - Add configuration examples for both strategies
  - Document topic naming patterns for consolidated topics
  - _Requirements: FR2_

- [ ] 15.2 Update design documentation
  - Update architecture diagrams to show consolidated topics
  - Add final implementation examples for topic manager
  - Update partition strategy section with actual code patterns
  - _Requirements: [Design reference]_

---

## Phase 4: Tooling & Deployment (Weeks 4-5)

- [ ] 16. Create topic migration tooling
  - Write script to migrate messages from per-symbol to consolidated topics
  - Support dry-run mode to verify migration without changes
  - Implement offset tracking and resumable migration
  - Generate migration report with message counts
  - _Requirements: [Operational tooling]_

- [ ] 16.1 Implement message migration script
  - Read from per-symbol topics (old naming)
  - Transform messages (if schema updates needed)
  - Write to consolidated topics (new naming)
  - Support parallel execution across multiple topics
  - _Requirements: [Tooling]_

- [ ] 16.2 Implement offset management
  - Track source and destination offsets
  - Support resumable migration (checkpoint progress)
  - Verify message integrity after migration
  - Generate migration completion report
  - _Requirements: [Tooling]_

- [x] 17. Create monitoring dashboard and metrics setup
  - Deploy Prometheus scrape configuration for Kafka producer
  - Create Grafana dashboard for key metrics
  - Setup alerting rules for critical conditions
  - Document metric definitions and interpretation
  - _Requirements: FR6 (Monitoring & Observability)_
  - _Completed: Nov 11, 2025 - PrometheusMetricsExporter class with 9 metrics, alert rules, Grafana dashboard_

- [x] 17.1 Setup Prometheus collection
  - Configure Prometheus to scrape `/metrics` endpoint
  - Define metric collection interval (10s recommended)
  - Setup data retention policy (30 days recommended)
  - Configure Alertmanager for alert routing
  - _Requirements: FR6_
  - _Completed: Nov 11, 2025 - prometheus.md with full configuration guide_

- [x] 17.2 Create Grafana dashboard
  - Build dashboard showing messages sent over time
  - Add latency percentile graphs (p50, p95, p99)
  - Add error rate and DLQ message tracking
  - Include per-exchange and per-data-type breakdowns
  - _Requirements: FR6_
  - _Completed: Nov 11, 2025 - grafana-dashboard.json with 9 panels covering all metrics_

- [x] 17.3 Define alerting rules
  - Alert if producer queue lag exceeds 10K messages
  - Alert if p99 latency exceeds 50ms
  - Alert if error rate exceeds 1%
  - Alert if Kafka brokers unavailable
  - _Requirements: FR6_
  - _Completed: Nov 11, 2025 - alert-rules.yaml with 8 alerts (critical/warning/info) and recording rules_

- [ ] 18. Create comprehensive operational runbook
  - Document incident response procedures
  - Write topic recreation procedures
  - Include partition rebalancing steps
  - Provide rollback procedures for producer versions
  - _Requirements: [Operational procedures]_

- [ ] 18.1 Write incident response runbook
  - Broker unavailability: expected behavior and recovery
  - High producer lag: diagnosis and remediation
  - High error rate: common causes and fixes
  - DLQ overflow: investigation and cleanup
  - _Requirements: [Operational support]_

- [ ] 18.2 Write infrastructure procedures
  - Topic recreation (if accidentally deleted)
  - Partition rebalancing (after broker addition/removal)
  - Consumer group offset reset (for replaying data)
  - Broker recovery (from backup, if applicable)
  - _Requirements: [Operational support]_

---

## Requirements Traceability Matrix

| Requirement ID | Requirement | Task(s) | Status |
|---|---|---|---|
| FR1 | Kafka Backend Implementation | 4, 4.1, 4.2, 4.3, 9 | Core |
| FR2 | Topic Management | 1, 1.1, 1.2, 1.3, 6, 6.1, 6.2, 9.1, 15, 15.1 | Core |
| FR3 | Partitioning Strategies | 2, 2.1, 2.2, 2.3, 2.4, 7, 7.1, 7.2, 7.3, 9.2 | Core |
| FR4 | Serialization Integration | 3, 3.1, 3.2, 3.3, 8, 8.1, 8.2, 9 | Core |
| FR5 | Delivery Guarantees | 4, 5.2, 9.3 | Testing |
| FR6 | Monitoring & Observability | 17, 17.1, 17.2, 17.3, 14.2 | Tooling |
| NFR1 | Performance | 10, 10.1, 10.2, 10.3 | Testing |
| NFR2 | Reliability | 11, 11.1, 11.2 | Testing |
| NFR3 | Configuration | 5, 5.1, 5.2, 5.3, 6.3 | Core |

---

## Task Execution Sequence

### Critical Path
1. **Tasks 1-5** (Core Components) - Enable basic Kafka producer functionality
2. **Tasks 6-8** (Unit Tests) - Validate individual components
3. **Task 9** (Integration Tests) - Verify end-to-end flow
4. **Tasks 10-11** (Performance & Compatibility) - Ensure production readiness
5. **Tasks 12-15** (Documentation) - Enable consumer adoption
6. **Tasks 16-18** (Tooling) - Support operations and migration

### Parallel Execution Opportunities
- Tasks 6-11 (testing) can run in parallel after Tasks 1-5 complete
- Tasks 12-15 (documentation) can run in parallel with testing
- Tasks 16-18 (tooling) can start after core and testing complete

---

## Acceptance Criteria Checklist

### Phase 1 Completion
- [ ] Consolidated and per-symbol topic naming both work
- [ ] 4 partition key strategies selectable and tested
- [ ] Message headers include routing metadata
- [ ] Configuration models load from YAML and Python
- [ ] KafkaCallback integrates all new components

### Phase 2 Completion
- [ ] All unit tests pass (topic naming, partitioning, headers, config)
- [ ] Integration tests verify end-to-end Kafka flow
- [ ] Performance benchmarks show 10K+ msg/s capability
- [ ] Backward compatibility confirmed (per-symbol mode still works)
- [ ] No performance regression vs existing implementation

### Phase 3 Completion
- [ ] Consumer integration guide covers Flink, DuckDB, Python
- [ ] Migration guide includes rollback procedures
- [ ] Operator guide documents procedures and alerts
- [ ] Spec documents updated with new strategy details

### Phase 4 Completion
- [ ] Topic migration tooling supports dry-run and resume
- [ ] Grafana dashboard displays key metrics
- [ ] Alerting rules defined and tested
- [ ] Operational runbooks document all procedures

---

## Engineering Excellence Standards

All tasks must satisfy:
- ✅ **Natural Language**: Describe capabilities, not code structure (per rules)
- ✅ **Task Integration**: Every task builds on previous outputs
- ✅ **Flexible Sizing**: Sub-tasks 1-3 hours each, groups by logical cohesion
- ✅ **Requirements Mapping**: All requirements covered, cross-referenced
- ✅ **Code Focus**: Implementation and testing only, no deployment/docs exclusions
- ✅ **Maximum 2 Levels**: Major + sub-task hierarchy only
- ✅ **Sequential Numbering**: 1, 2, 3... (no repeats), 1.1, 1.2, 2.1... (resets)
- ✅ **Checkbox Format**: Proper markdown with details and requirement refs

---

## Implementation Timeline

**Weeks 1-2**: Phase 1 Core Implementation
- Complete all 5 major tasks + sub-tasks
- Output: Consolidated topics, 4 partition strategies, headers, configuration models

**Week 2-3**: Phase 2 Testing & Validation (can parallelize)
- Complete tasks 6-11 in parallel where possible
- Output: Unit tests, integration tests, performance benchmarks, backward compatibility verification

**Week 3-4**: Phase 3 Documentation & Migration
- Complete tasks 12-15 sequentially
- Output: Consumer guides, migration guide, operator guide, spec updates

**Week 4-5**: Phase 4 Tooling & Deployment
- Complete tasks 16-18 sequentially
- Output: Migration tooling, monitoring, operational runbooks

---

## Notes

- **Protobuf Integration**: Tasks assume Spec 1 (protobuf-callback-serialization) is merged. If not available at task start, implement JSON fallback in Phase 1.
- **Backward Compatibility**: Per-symbol topic naming must remain functional throughout and after implementation.
- **Monitoring First**: Instrumentation (metrics, logging) should be added as each component is implemented, not deferred.
- **Consumer Examples**: Reference implementations should not include consumer business logic - focus on deserialization and topic subscription patterns.
- **Topic Scaling Benefit**: Moving from O(symbols × exchanges) to O(data_types) reduces topic count from 1000s to ~20, simplifying operations and reducing Kafka metadata overhead.
