# Phase 4: Producer-Centric Post-Merge Enhancement Roadmap (REFINED)

**Status**: PLANNED (deferred to post-merge)
**Timeline**: 3 weeks (15 working days)
**Effort**: 15 engineering days
**Priority**: High
**Architecture Principle**: Cryptofeed stops at Kafka. Consumers handle everything downstream.

---

## Scope Changes from Original Plan

### ❌ REMOVED (Consumer Responsibility)
- Task 12: Flink consumer guide
- Task 13: DuckDB consumer guide  
- Task 14: Python consumer guide

**Rationale**: Per cryptofeed architecture, consumers implement their own storage, analytics, and persistence. Producer documentation should focus on Kafka message contracts, not consumer implementations.

### ✅ RETAINED (Producer-Side Work)
- Task 10-10.3: Performance benchmarking (latency, throughput, memory, CPU)
- Task 17: Prometheus metrics integration (producer metrics: send rate, error rate, latency percentiles)
- Task 15: Migration guide (legacy → Phase 2 producer configuration, NOT consumer setup)
- Task 15.1-15.3: Operational documentation + rollback (producer operations)
- Task 16: Migration CLI tool (config translation for producer)

### 🆕 ADDED (Producer Enhancements)
- Task 17.1: Performance optimization based on benchmarks
- Task 17.2: Dead letter queue + circuit breaker patterns
- Task 17.3: Custom alerting rules + health checks
- Task 18: Schema registry integration documentation
- Task 18.1: Schema versioning guide
- Task 19: Producer tuning guide
- Task 19.1: Troubleshooting runbook

---

## Week 1: Performance Benchmarking (Days 1-3)

**Goal**: Establish baseline performance metrics and identify optimization opportunities.

### Task 10: End-to-end latency benchmarking
**Effort**: 1 day

**Activities**:
- Measure p50/p95/p99 latency from callback invocation to Kafka ACK
- Test with Trade (250 bytes), OrderBook (1000+ bytes), Ticker (400 bytes) messages
- Benchmark both consolidated and per-symbol topic strategies
- Test all 4 partition strategies (Composite, Symbol, Exchange, RoundRobin)

**Deliverables**:
- `tests/performance/benchmark_kafka_latency.py` - Latency measurement harness
- Baseline latency report: `docs/benchmarks/latency-baseline.md`

**Success Criteria**:
- [ ] p99 latency < 10ms (target from design)
- [ ] Latency measured across all data types (trades, orderbook, ticker)
- [ ] Per-strategy latency comparison documented

---

### Task 10.1: Throughput testing
**Effort**: 1 day

**Activities**:
- Benchmark sustained throughput (messages/second)
- Test with 3-broker Kafka cluster (production-like)
- Measure throughput across 10 exchanges × 100 symbols = 1000 streams
- Compare consolidated vs per-symbol topic throughput
- Test with compression enabled (snappy, lz4, gzip)

**Deliverables**:
- `tests/performance/benchmark_kafka_throughput.py` - Throughput test suite
- Throughput report: `docs/benchmarks/throughput-baseline.md`

**Success Criteria**:
- [ ] Sustained throughput > 100k msg/s (target from design)
- [ ] No message loss under sustained load
- [ ] Throughput stable over 10-minute test run

---

### Task 10.2: Memory profiling
**Effort**: 0.5 days

**Activities**:
- Profile memory usage under sustained load (1 hour test)
- Measure heap size, buffer usage, GC pressure
- Test with different batch sizes (100, 1000, 10000 messages)
- Identify memory leaks or unbounded growth

**Deliverables**:
- Memory profiling script: `tests/performance/profile_memory.py`
- Memory baseline report: `docs/benchmarks/memory-baseline.md`

**Success Criteria**:
- [ ] Memory usage stable (no leaks)
- [ ] Heap size < 500 MB for 1000 streams
- [ ] No memory warnings or OOM errors

---

### Task 10.3: CPU usage analysis
**Effort**: 0.5 days

**Activities**:
- Profile CPU usage during peak load
- Identify hot paths (profiling with py-spy or cProfile)
- Measure CPU usage per partition strategy
- Test with different serialization modes (protobuf vs JSON)

**Deliverables**:
- CPU profiling script: `tests/performance/profile_cpu.py`
- CPU baseline report: `docs/benchmarks/cpu-baseline.md`

**Success Criteria**:
- [ ] CPU usage < 50% for 1000 streams (4-core machine)
- [ ] Hot paths identified and documented
- [ ] No busy-wait or spin loops detected

---

## Week 2: Monitoring, Optimization & Reliability (Days 4-10)

**Goal**: Add production-grade observability, optimize based on benchmarks, and implement reliability patterns.

### Task 17: Prometheus metrics integration
**Effort**: 2 days

**Activities**:
- Export producer metrics via `/metrics` endpoint
- Implement metrics: `kafka_messages_sent_total`, `kafka_send_latency_seconds`, `kafka_errors_total`
- Add per-exchange and per-data-type labels
- Expose partition-level metrics (messages per partition)
- Test metrics with Prometheus scraping

**Deliverables**:
- Prometheus exporter in `cryptofeed/kafka_callback.py` (lines 900-950)
- Metrics documentation: `docs/monitoring/prometheus.md`
- Grafana dashboard template: `grafana/kafka-producer-dashboard.json`

**Success Criteria**:
- [ ] All key metrics exported (send rate, latency percentiles, error rate)
- [ ] Metrics scraped successfully by Prometheus
- [ ] Grafana dashboard displays real-time metrics

---

### Task 17.1: Performance optimization
**Effort**: 4 days

**Activities**:
- Analyze hot paths identified in Task 10.3 (CPU profiling)
- Optimize message serialization (cache protobuf descriptors)
- Optimize partition key generation (cache encoded keys)
- Optimize header enrichment (reduce allocations)
- Tune Kafka producer config (batch.size, linger.ms, compression.type)
- Rerun benchmarks (Tasks 10-10.3) to validate improvements

**Deliverables**:
- Optimized code in `cryptofeed/kafka_callback.py`
- Performance optimization report: `docs/benchmarks/optimization-results.md`
- Updated configuration guide: `docs/kafka/tuning-guide.md`

**Success Criteria**:
- [ ] p99 latency improved by 20% (target: <5ms vs baseline <10ms)
- [ ] Throughput improved by 15% (target: >115k msg/s vs baseline >100k)
- [ ] CPU usage reduced by 10%

---

### Task 17.2: Dead letter queue + circuit breaker patterns
**Effort**: 3 days

**Activities**:
- Implement DLQ for messages that fail after max retries
- Create circuit breaker for repeated Kafka broker failures
- Add exponential backoff for transient errors
- Test DLQ with simulated broker failures
- Document DLQ message format and retrieval

**Deliverables**:
- DLQ implementation in `cryptofeed/kafka_callback.py` (lines 950-1020)
- Circuit breaker implementation in `cryptofeed/kafka_callback.py` (lines 1020-1070)
- DLQ guide: `docs/kafka/dead-letter-queue.md`
- Circuit breaker guide: `docs/kafka/circuit-breaker.md`

**Success Criteria**:
- [ ] Failed messages routed to DLQ topic (`cryptofeed.dlq`)
- [ ] Circuit breaker opens after 5 consecutive failures
- [ ] Circuit breaker closes after 60-second cooldown
- [ ] No silent message drops (all failures logged)

---

### Task 17.3: Custom alerting rules + health checks
**Effort**: 1 day

**Activities**:
- Define alerting rules for Prometheus (high error rate, high latency, high lag)
- Implement health check endpoint (`/health`) for Kubernetes liveness/readiness probes
- Test health check with Kafka broker unavailability
- Document alerting runbook (alert definitions, thresholds, response procedures)

**Deliverables**:
- Alerting rules: `prometheus/kafka-producer-alerts.yml`
- Health check endpoint in `cryptofeed/kafka_callback.py` (lines 1070-1100)
- Alerting runbook: `docs/monitoring/alerting-runbook.md`

**Success Criteria**:
- [ ] Alerts fire correctly in test scenarios (simulated failures)
- [ ] Health check returns 200 when Kafka available, 503 when unavailable
- [ ] Alerting runbook reviewed by SRE team

---

## Week 3: Migration, Schema Management & Operations (Days 11-15)

**Goal**: Enable smooth migration from legacy backend, document schema management, and provide operational procedures.

### Task 18: Schema registry integration documentation
**Effort**: 2 days

**Activities**:
- Document Confluent Schema Registry integration
- Document Buf Schema Registry integration
- Provide protobuf schema upload procedures
- Test schema registry compatibility mode (backward, forward, full)

**Deliverables**:
- Schema registry guide: `docs/kafka/schema-registry.md`
- Buf Schema Registry example: `examples/kafka_buf_schema_registry.py`
- Confluent Schema Registry example: `examples/kafka_confluent_schema_registry.py`

**Success Criteria**:
- [ ] Protobuf schemas uploaded to Confluent Schema Registry
- [ ] Protobuf schemas uploaded to Buf Schema Registry
- [ ] Backward compatibility validated (v1 → v2 schema evolution)

---

### Task 18.1: Schema versioning guide
**Effort**: 1 day

**Activities**:
- Document protobuf schema versioning best practices
- Provide schema evolution examples (add field, deprecate field, rename field)
- Test backward/forward compatibility with old/new consumers
- Document schema version header usage

**Deliverables**:
- Schema versioning guide: `docs/kafka/schema-versioning.md`
- Schema evolution examples: `examples/schema_evolution/`

**Success Criteria**:
- [ ] Backward compatibility examples documented (v1 producer → v2 consumer)
- [ ] Forward compatibility examples documented (v2 producer → v1 consumer)
- [ ] Schema evolution tested with real Kafka messages

---

### Task 15: Migration guide (producer-focused)
**Effort**: 2 days

**Activities**:
- Document migration from legacy `cryptofeed/backends/kafka.py` to new `KafkaCallback`
- Provide configuration translation examples (old YAML → new YAML)
- Document dual-write strategy for zero-downtime migration
- Write rollback procedures (revert to legacy backend)
- **Exclude consumer migration** (consumers implement their own storage)

**Deliverables**:
- Migration guide: `docs/kafka/migration-guide.md`
- Configuration examples: `examples/kafka_migration/`

**Success Criteria**:
- [ ] Legacy config translated to new config (5+ real-world examples)
- [ ] Dual-write strategy documented with YAML examples
- [ ] Rollback procedure tested (revert to legacy backend)

---

### Task 15.1: Deprecation notice
**Effort**: 0.5 days

**Activities**:
- Add deprecation warnings to `cryptofeed/backends/kafka.py`
- Update CHANGELOG.md with deprecation timeline
- Add migration guide link to deprecation warning

**Deliverables**:
- Deprecation warnings in legacy backend
- Updated CHANGELOG.md

**Success Criteria**:
- [ ] Deprecation warnings logged when legacy backend used
- [ ] CHANGELOG.md updated with deprecation timeline (6 months)

---

### Task 15.2: Configuration translation examples
**Effort**: 1 day

**Activities**:
- Provide 5+ real-world config examples (Binance, Coinbase, Kraken, etc.)
- Show old config → new config translation side-by-side
- Document breaking changes (if any)
- Test translated configs with live exchanges

**Deliverables**:
- Configuration examples: `examples/kafka_migration/configs/`

**Success Criteria**:
- [ ] 5+ real-world configs translated
- [ ] All translated configs tested with live exchanges

---

### Task 15.3: Rollback procedures
**Effort**: 0.5 days

**Activities**:
- Document rollback steps (revert to legacy backend)
- Test rollback with dual-write scenario
- Document Kafka topic cleanup (delete new topics)

**Deliverables**:
- Rollback guide: `docs/kafka/rollback-guide.md`

**Success Criteria**:
- [ ] Rollback tested in staging environment
- [ ] Topic cleanup validated (no data loss)

---

### Task 16: Migration CLI tool
**Effort**: 2 days

**Activities**:
- Write CLI tool to translate old config → new config
- Support dry-run mode (preview changes without applying)
- Validate new config against schema
- Generate migration report (config diffs, breaking changes)

**Deliverables**:
- Migration CLI: `tools/migrate_kafka_config.py`
- Validation script: `tools/validate_kafka_config.py`

**Success Criteria**:
- [ ] CLI translates old config → new config correctly
- [ ] Dry-run mode works (no side effects)
- [ ] Validation script catches invalid configs

---

### Task 19: Producer tuning guide
**Effort**: 1 day

**Activities**:
- Document tuning parameters (batch.size, linger.ms, compression.type, acks)
- Provide tuning recommendations for different use cases (latency-sensitive, throughput-optimized)
- Test tuning parameters with benchmarks (reference Tasks 10-10.3)

**Deliverables**:
- Tuning guide: `docs/kafka/tuning-guide.md`

**Success Criteria**:
- [ ] Tuning recommendations validated with benchmarks
- [ ] Use case examples provided (low-latency, high-throughput, balanced)

---

### Task 19.1: Troubleshooting runbook
**Effort**: 1 day

**Activities**:
- Document common issues (high latency, message loss, broker unavailability)
- Provide diagnostic steps (check broker health, check topic lag, check error logs)
- Document resolution steps (increase batch size, increase partitions, restart producer)
- Test troubleshooting steps in staging environment

**Deliverables**:
- Troubleshooting runbook: `docs/kafka/troubleshooting-runbook.md`

**Success Criteria**:
- [ ] Common issues documented (5+ scenarios)
- [ ] Diagnostic steps validated in staging
- [ ] Resolution steps tested and verified

---

## Resource Allocation

**Team Composition** (Revised):
- **Performance Engineer** (Week 1: Days 1-3) - Benchmarking and profiling
- **Observability Specialist** (Week 2: Days 4-6) - Metrics and alerting
- **Reliability Engineer** (Week 2: Days 7-10) - Optimization, DLQ, circuit breaker
- **Migration Engineer** (Week 3: Days 11-15) - Migration guide, CLI tool, schema docs

**Estimated Effort**: 15 engineering days over 3 weeks (same as original)

---

## Gantt Timeline (Revised)

| Day | Week | Task | Owner | Status |
|-----|------|------|-------|--------|
| 1 | 1 | Task 10: Latency benchmarking | Performance Eng | ⏳ Planned |
| 2 | 1 | Task 10.1: Throughput testing | Performance Eng | ⏳ Planned |
| 3 | 1 | Task 10.2-10.3: Memory/CPU profiling | Performance Eng | ⏳ Planned |
| 4-5 | 2 | Task 17: Prometheus metrics | Observability Eng | ⏳ Planned |
| 6-9 | 2 | Task 17.1: Performance optimization | Reliability Eng | ⏳ Planned |
| 10-12 | 2 | Task 17.2: DLQ + Circuit breaker | Reliability Eng | ⏳ Planned |
| 13 | 2 | Task 17.3: Alerting + Health checks | Observability Eng | ⏳ Planned |
| 14-15 | 3 | Task 18-18.1: Schema registry + versioning | Migration Eng | ⏳ Planned |
| 16-17 | 3 | Task 15-15.3: Migration guide + rollback | Migration Eng | ⏳ Planned |
| 18-19 | 3 | Task 16: Migration CLI tool | Migration Eng | ⏳ Planned |
| 20 | 3 | Task 19-19.1: Tuning + troubleshooting | Migration Eng | ⏳ Planned |

---

## Task Summary (Refined)

**Total Tasks**: 16 (down from 18 in original plan)

### Week 1 (Performance)
- Task 10: Latency benchmarking
- Task 10.1: Throughput testing
- Task 10.2: Memory profiling
- Task 10.3: CPU usage analysis

### Week 2 (Monitoring & Reliability)
- Task 17: Prometheus metrics integration
- Task 17.1: Performance optimization (NEW)
- Task 17.2: DLQ + circuit breaker (NEW)
- Task 17.3: Alerting + health checks (NEW)

### Week 3 (Migration & Operations)
- Task 18: Schema registry integration (NEW)
- Task 18.1: Schema versioning guide (NEW)
- Task 15: Migration guide (producer-focused)
- Task 15.1: Deprecation notice
- Task 15.2: Configuration translation
- Task 15.3: Rollback procedures
- Task 16: Migration CLI tool
- Task 19: Producer tuning guide (NEW)
- Task 19.1: Troubleshooting runbook (NEW)

---

## Success Metrics (Producer-Centric)

### Performance Metrics
- [ ] p99 latency < 5ms (optimized from baseline <10ms)
- [ ] Throughput > 115k msg/s (improved from baseline >100k)
- [ ] Memory usage < 500 MB (1000 streams)
- [ ] CPU usage < 50% (1000 streams, 4-core machine)

### Reliability Metrics
- [ ] Zero silent message drops (all failures logged or DLQ'd)
- [ ] Circuit breaker opens/closes correctly
- [ ] Health check responds correctly (200 healthy, 503 unhealthy)

### Observability Metrics
- [ ] Prometheus metrics exported and scraped
- [ ] Grafana dashboard displays real-time metrics
- [ ] Alerting rules fire correctly in test scenarios

### Migration Metrics
- [ ] 5+ real-world configs translated successfully
- [ ] Rollback tested and validated
- [ ] CLI tool translates configs correctly

---

## Risk Mitigation (Revised)

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Performance targets not met | Low | Design basis solid; early benchmarking + optimization pass |
| DLQ implementation complexity | Medium | Simple append-to-DLQ-topic pattern; no complex routing |
| Schema registry integration issues | Low | Protobuf schemas already defined; registry upload is straightforward |
| Migration complexity | Medium | CLI tooling + dual-write reduces manual effort |

---

## Dependencies

- **Spec 0** (normalized-data-schema-crypto): ✅ COMPLETE
- **Spec 1** (protobuf-callback-serialization): ✅ COMPLETE
- **Phase 1-2**: ✅ COMPLETE (493+ tests passing)
- **External**: Kafka cluster (3+ brokers) for benchmarking
- **External**: Prometheus + Grafana for monitoring validation

---

## Comparison: Original vs Refined

| Aspect | Original | Refined | Change |
|--------|----------|---------|--------|
| **Total Tasks** | 18 | 16 | -2 (removed consumer guides) |
| **Consumer Guides** | 3 (Flink, DuckDB, Python) | 0 | ❌ Removed (consumer responsibility) |
| **Producer Enhancements** | 0 | 4 (optimization, DLQ, schema, tuning) | 🆕 Added |
| **Timeline** | 3 weeks | 3 weeks | ✅ Same |
| **Effort** | 15 days | 15 days | ✅ Same |
| **Scope** | Producer + Consumer | Producer Only | 🎯 Aligned with architecture |

---

## Post-Merge Execution Plan

1. **Create GitHub Issue**: "Phase 4: Producer-Centric Enhancements"
   - Link to this document
   - Assign to team lead
   - Milestone: v1.1.0 (post-merge)

2. **Create Feature Branch**: `feature/kafka-producer-phase4`
   - Branch from `main` (after Phase 1-2 merge)

3. **Generate Kiro Tasks**: `/kiro:spec-tasks market-data-kafka-producer --phase 4`
   - Generate 16 refined tasks
   - Assign to specialist agents

4. **Run Gap Analysis**: `/kiro:validate-gap market-data-kafka-producer --phase 4`
   - Identify missing implementations
   - Prioritize based on performance impact

5. **Execute Week-by-Week**:
   - Week 1: Performance Engineer executes Tasks 10-10.3
   - Week 2: Reliability + Observability Engineers execute Tasks 17-17.3
   - Week 3: Migration Engineer executes Tasks 15-19.1

6. **Validation & Merge**:
   - Run full test suite (493+ existing + 50+ new = 540+ tests)
   - Performance benchmarks meet targets
   - Code review by 2+ engineers
   - Merge to `main`

---

**Next Step**: Await user decision: Proceed with refined plan or revert to original plan?
