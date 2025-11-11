# Phase 4: Post-Merge Enhancement Roadmap (Refined)

**Status**: READY FOR EXECUTION
**Timeline**: 3 weeks post-merge
**Effort**: 15 engineering days
**Priority**: Medium (production optimization)
**Scope**: Producer-side enhancements only (consumers are separate concern)

---

## Overview

Phase 4 deferred work for market-data-kafka-producer specification. Phases 1-2 complete (493+ tests passing, 100% implementation). This refined document focuses on **producer-side capabilities** per cryptofeed architecture: "Cryptofeed stops at Kafka. Consumers handle everything downstream."

**Removed**: Consumer integration guides (Tasks 12-14) → Consumer responsibility
**Added**: Producer optimization, reliability, schema management, troubleshooting

---

## Task Breakdown (16 Tasks, 3 Weeks)

### Week 1: Performance Benchmarking (Days 1-3)

**Tasks 10-10.3**: Performance validation
- Task 10: End-to-end latency benchmarking (target: p99 <10ms)
- Task 10.1: Throughput testing (target: >100k msg/s)
- Task 10.2: Memory profiling under load
- Task 10.3: CPU usage analysis

**Deliverables**:
- `tests/performance/benchmark_kafka_producer.py` - Benchmark harness
- `docs/benchmarks/kafka-producer.md` - Baseline metrics + bottleneck analysis
- Performance profiling data (latency, throughput, memory, CPU)

**Success Criteria**:
- [ ] p99 latency < 10ms (consolidated topics)
- [ ] Throughput > 100k msg/s sustained
- [ ] Memory < 500MB per feed instance
- [ ] CPU usage < 50% under load

---

### Week 2: Monitoring, Optimization & Reliability (Days 4-13)

**Task 17**: Prometheus Metrics Integration (2 days)
- Producer metrics: `messages_produced_total`, `produce_latency_seconds`, `produce_errors_total`
- Kafka metrics: broker latency, partition lag, buffer utilization
- Serialization metrics: message size distribution, serialization latency

**Deliverables**:
- Prometheus exporter in `cryptofeed/backends/kafka_metrics.py`
- `docs/monitoring/prometheus.md` - Metrics documentation + alert rules
- Grafana dashboard JSON template

**Task 17.1**: Performance Optimization (4 days)
- Identify bottlenecks from Task 10-10.3 benchmarks
- Optimize hot paths: message serialization, buffer flushing, partition key generation
- Profile and optimize idempotent producer overhead
- Target: Achieve p99 <5ms (vs baseline <10ms)

**Deliverables**:
- Optimization commits with performance improvement reports
- Updated benchmarking results post-optimization

**Task 17.2**: Dead Letter Queue & Circuit Breaker Patterns (3 days)
- Implement DLQ for messages that fail Kafka produce (retries exhausted)
- Add circuit breaker for broker unavailability (fail-fast vs backoff)
- Exponential backoff strategy for transient errors
- Metrics: DLQ size, circuit breaker state changes

**Deliverables**:
- `cryptofeed/backends/kafka_dlq.py` - DLQ handler
- `cryptofeed/backends/kafka_circuit_breaker.py` - Circuit breaker pattern
- Integration tests for error scenarios

**Task 17.3**: Custom Alerting & Health Checks (1 day)
- Prometheus alert rules for: error rate >1%, latency p99 >15ms, partition lag >100
- Grafana alert notifications (email, Slack)
- `/health` endpoint for producer status (Kafka connectivity, buffer health)

**Deliverables**:
- Alert rules in `docs/monitoring/alert-rules.yaml`
- Health check endpoint in KafkaCallback

---

### Week 3: Schema Management, Migration & Operations (Days 14-20)

**Task 18**: Schema Registry Integration (2 days)
- Protobuf schema registration (Confluent Schema Registry or Buf)
- Schema versioning strategy (major.minor.patch)
- Schema ID embedding in message headers
- Validation: schema compatibility before produce

**Deliverables**:
- `cryptofeed/backends/kafka_schema.py` - Schema registry client
- `docs/kafka/schema-registry-setup.md` - Integration guide
- Schema registration automation (CI/CD integration)

**Task 18.1**: Schema Versioning Guide (1 day)
- Backward/forward compatibility rules
- Schema evolution examples (adding fields, deprecating fields)
- Multi-version producer support
- Testing schema changes

**Deliverables**:
- `docs/kafka/schema-versioning.md` - Best practices guide

**Task 15**: Migration Guide - Producer Focus (1 day)
- Config translation: legacy backend → Phase 2 KafkaCallback
- Dual-write strategy documentation (both backends simultaneously)
- Gradual cutover procedures (producer-side only)
- Rollback planning

**Deliverables**:
- `docs/kafka/migration-guide.md` - Producer migration strategy

**Task 15.1**: Deprecation Notice (0.5 days)
- Add deprecation warning to `cryptofeed/backends/kafka.py` (legacy backend)
- Log guidance to migrate to `KafkaCallback`

**Task 15.2**: Configuration Translation Examples (0.5 days)
- YAML examples: legacy config → Phase 2 config
- Topic strategy migration (single topic → consolidated topics)
- Partition strategy selection guide

**Deliverables**:
- `docs/kafka/config-translation-examples.md`

**Task 15.3**: Rollback Procedures (0.5 days)
- Emergency rollback from KafkaCallback → legacy backend
- Data recovery procedures (if applicable)
- Health check verification post-rollback

**Deliverables**:
- `docs/kafka/rollback-procedures.md`

**Task 16**: Migration CLI Tool (2 days)
- Config translator (legacy YAML → Phase 2 YAML)
- Config validator (compatibility checks)
- Dry-run mode (preview changes without applying)

**Deliverables**:
- `tools/migrate-kafka-config.py` - Migration CLI tool
- Integration tests with 10 real-world legacy configs

**Task 19**: Producer Tuning Guide (1 day)
- Configuration reference: batch.size, linger.ms, buffer.memory
- Use case profiles: latency-sensitive vs throughput-optimized
- Performance tuning checklist
- Monitoring-driven optimization workflow

**Deliverables**:
- `docs/kafka/producer-tuning.md` - Comprehensive tuning guide

**Task 19.1**: Troubleshooting Runbook (1 day)
- Common issues: broker unavailable, message loss, latency spikes
- Diagnostic steps and resolution procedures
- Log interpretation guide
- Alert response decision tree

**Deliverables**:
- `docs/kafka/troubleshooting.md` - Runbook for operators

---

## Resource Allocation (Refined)

**Team Composition**:
- 1 Performance Engineer (Week 1, part of Week 2)
- 1 Observability Specialist (Week 2: Tasks 17, 17.3)
- 1 Reliability Engineer (Week 2: Task 17.2, Week 3: Tasks 18-18.1)
- 1 Migration/Ops Engineer (Week 3: Tasks 15-16, 19-19.1)

**Estimated Effort**: 15 engineering days across 4 specialized roles

---

## Parallel Work Streams

| Stream | Duration | Tasks | Owner |
|--------|----------|-------|-------|
| **Performance** | Days 1-5 | 10, 10.1, 10.2, 10.3, 17.1 | Performance Engineer |
| **Observability** | Days 6-7 | 17, 17.3 | Observability Specialist |
| **Reliability** | Days 8-10 | 17.2, 18, 18.1 | Reliability Engineer |
| **Migration & Ops** | Days 11-20 | 15-16, 19-19.1 | Migration/Ops Engineer |

---

## Comparison: Original vs Refined

| Aspect | Original | Refined | Change |
|--------|----------|---------|--------|
| **Total Tasks** | 18 | 16 | -2 (removed consumer guides) |
| **Timeline** | 3 weeks | 3 weeks | Same |
| **Effort** | 15 days | 15 days | Same |
| **Producer Capabilities** | Basic | Advanced | +7 enhancements |
| **Architecture Alignment** | Partial | Full | Better |
| **Production Readiness** | Good | Excellent | Improved |

**Removed Tasks (Consumer Responsibility)**:
- ❌ Task 12: Flink consumer guide
- ❌ Task 13: DuckDB consumer guide
- ❌ Task 14: Python consumer guide

**Added Tasks (Producer Enhancement)**:
- ✅ Task 17.1: Performance optimization
- ✅ Task 17.2: DLQ + circuit breaker patterns
- ✅ Task 17.3: Custom alerting + health checks
- ✅ Task 18-18.1: Schema registry + versioning
- ✅ Task 19-19.1: Producer tuning + troubleshooting

---

## Risk Mitigation

| Risk | Probability | Mitigation |
|------|-------------|---------------|
| Performance targets not met | Medium | Dedicated optimization phase (Task 17.1) |
| Schema compatibility issues | Low | Backward/forward compatibility testing |
| Migration complexity | Medium | CLI tooling + validation automation |
| Kafka broker failures | Low | Circuit breaker + DLQ patterns (Task 17.2) |

---

## Success Criteria

- [ ] All benchmarks meet targets (p99 <10ms, >100k msg/s)
- [ ] Performance optimization achieves p99 <5ms
- [ ] Prometheus metrics operational with alert rules
- [ ] DLQ & circuit breaker patterns implemented + tested
- [ ] Schema registry integration tested end-to-end
- [ ] Migration CLI tool validates 10/10 legacy configs
- [ ] Operator runbook reviewed and approved
- [ ] Final validation score ≥8.5/10

---

## Dependencies

- Phase 1-2 implementation merged to main branch
- Kafka cluster with 3+ brokers (for benchmarking)
- Prometheus & Grafana (for observability)
- Schema Registry (Confluent or Buf) access

---

## Phase 1-2 Completion Summary

- **Completion Date**: November 10, 2025
- **Merge Score**: 8.6/10
- **Test Coverage**: 493+ tests passing
- **Implementation**: 100% complete (18/18 tasks)
- **Code Quality**: 8.5/10

---

## Execution Timeline

**Next Step**: Create feature branch and begin Phase 4 execution using kiro commands.

```bash
# Feature branch from main (after Phase 1-2 merge)
git checkout main && git pull
git checkout -b feature/market-data-kafka-producer-phase-4

# Generate Phase 4 tasks via kiro
/kiro:spec-tasks market-data-kafka-producer --phase=4

# Validate spec gaps before execution
/kiro:validate-gap market-data-kafka-producer --phase=4

# Execute weekly sprints with subagents
# Week 1: Performance Engineer runs Tasks 10-10.3
# Week 2: Observability + Reliability engineers run Tasks 17-17.2
# Week 3: Migration/Ops engineer runs Tasks 15-16, 19-19.1
```

---

**Status**: Ready for execution on `feature/market-data-kafka-producer-phase-4` branch post-merge
