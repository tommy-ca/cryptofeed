# Phase 4 Master Command Sequence - Complete Execution Workflow

**Status**: ACTIVE EXECUTION
**Feature Branch**: `feature/kafka-producer-phase-4`
**Timeline**: 3 weeks (21 days)
**Success Target**: Final validation score ≥8.5/10

---

## 🚀 PHASE 4 COMPLETE COMMAND SEQUENCE

### PHASE A: Pre-Execution Setup (Day 0) ✅ COMPLETE

```bash
# 1. Create feature branch
git checkout -b feature/kafka-producer-phase-4

# 2. Verify spec status
/kiro:spec-status market-data-kafka-producer

# 3. Validate implementation gaps
/kiro:validate-gap market-data-kafka-producer
```

**Status**: ✅ Pre-execution checks complete, feature branch ready

---

## PHASE B: WEEK 1 - Performance Benchmarking (Days 1-3)

### ✅ ACTIVE NOW

**Execute Week 1 Tasks** (10-10.3):

```bash
# CURRENTLY EXECUTING
/kiro:spec-impl market-data-kafka-producer 10 10.1 10.2 10.3
```

**Owner**: Performance Engineer
**Duration**: 3 days

**Expected Deliverables**:
- `tests/performance/benchmark_kafka_producer.py` - Latency/throughput harness
- `docs/benchmarks/kafka-producer.md` - Baseline metrics report
- Baseline metrics (p99 <10ms, >100k msg/s, <500MB, <50% CPU)

---

### Week 1 Validation Checkpoint (End of Day 3)

```bash
# EXECUTE AFTER DAY 3 COMPLETION
/kiro:validate-impl market-data-kafka-producer 10 10.1 10.2 10.3
```

**Decision Gate**: Score ≥7.5/10
- ✅ **PASS**: Proceed to Week 2
- ❌ **FAIL**: Iterate on failing tasks (extend Week 1)

**Commit Week 1 Results**:
```bash
git add tests/performance/ docs/benchmarks/
git commit -m "feat(kafka): Complete Week 1 - Performance benchmarking (Tasks 10-10.3)

- Benchmark harness implementation
- Latency, throughput, memory, CPU baselines
- Performance bottleneck analysis for optimization

Validation score: [INSERT SCORE]

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## PHASE C: WEEK 2 - Monitoring & Reliability (Days 4-13)

### ⏳ PENDING WEEK 1 VALIDATION (Score ≥7.5)

Execute Week 2 tasks sequentially (one task/command group at a time):

### Week 2a: Prometheus Metrics (Days 4-5, ~2 days)

```bash
# EXECUTE AFTER WEEK 1 VALIDATION PASSES
/kiro:spec-impl market-data-kafka-producer 17
```

**Owner**: Observability Specialist
**Task**: Task 17 - Prometheus metrics integration
**Deliverable**: `cryptofeed/backends/kafka_metrics.py`, `docs/monitoring/prometheus.md`, Grafana dashboard

**Commit After Task 17**:
```bash
git add cryptofeed/backends/kafka_metrics.py docs/monitoring/
git commit -m "feat(kafka): Add Prometheus metrics integration (Task 17)

- Producer metrics (messages_produced_total, produce_latency_seconds, produce_errors_total)
- Kafka metrics (broker latency, partition lag, buffer utilization)
- Grafana dashboard JSON template

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Week 2b: Performance Optimization (Days 6-9, ~4 days)

```bash
# EXECUTE AFTER TASK 17 COMPLETION
/kiro:spec-impl market-data-kafka-producer 17.1
```

**Owner**: Performance Engineer
**Task**: Task 17.1 - Performance optimization
**Deliverable**: Optimization commits, p99 <5ms achievement, updated benchmarks

**Prerequisites**: Week 1 benchmark bottleneck analysis

**Commit After Task 17.1**:
```bash
git add tests/performance/ docs/benchmarks/
git commit -m "perf(kafka): Optimize hot paths for p99 <5ms (Task 17.1)

- Optimize message serialization pipeline
- Cache partition keys and headers
- Tune buffer flushing strategy
- Target achieved: p99 <5ms (vs baseline <10ms)

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Week 2c: DLQ & Circuit Breaker (Days 10-12, ~3 days)

```bash
# EXECUTE AFTER TASK 17.1 COMPLETION
/kiro:spec-impl market-data-kafka-producer 17.2
```

**Owner**: Reliability Engineer
**Task**: Task 17.2 - DLQ + circuit breaker patterns
**Deliverable**: `cryptofeed/backends/kafka_dlq.py`, `cryptofeed/backends/kafka_circuit_breaker.py`, integration tests

**Commit After Task 17.2**:
```bash
git add cryptofeed/backends/kafka_dlq.py cryptofeed/backends/kafka_circuit_breaker.py tests/
git commit -m "feat(kafka): Add dead letter queue and circuit breaker patterns (Task 17.2)

- DLQ handler for retry-exhausted messages
- Circuit breaker for broker unavailability
- Exponential backoff for transient errors
- Integration tests for error scenarios

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Week 2d: Alerting & Health Checks (Day 13, ~1 day)

```bash
# EXECUTE AFTER TASK 17.2 COMPLETION
/kiro:spec-impl market-data-kafka-producer 17.3
```

**Owner**: Observability Specialist
**Task**: Task 17.3 - Alerting + health checks
**Deliverable**: Alert rules, health check endpoints, Grafana notifications

**Commit After Task 17.3**:
```bash
git add docs/monitoring/alert-rules.yaml cryptofeed/
git commit -m "feat(kafka): Add custom alerting rules and health checks (Task 17.3)

- Prometheus alert rules (error rate >1%, latency p99 >15ms, lag >100)
- Grafana alert notifications (email, Slack)
- /health endpoint for producer status

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Week 2 Validation Checkpoint (End of Day 13)

```bash
# EXECUTE AFTER ALL WEEK 2 TASKS COMPLETE
/kiro:validate-impl market-data-kafka-producer 17 17.1 17.2 17.3
```

**Decision Gate**: Score ≥8.0/10
- ✅ **PASS**: Proceed to Week 3
- ❌ **FAIL**: Iterate on failing tasks (extend Week 2)

---

## PHASE D: WEEK 3 - Schema, Migration & Operations (Days 14-20)

### ⏳ PENDING WEEK 2 VALIDATION (Score ≥8.0)

Execute Week 3 tasks sequentially:

### Week 3a: Schema Registry & Versioning (Days 14-16, ~3 days)

```bash
# EXECUTE AFTER WEEK 2 VALIDATION PASSES
/kiro:spec-impl market-data-kafka-producer 18 18.1
```

**Owner**: Reliability Engineer
**Tasks**: Task 18 + 18.1 (Schema registry + versioning)
**Deliverables**:
- `cryptofeed/backends/kafka_schema.py`
- `docs/kafka/schema-registry-setup.md`
- `docs/kafka/schema-versioning.md`

**Commit After Tasks 18-18.1**:
```bash
git add cryptofeed/backends/kafka_schema.py docs/kafka/schema-*
git commit -m "feat(kafka): Add schema registry integration and versioning guide (Tasks 18-18.1)

- Protobuf schema registration (Confluent Schema Registry / Buf)
- Schema ID embedding in message headers
- Schema compatibility validation before produce
- Backward/forward compatibility rules
- Schema evolution examples

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Week 3b: Migration Guide & CLI Tool (Days 17-19, ~4.5 days)

```bash
# EXECUTE AFTER TASKS 18-18.1 COMPLETION
/kiro:spec-impl market-data-kafka-producer 15 15.1 15.2 15.3 16
```

**Owner**: Migration/Ops Engineer
**Tasks**: Task 15, 15.1, 15.2, 15.3, 16 (Migration guide + CLI)
**Deliverables**:
- `docs/kafka/migration-guide.md`
- Deprecation warning in `cryptofeed/backends/kafka.py`
- `docs/kafka/config-translation-examples.md`
- `docs/kafka/rollback-procedures.md`
- `tools/migrate-kafka-config.py` (CLI tool)

**Commit After Tasks 15-16**:
```bash
git add docs/kafka/migration-* tools/migrate-kafka-config.py cryptofeed/backends/kafka.py
git commit -m "feat(kafka): Add migration guide and CLI tool for legacy backend (Tasks 15-16)

- Producer migration strategy (legacy → Phase 2)
- Config translation automation (YAML → YAML)
- Config validator with dry-run mode
- Deprecation notice in legacy backend
- Rollback procedures for emergency cutover
- Validated against 10 real-world legacy configs

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Week 3c: Producer Tuning & Troubleshooting (Days 20-21, ~2 days)

```bash
# EXECUTE AFTER TASKS 15-16 COMPLETION
/kiro:spec-impl market-data-kafka-producer 19 19.1
```

**Owner**: Migration/Ops Engineer
**Tasks**: Task 19 + 19.1 (Tuning + troubleshooting)
**Deliverables**:
- `docs/kafka/producer-tuning.md`
- `docs/kafka/troubleshooting.md`

**Commit After Tasks 19-19.1**:
```bash
git add docs/kafka/producer-tuning.md docs/kafka/troubleshooting.md
git commit -m "docs(kafka): Add producer tuning and troubleshooting guides (Tasks 19-19.1)

- Configuration reference (batch.size, linger.ms, buffer.memory, compression)
- Use case profiles (latency-sensitive vs throughput-optimized)
- Performance tuning checklist
- Common issues and diagnostic steps
- Log interpretation guide
- Alert response decision tree

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Week 3 Validation Checkpoint (End of Day 20)

```bash
# EXECUTE AFTER ALL WEEK 3 TASKS COMPLETE
/kiro:validate-impl market-data-kafka-producer 18 18.1 15 15.1 15.2 15.3 16 19 19.1
```

**Decision Gate**: Score ≥8.5/10
- ✅ **PASS**: Proceed to finalization
- ❌ **FAIL**: Iterate on failing tasks (extend Week 3)

---

## PHASE E: FINALIZATION (Day 21)

### ⏳ PENDING WEEK 3 VALIDATION (Score ≥8.5)

### Final Validation & Merge

```bash
# EXECUTE AFTER WEEK 3 VALIDATION PASSES
/kiro:validate-impl market-data-kafka-producer
/kiro:spec-status market-data-kafka-producer
```

**Expected Final Score**: ≥8.5/10
**Expected Final Status**: PRODUCTION READY - PHASE 4 COMPLETE

---

### Merge to Main Branch

```bash
# 1. Verify all changes committed
git status

# 2. Create final merge commit
git checkout next
git merge feature/kafka-producer-phase-4 --no-ff -m "feat(kafka): Complete Phase 4 - Production enhancements

Complete 16 producer-focused enhancement tasks across 3 weeks:

Week 1: Performance Benchmarking (Tasks 10-10.3)
- End-to-end latency benchmarking (p99 <10ms baseline)
- Throughput testing (>100k msg/s baseline)
- Memory profiling (<500MB target)
- CPU usage analysis (<50% target)

Week 2: Monitoring & Reliability (Tasks 17, 17.1-17.3)
- Prometheus metrics integration
- Performance optimization (p99 <5ms achieved)
- Dead letter queue + circuit breaker patterns
- Custom alerting rules + health checks

Week 3: Schema, Migration & Operations (Tasks 18-19.1)
- Schema registry integration (Confluent/Buf support)
- Schema versioning guide (backward/forward compatibility)
- Producer migration guide (legacy → Phase 2)
- Migration CLI tool (10/10 configs validated)
- Producer tuning guide + troubleshooting runbook

Final Validation Score: [INSERT SCORE ≥8.5]

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"

# 3. Push to remote
git push origin next

# 4. Create PR to main
gh pr create --base main --head next \
  --title "feat(kafka): Market Data Kafka Producer - Phase 4 Production Enhancements" \
  --body "Complete 16 producer-focused enhancement tasks: performance benchmarking, Prometheus metrics, DLQ/circuit breaker, schema registry, migration tooling, operational guides. Validation score: [INSERT SCORE]. Ready for production deployment."
```

---

## 📊 Complete Execution Timeline

```
WEEK 1 (Days 1-3):
  Day 1: Execute Tasks 10-10.3 via /kiro:spec-impl
  Day 3: Run /kiro:validate-impl (gate: ≥7.5)

WEEK 2 (Days 4-13):
  Days 4-5:  Execute Task 17 via /kiro:spec-impl
  Days 6-9:  Execute Task 17.1 via /kiro:spec-impl
  Days 10-12: Execute Task 17.2 via /kiro:spec-impl
  Day 13:    Execute Task 17.3 via /kiro:spec-impl
  Day 13:    Run /kiro:validate-impl (gate: ≥8.0)

WEEK 3 (Days 14-20):
  Days 14-16: Execute Tasks 18-18.1 via /kiro:spec-impl
  Days 17-19: Execute Tasks 15-16 via /kiro:spec-impl
  Days 20-21: Execute Tasks 19-19.1 via /kiro:spec-impl
  Day 20:    Run /kiro:validate-impl (gate: ≥8.5)

FINALIZATION (Day 21):
  Run final /kiro:validate-impl
  Merge feature branch to next
  Push to remote
  Create PR to main
```

---

## 🎯 Execution Checkpoints

| Week | Tasks | Validation | Gate | Status |
|------|-------|-----------|------|--------|
| 1 | 10-10.3 | /kiro:validate-impl | ≥7.5 | 🚀 EXECUTING |
| 2 | 17-17.3 | /kiro:validate-impl | ≥8.0 | ⏳ PENDING |
| 3 | 18-19.1 | /kiro:validate-impl | ≥8.5 | ⏳ PENDING |
| Final | All | /kiro:validate-impl | ≥8.5 | ⏳ PENDING |

---

## ✅ Success Criteria

**Week 1**: Baseline metrics meet targets (p99 <10ms, >100k msg/s, <500MB, <50% CPU)
**Week 2**: Monitoring operational, performance optimized (p99 <5ms), reliability patterns tested
**Week 3**: Schema registry working, migration CLI validates 10/10 configs, operational guides complete
**Final**: All 16 tasks complete, validation score ≥8.5/10, ready to merge to main

---

**Current Status**: Week 1 EXECUTING via `/kiro:spec-impl market-data-kafka-producer 10 10.1 10.2 10.3`

Next action: Wait for Week 1 completion (3 days), then run Week 1 validation checkpoint