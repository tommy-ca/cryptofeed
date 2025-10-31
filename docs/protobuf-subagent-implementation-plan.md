# Protobuf Callback Serialization - Subagent Implementation Plan

## Executive Summary

This document provides a comprehensive plan for implementing protobuf-callback-serialization using subagent orchestration. The approach leverages parallel work streams, specialized subagent roles, and clear handoff protocols to maximize efficiency.

**Approach**: Compound Engineering with Subagent Orchestration
**Timeline**: 18-23 days (with 2 parallel subagent streams)
**Total Tasks**: 16 tasks across 4 phases
**Subagents**: 2 primary + 1 verification agent

---

## Subagent Architecture

### Subagent Roles

#### **Subagent Alpha (Core Implementation)**
**Focus**: Foundation and core serialization logic
**Responsibilities**:
- Phase 0: Exception classes
- Phase 1: Serialization abstraction (Serializer ABC, JSONSerializer)
- Phase 2: ProtobufSerializer, core wrapper implementations
- Phase 3: Performance benchmarking, E2E testing

**Characteristics**:
- Strong focus on SOLID principles
- TDD approach (write tests first)
- Deep understanding of C extensions and Python wrappers
- Performance-oriented

#### **Subagent Beta (Configuration & Data Types)**
**Focus**: Configuration, Kafka integration, and data type wrappers
**Responsibilities**:
- Phase 1: Configuration loading, Kafka routing
- Phase 2: Test infrastructure, parallel wrapper implementations
- Phase 3: Profiling, documentation

**Characteristics**:
- Configuration and infrastructure expertise
- DRY principle focus (reusable test fixtures)
- Kafka and backend integration knowledge
- Documentation skills

#### **Subagent Gamma (Verification & Quality)**
**Focus**: Continuous validation and integration
**Responsibilities**:
- Validate each completed task against requirements
- Run integration tests at phase boundaries
- Verify SOLID principle adherence
- Check for breaking changes
- Coordinate handoffs between Alpha and Beta

**Characteristics**:
- Quality assurance mindset
- Requirements traceability focus
- Integration testing expertise
- Non-blocking verification (parallel with next tasks)

---

## Phase-by-Phase Execution Plan

### Phase 0: Prerequisites (Day 1, 0.5 days)

#### Task 1.0: Exception Classes
**Assigned To**: Subagent Alpha
**Duration**: 0.5 day
**Dependencies**: None
**Deliverables**:
- `cryptofeed/exceptions.py`
- `tests/unit/test_exceptions.py`
- 4/4 tests passing

**Execution**:
```bash
# Subagent Alpha
1. Read requirements R4.5 (Exception Handling)
2. Read engineering guidelines (SOLID, KISS)
3. Create exception hierarchy (TDD):
   - Write test_cryptofeed_serialization_exception_base()
   - Write test_serialization_error_with_data_type()
   - Write test_protobuf_encode_error_with_context()
   - Write test_exception_chain_preserved()
4. Implement exceptions to pass tests
5. Verify 100% test coverage
6. Commit: "feat(serialization): add exception classes for protobuf serialization"
```

**Acceptance Criteria Verification** (Subagent Gamma):
- ✅ All exceptions inherit from CryptofeedSerializationException
- ✅ Error messages include context (data type, schema name, version)
- ✅ Exception chains preserved with `from` clause
- ✅ 100% test coverage

**Handoff**: Subagent Alpha → Subagent Beta (1.3.0) and Alpha continues (1.1)

---

### Phase 1: Foundation (Days 2-6, 5-7 days)

#### Parallel Stream 1 (Subagent Alpha): Serialization Abstraction

**Day 2-3: Task 1.1 - Serializer ABC**
```bash
# Subagent Alpha
1. Read requirements R2, design doc section "Serializer Interface"
2. Create TDD tests:
   - test_serializer_cannot_be_instantiated()
   - test_serializer_has_abstract_methods()
   - test_incomplete_implementation_fails()
   - test_complete_implementation_succeeds()
3. Implement Serializer ABC
4. Verify mypy strict mode passes
5. Commit: "feat(serialization): add Serializer abstract base class"
```

**Day 4-5: Task 1.2 - JSONSerializer**
```bash
# Subagent Alpha
1. Read requirements R3 (Backward Compatibility)
2. Create TDD tests:
   - test_json_serializer_trade_basic()
   - test_json_serializer_decimal_precision()
   - test_json_serializer_content_type()
   - test_json_serializer_error_handling()
3. Implement JSONSerializer
4. Verify output identical to pre-refactor to_dict()
5. Commit: "feat(serialization): add JSONSerializer for backward compatibility"
```

#### Parallel Stream 2 (Subagent Beta): Configuration & Kafka

**Day 2: Task 1.3.0 - Configuration Loading**
```bash
# Subagent Beta (parallel with Alpha 1.1)
1. Read requirements R3.7-3.10 (Config validation)
2. Create TDD tests:
   - test_config_valid_formats()
   - test_config_case_insensitive()
   - test_config_invalid_format()
   - test_env_var_precedence()
3. Implement validate_serialization_format()
4. Implement get_serialization_format_from_env()
5. Commit: "feat(config): add serialization format validation and loading"
```

**Day 3: Task 1.3.1 - Kafka Topic Routing**
```bash
# Subagent Beta (parallel with Alpha 1.2)
1. Read requirements R4.10-4.14 (Kafka routing)
2. Read current Kafka backend implementation
3. Create TDD tests:
   - test_kafka_protobuf_topic_routing()
   - test_kafka_json_topic_routing()
   - test_kafka_partition_key_protobuf()
   - test_kafka_partition_key_json()
4. Implement topic() and partition_key() methods
5. Verify backward compatibility for JSON format
6. Commit: "feat(kafka): add protobuf topic routing and partitioning"
```

#### Convergence: Task 1.3.3 - BackendCallback Integration

**Day 6: Task 1.3.3 - BackendCallback Integration**
```bash
# Subagent Alpha (requires 1.0, 1.1, 1.2, 1.3.0)
1. Read requirements R2, R3
2. Read design doc "BackendCallback Modifications"
3. Create TDD tests:
   - test_callback_json_serialization()
   - test_callback_default_format()
   - test_callback_invalid_format()
   - test_callback_error_handling()
4. Modify BackendCallback to accept serialization_format
5. Implement _get_serializer() factory method
6. Integrate with existing write() flow
7. Verify backward compatibility (JSON default)
8. Commit: "feat(backend): integrate serialization format selection"
```

**Phase 1 Verification** (Subagent Gamma):
- ✅ All foundation tests passing
- ✅ Backward compatibility verified
- ✅ Configuration loading works (YAML + env vars)
- ✅ Kafka routing implemented
- ✅ No breaking changes

**Phase 1 Deliverables**:
- 3 new modules (exceptions, serializers/base, serializers/json)
- 1 modified module (backends/backend)
- 1 modified module (backends/kafka)
- ~500 LOC, 16 tests passing

---

### Phase 2: Protobuf Integration (Days 7-16, 10-14 days)

#### Task 1.4: Generate Protobuf Bindings (Day 7)

**Assigned To**: Subagent Beta
**Duration**: 1 day
```bash
# Subagent Beta
1. Read requirements R5 (Schema Alignment)
2. Verify normalized-data-schema-crypto v0.1.0 available
3. Run: buf generate proto/
4. Create import wrapper: cryptofeed/proto_bindings/__init__.py
5. Create TDD tests:
   - test_protobuf_bindings_importable()
   - test_protobuf_message_instantiation()
   - test_protobuf_serialization()
6. Verify all 20 proto schemas generated
7. Commit: "feat(protobuf): generate Python bindings from normalized schemas"
```

#### Parallel Stream: Serializer + Test Infrastructure (Days 8-10)

**Task 1.5: ProtobufSerializer (Days 8-10)**
```bash
# Subagent Alpha
1. Read requirements R6 (Type Safety), R4.5 (Exception Handling)
2. Read design doc "ProtobufSerializer"
3. Create TDD tests:
   - test_protobuf_serializer_basic()
   - test_protobuf_serializer_missing_method()
   - test_protobuf_serializer_content_type()
   - test_protobuf_serializer_invalid_return()
4. Implement ProtobufSerializer class
5. Integrate with exception classes from Task 1.0
6. Verify type safety with mypy strict mode
7. Commit: "feat(serialization): add ProtobufSerializer implementation"
```

**Task 1.5.1: Test Infrastructure (Day 10)**
```bash
# Subagent Beta (parallel with Alpha 1.5 days 8-9)
1. Read DRY principle requirements
2. Create reusable test fixtures:
   - sample_trade(), sample_orderbook(), sample_ticker()
   - ... for all 14 data types
3. Create test helpers:
   - protobuf_round_trip_helper()
   - decimal_comparison()
   - timestamp_comparison()
4. Document fixture usage patterns
5. Commit: "test(fixtures): add reusable test infrastructure for wrappers"
```

#### Wrapper Implementation (Days 11-16)

**Task 1.6: Trade + OrderBook Wrappers (Days 11-14)**
```bash
# Subagent Alpha
1. Read requirements R1, R7.5 (Wrapper Adapter)
2. Read C extension implementation (cryptofeed/types.so)
3. Create TDD tests (using fixtures from 1.5.1):
   - test_trade_wrapper_to_proto()
   - test_trade_decimal_precision()
   - test_trade_timestamp_conversion()
   - test_trade_roundtrip()
   - ... (same pattern for OrderBook)
4. Implement TradeWrapper class
5. Implement OrderBookWrapper class
6. Verify round-trip serialization
7. Commit: "feat(wrappers): add Trade and OrderBook protobuf wrappers"
```

**Task 1.6.1: Wrapper Adapter (Day 15)**
```bash
# Subagent Alpha
1. Read requirements R7.5 (C Extension Wrapper Adapter)
2. Create TDD tests:
   - test_adapter_trade_wrapper()
   - test_adapter_unsupported_type()
   - test_adapter_stateless()
   - test_adapter_performance_overhead()
3. Implement wrap_for_serialization() function
4. Create type registry
5. Verify <100µs overhead per message
6. Commit: "feat(adapter): add C extension wrapper adapter layer"
```

#### Parallel Data Type Implementation (Days 16-20)

**Task 1.7: Ticker + Candle + Funding (Days 16-18)**
```bash
# Subagent Beta (parallel with Alpha 1.8)
1. Follow pattern from Task 1.6
2. Use test fixtures from Task 1.5.1
3. Implement:
   - TickerWrapper + tests
   - CandleWrapper + tests
   - FundingWrapper + tests
4. Register in adapter
5. Commit: "feat(wrappers): add Ticker, Candle, Funding protobuf wrappers"
```

**Task 1.8: 9 Remaining Types (Days 16-20)**
```bash
# Subagent Alpha (parallel with Beta 1.7)
1. Follow pattern from Task 1.6
2. Use test fixtures from Task 1.5.1
3. Implement (in order of priority):
   - LiquidationWrapper + tests
   - OpenInterestWrapper + tests
   - IndexWrapper + tests
   - BalanceWrapper + tests
   - PositionWrapper + tests
   - FillWrapper + tests
   - OrderInfoWrapper + tests
   - TransactionWrapper + tests
   - OrderWrapper + tests
4. Register all in adapter
5. Commit: "feat(wrappers): add remaining 9 data type protobuf wrappers"
```

**Phase 2 Verification** (Subagent Gamma):
- ✅ All 14 wrappers implemented
- ✅ All round-trip tests passing
- ✅ Adapter overhead <100µs per message
- ✅ 100% test coverage for wrapper classes
- ✅ Type safety verified (mypy strict)

**Phase 2 Deliverables**:
- 1 serializer module (serializers/protobuf)
- 14 wrapper modules (proto_adapters/*.py)
- 1 adapter module (proto_adapters/adapter.py)
- 1 test infrastructure module (tests/fixtures/protobuf_fixtures.py)
- ~2,000 LOC, 80+ tests passing

---

### Phase 3: Production Readiness (Days 21-27, 5-7 days)

#### Task 1.9: Performance Benchmarking (Days 21-23)

**Assigned To**: Subagent Alpha
**Duration**: 2-3 days
```bash
# Subagent Alpha
1. Read requirements R8 (Performance Characteristics)
2. Create baseline datasets:
   - TRADE_WORKLOAD (10,000 Trade messages)
   - ORDERBOOK_WORKLOAD (1,000 OrderBook messages)
   - MIXED_WORKLOAD (7,000 Trade + 2,000 OrderBook + 1,000 Ticker)
3. Implement benchmark tests:
   - test_latency_percentiles() (p50, p95, p99)
   - test_throughput_trade_workload()
   - test_throughput_mixed_workload()
   - test_memory_stability()
   - test_size_reduction()
4. Run benchmarks and collect metrics
5. Verify targets met:
   - p99 <1ms for Trade
   - p99 <2ms for OrderBook
   - Throughput ≥10k msg/s
   - Size reduction 50-60%
6. Commit: "test(performance): add baseline benchmarking suite"
```

#### Parallel: Profiling + Documentation (Days 24-25)

**Task 1.9.1: Profiling & Optimization (Day 24)**
```bash
# Subagent Beta (parallel with Alpha 1.10 setup)
1. Create profiling script (tools/profile_serialization.py)
2. Run cProfile on serialization hot paths
3. Identify functions consuming >5% cumtime
4. Document findings in docs/performance-baseline.md
5. Optimize if Decimal/timestamp conversion >10%
6. Commit: "perf(profiling): add performance profiling and optimization"
```

**Task 1.11: User Documentation (Days 24-26)**
```bash
# Subagent Beta (parallel with Alpha 1.10)
1. Read requirements R9 (Documentation)
2. Create docs/protobuf-serialization-user-guide.md:
   - Introduction and benefits
   - Configuration (YAML + env vars + API)
   - Kafka topic naming conventions
   - Migration guide (JSON → Protobuf)
   - Troubleshooting guide
3. Create docs/consumer-integration-guide.md:
   - Python consumer example
   - Flink consumer reference (Java)
   - DuckDB consumer reference (SQL)
4. Create examples/kafka_protobuf_producer.py
5. Create examples/kafka_protobuf_consumer.py
6. Commit: "docs(protobuf): add user guide and consumer integration examples"
```

#### Task 1.10: Kafka E2E Integration (Days 24-27)

**Assigned To**: Subagent Alpha
**Duration**: 3-4 days
```bash
# Subagent Alpha
1. Read requirements R4 (Kafka Routing), R7 (Testing Coverage)
2. Create docker-compose.test.yml (Kafka + Zookeeper)
3. Create integration tests:
   - test_kafka_trade_roundtrip()
   - test_kafka_no_message_loss()
   - test_kafka_multi_exchange()
   - test_kafka_partition_routing()
   - test_kafka_protobuf_vs_json_coexistence()
4. Start Kafka cluster: docker-compose up -d
5. Run E2E tests with real Kafka
6. Verify all 14 data types work E2E
7. Commit: "test(e2e): add Kafka integration tests for protobuf serialization"
```

**Phase 3 Verification** (Subagent Gamma):
- ✅ Performance targets met
- ✅ E2E tests passing with real Kafka
- ✅ Documentation complete
- ✅ No breaking changes
- ✅ Backward compatibility verified

**Phase 3 Deliverables**:
- Performance baseline docs
- User guide + consumer integration guide
- 2 code examples
- E2E test suite
- docker-compose for testing
- ~800 LOC, 15 tests passing

---

## Subagent Coordination Protocol

### Daily Standup (Asynchronous)

**Each Subagent Reports**:
1. Yesterday: Tasks completed, blockers encountered
2. Today: Tasks in progress, expected completion
3. Blockers: Dependencies needed, questions for other subagents

**Coordination via**:
- Git commits with conventional format
- Task status updates in shared document
- Clear handoff messages when task completes

### Handoff Protocol

**When Task Completes**:
1. Subagent commits with conventional message
2. Subagent runs local tests (pytest -v)
3. Subagent updates task status: "COMPLETED"
4. Subagent notifies downstream dependencies
5. Subagent Gamma validates (non-blocking)

**Handoff Message Template**:
```
Task X.Y COMPLETED by Subagent {Alpha|Beta}

Deliverables:
- File 1 created: path/to/file1.py
- File 2 modified: path/to/file2.py
- Tests: X/X passing
- Coverage: XX%

Downstream Dependencies Unblocked:
- Task X.Z (Subagent {Alpha|Beta})
- Task X.W (Subagent {Alpha|Beta})

Verification Status: PENDING (Subagent Gamma)
```

### Conflict Resolution

**If Merge Conflict**:
1. Subagent detects conflict during git pull
2. Subagent analyzes conflict context
3. Subagent resolves based on:
   - Timestamps (later wins if same scope)
   - SOLID principles (better design wins)
   - Requirements (requirement-aligned wins)
4. Subagent commits resolution with explanation
5. Subagent Gamma validates resolution

**If Requirement Ambiguity**:
1. Subagent flags ambiguity with QUESTION tag
2. Main agent or domain expert clarifies
3. Subagent proceeds with clarification
4. Document clarification for future reference

---

## Quality Gates

### Gate 1: Phase 1 Complete (Day 6)

**Verification Checklist** (Subagent Gamma):
- [ ] All foundation tests passing (16 tests)
- [ ] Backward compatibility verified (existing JSON works)
- [ ] Configuration loading tested (YAML + env vars)
- [ ] Kafka routing tested (topic + partition key)
- [ ] No mypy errors (strict mode)
- [ ] No breaking changes in BackendCallback
- [ ] Conventional commits followed

**Exit Criteria**: All checks pass → Proceed to Phase 2

### Gate 2: Phase 2 Complete (Day 20)

**Verification Checklist** (Subagent Gamma):
- [ ] All 14 wrappers implemented
- [ ] All wrapper tests passing (80+ tests)
- [ ] Round-trip tests passing for all types
- [ ] Adapter overhead <100µs per message
- [ ] Test fixtures reusable and documented
- [ ] 100% test coverage for wrappers
- [ ] No mypy errors (strict mode)

**Exit Criteria**: All checks pass → Proceed to Phase 3

### Gate 3: Phase 3 Complete (Day 27)

**Verification Checklist** (Subagent Gamma):
- [ ] Performance benchmarks meet targets
- [ ] E2E tests passing with real Kafka
- [ ] Documentation complete and accurate
- [ ] Examples run successfully
- [ ] No regressions in existing functionality
- [ ] Zero breaking changes
- [ ] Ready for production deployment

**Exit Criteria**: All checks pass → Specification COMPLETE

---

## Execution Timeline

### Parallel Execution (2 Subagents)

```
Day 1:   [Alpha: 1.0] ─────────────────────────────────┐
                                                        ▼
Day 2:   [Alpha: 1.1──────] [Beta: 1.3.0─────] ────▶ Handoff
Day 3:   [Alpha: 1.1──────] [Beta: 1.3.1─────]
Day 4:   [Alpha: 1.2──────] [Beta: 1.3.1─────]
Day 5:   [Alpha: 1.2──────]
Day 6:   [Alpha: 1.3.3─────────────────────] ────▶ Gate 1
                                                        ▼
Day 7:   [Beta: 1.4────────────────]
Day 8:   [Alpha: 1.5──────] [Beta: 1.5.1────]
Day 9:   [Alpha: 1.5──────]
Day 10:  [Alpha: 1.5──────] [Beta: 1.5.1────]
Day 11:  [Alpha: 1.6──────────────]
Day 12:  [Alpha: 1.6──────────────]
Day 13:  [Alpha: 1.6──────────────]
Day 14:  [Alpha: 1.6──────────────]
Day 15:  [Alpha: 1.6.1────]
Day 16:  [Alpha: 1.8──────────────] [Beta: 1.7────────]
Day 17:  [Alpha: 1.8──────────────] [Beta: 1.7────────]
Day 18:  [Alpha: 1.8──────────────] [Beta: 1.7────────]
Day 19:  [Alpha: 1.8──────────────]
Day 20:  [Alpha: 1.8──────────────] ────▶ Gate 2
                                                        ▼
Day 21:  [Alpha: 1.9──────────────]
Day 22:  [Alpha: 1.9──────────────]
Day 23:  [Alpha: 1.9──────────────]
Day 24:  [Alpha: 1.10─────────────] [Beta: 1.9.1──+ 1.11─────]
Day 25:  [Alpha: 1.10─────────────] [Beta: 1.11────────────]
Day 26:  [Alpha: 1.10─────────────] [Beta: 1.11────────────]
Day 27:  [Alpha: 1.10─────────────] ────▶ Gate 3
                                                        ▼
                                                   COMPLETE
```

**Duration**: 27 days (20-28 day estimate)
**Critical Path**: Alpha tasks (longer path)
**Parallelization**: Beta tasks fill gaps in Alpha schedule

---

## Risk Mitigation

### Risk: C Extension Complexity

**Mitigation**:
- Task 1.6 explicitly explores C extension structure
- Wrapper pattern isolates C extension complexity
- Test fixtures provide known-good data
- Round-trip tests verify correctness

**Contingency**:
- If C extension modification needed, escalate to main agent
- Fallback: Pure Python wrapper with __getattr__ delegation

### Risk: Performance Targets Not Met

**Mitigation**:
- Task 1.9.1 profiles hot paths early
- Optimization opportunities documented
- Decimal/timestamp conversion optimization targeted

**Contingency**:
- If p99 >1ms, analyze profiling results
- Optimize Decimal-to-string conversion
- Consider Cython for hot paths

### Risk: Merge Conflicts

**Mitigation**:
- Clear file ownership per subagent
- Alpha owns serializers/, Beta owns fixtures/
- Frequent commits with pull before push
- Conflict resolution protocol defined

**Contingency**:
- Subagent Gamma mediates conflicts
- SOLID principles guide resolution
- Requirements take precedence

### Risk: Task Dependency Blocking

**Mitigation**:
- Parallel tasks minimize blocking
- Critical path optimized (Alpha)
- Beta fills gaps in Alpha schedule

**Contingency**:
- If Alpha blocked, Beta takes additional tasks
- If Beta blocked, Alpha continues with Beta tasks
- Flexible task reassignment

---

## Success Metrics

### Quantitative Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Timeline** | 18-23 days | Actual completion date |
| **Test Coverage** | ≥95% | pytest --cov |
| **Performance (Trade p99)** | <1ms | Benchmark suite |
| **Performance (OrderBook p99)** | <2ms | Benchmark suite |
| **Throughput** | ≥10k msg/s | Benchmark suite |
| **Size Reduction** | 50-60% | Benchmark suite |
| **Memory Stability** | <5% growth | Memory profiler |
| **Adapter Overhead** | <100µs | Performance tests |

### Qualitative Metrics

| Metric | Target | Verification |
|--------|--------|--------------|
| **SOLID Adherence** | 100% | Code review (Gamma) |
| **TDD Compliance** | 100% | Tests written first |
| **Backward Compatibility** | 100% | Regression tests |
| **Documentation Quality** | Complete | User guide reviewed |
| **Engineering Principles** | 94/100+ | Scorecard (Gamma) |

---

## Deliverables Summary

### Code Artifacts

1. **Core Serialization** (Phase 1):
   - `cryptofeed/exceptions.py` (100 LOC)
   - `cryptofeed/serializers/base.py` (50 LOC)
   - `cryptofeed/serializers/json.py` (80 LOC)
   - `cryptofeed/serializers/protobuf.py` (100 LOC) [Phase 2]

2. **Backend Integration** (Phase 1):
   - `cryptofeed/backends/backend.py` (modified, +150 LOC)
   - `cryptofeed/backends/kafka.py` (modified, +100 LOC)
   - `cryptofeed/config.py` (new or modified, +80 LOC)

3. **Protobuf Wrappers** (Phase 2):
   - `cryptofeed/proto_bindings/__init__.py` (50 LOC)
   - `cryptofeed/proto_adapters/adapter.py` (150 LOC)
   - `cryptofeed/proto_adapters/*.py` (14 files, ~1,400 LOC)

4. **Test Infrastructure** (Phase 2):
   - `tests/fixtures/protobuf_fixtures.py` (400 LOC)
   - `tests/helpers/*.py` (200 LOC)

5. **Tests** (All Phases):
   - Unit tests: ~2,000 LOC (100+ tests)
   - Integration tests: ~800 LOC (15 tests)
   - Benchmark tests: ~600 LOC

6. **Documentation** (Phase 3):
   - `docs/protobuf-serialization-user-guide.md` (500 lines)
   - `docs/consumer-integration-guide.md` (400 lines)
   - `docs/performance-baseline.md` (200 lines)

7. **Examples** (Phase 3):
   - `examples/kafka_protobuf_producer.py` (150 LOC)
   - `examples/kafka_protobuf_consumer.py` (150 LOC)

**Total**: ~7,500 LOC (production + tests + docs + examples)

---

## Execution Command

### Start Implementation

```bash
# Initialize implementation
/kiro:spec-impl protobuf-callback-serialization --mode=subagent --parallel=2

# Subagent Alpha starts with Task 1.0
# Subagent Beta waits for handoff at Task 1.3.0
# Subagent Gamma monitors and validates
```

### Monitor Progress

```bash
# Check task status
cat .kiro/specs/protobuf-callback-serialization/status.md

# View subagent activity
git log --oneline --author="Subagent Alpha" --since="1 week ago"
git log --oneline --author="Subagent Beta" --since="1 week ago"

# Run verification
pytest -v --cov=cryptofeed
```

### Quality Gates

```bash
# Phase 1 Gate
pytest tests/unit/serializers/ tests/unit/test_exceptions.py -v
mypy cryptofeed/serializers cryptofeed/exceptions.py --strict

# Phase 2 Gate
pytest tests/unit/proto_adapters/ tests/fixtures/ -v
pytest -k "round_trip" -v

# Phase 3 Gate
pytest tests/benchmarks/ tests/integration/ -v
docker-compose -f docker-compose.test.yml up -d
pytest tests/integration/test_kafka_serialization_e2e.py -v
```

---

## Conclusion

This subagent implementation plan leverages:
- ✅ **Compound Engineering**: 2 parallel work streams
- ✅ **Clear Handoffs**: Explicit coordination protocol
- ✅ **Quality Gates**: Verification at phase boundaries
- ✅ **Risk Mitigation**: Contingencies for common issues
- ✅ **SOLID Principles**: Maintained throughout
- ✅ **TDD Approach**: Tests written first, always
- ✅ **Engineering Excellence**: 94/100+ score target

**Timeline**: 18-23 days (35-40% faster than sequential)
**Quality**: Production-ready, zero breaking changes
**Documentation**: Complete user and integration guides

**Ready to Execute**: ✅ All subagent roles defined, tasks assigned, coordination protocol established.
