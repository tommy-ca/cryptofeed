# Kafka & Protobuf Reorganization - Migration Checklist

**Quick Reference for Implementation**

---

## Phase 1.0: Legacy Backend Status Update (0.5 days)

- [ ] Remove `warnings.warn()` from `cryptofeed/backends/kafka.py`
- [ ] Update module docstring to "MAINTAINED" status
- [ ] Remove deprecation notice from docstring
- [ ] Update migration guide to clarify maintenance status
- [ ] Run tests to verify no breakage

---

## Phase 1.1: Protobuf Reorganization (2-3 days)

### Setup
- [ ] Create `cryptofeed/backends/protobuf/` directory
- [ ] Create `__init__.py` with public API exports

### File Moves
- [ ] Move `protobuf_helpers.py` → `protobuf/helpers.py`
- [ ] Extract converter functions → `protobuf/converters.py`
- [ ] Extract serialization logic → `protobuf/serialization.py`
- [ ] Move `proto_bindings/__init__.py` logic → `protobuf/bindings.py`

### Compatibility Shims
- [ ] Create `proto_bindings/__init__.py` shim (with deprecation warning)
- [ ] Create `backends/protobuf_helpers.py` shim (with deprecation warning)
- [ ] Test backward compatibility imports

### Testing
- [ ] Update internal imports
- [ ] Run full test suite
- [ ] Verify all protobuf tests pass
- [ ] Check import compatibility

---

## Phase 1.2: Kafka Reorganization (4-5 days)

### Setup
- [ ] Create `cryptofeed/backends/kafka/` directory
- [ ] Create `__init__.py` placeholder

### Phase 1.2a: Extract TopicManager (1 day)
- [ ] Create `topic_manager.py`
- [ ] Move `TopicManager` class
- [ ] Move related constants (`SUPPORTED_DATA_TYPES`, etc.)
- [ ] Update imports in `kafka_callback.py`
- [ ] Run tests
- [ ] Verify no regressions

### Phase 1.2b: Extract Partitioner (1 day)
- [ ] Create `partitioner.py`
- [ ] Move `Partitioner` base class
- [ ] Move all partitioner implementations:
  - [ ] `CompositePartitioner`
  - [ ] `SymbolPartitioner`
  - [ ] `ExchangePartitioner`
  - [ ] `RoundRobinPartitioner`
- [ ] Move `PartitionerFactory`
- [ ] Update imports in `kafka_callback.py`
- [ ] Run tests
- [ ] Verify no regressions

### Phase 1.2c: Extract HeaderEnricher (1 day)
- [ ] Create `headers.py`
- [ ] Move `MessageHeaders` class
- [ ] Move `OptionalHeaders` class
- [ ] Move `HeaderEnricher` class
- [ ] Update imports in `kafka_callback.py`
- [ ] Run tests
- [ ] Verify no regressions

### Phase 1.2d: Move Remaining Files (1 day)
- [ ] Move `kafka_producer.py` → `kafka/producer.py`
- [ ] Move `kafka_config.py` → `kafka/config.py`
- [ ] Move `kafka_callback.py` → `kafka/callback.py` (refactor)
- [ ] Move `backends/kafka.py` → `kafka/legacy.py`
- [ ] Create `kafka/__init__.py` with re-exports
- [ ] Create root-level compatibility shims:
  - [ ] `cryptofeed/kafka_callback.py` shim
  - [ ] `cryptofeed/kafka_producer.py` shim
  - [ ] `cryptofeed/kafka_config.py` shim
- [ ] Update all internal imports
- [ ] Run full test suite
- [ ] Verify backward compatibility

---

## Phase 1.3: Protobuf Backend Creation (3-4 days)

### Base Class
- [ ] Create `kafka/base.py`
- [ ] Implement `KafkaBackendBase` class
- [ ] Move shared infrastructure:
  - [ ] TopicManager initialization
  - [ ] Partitioner initialization
  - [ ] HeaderEnricher initialization
  - [ ] KafkaProducer initialization
  - [ ] Queue management
  - [ ] Writer loop
- [ ] Test base class in isolation

### Protobuf Backend
- [ ] Create `kafka/protobuf.py`
- [ ] Implement `KafkaProtobufCallback` inheriting from `KafkaBackendBase`
- [ ] Force protobuf serialization (no JSON support)
- [ ] Add protobuf-specific optimizations
- [ ] Integrate schema validation
- [ ] Write unit tests
- [ ] Write integration tests

### Unified Callback Refactor
- [ ] Refactor `kafka/callback.py` to inherit from `KafkaBackendBase`
- [ ] Maintain JSON + Protobuf support
- [ ] Remove code duplication
- [ ] Run tests
- [ ] Verify backward compatibility

### Testing
- [ ] Comprehensive unit tests for protobuf backend
- [ ] Integration tests with real Kafka
- [ ] Performance benchmarks
- [ ] Verify all existing tests pass

---

## Phase 1.4: Metrics & Documentation (2-3 days)

### Metrics
- [ ] Create `kafka/metrics.py`
- [ ] Implement Prometheus metrics export
- [ ] Add metrics:
  - [ ] `kafka_messages_produced_total`
  - [ ] `kafka_message_latency_seconds`
  - [ ] `kafka_queue_depth`
  - [ ] `kafka_errors_total`
  - [ ] `kafka_partition_key_cache_hits_total`
  - [ ] `kafka_partition_key_cache_misses_total`
- [ ] Integrate with health check system
- [ ] Test metrics collection

### Documentation
- [ ] Create `docs/protobuf/schema-versioning.md`
- [ ] Document versioning policy
- [ ] Create migration guide
- [ ] Update API documentation
- [ ] Update user guides
- [ ] Create examples for new structure

---

## Phase 1.5: Test Migration (1-2 days)

### Test Reorganization
- [ ] Reorganize `tests/unit/kafka/` structure:
  - [ ] `test_legacy.py`
  - [ ] `test_protobuf.py`
  - [ ] `test_callback.py`
  - [ ] `test_topic_manager.py`
  - [ ] `test_partitioner.py`
  - [ ] `test_headers.py`
- [ ] Reorganize `tests/unit/protobuf/` structure:
  - [ ] `test_converters.py`
  - [ ] `test_serialization.py`
  - [ ] `test_validation.py`
  - [ ] `test_bindings.py`

### Import Updates
- [ ] Phase 1: Update tests to use new imports (via compatibility shims)
- [ ] Phase 2: Update tests to use direct new imports
- [ ] Phase 3: Verify all tests pass

### Coverage Verification
- [ ] Run full test suite after each phase
- [ ] Verify test coverage maintained (>90%)
- [ ] Check for any test failures
- [ ] Update test fixtures if needed

---

## Final Verification

### Code Quality
- [ ] All linters pass (`ruff`, `mypy`)
- [ ] Type coverage 100%
- [ ] No deprecation warnings in new code
- [ ] All docstrings updated

### Backward Compatibility
- [ ] All old imports work (via shims)
- [ ] Legacy backend functional
- [ ] No breaking changes
- [ ] Migration guide complete

### Documentation
- [ ] All public APIs documented
- [ ] Examples updated
- [ ] Migration guides complete
- [ ] Architecture diagrams updated

### Testing
- [ ] All 628+ tests passing
- [ ] Test coverage >90%
- [ ] Integration tests pass
- [ ] Performance tests pass

---

## Rollback Plan

If issues arise:
1. [ ] Revert to previous commit
2. [ ] Document issues encountered
3. [ ] Update plan with lessons learned
4. [ ] Re-attempt with fixes

---

**Status**: Ready for Implementation  
**Estimated Total Time**: 13-17 days  
**Last Updated**: January 2025
