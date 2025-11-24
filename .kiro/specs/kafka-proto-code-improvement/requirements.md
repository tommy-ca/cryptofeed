# Requirements: Kafka & Protobuf Code Improvement

**Spec Name**: `kafka-proto-code-improvement`  
**Version**: 0.1.0  
**Created**: January 15, 2025  
**Status**: Requirements Phase

---

## Overview

Reorganize Kafka and Protobuf-related code to improve maintainability, enforce separation of concerns, and colocate related files following engineering principles (SOLID, KISS, DRY, NO LEGACY).

**Scope**: Code organization and structure improvements for Kafka backend and Protobuf serialization modules.

**Strategy**: 
- Preserve legacy backend (maintain, don't deprecate)
- Create isolated protobuf backend
- Colocate related files in module directories
- Maintain 100% backward compatibility via compatibility shims

---

## Goals

1. **File Organization**: Colocate Kafka files in `backends/kafka/` and Protobuf files in `backends/protobuf/`
2. **Separation of Concerns**: Isolate protobuf backend from unified callback
3. **Backward Compatibility**: Maintain all existing imports via compatibility shims
4. **Code Reuse**: Share infrastructure via base classes (DRY principle)
5. **Maintainability**: Clear module boundaries, easier navigation

---

## Scope Boundaries

### IN-SCOPE

#### Kafka Module Reorganization
- Move Kafka-related files to `cryptofeed/backends/kafka/`
- Extract components from unified callback (TopicManager, Partitioner, HeaderEnricher)
- Create `KafkaBackendBase` for shared infrastructure
- Preserve legacy backend (`legacy.py`)
- Create isolated protobuf backend (`protobuf.py`)
- Refactor unified callback to use base class
- Create compatibility shims for root-level imports

#### Protobuf Module Reorganization
- Move Protobuf-related files to `cryptofeed/backends/protobuf/`
- Extract converters from `protobuf_helpers.py`
- Extract serialization logic
- Move bindings import logic
- Create compatibility shims for `proto_bindings` and `protobuf_helpers`

#### Legacy Backend Status
- Remove deprecation warning from `backends/kafka.py`
- Update documentation to "MAINTAINED" status
- Preserve full functionality

#### Testing
- Update test imports gradually
- Reorganize test files to match new structure
- Maintain test coverage >90%

### OUT-OF-SCOPE

- New features or functionality
- Performance optimizations (separate effort)
- Schema changes
- Breaking changes to public APIs
- Consumer-side changes

---

## Functional Requirements

### FR1: Legacy Backend Status Update

**Objective**: As a developer, I want the legacy backend to be clearly marked as maintained (not deprecated), so there's no confusion about its status.

#### Acceptance Criteria

1. **WHEN** a developer imports from `cryptofeed.backends.kafka` **THEN** no deprecation warning SHALL be shown
2. **WHEN** a developer reads the module docstring **THEN** it SHALL clearly state "MAINTAINED for backward compatibility"
3. **WHEN** migration documentation is consulted **THEN** it SHALL clarify that legacy backend is preserved indefinitely

---

### FR2: Protobuf Module Reorganization

**Objective**: As a developer, I want all protobuf-related code colocated in `backends/protobuf/`, so I can easily find and maintain protobuf functionality.

#### Acceptance Criteria

1. **WHEN** protobuf code is reorganized **THEN** all files SHALL be in `cryptofeed/backends/protobuf/`
2. **WHEN** converters are extracted **THEN** they SHALL be in `protobuf/converters.py`
3. **WHEN** serialization logic is extracted **THEN** it SHALL be in `protobuf/serialization.py`
4. **WHEN** bindings import logic is moved **THEN** it SHALL be in `protobuf/bindings.py`
5. **WHEN** old imports are used **THEN** they SHALL work via compatibility shims
6. **WHEN** compatibility shims are used **THEN** deprecation warnings SHALL guide users to new imports

---

### FR3: Kafka Module Reorganization

**Objective**: As a developer, I want all Kafka-related code colocated in `backends/kafka/`, so I can easily navigate and maintain Kafka functionality.

#### Acceptance Criteria

1. **WHEN** Kafka code is reorganized **THEN** all files SHALL be in `cryptofeed/backends/kafka/`
2. **WHEN** TopicManager is extracted **THEN** it SHALL be in `kafka/topic_manager.py`
3. **WHEN** Partitioner is extracted **THEN** it SHALL be in `kafka/partitioner.py`
4. **WHEN** HeaderEnricher is extracted **THEN** it SHALL be in `kafka/headers.py`
5. **WHEN** legacy backend is moved **THEN** it SHALL be in `kafka/legacy.py`
6. **WHEN** old imports are used **THEN** they SHALL work via compatibility shims

---

### FR4: Isolated Protobuf Backend

**Objective**: As a developer, I want a dedicated protobuf-only Kafka backend, so I can optimize for protobuf-specific features without JSON complexity.

#### Acceptance Criteria

1. **WHEN** `KafkaProtobufCallback` is created **THEN** it SHALL inherit from `KafkaBackendBase`
2. **WHEN** `KafkaProtobufCallback` is instantiated **THEN** protobuf serialization SHALL be forced (no JSON support)
3. **WHEN** `KafkaProtobufCallback` processes messages **THEN** it SHALL use protobuf-specific optimizations
4. **WHEN** schema validation is needed **THEN** it SHALL be integrated with protobuf backend
5. **WHEN** unified callback is refactored **THEN** it SHALL also inherit from `KafkaBackendBase`

---

### FR5: Shared Infrastructure Base Class

**Objective**: As a developer, I want shared Kafka infrastructure in a base class, so protobuf and unified callbacks can reuse code without duplication.

#### Acceptance Criteria

1. **WHEN** `KafkaBackendBase` is created **THEN** it SHALL contain:
   - TopicManager initialization
   - Partitioner initialization
   - HeaderEnricher initialization
   - KafkaProducer initialization
   - Queue management
   - Writer loop
2. **WHEN** `KafkaProtobufCallback` inherits from base **THEN** it SHALL reuse all shared infrastructure
3. **WHEN** `KafkaCallback` inherits from base **THEN** it SHALL reuse all shared infrastructure
4. **WHEN** code is shared **THEN** duplication SHALL be eliminated (DRY principle)

---

### FR6: Backward Compatibility

**Objective**: As a user, I want all existing imports to continue working, so my code doesn't break during the reorganization.

#### Acceptance Criteria

1. **WHEN** old imports are used **THEN** they SHALL work via compatibility shims:
   - `from cryptofeed.kafka_callback import KafkaCallback`
   - `from cryptofeed.kafka_producer import KafkaProducer`
   - `from cryptofeed.kafka_config import KafkaConfig`
   - `from cryptofeed.proto_bindings import trade_pb2`
   - `from cryptofeed.backends.protobuf_helpers import serialize_to_protobuf`
   - `from cryptofeed.backends.kafka import TradeKafka`
2. **WHEN** compatibility shims are used **THEN** deprecation warnings SHALL guide users to new imports
3. **WHEN** new imports are used **THEN** no deprecation warnings SHALL appear
4. **WHEN** legacy backend is used **THEN** it SHALL function identically to before

---

## Non-Functional Requirements

### NFR1: Code Quality

- **Test Coverage**: Maintain >90% coverage during reorganization
- **Type Coverage**: 100% type hints for new code
- **Linting**: All code passes `ruff` and `mypy` checks
- **Documentation**: All public APIs documented

### NFR2: Performance

- **Latency**: No performance regression (maintain P99 <5ms)
- **Throughput**: No throughput regression (maintain ≥100k msg/s)
- **Memory**: No memory leaks introduced

### NFR3: Maintainability

- **Module Boundaries**: Clear separation of concerns
- **File Organization**: Related files colocated
- **Code Reuse**: Shared infrastructure via base classes
- **Documentation**: Clear migration guides

---

## Dependencies

### Required
- `market-data-kafka-producer`: Existing Kafka implementation
- `protobuf-callback-serialization`: Existing protobuf serialization

### External
- None

---

## Success Criteria

1. ✅ All Kafka files colocated in `backends/kafka/`
2. ✅ All Protobuf files colocated in `backends/protobuf/`
3. ✅ Isolated protobuf backend created and functional
4. ✅ Legacy backend preserved and functional
5. ✅ All existing imports work via compatibility shims
6. ✅ Test coverage maintained >90%
7. ✅ No performance regression
8. ✅ All tests passing (628+ tests)
9. ✅ Documentation updated with new structure
10. ✅ Migration guides created

---

## Out of Scope

- New features or functionality
- Performance optimizations (separate effort)
- Schema evolution
- Breaking API changes
- Consumer-side changes
- Storage layer changes

---

## Engineering Principles Applied

- **Single Responsibility**: Each module has one clear purpose
- **Open/Closed**: New protobuf backend extends without modifying legacy
- **Liskov Substitution**: All backends implement `BackendCallback` interface
- **Interface Segregation**: Separate interfaces for JSON vs Protobuf
- **Dependency Inversion**: Depend on abstractions (`BackendCallback`)
- **DRY**: Shared infrastructure via `KafkaBackendBase`
- **KISS**: Clear module boundaries, no unnecessary complexity
- **NO LEGACY**: Legacy preserved but isolated, new code follows modern patterns
- **Colocation**: Related files grouped together

---

**Status**: Requirements Draft  
**Next Step**: Generate design document with `kiro:spec-design`
