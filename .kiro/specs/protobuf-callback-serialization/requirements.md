# Requirements Document: Protobuf Callback Serialization

## Project Description (Input)

Protobuf Serialization for Data Feed Callbacks - Foundation layer for streaming lakehouse architecture. Add native Protocol Buffer serialization format as alternative to JSON for backend callbacks (Kafka, Redis, etc.), leveraging existing normalized-data-schema-crypto protos. Implement to_proto() conversion methods on all 20 data types (Trade, Ticker, FundingRate, OrderBook delta, etc.) and extend BackendCallback to support binary protobuf output with JSON as default for backward compatibility. Enable type-safe, space-efficient binary serialization while maintaining full backward compatibility with existing JSON-based systems. Core foundation for subsequent QuixStreams integration and lakehouse backend implementation.

## Specification Overview

This specification establishes the foundation for protobuf-native data serialization in cryptofeed backend callbacks, enabling:
- Type-safe binary serialization using Protocol Buffer format
- Reduced payload size (~50% compared to JSON)
- Full backward compatibility with existing JSON systems
- Foundation for downstream QuixStreams and Lakehouse integration

## Context & Motivation

### Current State
- Backend callbacks currently serialize all data to JSON via `BackendCallback.__call__()` at line 93 of `cryptofeed/backends/backend.py`
- 20 protobuf schemas already exist in `proto/cryptofeed/normalized/v1/` (Trade, Ticker, BookSnapshot, etc.)
- Generated Python bindings ready in `gen/python/cryptofeed/normalized/v1/`
- 13 backend implementations (Kafka, Redis, RabbitMQ, S3, etc.) use custom `value_serializer` pattern

### Why Now
- Protobuf schemas are production-ready but unused
- QuixStreams integration requires efficient binary format for stream processing
- Lakehouse architecture depends on protobuf as unified data format
- Users requested smaller payload sizes and better type safety

## Requirements

### Functional Requirements

**FR1: Protobuf Conversion Methods**
- Implement `to_proto()` methods on all 20 cryptofeed data types
- Map fields from existing Python objects to protobuf message equivalents
- Handle numeric precision (Decimal → string for financial values)
- Preserve all data fidelity in binary format

**FR2: Backend Callback Serialization**
- Extend `BackendCallback` to accept `serialization_format` parameter
- Support JSON (default) and PROTOBUF formats
- Implement `ProtobufSerializer` class mirroring JSON serializer pattern
- Configure via YAML: `serialization_format: protobuf` or `serialization_format: json`

**FR3: Backward Compatibility**
- All backend callbacks default to JSON serialization
- No breaking changes to existing callback APIs
- Both formats available simultaneously for migration path
- Configuration-driven format selection

**FR4: Type Safety**
- Use generated protobuf bindings with full type hints
- Leverage mypy for end-to-end type checking
- Validate message creation before serialization

### Technical Requirements

**TR1: Data Type Coverage**
- Trade, Ticker, FundingRate, OrderBook (Snapshot + Deltas)
- Liquidation, Open Interest, Funding Rate Change, Candle
- Index, Implied Volatility, Greeks, Open Interest Change
- Price Snapshot, Volume Snapshot

**TR2: Integration Points**
- Modify `cryptofeed/backends/backend.py` callback serialization path
- Update `KafkaCallback`, `RedisCallback`, and other backends to use serializer
- No changes required to exchange adapters or data flow

**TR3: Configuration**
- YAML: `serialization_format: protobuf`
- Python API: `KafkaCallback(symbol=..., serialization_format="protobuf")`
- Environment variable: `CRYPTOFEED_CALLBACK_FORMAT=protobuf`

### Non-Functional Requirements

**NFR1: Performance**
- Protobuf serialization < 1ms per message for typical payloads
- Deserialization < 500µs per message
- No memory overhead vs JSON serialization

**NFR2: Data Efficiency**
- Target 40-50% size reduction vs JSON for typical trade messages
- Maintain compression compatibility (gzip/snappy with Kafka)

**NFR3: Maintainability**
- Centralized serialization logic in single module
- Clear separation between format concerns
- Comprehensive test coverage for both formats

## Dependencies & Related Specifications

- **Upstream**: `normalized-data-schema-crypto` (provides .proto schemas) - v0.1.0 released
- **Downstream**: `quixstreams-integration` (uses protobuf serialization for stream processing)
- **Downstream**: `lakehouse-backend-adapter` (uses protobuf for Parquet storage)

## Success Criteria

1. ✅ All 20 data types implement `to_proto()` method
2. ✅ `ProtobufSerializer` class created and tested
3. ✅ `BackendCallback` supports both JSON and Protobuf formats
4. ✅ Configuration examples for all 13 backend types
5. ✅ 100% test coverage for serialization layer
6. ✅ Size reduction benchmarks show 40-50% improvement
7. ✅ Zero breaking changes to existing APIs
8. ✅ Production integration guide documented

## Next Steps

1. **Requirements Approval**: Review and approve this requirements document
2. **Design Phase**: Create design.md specifying implementation architecture
3. **Tasks Generation**: Generate implementation tasks and timeline
4. **Implementation**: Execute tasks using TDD methodology

---

**Phase**: Initialized
**Timeline**: 1-2 weeks (foundation layer)
**Owner**: Development Team
