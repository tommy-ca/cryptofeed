# Protobuf Serialization Implementation - Next Steps

## Summary

This document outlines the next steps for implementing protobuf output format support in cryptofeed, enabling dual-format serialization (JSON + Protobuf) for backend callbacks.

**Status**: Tasks generated and ready for implementation
**Spec**: `.kiro/specs/protobuf-callback-serialization/`
**Effort**: 2-3 weeks (24-33 days)
**Blocking**: None - normalized-data-schema-crypto v0.1.0 merged

---

## Implementation Approach

### Critical Discovery: C Extension Data Types

**Finding**: Cryptofeed data types (`Trade`, `OrderBook`, `Ticker`, etc.) are implemented as C extensions (`cryptofeed.types.cpython-312-x86_64-linux-gnu.so`), not pure Python classes.

**Solution**: Create Python wrapper classes in `cryptofeed/proto_adapters/` that:
- Wrap C extension objects
- Provide `to_proto()` methods
- Delegate field access to underlying C objects
- Convert Decimal → string and float timestamp → int64 microseconds

**Benefits**:
- ✅ No C extension source modification required
- ✅ Backward compatible (existing code unchanged)
- ✅ Type-safe with protobuf bindings
- ✅ Testable with pure Python unit tests

---

## Task Breakdown (10 Tasks)

### **Phase 1: Foundation** (4-5 days)
1. **Task 1.1**: Serializer Abstract Base Class (1-2 days)
2. **Task 1.2**: JSONSerializer for backward compatibility (1-2 days)
3. **Task 1.3**: BackendCallback integration with format selection (2-3 days)

### **Phase 2: Protobuf Integration** (12-16 days, can parallelize)
4. **Task 1.4**: Generate Python protobuf bindings from `.proto` files (1 day)
5. **Task 1.5**: ProtobufSerializer implementation (2-3 days)
6. **Task 1.6**: Trade + OrderBook wrappers with `to_proto()` (3-4 days)
7. **Task 1.7**: Ticker + Candle + Funding wrappers (2-3 days, parallel with 1.8)
8. **Task 1.8**: 9 remaining type wrappers (Balance, Position, Fill, etc.) (4-5 days, parallel with 1.7)

### **Phase 3: Production Readiness** (6-8 days)
9. **Task 1.9**: Performance benchmarking and optimization (2-3 days)
10. **Task 1.10**: Kafka integration E2E testing (3-4 days)

**Total**: 24-33 days estimated, 2-3 weeks with parallelization

---

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│ Cryptofeed Core                             │
│                                             │
│  ┌──────────────┐    ┌──────────────┐     │
│  │ Exchange     │───▶│ C Extension  │     │
│  │ Adapters     │    │ Data Types   │     │
│  └──────────────┘    └──────┬───────┘     │
│                              │             │
└──────────────────────────────┼─────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │ Python Wrappers  │
                    │ (proto_adapters) │
                    └────────┬─────────┘
                             │
                  ┌──────────┴──────────┐
                  │                     │
                  ▼                     ▼
         ┌────────────────┐   ┌────────────────┐
         │ JSONSerializer │   │ ProtobufSerializer│
         │ (to_dict())    │   │ (to_proto())     │
         └────────┬───────┘   └────────┬─────────┘
                  │                     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌──────────────────┐
                  │ BackendCallback  │
                  │ (format selector)│
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Kafka/Redis/etc. │
                  │ (binary output)  │
                  └──────────────────┘
```

---

## Key Design Decisions

### 1. Wrapper Pattern for C Extensions

**Decision**: Use Python wrapper classes instead of modifying C extension source.

**Rationale**:
- Cryptofeed types are C extensions (`.so` files), not pure Python
- Modifying C source requires Cython expertise and rebuild pipeline
- Wrappers provide clean separation and easier testing
- Backward compatible with existing code

**Implementation**:
```python
# cryptofeed/proto_adapters/trade.py
from cryptofeed.types import Trade as CExtTrade
from cryptofeed.proto_bindings import trade_pb2

class TradeWrapper:
    def __init__(self, trade: CExtTrade):
        self._trade = trade
    
    def to_proto(self) -> trade_pb2.Trade:
        return trade_pb2.Trade(
            symbol=self._trade.symbol,
            price=str(self._trade.price),  # Decimal → string
            amount=str(self._trade.amount),
            timestamp_us=int(self._trade.timestamp * 1_000_000),  # float sec → int64 µsec
            side=self._trade.side,
            exchange=self._trade.exchange
        )
```

### 2. Serializer Abstraction

**Decision**: Abstract `Serializer` base class with `serialize()` and `content_type()` methods.

**Rationale**:
- SOLID: Single Responsibility, Open/Closed, Liskov Substitution
- Easy to add new formats (Avro, MessagePack) in future
- Clear contract for all serializers
- Type-safe with Python ABC

### 3. Dual-Format Support

**Decision**: Support JSON and Protobuf simultaneously via configuration.

**Rationale**:
- Backward compatibility: existing configs use JSON (default)
- Incremental migration: operators can run both formats side-by-side
- Zero breaking changes: JSON remains default
- Format selection per-callback, not global

**Configuration**:
```yaml
backends:
  kafka_protobuf:
    type: kafka
    serialization_format: protobuf  # NEW parameter
    bootstrap_servers: localhost:9092
  
  redis_json:
    type: redis
    serialization_format: json  # Explicit (or omit for default)
    host: localhost
```

### 4. Decimal Precision Preservation

**Decision**: Encode `Decimal` as string, not float64.

**Rationale**:
- Protobuf `double` (IEEE 754) loses precision (15-17 significant digits)
- Financial data requires arbitrary precision
- String encoding preserves full decimal places
- Negligible size penalty (~20-30 bytes per message)

**Example**:
- Input: `Decimal('50000.123456789')`
- Protobuf: `price: "50000.123456789"` (string field)
- Round-trip: `Decimal('50000.123456789')` ✅ No loss

### 5. Timestamp Conversion

**Decision**: Convert float seconds → int64 microseconds.

**Rationale**:
- Protobuf int64 has no precision loss (unlike float)
- Microsecond precision matches industry standard
- Consistent with other market data systems (Tardis, DBN)

**Conversion**:
```python
timestamp_float = 1700000000.123  # float seconds
timestamp_us = int(timestamp_float * 1_000_000)  # 1700000000123000 µsec
```

---

## Next Immediate Steps

### **Step 1: Start with Task 1.1** (Serializer ABC)
```bash
cd /home/tommyk/projects/quant/data-sources/crypto-data/cryptofeed

# Create serializers module
mkdir -p cryptofeed/serializers
touch cryptofeed/serializers/__init__.py
touch cryptofeed/serializers/base.py

# Write tests first (TDD)
mkdir -p tests/unit/serializers
touch tests/unit/serializers/test_serializer_base.py

# Implement Serializer ABC
# Run tests
pytest tests/unit/serializers/test_serializer_base.py -v
```

**Task 1.1 Acceptance**:
- `Serializer` abstract base class with `serialize()` and `content_type()` methods
- Cannot instantiate directly (raises `TypeError`)
- Subclasses must implement both methods
- 4/4 unit tests passing

---

### **Step 2: Task 1.2** (JSONSerializer)
```bash
# Create JSONSerializer
touch cryptofeed/serializers/json.py
touch tests/unit/serializers/test_json_serializer.py

# Implement JSONSerializer (backward compatible)
# Run tests
pytest tests/unit/serializers/test_json_serializer.py -v
```

**Task 1.2 Acceptance**:
- `JSONSerializer` uses existing `to_dict()` method
- Byte-identical output to pre-refactor JSON
- 4/4 unit tests passing
- 100% code coverage

---

### **Step 3: Task 1.3** (BackendCallback Integration)
```bash
# Modify BackendCallback
# Add serialization_format parameter
# Add _get_serializer() factory method

# Test integration
touch tests/integration/test_callback_serialization.py
pytest tests/integration/test_callback_serialization.py -v
```

**Task 1.3 Acceptance**:
- BackendCallback accepts `serialization_format` parameter
- Defaults to `'json'` (backward compatible)
- Raises `ValueError` for unknown formats
- 4/4 integration tests passing

---

### **Step 4: Task 1.4** (Generate Protobuf Bindings)
```bash
# Generate Python bindings from .proto files
buf generate proto/

# Verify generation
ls -la gen/python/cryptofeed/normalized/v1/

# Create import wrapper
mkdir -p cryptofeed/proto_bindings
touch cryptofeed/proto_bindings/__init__.py

# Test imports
python3 -c "from cryptofeed.proto_bindings import trade_pb2; print(trade_pb2.Trade)"

# Run tests
touch tests/unit/proto/test_protobuf_bindings.py
pytest tests/unit/proto/test_protobuf_bindings.py -v
```

**Task 1.4 Acceptance**:
- 20 `*_pb2.py` files generated in `gen/python/`
- All protobuf message classes importable
- 3/3 unit tests passing
- No import errors

---

### **Step 5: Task 1.5** (ProtobufSerializer)
```bash
# Create ProtobufSerializer
touch cryptofeed/serializers/protobuf.py
touch tests/unit/serializers/test_protobuf_serializer.py

# Implement ProtobufSerializer
# Run tests
pytest tests/unit/serializers/test_protobuf_serializer.py -v --cov=cryptofeed.serializers.protobuf
```

**Task 1.5 Acceptance**:
- `ProtobufSerializer` invokes `obj.to_proto()` and serializes to bytes
- Raises `SerializationError` for missing `to_proto()`
- 4/4 unit tests passing
- 100% code coverage

---

## Testing Strategy

### Unit Tests (TDD - Write First)
- **Serializer ABC**: Cannot instantiate, requires implementations
- **JSONSerializer**: Backward compatibility, identical output
- **ProtobufSerializer**: Handles to_proto(), raises errors
- **Data type wrappers**: Round-trip serialization, precision preservation

### Integration Tests
- **Callback integration**: Format selection, error handling
- **Multi-format coexistence**: JSON + Protobuf side-by-side
- **Kafka integration**: Real Kafka topics, consumer deserialization

### Performance Tests
- **Latency benchmarks**: <1ms p99 for protobuf, <2ms for JSON
- **Size reduction**: 50-60% smaller than JSON
- **Throughput**: 10,000 messages/sec minimum

---

## Success Metrics

### Functional Success
- ✅ All 14 data types have `to_proto()` methods (via wrappers)
- ✅ ProtobufSerializer produces valid protobuf bytes
- ✅ JSONSerializer output matches pre-refactor exactly
- ✅ Backward compatible (zero breaking changes)
- ✅ Configuration via YAML and programmatic API

### Performance Success
- ✅ Protobuf serialization p99 < 1ms per message
- ✅ JSON serialization p99 < 2ms per message
- ✅ Protobuf 50-70% smaller than JSON
- ✅ Memory overhead < 10MB per callback instance

### Quality Success
- ✅ 95%+ test coverage for new code
- ✅ 100% type hint coverage (mypy strict)
- ✅ Zero breaking changes
- ✅ Clear documentation and examples

---

## Dependencies

### Required
- ✅ `normalized-data-schema-crypto` v0.1.0 (merged, schemas available)
- ✅ `protobuf>=5.0.0` (Python protobuf library)
- ✅ `buf` CLI (for schema generation)

### Optional
- `confluent-kafka>=2.3.0` (for Kafka integration in Phase 3)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| C extension modification complexity | Use Python wrappers instead |
| Performance overhead from wrappers | Benchmark early (Task 1.9), optimize hot paths |
| Backward compatibility broken | Default to JSON, comprehensive regression tests |
| Decimal precision loss | Use string encoding, round-trip test coverage |
| Consumer deserialization errors | Provide reference implementations, clear docs |

---

## Documentation Deliverables

1. **User Guide**: `docs/protobuf-serialization.md`
   - Configuration examples (YAML + Python API)
   - Migration guide (JSON → Protobuf)
   - Troubleshooting guide

2. **Consumer Integration Guide**: `docs/consumer-integration-guide.md`
   - Python consumer example (Kafka + protobuf)
   - Flink consumer reference
   - DuckDB consumer reference

3. **Code Examples**:
   - `examples/kafka_protobuf.py` - Protobuf Kafka producer
   - `examples/kafka_dual_format.py` - JSON + Protobuf side-by-side

---

## Questions?

**Slack**: #cryptofeed-dev
**Spec**: `.kiro/specs/protobuf-callback-serialization/`
**Tasks**: `.kiro/specs/protobuf-callback-serialization/tasks.md`

Ready to start with Task 1.1! 🚀
