# Implementation Tasks: Shift Left Streaming Lakehouse Integration

- [ ] 1. Define v2 Protobuf Schemas (Phase 1)
  - Create `proto/cryptofeed/normalized/v2/` directory structure
  - Define `trade.proto` with native types (`double`, `int64`) and `google.protobuf.Timestamp`
  - Define `ticker.proto`, `book.proto`, and `candle.proto` with consistent native type patterns
  - Configure `syntax = "proto3"` and proper package names in all files
  - Add `sequence_number` field to all message types for gap detection
  - Add a per-message field matrix section documenting chosen numeric type (`double` vs `bytes`) and, if `bytes`, the shared `scale` field per REQ-011; reserve any v1 field numbers that are not reused
  - Author the field matrix in `proto/cryptofeed/normalized/v2/README.md` and keep it in sync with `.proto` definitions
  - Add a launch decision table marking which fields (and exchanges, if applicable) will use `bytes+scale` at Day 1; otherwise default to `double`
  - Fix and document the `scale` field number (use `15` across all messages when present) and mark any unused numbers as reserved in the `.proto` files
  - Run `buf lint proto/cryptofeed/normalized/v2` to ensure schema hygiene
  - _Requirements: REQ-005, REQ-007, REQ-010, NFR-004_

- [ ] 2. Implement v2 Protobuf Helpers (Phase 2)
  - Create `cryptofeed/backends/protobuf_helpers_v2.py` module
  - Implement `trade_to_proto_v2` function with `Decimal` to `float` casting
  - Implement timestamp conversion helper to populate `google.protobuf.Timestamp`
  - Implement conversion functions for Ticker, Book, and Candle types
  - Add unit tests for value precision and timestamp accuracy
  - _Requirements: REQ-006, REQ-005, REQ-007_

- [ ] 3. Enhance Schema Registry Client (Phase 3)
  - Verify `cryptofeed.backends.kafka_schema.SchemaRegistry` thread-safety for async execution
  - Enhance `_schema_cache` to ensure atomic updates or thread-safe access
  - Verify support for Basic Auth and mTLS in the underlying request configuration
  - _Requirements: REQ-001, NFR-001, NFR-005_

- [ ] 4. Integrate Registry in KafkaCallback (Phase 3)
  - Update `cryptofeed/kafka_callback.py` to parse `schema_registry` configuration
  - Implement `_get_schema_id` using `loop.run_in_executor` for async registry operations
  - Implement Confluent Wire Format framing (Magic Byte + Schema ID + Payload)
  - Integrate `protobuf_helpers_v2` for serialization when registry mode is active
  - Implement error handling strategy (buffer/fail) for registry unavailability
  - _Requirements: REQ-002, REQ-003, REQ-004, NFR-001_

- [ ] 5. Implement Context & Dual Production (Phase 3)
  - Add standard headers (`exchange`, `symbol`, `data_type`, `schema_version`) to Kafka records
  - Implement composite key generation (e.g., `<exchange>-<symbol>`) for partition ordering
  - Add logic to support dual production to v1 (legacy) and v2 (registry) topics simultaneously
  - _Requirements: REQ-008, REQ-009, NFR-003_

- [ ] 6. Verification & Documentation (Phase 4)
  - Create end-to-end integration test using a mock or local Schema Registry
  - Implement benchmark script to measure v1 vs v2 message size reduction (>30%)
  - Update `docs/consumer-integration-guide.md` with v2 consumption examples
  - _Requirements: NFR-002_
