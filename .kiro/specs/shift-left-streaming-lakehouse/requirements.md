# Requirements Document: Shift Left Streaming Lakehouse Integration

## Project Description
This initiative aims to "shift left" the data quality and schema enforcement responsibilities to the ingestion layer (Cryptofeed). Currently, consumers receive raw Protobuf messages with string-based types and must handle schema validation and type conversion manually. By implementing Confluent Schema Registry integration, moving to v2 Protobuf schemas with native types, and enriching message context, we enable a seamless Flink -> Iceberg streaming lakehouse pattern.

## Functional Requirements

### Schema Registry Integration (Contract)
- **REQ-001**: WHEN the Kafka Producer initializes, THEN the system SHALL verify connectivity to the configured Confluent Schema Registry.
- **REQ-002**: WHEN publishing a message, IF the schema is not registered, THEN the system SHALL register the Protobuf schema version with the Registry.
- **REQ-003**: WHEN publishing a message, THEN the system SHALL serialize the payload using the Confluent Wire Format (Magic Byte + Schema ID + Payload).
- **REQ-004**: WHERE the Schema Registry is unavailable, THEN the system SHALL fallback to a configurable error handling strategy (buffer or fail-fast).

### Native v2 Types (Compute)
- **REQ-005**: WHEN generating v2 Protobuf schemas, THEN the system SHALL use `double` or `bytes` for numeric fields (Price, Amount) instead of `string`.
- **REQ-006**: WHEN transforming internal data structures to v2 Protobuf messages, THEN the system SHALL perform efficient type conversion (e.g., Decimal to double/bytes).
- **REQ-007**: IF a field represents a timestamp, THEN the system SHALL use `google.protobuf.Timestamp` or `int64` (nanoseconds) in the v2 schema.
- **REQ-011**: WHEN a field uses `bytes` to preserve Decimal fidelity, the schema SHALL also define a message-level `int32 scale` field documenting the exponent used during quantization; if `double` is chosen, the design MUST record that the field is lossy but acceptable for the data type.

### Stream ID Context (Context)
- **REQ-008**: WHEN publishing a Kafka message, THEN the system SHALL include standard headers for `exchange`, `symbol`, `data_type`, and `schema_version`.
- **REQ-009**: WHEN constructing the Kafka record key, THEN the system SHALL use a consistent composite key (e.g., `<exchange>-<symbol>`) to ensure partition ordering.
- **REQ-010**: WHERE the data source provides a sequence number, THEN the system SHALL include it in the message payload to allow gap detection by consumers.

## Non-Functional Requirements

### Performance
- **NFR-001**: The overhead of Schema Registry lookups SHALL be minimized by caching Schema IDs locally (target: < 1ms overhead per message after cache warmup).
- **NFR-002**: Binary serialization with native types SHOULD result in a message size reduction of at least 30% compared to string-based v1 schemas.

### Compatibility
- **NFR-003**: The system SHALL support parallel production of v1 (legacy) and v2 (schema-registry) topics during the migration phase.
- **NFR-004**: The v2 schemas SHALL follow Protobuf best practices to allow for forward and backward compatibility (e.g., reserved fields, no required fields).

### Reliability
- **NFR-005**: The integration SHALL support standard Schema Registry authentication methods (Basic Auth, mTLS).

## Implementation Plan

### Phase 1: Schema Definition (v2)
- Define `v2` Protobuf schemas in `proto/cryptofeed/normalized/v2/`.
- Replace string-based numeric types with native types (`double` for float efficiency or `bytes` for decimal precision).
- Standardize timestamp fields.
- Produce a per-message field matrix (trade, ticker, book, candle) that records the exact type choice (`double` vs `bytes`), any shared `scale` field, and reserved field numbers inherited from v1 for backward compatibility.

### Phase 2: Schema Registry Client
- Integrate `confluent-kafka` python client or compatible library.
- Implement a `SchemaRegistryService` within Cryptofeed to handle registration and ID caching.
- Add configuration options for Schema Registry URL and credentials.

### Phase 3: Producer Update
- Update `KafkaCallback` to support a "Schema Registry Mode".
- Implement the serialization logic using the Schema Registry serializer.
- Inject standard headers (`exchange`, `symbol`, etc.) into the Kafka record.

### Phase 4: Validation & Documentation
- Verify Flink compatibility by consuming v2 topics with a simple Flink job.
- Update `docs/consumer-integration-guide.md` with instructions for consuming v2 topics.
- Benchmark performance difference between v1 and v2.
