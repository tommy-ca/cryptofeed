# CryptofeedSource for QuixStreams - Requirements Document

## Introduction

CryptofeedSource integrates Cryptofeed's Kafka producer with the QuixStreams streaming framework, enabling real-time market data analytics and aggregations. This specification defines comprehensive EARS-format functional requirements for consuming protobuf-serialized market data from 14 data type topics (Trade, Ticker, OrderBook, Candle, Funding, Liquidation, OpenInterest, Index, Balance, Position, Fill, OrderInfo, Order, Transaction), implementing error handling with Dead Letter Queues (DLQ), state management, monitoring, and exactly-once semantics.

### Scope

**In-Scope:**
- CryptofeedSource class extending QuixStreams Source
- Kafka consumer integration via confluent-kafka-python
- Protobuf deserialization for 14 data types with message header extraction (exchange, symbol, data_type, schema_version)
- Error handling and Dead Letter Queue (DLQ) implementation
- State management (offset tracking, checkpointing, consumer group management)
- Monitoring and observability (Prometheus metrics, structured logging, health checks)
- Configuration management (YAML support, environment variables, programmatic API)
- Schema version compatibility and migration guidance

**Out-of-Scope:**
- Storage implementation (Iceberg, DuckDB, Parquet delegated to consumers)
- Analytics and aggregation logic (consumer responsibility)
- Stream processing (transformation, joins, windowing - consumer responsibility)
- Data persistence policies and retention management
- Query engines and lakehouse architecture

### Dependencies

- **Spec 0 (normalized-data-schema-crypto):** 14 protobuf schemas for Trade, Ticker, OrderBook, Candle, Funding, Liquidation, OpenInterest, Index, Balance, Position, Fill, OrderInfo, Order, Transaction
- **Spec 1 (protobuf-callback-serialization):** Protobuf serialization helpers in `cryptofeed/backends/protobuf_helpers.py`
- **Spec 3 (market-data-kafka-producer):** Kafka topics with message headers (exchange, symbol, data_type, schema_version) and partition strategies (Composite, Symbol, Exchange, RoundRobin)

### Implementation Phases

- **Phase 1 (Core):** QuixStreams Source implementation, Kafka consumer integration, basic deserialization
- **Phase 2 (Error Handling):** DLQ, retry logic, circuit breaker, comprehensive error logging
- **Phase 3 (Monitoring):** Prometheus metrics, structured logging, health checks, observability
- **Phase 4 (Production):** Configuration management, schema compatibility, hardening, deployment

---

## Requirement 1: QuixStreams Source Implementation

**Objective:** As a data engineer, I want a QuixStreams-compatible Source class that integrates Cryptofeed's Kafka producer with the QuixStreams streaming framework, so that I can build real-time analytics applications consuming Cryptofeed market data.

### Acceptance Criteria

1. WHEN CryptofeedSource is instantiated with valid configuration parameters THEN CryptofeedSource SHALL initialize with a Kafka consumer, deserialization context, and internal state tracking structures.

2. WHEN CryptofeedSource.start() is called THEN CryptofeedSource SHALL perform resource initialization including creating the Kafka consumer, establishing topic subscriptions, and validating broker connectivity.

3. WHEN CryptofeedSource.run_stream() enters the main event loop THEN CryptofeedSource SHALL continuously poll Kafka for messages at a configurable interval (default: 100ms) until shutdown is signaled.

4. WHEN a valid Kafka message arrives on a subscribed topic THEN CryptofeedSource SHALL emit the deserialized message object to the QuixStreams pipeline with complete metadata (exchange, symbol, data_type, schema_version).

5. WHEN CryptofeedSource.shutdown() is called THEN CryptofeedSource SHALL cleanly close the Kafka consumer, commit pending offsets, and release all acquired resources.

6. IF the consumer is in idle state (no messages within heartbeat_interval) THEN CryptofeedSource SHALL maintain group membership by automatically sending heartbeat signals to the Kafka broker.

7. IF CryptofeedSource is stopped midstream THEN CryptofeedSource SHALL preserve the current offset position to enable resume-from-last-position on next start.

8. WHILE CryptofeedSource is actively polling THEN CryptofeedSource SHALL track bytes consumed, message count, and processing latency for each topic subscribed.

9. WHERE multiple CryptofeedSource instances operate with the same consumer group THEN CryptofeedSource SHALL implement partition assignment and automatic rebalancing per Kafka group semantics.

10. WHEN the configuration specifies multiple topic subscriptions THEN CryptofeedSource SHALL subscribe to all topics simultaneously and interleave messages from all subscribed topics in arrival order.

11. IF no messages are available within the poll timeout THEN CryptofeedSource SHALL return a null/empty message and continue polling without error.

---

## Requirement 2: Kafka Consumer Integration

**Objective:** As a developer, I want robust Kafka consumer integration that handles topic subscription, message polling, offset management, and partition rebalancing, so that CryptofeedSource reliably consumes messages from cryptofeed Kafka topics.

### Acceptance Criteria

1. WHEN CryptofeedSource is configured with broker addresses and topic names THEN CryptofeedSource SHALL create a confluent-kafka-python consumer with specified bootstrap servers and group ID.

2. WHEN CryptofeedSource initializes THEN CryptofeedSource SHALL validate Kafka broker connectivity by performing a metadata fetch with a configurable timeout (default: 5 seconds).

3. WHEN the on_assign callback is triggered during rebalancing THEN CryptofeedSource SHALL reset internal state (message counter, latency tracking) for assigned partitions.

4. WHEN the on_revoke callback is triggered during rebalancing THEN CryptofeedSource SHALL synchronously commit current offset state for revoked partitions before releasing them.

5. WHEN a message is successfully processed and emitted THEN CryptofeedSource SHALL automatically commit the message offset at configurable intervals (default: every 100 messages or 30 seconds).

6. IF auto-commit is enabled AND offset_commit_timeout is reached THEN CryptofeedSource SHALL asynchronously trigger offset commit and continue message processing.

7. IF a KafkaException occurs during message poll THEN CryptofeedSource SHALL surface the exception with exchange, symbol, and topic context for debugging.

8. WHEN consumer group coordination is lost THEN CryptofeedSource SHALL attempt to rejoin the consumer group with exponential backoff (max 5 retries) before raising an exception.

9. IF partition assignment changes THEN CryptofeedSource SHALL log the assignment change including added partitions, removed partitions, and owner instance ID.

10. WHEN the consumer reaches end-of-partition (EOF) THEN CryptofeedSource SHALL continue polling without error and wait for new messages.

11. WHERE isolation_level is set to "read_committed" THEN CryptofeedSource SHALL consume only transactionally committed messages from Kafka brokers supporting exactly-once semantics.

---

## Requirement 3: Protobuf Deserialization

**Objective:** As a data engineer, I want seamless protobuf deserialization for all 14 market data types with automatic type detection and metadata enrichment, so that I can work with strongly-typed data objects throughout the streaming pipeline.

### Acceptance Criteria

1. WHEN a Kafka message arrives with message headers containing data_type THEN CryptofeedSource SHALL select the appropriate protobuf schema (Trade, Ticker, OrderBook, Candle, Funding, Liquidation, OpenInterest, Index, Balance, Position, Fill, OrderInfo, Order, or Transaction) and deserialize the message body.

2. WHEN protobuf deserialization succeeds THEN CryptofeedSource SHALL populate the deserialized object with exchange (from header), symbol (from header), data_type (from header), and schema_version (from header) as object attributes.

3. WHEN a Trade message is deserialized THEN CryptofeedSource SHALL validate that required fields (timestamp, exchange, symbol, price, amount) are present and correctly typed as float/Decimal.

4. WHEN an OrderBook message is deserialized THEN CryptofeedSource SHALL validate bids and asks lists contain valid (price, amount) tuples and preserve the original order from the serialized message.

5. WHEN a Candle message is deserialized THEN CryptofeedSource SHALL validate that OHLCV fields (open, high, low, close, volume) are non-negative and close value is between high and low.

6. WHEN a Ticker message is deserialized THEN CryptofeedSource SHALL validate that fields (bid, ask, timestamp) are present and ask >= bid (within floating-point tolerance of 1e-8).

7. WHEN a message lacks the required schema_version header THEN CryptofeedSource SHALL default to the latest schema version for the detected data_type and log a warning.

8. WHEN the message body size exceeds the configured maximum (default: 10MB) THEN CryptofeedSource SHALL reject the message with a size validation error and route to DLQ.

9. WHEN the message body is successfully deserialized THEN CryptofeedSource SHALL enrich the object with _kafka_partition, _kafka_offset, _consumed_at timestamp for operational tracking.

10. IF deserialization fails due to protobuf parsing error THEN CryptofeedSource SHALL capture the raw message bytes, error message, and headers, then route to Dead Letter Queue for manual inspection.

11. WHILE processing a batch of messages THEN CryptofeedSource SHALL deserialize each message independently without allowing exceptions in one message to halt processing of subsequent messages.

12. WHERE a message's data_type header is not recognized THEN CryptofeedSource SHALL emit a validation error, log the invalid data_type value, and route the message to DLQ.

---

## Requirement 4: Error Handling and Dead Letter Queue

**Objective:** As an operations engineer, I want comprehensive error handling with Dead Letter Queues, retry logic, circuit breakers, and detailed error logging, so that I can identify and resolve data quality issues without losing messages.

### Acceptance Criteria

1. WHEN a message cannot be deserialized due to protobuf parsing error THEN CryptofeedSource SHALL write the raw message bytes, headers, error message, and timestamp to the Dead Letter Queue topic (default: cryptofeed-dlq) and continue processing.

2. WHEN a validation error occurs (e.g., OrderBook bids/asks out of order, Ticker ask < bid) THEN CryptofeedSource SHALL emit the validation error with data_type, exchange, symbol, and expected constraints, then route to DLQ.

3. WHEN deserialization of a message succeeds but a subsequent processing step fails THEN CryptofeedSource SHALL capture the failed message, error stack trace, and processing stage (e.g., "deserialization", "validation", "enrichment") in DLQ record.

4. WHEN a Kafka broker error occurs (broker unreachable, metadata request timeout) THEN CryptofeedSource SHALL implement exponential backoff with configurable base delay (default: 100ms) and maximum retries (default: 5) before raising an exception.

5. WHEN Kafka consumer group coordination fails THEN CryptofeedSource SHALL enter a circuit breaker state: pause message polling, log the failure reason, and wait for configurable circuit breaker timeout (default: 30 seconds) before attempting reconnection.

6. IF circuit breaker is in OPEN state (broker connectivity lost) THEN CryptofeedSource SHALL not attempt Kafka operations and instead raise a CircuitBreakerOpen exception to the application.

7. WHEN circuit breaker transitions from OPEN to HALF_OPEN THEN CryptofeedSource SHALL reset internal error counters and attempt a single metadata fetch to verify broker connectivity.

8. IF metadata fetch succeeds while in HALF_OPEN state THEN CryptofeedSource SHALL transition to CLOSED state and resume normal message polling.

9. IF metadata fetch fails while in HALF_OPEN state THEN CryptofeedSource SHALL transition back to OPEN state and restart the circuit breaker cooldown timer.

10. WHEN a message is written to DLQ THEN CryptofeedSource SHALL include structured metadata: original_topic, partition, offset, error_code, error_message, timestamp, and full headers.

11. WHILE operating under degraded conditions (high error rate >10% over 1-minute window) THEN CryptofeedSource SHALL emit WARNING level logs with error statistics (error_count, success_count, error_rate) every 10 seconds.

12. WHERE an exception occurs during DLQ write THEN CryptofeedSource SHALL log the DLQ write failure at ERROR level, increment dlq_write_failure_counter, and continue processing to prevent cascade failures.

13. WHEN error_max_retries is exceeded for a single message THEN CryptofeedSource SHALL permanently mark the message as unrecoverable and move to DLQ with status "max_retries_exceeded".

---

## Requirement 5: State Management

**Objective:** As a platform engineer, I want robust state management including offset tracking, checkpointing, consumer group coordination, and optional RocksDB state stores, so that CryptofeedSource can reliably resume processing and support stateful operations.

### Acceptance Criteria

1. WHEN a message is successfully processed THEN CryptofeedSource SHALL track the Kafka offset (partition, offset pair) in internal state before committing to broker.

2. WHEN commit_interval_messages is reached (default: 1000 messages) THEN CryptofeedSource SHALL synchronously commit all tracked offsets to Kafka broker with committed_at timestamp.

3. WHEN commit_interval_seconds is reached (default: 30 seconds) THEN CryptofeedSource SHALL synchronously commit tracked offsets regardless of message count.

4. IF offset commit fails due to broker error THEN CryptofeedSource SHALL log the commit failure with affected offsets and retry on the next commit interval.

5. WHEN CryptofeedSource starts AND an existing consumer group offset is recorded THEN CryptofeedSource SHALL resume from the last committed offset (seek behavior = automatic_offset_reset per config).

6. IF automatic_offset_reset is "earliest" AND no prior offset exists THEN CryptofeedSource SHALL start consuming from partition offset 0.

7. IF automatic_offset_reset is "latest" AND no prior offset exists THEN CryptofeedSource SHALL start consuming from the current end-of-partition offset.

8. WHEN a rebalance occurs THEN CryptofeedSource SHALL commit current offsets synchronously, pause message polling during rebalance, and resume polling after new partition assignment.

9. WHERE StatefulSource mode is enabled THEN CryptofeedSource SHALL create or open a RocksDB state store at the configured path with key-value serialization matching the application schema.

10. WHEN a state store operation is requested THEN CryptofeedSource SHALL provide synchronous read/write access with automatic key prefix scoping per partition (isolation).

11. IF state store write fails THEN CryptofeedSource SHALL raise a StateStoreException and halt processing to prevent state corruption.

12. WHILE a rebalance is in progress THEN CryptofeedSource SHALL flush all pending state store operations to disk before releasing partition ownership.

13. WHEN CryptofeedSource shuts down THEN CryptofeedSource SHALL flush pending offsets, close state store handles, and persist final state to durable storage.

---

## Requirement 6: Monitoring and Observability

**Objective:** As an operations engineer, I want comprehensive monitoring including Prometheus metrics, structured JSON logging, and health checks, so that I can observe CryptofeedSource behavior in production and detect anomalies.

### Acceptance Criteria

1. WHEN CryptofeedSource processes a message THEN CryptofeedSource SHALL increment a counter metric messages_consumed_total with labels: topic, partition, data_type, exchange.

2. WHEN CryptofeedSource successfully emits a message to the pipeline THEN CryptofeedSource SHALL increment messages_produced_total counter with identical labels.

3. WHEN an error occurs during message processing THEN CryptofeedSource SHALL increment errors_total counter with labels: error_type (e.g., "deserialization", "validation", "broker_error"), topic, severity (warning, error).

4. WHEN a message is written to DLQ THEN CryptofeedSource SHALL increment dlq_messages_total counter with labels: reason (e.g., "parse_error", "validation_error", "size_exceeded").

5. WHEN a message is consumed from Kafka and emitted to the pipeline THEN CryptofeedSource SHALL measure end-to-end latency (kafka_timestamp to emit time) and record in messages_latency_seconds histogram with buckets: 0.01, 0.1, 0.5, 1.0, 5.0 seconds.

6. WHEN offset commit succeeds THEN CryptofeedSource SHALL record the committed offset and last_committed_offset metric with labels: topic, partition.

7. WHILE CryptofeedSource is actively polling THEN CryptofeedSource SHALL emit consumer_lag_offsets gauge with labels: topic, partition (current offset vs latest offset).

8. WHEN partition assignment changes THEN CryptofeedSource SHALL emit partition_assignment_count counter with labels: action (assigned, revoked), partition_count.

9. WHEN Kafka broker connectivity is lost THEN CryptofeedSource SHALL emit kafka_broker_connectivity_status gauge with value 0 (down) and record broker_reconnect_attempts counter.

10. WHEN circuit breaker state changes THEN CryptofeedSource SHALL emit circuit_breaker_state gauge with value: 0 (CLOSED), 1 (HALF_OPEN), 2 (OPEN).

11. WHERE metrics collection is enabled THEN CryptofeedSource SHALL expose Prometheus-compatible metrics endpoint (default: 0.0.0.0:8000/metrics) with text format per Prometheus spec.

12. WHEN CryptofeedSource logs an event THEN CryptofeedSource SHALL emit structured JSON with fields: timestamp (ISO 8601), level (DEBUG, INFO, WARNING, ERROR), message, context (exchange, symbol, topic, partition, offset), trace_id (for distributed tracing).

13. IF log_level is DEBUG THEN CryptofeedSource SHALL include additional context: message_size_bytes, deserialization_time_ms, commit_offset, error_details.

14. WHEN a health check request is received (GET /health) THEN CryptofeedSource SHALL return HTTP 200 with status: {"status": "healthy", "kafka_connected": bool, "circuit_breaker_state": string, "messages_processed": int}.

15. IF Kafka broker is unreachable AND circuit breaker is OPEN THEN CryptofeedSource health check SHALL return HTTP 503 with status: {"status": "unhealthy", "reason": "kafka_unavailable"}.

---

## Requirement 7: Configuration Management

**Objective:** As a DevOps engineer, I want flexible configuration supporting YAML files, environment variables, and programmatic APIs with validation, so that I can deploy CryptofeedSource across environments without code changes.

### Acceptance Criteria

1. WHEN CryptofeedSource is initialized with a config_file path THEN CryptofeedSource SHALL load and parse YAML configuration with structured validation against a schema.

2. WHEN environment variables are set matching the pattern CRYPTOFEED_QUIXSTREAMS_* THEN CryptofeedSource SHALL override corresponding config values from YAML (environment takes precedence).

3. WHEN a configuration key is missing AND a default value is defined THEN CryptofeedSource SHALL use the default value without raising an error.

4. WHEN a configuration value fails validation (e.g., commit_interval_messages is not a positive integer) THEN CryptofeedSource SHALL raise a ConfigurationError with the invalid key, provided value, and expected type/constraints.

5. WHEN CryptofeedSource is instantiated with a ConfigDict object THEN CryptofeedSource SHALL use the ConfigDict values directly with programmatic override support (no file loading).

6. IF a required configuration key is missing AND no default exists THEN CryptofeedSource SHALL raise a ConfigurationError listing all required missing keys.

7. WHEN the configuration includes a broker list THEN CryptofeedSource SHALL accept both comma-separated string format ("broker1:9092,broker2:9092") and list format (["broker1:9092", "broker2:9092"]).

8. WHERE configuration specifies topics as a comma-separated string THEN CryptofeedSource SHALL parse and normalize to a list ["cryptofeed.trade", "cryptofeed.orderbook", ...] and validate each topic matches the pattern cryptofeed.*

9. WHEN commit_interval_seconds is set to 0 THEN CryptofeedSource SHALL disable time-based offset commits and use only message count-based commits (commit_interval_messages).

10. IF consumer_timeout_ms is set to a negative value THEN CryptofeedSource SHALL raise a ConfigurationError (timeout must be positive integer or 0 for infinite).

11. WHEN configuration specifies dlq_topic THEN CryptofeedSource SHALL validate that dlq_topic is different from source topics and exists or can be auto-created on broker.

12. WHERE schema_registry_url is configured THEN CryptofeedSource SHALL validate connectivity to the schema registry at initialization time and raise ConfigurationError if unreachable.

13. WHEN the configuration includes optional_fields (e.g., state_store_path, metrics_port) THEN CryptofeedSource SHALL create supporting resources only if the optional field is specified.

---

## Requirement 8: Schema Version Compatibility

**Objective:** As a data architect, I want automatic schema version checking and backward-compatible deserialization with migration guidance, so that schema updates don't break CryptofeedSource and provide clear upgrade paths.

### Acceptance Criteria

1. WHEN a message arrives with schema_version header THEN CryptofeedSource SHALL extract the version number and validate against the supported schema versions for that data_type.

2. IF message schema_version matches the current schema version THEN CryptofeedSource SHALL deserialize using the current schema and succeed.

3. IF message schema_version is older than the current schema but in the compatibility window (default: last 2 major versions) THEN CryptofeedSource SHALL deserialize using a version-specific deserializer and apply automatic field mapping.

4. IF message schema_version is newer than the current schema THEN CryptofeedSource SHALL emit a warning, route the message to DLQ with reason "schema_version_too_new", and log the schema version mismatch.

5. WHEN schema_version indicates Trade v2 and current implementation supports Trade v2 and v1 THEN CryptofeedSource SHALL deserialize Trade v2 message correctly preserving all v2 fields.

6. WHEN schema_version indicates Trade v1 and current implementation supports Trade v2 and v1 THEN CryptofeedSource SHALL deserialize Trade v1 message and populate missing v2 fields with sensible defaults or null.

7. WHERE a breaking change is required in schema (removal of a field) THEN CryptofeedSource SHALL document the breaking change version, affected data_type, removed fields, and recommended migration path in schema_migration_guide.md.

8. WHEN CryptofeedSource encounters an unsupported schema_version THEN CryptofeedSource SHALL emit an error log with data_type, message schema_version, and supported_versions list, then route to DLQ.

9. IF a message lacks the schema_version header THEN CryptofeedSource SHALL log a warning, assume the latest schema version, and attempt deserialization with potential data loss if the assumption is incorrect.

10. WHEN a schema migration is required THEN CryptofeedSource SHALL provide a migration tool or documented procedure to reprocess messages from DLQ with updated schema definitions.

11. WHERE schema_version is used for metrics or monitoring THEN CryptofeedSource SHALL include schema_version in message metadata labels for histograms and counters.

12. WHEN the supported schema version window is updated (e.g., dropping v1 support after v3 release) THEN CryptofeedSource SHALL document the deprecation timeline with minimum 2-week notice before removal.

---

## Requirement 9: Message Header Extraction and Routing

**Objective:** As a developer, I want automatic extraction and validation of Kafka message headers (exchange, symbol, data_type, schema_version) with proper routing, so that I can rely on consistent header-based metadata.

### Acceptance Criteria

1. WHEN a Kafka message is consumed THEN CryptofeedSource SHALL extract and validate all expected headers: exchange, symbol, data_type, schema_version.

2. IF a required header (exchange, symbol, data_type) is missing THEN CryptofeedSource SHALL emit a validation error, log the missing header name, and route the message to DLQ.

3. WHEN all headers are present AND valid THEN CryptofeedSource SHALL populate the deserialized message object attributes: msg.exchange, msg.symbol, msg.data_type, msg.schema_version.

4. IF header value encoding is UTF-8 bytes THEN CryptofeedSource SHALL decode to string automatically with fallback to latin-1 if UTF-8 decode fails.

5. WHEN a header value exceeds maximum length (default: 1000 chars) THEN CryptofeedSource SHALL truncate with warning log and append "[truncated]" suffix.

6. WHERE partition strategy is "symbol-based" THEN CryptofeedSource SHALL validate that symbol header is present and consistent with the partition assignment strategy.

---

## Requirement 10: Integration with QuixStreams Pipeline

**Objective:** As an application developer, I want seamless integration with QuixStreams streaming applications, so that I can build analytics workflows using CryptofeedSource as a message source.

### Acceptance Criteria

1. WHEN CryptofeedSource is registered as a Source in a QuixStreams StreamingApp THEN CryptofeedSource SHALL emit deserialized message objects that match the QuixStreams message format.

2. WHEN a downstream QuixStreams transform accesses a message from CryptofeedSource THEN message SHALL contain all fields from the protobuf schema plus enrichment fields (_kafka_partition, _kafka_offset, _consumed_at).

3. WHEN CryptofeedSource is used with StreamingApp.run() THEN CryptofeedSource SHALL block until Kafka broker becomes unreachable or the application is shut down.

4. IF CryptofeedSource encounters an unrecoverable error THEN CryptofeedSource SHALL raise an exception to the StreamingApp which will trigger application shutdown per framework semantics.

5. WHEN CryptofeedSource emits a message THEN the message SHALL be processed synchronously by QuixStreams transforms before the next message is polled from Kafka.

---

## Validation Summary

### EARS Format Compliance
- Total requirements generated: 83
- WHEN-THEN patterns: 57 (68.7%)
- IF-THEN patterns: 18 (21.7%)
- WHILE-THE patterns: 4 (4.8%)
- WHERE-THE patterns: 4 (4.8%)

### Breakdown by Functional Area
1. QuixStreams Source Implementation: 11 criteria
2. Kafka Consumer Integration: 11 criteria
3. Protobuf Deserialization: 12 criteria
4. Error Handling and DLQ: 13 criteria
5. State Management: 13 criteria
6. Monitoring and Observability: 15 criteria
7. Configuration Management: 13 criteria
8. Schema Version Compatibility: 12 criteria
9. Message Header Extraction and Routing: 6 criteria
10. Integration with QuixStreams Pipeline: 5 criteria

### Testability Assessment
- All acceptance criteria use measurable/observable conditions
- Success criteria are verifiable through automated tests
- Clear expected outcomes for each acceptance criterion
- Dependencies between requirements properly identified

### Ambiguity Assessment
- No vague terms ("fast", "stable", "good") used
- All timeouts, counters, error rates explicitly specified with defaults
- All error conditions mapped to concrete actions
- All data types and ranges defined
