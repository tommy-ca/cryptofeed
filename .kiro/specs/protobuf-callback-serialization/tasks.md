# Implementation Tasks

- [x] 1. Establish serialization foundation
- [x] 1.1 Define serialization exception hierarchy
  - Provide a base serialization exception that captures data type, schema, and format context for observability.
  - Create specialized errors for encoding failures and missing converters with actionable remediation guidance.
  - Preserve exception chaining so upstream handlers receive both the root cause and high-level context.
  - _Requirements: 4.5_

- [x] 1.2 Introduce serializer abstraction with typed interface
  - Provide an abstract serializer contract that enforces binary outputs and surfaces MIME metadata for transports.
  - Supply a JSON serializer implementation that adheres to the contract to maintain the existing default behavior.
  - Enforce strict typing and static analysis so serializer implementations satisfy mypy checks.
  - _Requirements: 2, 3, 6_

- [x] 2. Deliver configuration-driven format selection
- [x] 2.1 Expose serialization format across configuration surfaces
  - Accept explicit format selection within backend configuration documents with validation of supported values.
  - Honor environment overrides with predictable precedence over static configuration.
  - Reflect configuration choices inside programmatic callback builders for parity with YAML usage.
  - _Requirements: 2, 3_

- [x] 2.2 Preserve backward-compatible defaults
  - Default callbacks to JSON when no format is configured to avoid breaking existing deployments.
  - Log format selection at startup for observability without overwhelming production logs.
  - Prevent unsupported formats by emitting clear errors that describe allowed values.
  - _Requirements: 2, 3_

- [x] 2.3 Validate runtime toggle interactions
  - Confirm format selection can vary per callback without cross-contamination of serializer state.
  - Ensure mixed JSON and protobuf callbacks operate concurrently within the same feed handler.
  - Provide guardrails preventing serialization format changes after callback initialization.
  - _Requirements: 2, 3, 6_

- [x] 3. Integrate serialization into backend callbacks
- [x] 3.1 Inject serializer selection into callback lifecycle
  - Resolve serializer implementations lazily based on configured format for each invocation.
  - Route backend writes through serializer outputs while preserving existing queue semantics.
  - Record serialization errors with data type identifiers before surfacing exceptions upstream.
  - _Requirements: 2, 4.5, 6, 7_

- [x] 3.2 Implement Kafka topic and partition strategy
  - Produce topics following the hierarchy `cryptofeed.market.{data_type}.{exchange}` when protobuf payloads are enabled.
  - Derive partition keys from normalized symbols to align with consumer sharding expectations.
  - Confirm topic naming remains backward compatible for JSON callbacks unless protobuf is actively selected.
  - _Requirements: 4_

- [x] 3.3 Enable binary payload delivery for alternate backends
  - Ensure Redis and ZMQ transports accept binary messages without implicit JSON conversion.
  - Surface content type metadata to downstream consumers when the transport supports headers.
  - Maintain compatibility with legacy transports by falling back to JSON when binary payloads are unsupported.
  - _Requirements: 2, 3_

- [x] 4. Align protobuf schema integration
- [x] 4.1 Load normalized schema bindings and version guardrails
  - Import generated protobuf bindings from the normalized data schema release and verify availability during startup.
  - Validate message classes before serialization begins, raising targeted errors when bindings are missing.
  - Capture schema version information for inclusion within logging and exception metadata.
  - _Requirements: 5, 6_

- [ ] 4.2 Manage schema registry publication workflow
  - Publish protobuf descriptors to configured registry endpoints whenever protobuf serialization is enabled.
  - Coordinate updates so schema publication completes before the first message is emitted.
  - Detect schema drift between bindings and registry contents, failing fast with remediation guidance.
  - _Requirements: 5_

- [x] 5. Build adapter layer for C extension types
- [x] 5.1 Create wrapper registry for data types
  - Detect inbound C extension instances and wrap them with Python adapters that expose `to_proto` behavior.
  - Provide a registry lookup keyed by normalized data type identifiers with a stateless implementation.
  - Emit descriptive errors whenever a data type lacks wrapper coverage to guide follow-up work.
  - _Requirements: 1, 7.5_

- [x] 5.2 Normalize field conversions within adapters
  - Convert Decimal values to string representations that preserve arbitrary precision across all data types.
  - Translate timestamps from float seconds to integer microseconds expected by protobuf schemas.
  - Map enum-like fields to protobuf enumerations while validating allowed values and raising precise errors.
  - _Requirements: 1, 6, 7.5_

- [x] 6. Implement to_proto conversions for market data categories
- [x] 6.1 Support trade and order flow events
  - Provide protobuf conversions for trades, fills, and order acknowledgements using the adapter registry.
  - Implement order book snapshot and delta conversions that preserve depth levels and metadata.
  - Ensure trade-related messages include exchange, symbol, side, price, amount, and identifiers without loss.
  - _Requirements: 1, 5, 7_

- [x] 6.2 Support pricing and rate surfaces
  - Deliver protobuf conversions for ticker, candle, funding rate, and mark price events with full precision.
  - Encode price and volume series alongside interval metadata required by downstream analytics.
  - Include next funding timestamps and rate calculations consistent with schema expectations.
  - _Requirements: 1, 5, 6_

- [x] 6.3 Support portfolio and index data
  - Implement conversions for open interest, index values, positions, balances, and transaction records.
  - Ensure optional fields such as leverage, pnl, and funding impact are populated when available.
  - Provide graceful handling for instruments not yet supported by emitting targeted errors and guidance.
  - _Requirements: 1, 5, 7.5_

- [x] 7. Validate serialization correctness and safety
- [x] 7.1 Establish unit and property-based test suites
  - Cover round-trip serialization and deserialization for each supported data type under normal and edge conditions.
  - Include stress tests with large order books and high-precision decimal values to guard against regressions.
  - Compare protobuf and JSON payload sizes to confirm expected reductions for key data sets.
  - _Requirements: 7_

- [x] 7.2 Enforce static and runtime validation
  - Run strict type checking across serialization modules with zero tolerated errors.
  - Assert adapter registry completeness during test setup to detect missing wrappers immediately.
  - Simulate malformed inputs to confirm defensive error handling and structured logging.
  - _Requirements: 6, 7, 7.5_

- [x] 7.3 Verify callback integration scenarios
  - Execute integration tests covering mixed-format callbacks within a single feed handler run.
  - Validate Kafka publishing flows including topic naming, partitioning, and optional header propagation.
  - Exercise Redis and ZMQ transports to ensure binary payloads are emitted without corruption or data loss.
  - _Requirements: 2, 3, 4, 7_

- [x] 8. Benchmark serialization performance
- [x] 8.1 Build reproducible benchmarking harness
  - Generate baseline datasets for trade, order book, and mixed workloads matching requirement definitions.
  - Measure latency percentiles for protobuf serialization across datasets and compare results with JSON baselines.
  - Track throughput and memory usage over sustained bursts exceeding ten thousand events.
  - Capture size reduction metrics for uncompressed payloads plus lz4 and zstd compressed outputs side-by-side.
  - _Requirements: 8_

- [x] 8.2 Optimize hot paths based on findings
  - Identify hotspots in adapters or serializer loops through profiling and refactor to meet latency targets.
  - Validate that optimizations preserve type safety, precision, and configuration guarantees.
  - Record benchmark outcomes alongside regression thresholds for future releases, highlighting lz4 and zstd comparisons.
  - _Requirements: 8, 6_

- [x] 9. Publish documentation and integration guidance
- [x] 9.1 Document configuration and operational guidance
  - Update user guidance with configuration examples for enabling protobuf serialization.
  - Explain migration paths from JSON-only deployments while highlighting backward compatibility assurances.
  - Provide troubleshooting steps for common errors such as missing schemas or unsupported data types.
  - _Requirements: 3, 9_

- [x] 9.2 Deliver consumer reference materials
  - Produce sample producer and consumer flows that demonstrate protobuf payload handling end to end.
  - Document Kafka topic conventions, partition strategies, and schema registry expectations for downstream teams.
  - Provide notes for downstream systems on deserialization patterns and performance characteristics.
  - _Requirements: 4, 5, 9_
