# Requirements Document

## Project Description (Input)
Pluggable filter pipeline architecture for screening cryptocurrency symbols based on real-time market data. Users compose linear or branching filter chains to build custom screening strategies. Filtered universes published to Kafka topics for downstream consumption.

## Introduction

This specification defines a **universe screener pipeline extension** for Cryptofeed that enables real-time cryptocurrency symbol screening through composable filter chains. The extension operates as a standalone package (`cryptofeed_screener`) that consumes normalized market data and publishes filtered symbol universes to Kafka topics, maintaining strict architectural boundaries with zero modifications to core Cryptofeed code.

The implementation follows a **simplified architecture** emphasizing linear filter chains, filter-owned state, stdlib usage, and JSON serialization. This approach reduces complexity from an estimated 3,000 LOC to 500-800 LOC while maintaining production-grade reliability and performance.

## Requirements

### Requirement 1: Extension Package Architecture
**Objective:** As a Cryptofeed developer, I want a standalone extension package for universe screening, so that I can add screening capabilities without modifying core Cryptofeed code.

#### Acceptance Criteria
1. WHEN a developer installs the extension package THEN the Screener Pipeline SHALL install as an independent Python package with `cryptofeed` as a declared dependency
2. WHEN the extension package imports Cryptofeed types THEN the Screener Pipeline SHALL import `Trade`, `Ticker`, `OrderBook`, `AggregateCallback`, and `KafkaCallback` from core Cryptofeed without modification
3. WHERE code changes are proposed to the extension package THE Screener Pipeline SHALL enforce zero modifications to files under `cryptofeed/` directory via automated CI checks
4. IF a pull request modifies any file in `cryptofeed/` directory THEN the CI pipeline SHALL fail with an explicit extension boundary violation message
5. WHEN the extension package is deployed THEN the Screener Pipeline SHALL verify compatibility with Cryptofeed versions ≥2.5.0 via semantic versioning constraints

### Requirement 2: Linear Filter Pipeline Execution
**Objective:** As a quant researcher, I want to compose linear filter chains from YAML configuration, so that I can build custom screening strategies without writing code.

#### Acceptance Criteria
1. WHEN a user configures a filter chain in YAML THEN the Screener Pipeline SHALL execute filters sequentially in the order specified (e.g., VolumeFilter → MomentumFilter → RSIFilter)
2. WHEN market data arrives at the pipeline THEN the Screener Pipeline SHALL broadcast each data event to all filters via their `update()` method
3. IF a filter's `update()` method raises an exception THEN the Screener Pipeline SHALL log the error with structured context (filter name, symbol, error message) and continue processing remaining filters
4. WHEN the emit interval elapses THEN the Screener Pipeline SHALL evaluate all tracked symbols against the filter chain using the `should_include()` method of each filter
5. IF all filters return `True` for a symbol THEN the Screener Pipeline SHALL include the symbol in the published universe result
6. WHILE the pipeline is running THE Screener Pipeline SHALL maintain a set of all symbols encountered for periodic evaluation

### Requirement 3: Filter Base Contract
**Objective:** As a filter developer, I want a simple, well-defined filter interface, so that I can implement custom screening logic with minimal complexity.

#### Acceptance Criteria
1. WHEN a developer creates a new filter THEN the filter SHALL implement the abstract `Filter` base class with `update()` and `should_include()` methods
2. WHEN market data arrives THEN the filter's `update(data)` method SHALL receive `Trade`, `Ticker`, or `OrderBook` objects and update internal state accordingly
3. WHEN the pipeline evaluates a symbol THEN the filter's `should_include(symbol: str) -> bool` method SHALL return `True` if the symbol passes the filter's criteria, `False` otherwise
4. WHERE a filter requires historical data THE filter SHALL maintain its own state using appropriate data structures (e.g., `defaultdict`, `deque`)
5. IF a filter has insufficient data to make a decision THEN the filter's `should_include()` method SHALL return `False` to exclude the symbol

### Requirement 4: Stateless Volume Filter
**Objective:** As a user, I want to filter symbols by minimum 24-hour trading volume, so that I can focus on liquid markets with sufficient trading activity.

#### Acceptance Criteria
1. WHEN the VolumeFilter is configured with a `min_volume` threshold THEN the filter SHALL track cumulative trade volume per symbol using a `defaultdict(Decimal)`
2. WHEN a `Trade` object is received THEN the VolumeFilter SHALL accumulate the trade amount (volume) to the symbol's total using `Decimal` arithmetic for precision
3. WHEN the pipeline queries `should_include(symbol)` THEN the VolumeFilter SHALL return `True` if the symbol's accumulated volume is greater than or equal to `min_volume`, otherwise `False`
4. IF a symbol has never been seen THEN the VolumeFilter SHALL return `False` (default volume is zero)
5. WHERE the data object does not have `symbol` or `amount` attributes THE VolumeFilter SHALL skip the update without raising an exception

### Requirement 5: Stateful RSI Filter with Pandas Integration
**Objective:** As a technical analyst, I want to filter symbols by RSI (Relative Strength Index) range, so that I can identify oversold or overbought conditions matching TradingView indicators.

#### Acceptance Criteria
1. WHEN the RSIFilter is initialized THEN the filter SHALL configure `min_rsi`, `max_rsi`, and `period` parameters with defaults of 30, 70, and 14 respectively
2. WHEN a `Trade` object is received THEN the RSIFilter SHALL append the trade price to a symbol-specific price history maintained as `deque(maxlen=100)` for bounded memory
3. WHEN the pipeline queries `should_include(symbol)` THEN the RSIFilter SHALL compute the RSI using pandas `.ewm()` with Wilder's smoothing (adjust=False)
4. IF the computed RSI value falls within the range `[min_rsi, max_rsi]` THEN the RSIFilter SHALL return `True`, otherwise `False`
5. IF the symbol has fewer than `period` price samples THEN the RSIFilter SHALL return `False` (insufficient data)
6. WHERE the RSI calculation produces `NaN` or `None` THE RSIFilter SHALL return `False` to exclude the symbol
7. WHEN tested against TradingView fixture data THEN the RSIFilter SHALL produce RSI values within ±0.5% of TradingView's RSI indicator for the same input

### Requirement 6: Configuration Management with Pydantic
**Objective:** As a DevOps engineer, I want to load screener configurations from YAML files with automatic validation, so that I can catch configuration errors before runtime.

#### Acceptance Criteria
1. WHEN a user loads a YAML configuration file THEN the ScreenerConfig SHALL parse the file using `yaml.safe_load()` (never `yaml.load()` to prevent code execution vulnerabilities)
2. WHEN the configuration is parsed THEN the ScreenerConfig SHALL validate field types and constraints using Pydantic v2 models with strict mode enabled
3. IF the `name` field does not match the regex pattern `^[a-z0-9_]+$` THEN the ScreenerConfig SHALL raise a validation error
4. IF the `topic` field does not match the regex pattern `^[a-z0-9._-]+$` THEN the ScreenerConfig SHALL raise a validation error
5. IF the `emit_interval` is not between 1 and 3600 seconds THEN the ScreenerConfig SHALL raise a validation error
6. IF the `filters` list is empty or contains more than 100 filters THEN the ScreenerConfig SHALL raise a validation error
7. WHEN the `build_filters()` method is called THEN the ScreenerConfig SHALL instantiate filter objects using an allowlist registry pattern (never `eval()` or `__import__()`)
8. IF a filter type is not in the registry THEN the ScreenerConfig SHALL raise a descriptive error identifying the unknown filter type

### Requirement 7: Kafka Output Integration
**Objective:** As a downstream consumer, I want filtered symbol universes published to Kafka topics in JSON format, so that I can consume screening results for strategy execution or dashboards.

#### Acceptance Criteria
1. WHEN the ScreenerPipeline is initialized THEN the pipeline SHALL accept a `KafkaCallback` instance configured with bootstrap servers, security settings, and performance optimizations
2. WHEN the emit interval elapses THEN the ScreenerPipeline SHALL serialize the filtered universe as JSON with fields: `timestamp` (float seconds), `symbols` (list of strings), `total` (integer count)
3. WHEN publishing to Kafka THEN the ScreenerPipeline SHALL call `kafka.write(topic, json.dumps(result).encode())` with the configured topic name
4. WHERE Kafka security is required THE KafkaCallback SHALL support SASL/SCRAM authentication using environment variables for credentials (username, password)
5. IF the Kafka `write()` operation fails THEN the ScreenerPipeline SHALL log the error with structured context (topic, symbol count, error message) without crashing the pipeline
6. WHEN the pipeline publishes messages THEN the KafkaCallback SHALL use performance optimizations including `acks="all"`, `enable_idempotence=True`, `poll_batch_size=100`, and `compression_type="lz4"`

### Requirement 8: Bounded Memory and Resource Limits
**Objective:** As a production engineer, I want resource usage to be bounded and predictable, so that the screener can run reliably in memory-constrained environments.

#### Acceptance Criteria
1. WHEN stateful filters use rolling windows THEN the filters SHALL use `collections.deque(maxlen=N)` to automatically evict old data and prevent unbounded memory growth
2. WHEN the configuration specifies `max_symbols` THEN the ScreenerPipeline SHALL enforce a limit on the number of symbols tracked (default: 10,000, maximum: 100,000)
3. WHEN the configuration specifies `max_filters` THEN the ScreenerConfig SHALL reject configurations with more than 100 filters
4. IF the symbol count exceeds `max_symbols` THEN the ScreenerPipeline SHALL log a warning and either reject new symbols or evict the least recently updated symbols
5. WHILE the pipeline processes 10,000 symbols over 24 hours THE ScreenerPipeline SHALL maintain total memory usage below 1GB (measured as resident set size)

### Requirement 9: Error Handling and Fault Isolation
**Objective:** As a reliability engineer, I want filter errors to be isolated and logged without crashing the entire pipeline, so that one faulty filter doesn't disrupt other filters or the data stream.

#### Acceptance Criteria
1. WHEN a filter's `update()` method raises an exception THEN the ScreenerPipeline SHALL catch the exception and wrap it in a try/except block
2. WHEN an exception is caught THEN the ScreenerPipeline SHALL log the error using structured logging with fields: `filter_name`, `symbol`, `error_type`, `error_message`
3. IF a filter fails during update THEN the ScreenerPipeline SHALL continue processing the event through remaining filters (exception boundaries)
4. WHEN a filter's `should_include()` method raises an exception THEN the ScreenerPipeline SHALL treat the result as `False` and log the error
5. WHERE Kafka publishing fails THE ScreenerPipeline SHALL log the failure but continue accepting and processing new market data
6. WHILE the pipeline is running THE ScreenerPipeline SHALL not propagate exceptions to the Cryptofeed FeedHandler (graceful degradation)

### Requirement 10: Observability and Structured Logging
**Objective:** As an operations engineer, I want structured, queryable logs with relevant context, so that I can monitor pipeline health and debug issues in production.

#### Acceptance Criteria
1. WHEN the pipeline emits a filtered universe THEN the ScreenerPipeline SHALL log an event with structured fields: `topic`, `input_symbols` (total tracked), `passed_symbols` (filtered count), `filter_count`, `timestamp`
2. WHEN a filter update fails THEN the ScreenerPipeline SHALL log an error event with structured fields: `filter_name`, `symbol`, `error_type`, `error_message`, `timestamp`
3. WHEN the pipeline starts THEN the ScreenerPipeline SHALL log a startup event with configuration summary: `name`, `topic`, `emit_interval`, `filter_types` (list of filter class names)
4. WHEN the pipeline stops THEN the ScreenerPipeline SHALL log a shutdown event with runtime statistics: `total_updates`, `total_emits`, `uptime_seconds`
5. WHERE structured logging is enabled THE ScreenerPipeline SHALL use `structlog` for JSON-formatted logs compatible with log aggregation systems (e.g., ELK, Splunk)

### Requirement 11: Test Coverage and NO MOCKS Principle
**Objective:** As a quality engineer, I want comprehensive test coverage using real implementations, so that tests validate actual behavior rather than mock interactions.

#### Acceptance Criteria
1. WHEN the test suite runs THEN the test coverage SHALL achieve ≥85% line coverage across all modules (pipeline, filters, config)
2. WHEN testing filter logic THEN the tests SHALL use real `Trade`, `Ticker`, and `OrderBook` fixture objects (not mocks)
3. WHEN testing Kafka integration THEN the integration tests SHALL use real Kafka or Redpanda via docker-compose (not mocked KafkaCallback)
4. WHEN testing RSI calculations THEN the tests SHALL validate against TradingView fixture data with ±0.5% tolerance for numerical accuracy
5. IF a filter implementation changes THEN the tests SHALL detect behavioral regressions without requiring test updates (tests validate behavior, not implementation)
6. WHEN the test suite completes THEN all tests SHALL pass with 100% success rate (zero flaky tests)
7. WHERE performance is critical THE test suite SHALL include benchmark tests measuring filter latency (<1ms per symbol) and pipeline throughput (≥10,000 symbols processed per emit cycle)

### Requirement 12: Extension Boundary CI Enforcement
**Objective:** As a project maintainer, I want automated CI checks to prevent accidental modifications to core Cryptofeed code, so that the extension boundary remains intact across all contributions.

#### Acceptance Criteria
1. WHEN a pull request is submitted THEN the CI pipeline SHALL run a git diff check comparing changed files against the `cryptofeed/` directory pattern
2. IF any file matching the pattern `cryptofeed/**/*` is modified THEN the CI check SHALL fail with exit code 1 and an error message: "Extension boundary violated: core Cryptofeed files modified"
3. IF no core files are modified THEN the CI check SHALL pass with exit code 0
4. WHEN the CI check is implemented THEN the workflow SHALL use the command: `git diff origin/master --name-only | grep '^cryptofeed/' && exit 1 || exit 0`
5. WHERE the check fails THE CI pipeline SHALL block the pull request from merging until core file changes are reverted

### Requirement 13: Documentation and User Guidance
**Objective:** As a new user, I want clear, comprehensive documentation with examples, so that I can configure and run a screener pipeline in less than 10 minutes.

#### Acceptance Criteria
1. WHEN a user reads the README THEN the documentation SHALL provide a quick start guide covering: installation (`pip install cryptofeed-screener`), configuration (YAML example), and execution (Python script)
2. WHEN a user needs configuration reference THEN the documentation SHALL list all filter types with parameter descriptions, defaults, and valid ranges
3. WHEN a user wants to build a custom filter THEN the documentation SHALL provide a step-by-step guide with code examples showing how to subclass `Filter` and implement `update()` and `should_include()` methods
4. WHERE Kafka integration is required THE documentation SHALL provide a complete example showing KafkaCallback initialization with security settings (SASL/SCRAM)
5. WHEN a user examines the examples directory THEN the documentation SHALL include at least one complete, runnable example demonstrating a momentum screener (Volume → Price Change → RSI)
6. IF a user encounters an error THEN the documentation SHALL include a troubleshooting section covering common issues: YAML syntax errors, Kafka connection failures, insufficient data for indicators

### Requirement 14: Graceful Lifecycle Management
**Objective:** As a system operator, I want the screener pipeline to start, run, and stop gracefully, so that deployments and restarts do not lose data or leave resources hanging.

#### Acceptance Criteria
1. WHEN the FeedHandler starts THEN the ScreenerPipeline SHALL initialize all filters and begin tracking symbols from the first market data event
2. WHEN the FeedHandler stops THEN the ScreenerPipeline SHALL perform a final emit if the emit interval has not elapsed (flush pending results)
3. WHERE the pipeline is running THE ScreenerPipeline SHALL accept shutdown signals (SIGTERM, SIGINT) and complete gracefully within 5 seconds
4. WHEN the pipeline shuts down THEN the ScreenerPipeline SHALL call `await kafka.stop()` to flush any buffered Kafka messages before terminating
5. IF the FeedHandler is restarted THEN the ScreenerPipeline SHALL rebuild filter state from incoming market data (no persistent state required for MVP)
6. WHILE the pipeline is shutting down THE ScreenerPipeline SHALL not accept new market data events after the shutdown signal is received

### Requirement 15: Performance and Latency Targets
**Objective:** As a performance engineer, I want predictable, low-latency screening performance, so that the pipeline can handle high-volume market data streams without introducing significant delay.

#### Acceptance Criteria
1. WHEN a single filter processes a symbol THEN the stateless filter (VolumeFilter, MomentumFilter) SHALL complete `should_include()` in <0.1 microseconds (measured via pytest-benchmark)
2. WHEN a stateful filter processes a symbol THEN the filter (RSIFilter, MACDFilter) SHALL complete `should_include()` in <0.5 microseconds including pandas calculations
3. WHEN the pipeline evaluates 10,000 symbols through a 10-filter chain THEN the total evaluation time SHALL be <10 milliseconds (1ms per symbol target)
4. WHEN the pipeline publishes to Kafka THEN the publish latency SHALL be <100ms at p99 under normal load (measured from emit trigger to Kafka acknowledge)
5. WHILE the pipeline processes 100,000 trades per second THE ScreenerPipeline SHALL maintain filter update latency below 1ms per update at p99
6. WHERE memory is constrained THE ScreenerPipeline SHALL operate within 500MB for 1,000 symbols and scale linearly to 1GB for 10,000 symbols

### Requirement 16: Security Hardening
**Objective:** As a security engineer, I want the screener extension to follow secure coding practices, so that configuration loading, filter instantiation, and Kafka integration do not introduce vulnerabilities.

#### Acceptance Criteria
1. WHEN loading YAML configuration THEN the ScreenerConfig SHALL exclusively use `yaml.safe_load()` to prevent arbitrary code execution via YAML deserialization
2. WHEN instantiating filters from configuration THEN the ScreenerConfig SHALL use an allowlist registry pattern matching filter type strings to predefined classes (no dynamic imports via `eval()`, `exec()`, or `__import__()`)
3. WHEN connecting to Kafka THEN the KafkaCallback SHALL support SASL/SCRAM authentication with credentials loaded from environment variables (never hardcoded in config files)
4. WHERE TLS is required THE KafkaCallback SHALL support `security_protocol="SASL_SSL"` with certificate verification enabled
5. IF invalid input is detected during config parsing THEN the ScreenerConfig SHALL raise descriptive validation errors without exposing internal system details
6. WHEN processing user-provided symbol names THEN the ScreenerPipeline SHALL validate symbols against a safe character set (alphanumeric, dash, underscore) to prevent injection attacks

## Traceability Matrix

| Requirement ID | EARS Pattern | Priority | Validation Method |
|----------------|--------------|----------|-------------------|
| REQ-1 | Event-Driven, State-Based | CRITICAL | CI boundary check, unit tests |
| REQ-2 | Event-Driven, State-Based | HIGH | Unit + integration tests |
| REQ-3 | Event-Driven | HIGH | Unit tests with fixtures |
| REQ-4 | Event-Driven | MEDIUM | Unit tests with Trade fixtures |
| REQ-5 | Event-Driven, State-Based | HIGH | Unit tests + TradingView validation |
| REQ-6 | Event-Driven, State-Based | HIGH | Unit tests + fuzzing |
| REQ-7 | Event-Driven | HIGH | Integration tests (docker-compose Kafka) |
| REQ-8 | Continuous Behavior | HIGH | Load tests + memory profiling |
| REQ-9 | Event-Driven | CRITICAL | Chaos engineering tests |
| REQ-10 | Event-Driven | MEDIUM | Log output validation |
| REQ-11 | Continuous Behavior | CRITICAL | pytest-cov + code review |
| REQ-12 | Event-Driven | CRITICAL | CI workflow |
| REQ-13 | State-Based | MEDIUM | Manual review + user testing |
| REQ-14 | Event-Driven | HIGH | Integration tests + signal tests |
| REQ-15 | Continuous Behavior | MEDIUM | pytest-benchmark + load tests |
| REQ-16 | Event-Driven, State-Based | CRITICAL | Security audit + static analysis |

## Success Criteria Summary

The universe screener pipeline extension will be considered complete when:

1. **Architectural Integrity**: Zero modifications to `cryptofeed/` directory (enforced by CI)
2. **Functional Completeness**: All 16 requirements validated through 85%+ test coverage
3. **Performance Targets**: <1ms filter latency, <10ms pipeline evaluation for 10K symbols
4. **Production Readiness**: Documentation enables 10-minute setup, graceful lifecycle management
5. **Security Compliance**: YAML safe loading, filter allowlist registry, Kafka SASL/SCRAM support

## Out of Scope

The following capabilities are explicitly excluded from this specification:

1. **DAG Infrastructure**: No directed acyclic graph execution (linear chains only)
2. **Branching Pipelines**: No multi-output support (one pipeline = one topic; users run multiple pipelines for multiple strategies)
3. **State Persistence**: No filter state saving/restoration across restarts
4. **Protobuf Serialization**: JSON output only (protobuf deferred to future)
5. **Custom Indicator Classes**: Use pandas/talib directly (no RSICalculator, MACDCalculator wrappers)
6. **MetricsAggregator**: Filters own their state (no centralized aggregator)
7. **Pre-built Templates**: Configuration templates deferred to Phase 5 (post-production feedback)
8. **Web UI**: No visual pipeline builder or monitoring dashboard (logs + Kafka consumer suffice)
9. **Backtesting**: No historical simulation mode (use separate backtesting tools)
10. **Auto-optimization**: No automatic filter reordering or performance tuning

## Dependencies

**Hard Dependencies**:
- cryptofeed ≥2.5.0 (stable AggregateCallback interface)
- confluent-kafka ≥2.0.0 (Kafka client)
- pydantic ≥2.0.0 (config validation)
- PyYAML ≥6.0 (YAML parsing)
- pandas ≥1.24.0 (indicator calculations)
- structlog ≥23.0.0 (structured logging)

**Development Dependencies**:
- pytest ≥7.0.0
- pytest-asyncio ≥0.21.0
- pytest-cov ≥4.0.0
- pytest-benchmark ≥4.0.0
- docker-compose (for Kafka integration tests)

## Risk Mitigation

| Risk | Mitigation Strategy (Requirements Coverage) |
|------|---------------------------------------------|
| Extension boundary violation | REQ-12 (CI enforcement), REQ-1 (package architecture) |
| YAML code execution | REQ-6 (yaml.safe_load), REQ-16 (security hardening) |
| Filter code injection | REQ-6 (allowlist registry), REQ-16 (no dynamic imports) |
| Unbounded memory growth | REQ-8 (deque maxlen, max_symbols limit) |
| Silent filter failures | REQ-9 (exception boundaries, structured logging) |
| Numerical instability (RSI/MACD) | REQ-5 (pandas .ewm, TradingView validation) |
| Kafka authentication bypass | REQ-7 (SASL/SCRAM), REQ-16 (environment variables) |
| Poor test coverage | REQ-11 (85%+ coverage, NO MOCKS) |
