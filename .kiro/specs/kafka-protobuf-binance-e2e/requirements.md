# Kafka Protobuf Binance E2E - Requirements

## Introduction

Validate an end-to-end ingestion pipeline that starts from Binance public market data (REST + WebSocket), normalizes events through existing `cryptofeed` exchange connectors, and publishes Protobuf-serialized messages to Kafka via the modern Kafka backend, using normalized protobuf schemas and a Redpanda-based integration environment.

This specification focuses on **validation and test infrastructure** rather than new production features. All runtime behavior must align with existing specifications:
- `market-data-kafka-producer` (Kafka backend behavior)
- `protobuf-callback-serialization` (Protobuf serialization helpers)
- `normalized-data-schema-crypto` (normalized protobuf schemas)

The goal is to provide high-confidence, reproducible tests that prove the Binance→Kafka Protobuf path behaves as designed, without introducing new ingestion responsibilities beyond Kafka topic publication.

## Functional Requirements

### FR1: End-to-End Trade Pipeline (Binance → Kafka Protobuf)

As an engineer, I want a live-path integration test that exercises Binance trades from WebSocket through normalization to Kafka Protobuf topics, so that we can validate the full production path with realistic data.

**Acceptance Criteria**
1. WHEN a Binance `TRADES` channel is subscribed via `FeedHandler` and `Binance` THEN at least one `Trade` event SHALL be produced to a Kafka topic using `KafkaProtobufCallback`.
2. WHEN the produced Kafka message is consumed via a Kafka client THEN the payload SHALL decode successfully using the generated trade Protobuf bindings and match core fields from the normalized `Trade` dataclass (exchange, symbol, side, amount, price, timestamp, id where present).
3. WHEN the message headers are inspected THEN they SHALL include at minimum `content-type`, `exchange`, `symbol`, `data_type`, `schema_version`, and `cf.serialization_format`, with values consistent with existing Kafka Protobuf E2E tests.
4. WHILE the test runs under a healthy Redpanda + Binance environment THEN it SHALL complete within a bounded timeout (e.g., ≤ 60 seconds) without flakiness under normal network conditions.

### FR2: L2 Order Book Pipeline (REST Snapshot + WS Deltas → Kafka Protobuf) *(Optional but Recommended)*

As an engineer, I want the Binance order book path (REST snapshot + WebSocket deltas) to be exercised through Kafka Protobuf, so that we validate a more complex path that relies on both HTTP and WS layers.

**Acceptance Criteria**
1. WHEN a Binance feed is configured for `L2_BOOK` with the existing snapshot+delta mechanism THEN at least one normalized `OrderBook` snapshot or delta event SHALL be published to Kafka via `KafkaProtobufCallback`.
2. WHEN the corresponding Kafka message is consumed and decoded with the order book Protobuf bindings THEN the top-of-book levels (best bid/ask) and symbol/exchange fields SHALL match the normalized `OrderBook` instance used by the backend.
3. IF a REST snapshot fails during the test environment setup (HTTP error, timeout) THEN the test SHALL skip gracefully with a clear reason rather than failing unpredictably.

### FR3: Topic and Partition Strategy Alignment

As an operator, I want the Binance E2E tests to respect the Kafka topic and partitioning strategies defined by the Kafka backend specs, so that the tests remain valid as configuration evolves.

**Acceptance Criteria**
1. WHEN the E2E test configures `KafkaProtobufCallback` for per-symbol routing THEN trade events originating from Binance SHALL land on topics that match the established naming pattern (e.g., `cryptofeed.trade.binance.btc-usdt`), consistent with existing Protobuf E2E tests.
2. WHEN the E2E test configures the partitioner as round-robin for at least one scenario THEN the produced Kafka record key SHALL be `None`, matching existing round-robin semantics tests.
3. WHEN consolidated topics are used in future extensions of this spec THEN the E2E tests SHALL treat topic names as configuration, not constants, and SHALL continue to validate headers for routing metadata (exchange, symbol, data_type) rather than hard-coding topic layouts.

### FR4: Protobuf Schema and Serialization Parity

As a quality engineer, I want the Binance E2E tests to reuse existing Protobuf schemas and serialization helpers, so that they validate parity rather than introducing parallel serialization logic.

**Acceptance Criteria**
1. WHEN Kafka messages from the Binance pipeline are decoded THEN they SHALL use the existing generated Protobuf bindings under `cryptofeed.proto_bindings` for the appropriate data type (e.g., `trade_pb2.Trade`).
2. WHEN payload fields are compared in tests THEN numeric and timestamp fields SHALL match the semantics validated by `schema-parity-hardening` tests (precision, timestamp units, required/optional fields), without re-defining parity rules in this spec.
3. WHEN serialization errors occur within `serialize_to_protobuf` for Binance-derived events during tests THEN the system SHALL surface `ProtobufEncodeError` or the existing serialization exception types, and the test SHALL assert on those errors rather than introducing new exception classes.

### FR5: Environment-Controlled Execution

As a CI maintainer, I want the Binance E2E tests to be clearly marked and skippable, so that pipelines can run quickly by default while still allowing full E2E validation when desired.

**Acceptance Criteria**
1. WHEN the test suite is run without explicit opt-in for live Binance E2E THEN all Binance→Kafka Protobuf integration tests SHALL be skipped, with a clear skip reason referencing the required environment conditions.
2. WHEN a designated marker (e.g., `@pytest.mark.binance_live`) and/or environment variable (e.g., `CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true`) is set THEN the Binance E2E tests SHALL execute, assuming Docker and network prerequisites are satisfied.
3. IF Docker or `docker compose` is unavailable in the environment THEN the tests SHALL skip with a message aligned with the existing Redpanda E2E tests (reusing or matching the `_docker_compose_available` check behavior).

### FR6: Non-Invasive Use of Existing Backends and Connectors

As a maintainer, I want the E2E tests to exercise existing production code paths (Binance exchange connector, FeedHandler, KafkaProtobufCallback) without intrusive changes, so that the tests remain stable as long as public APIs are stable.

**Acceptance Criteria**
1. WHEN implementing the E2E tests THEN they SHALL instantiate `FeedHandler`, `Binance`, and `KafkaProtobufCallback` using public configuration surfaces (constructor arguments, callbacks mapping, config objects) rather than modifying core modules to add test-only hooks.
2. IF minor test-only configuration overrides are required (e.g., per-symbol topic strategy, disabling metrics) THEN they SHALL be applied via configuration arguments or attribute overrides already used in existing tests (such as `test_kafka_protobuf_e2e.py`).
3. WHILE this spec is implemented THEN it SHALL not introduce new public APIs solely for test control unless coordinated with `kafka-backend-maintenance` or `market-data-kafka-producer` specs.

### FR7: Proxy-Aware Execution (HTTP + WebSocket)

As an operator, I want the Binance E2E harness to honor the proxy system (including pools) so we can validate or reproduce proxy-routed runs without breaking the direct path.

**Acceptance Criteria**
1. WHEN `CRYPTOFEED_PROXY_*` env vars or `ProxySettings`/`FeedHandler(proxy_settings=...)` are provided THEN the E2E setup SHALL load them (env > YAML > programmatic) and apply them to Binance HTTP and WebSocket connections via the existing proxy injector.
2. WHEN a SOCKS or HTTP proxy is configured for Binance WS/REST THEN the test harness SHALL assert proxy resolution (e.g., via `get_proxy_injector().get_http_proxy_url('binance')` / `get_websocket_proxy_url`) and skip with a clear message if the required dependency (`python-socks` for SOCKS WS) is unavailable.
3. WHEN a proxy pool is provided (e.g., `CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__POOL__PROXIES__0__URL=...`) THEN the harness SHALL accept it without crash and verify that a proxy entry is selected (round-robin or configured strategy) for Binance connections.
4. IF no proxy configuration is provided THEN the E2E tests SHALL continue to run direct and MUST NOT regress existing direct-path behavior or skip conditions.
5. Symbol metadata bootstrap (`exchangeInfo` / `symbol_mapping`) and private listen-key acquisition/refresh MUST honor ProxySettings and use non-blocking, timeout-bound HTTP calls (aiohttp or equivalent); sync `requests` fallbacks that bypass the ProxyInjector are disallowed for Binance paths.

### FR8: Topic Auto-Provision for E2E

As an engineer, I want the Binance Kafka E2E harness to auto-provision required Kafka topics in Redpanda so runs are reproducible without manual setup.

**Acceptance Criteria**
1. GIVEN a Redpanda/Kafka bootstrap address WHEN the E2E starts THEN it SHALL idempotently ensure the expected topics exist (per-symbol or consolidated based on the configured strategy) before producing.
2. WHEN topic creation fails (insufficient permissions or broker offline) THEN the tests SHALL skip with a clear message, not hang or partially run.
3. Partitions/replication SHALL be configurable via env/fixture defaults (sane defaults acceptable for local Redpanda: partitions≥1, replication=1).
4. Topic names SHALL align with the configured topic strategy (per_symbol by default) and match assertions in the tests.

## Non-Functional Requirements

### NFR1: Opt-in, Skippable, Deterministic

As an engineer, I want the Binance Kafka E2E tests to be opt-in, deterministic, and skip when prerequisites are missing, to avoid flaky CI and developer frustration.

1. The Binance E2E tests SHALL include deterministic timeouts and robust skip conditions, so that intermittent external issues (network blips, rate limits) degrade into skipped tests rather than flakiness.
2. The E2E tests SHOULD minimize the amount of data required (e.g., succeed after a small number of messages) to reduce load on Binance and execution time in CI.
3. The E2E tests SHOULD avoid assumptions about specific trade activity beyond “at least one message arrives within the timeout window”.

**Notes**
- Proxies, Redpanda, and Binance WS all need to be reachable for the test to pass; missing dependencies should cause skips, not failures.
- REST symbol metadata uses `requests`; when proxies are configured, `HTTP_PROXY` / `HTTPS_PROXY` MUST be set so REST bootstrap (exchangeInfo) is proxied consistently with WS.

### NFR2: Alignment with Existing Specs and Docs

1. This spec SHALL reuse terminology and expectations from `docs/kafka/technical-specification.md`, `docs/kafka/user-guide.md`, and `docs/kafka/INTEGRATION_GUIDE.md` when describing headers, topics, and partitioning behavior.
2. Any new documentation or comments authored as part of this spec SHALL clearly reference this spec name (`kafka-protobuf-binance-e2e`) and the related Kafka/Protobuf specs where appropriate.

### NFR3: Scope Boundaries

1. This spec SHALL NOT introduce new responsibilities beyond Kafka topic publication; downstream consumer behavior (storage, analytics, retention) remains out of scope and is covered by consumer-side specs and docs.
2. This spec SHALL NOT modify the underlying Binance connector’s normalization logic except in the event of a discovered bug that prevents Protobuf serialization; such bugs, if found, SHALL be cross-referenced with the relevant core exchange or normalization specs.

## Dependencies

- **market-data-kafka-producer**: Defines Kafka backend behavior, topic/partition strategies, and monitoring expectations.
- **protobuf-callback-serialization**: Provides Protobuf serialization helpers and error semantics used by `KafkaProtobufCallback`.
- **normalized-data-schema-crypto**: Supplies the normalized Protobuf schemas referenced by `cryptofeed.proto_bindings`.
- **schema-parity-hardening**: Defines parity expectations between dataclasses and Protobuf representations; Binance E2E tests SHALL be consistent with these expectations.

## Compound Workstreams & Ownership (C.1)
- Upstream contracts: schemas (normalized-data-schema-crypto), serialization helpers (protobuf-callback-serialization), Kafka backend topics/headers/partitioning (market-data-kafka-producer). This spec only **consumes** those contracts for validation.
- Local ownership: Binance→Kafka Protobuf test harness (pytest modules under `tests/integration/kafka/`), shared Redpanda fixtures/helpers, Makefile targets for Kafka E2E, and lightweight docs describing how to run the suite.
- Downstream consumers (QuixStreams, Flink, etc.) remain out of scope; their specs own storage/analytics.

## AI Agent Boundaries (C.2)
- Allowed edit surface for this spec: `tests/integration/kafka/*` (new/updated E2E tests, fixtures, helpers), `Makefile` (Kafka/Redpanda targets), and documentation snippets referencing how to run the opt-in E2E suite (e.g., `docs/kafka/user-guide.md`).
- Disallowed without upstream spec change: core exchange normalization (`cryptofeed/exchanges/binance.py`), Kafka backend production code, protobuf schemas or serialization helpers. Any discovered defects must be linked to their owning spec before code changes.
- No new public APIs for testing; use existing FeedHandler/callback surfaces and Redpanda docker compose stack. Skips must guard missing Docker/network/env.

## Compound Engineering Alignment

- **Parallel Workstreams**:
  - Kafka backend core and topic/partition semantics (`market-data-kafka-producer`).
  - Protobuf serialization and converter registry (`protobuf-callback-serialization`).
  - Canonical normalized schemas (`normalized-data-schema-crypto`).
  - Schema regression and parity tooling (`schema-parity-hardening`).
  - This spec’s E2E validation harness (`kafka-protobuf-binance-e2e`).
- **Upstream Dependencies**:
  - This spec SHALL treat the Kafka backend, Protobuf helpers, and schemas as *fixed contracts* and MUST NOT redefine their behavior; it only verifies that Binance end-to-end flows respect those contracts.
- **Downstream Consumers**:
  - Outputs of this spec (E2E tests, fixtures, and documentation) MAY be reused by future consumer-integration specs (e.g., QuixStreams, Flink), but those specs remain responsible for downstream storage and analytics.

## AI Agentic Implementation Constraints

- AI agents implementing this spec MUST:
  - Restrict code changes to test harnesses, fixtures, and documentation explicitly referenced in this spec (e.g., `tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py`) and avoid modifying production modules (Binance connector, Kafka backend, Protobuf helpers) unless a separate spec authorizes those changes.
  - Use existing patterns for Redpanda Docker infra and Kafka Protobuf tests (e.g., reuse the `redpanda` fixture and `_consume_one` helper) rather than creating parallel infrastructure.
  - Preserve the **ingestion boundary** principle: validation stops at Kafka topic production with Protobuf payloads; any consumer/storage behaviors remain out of scope.
- If an agent discovers a bug in core components while working on this spec, it SHALL:
  - Capture the issue and link it to the appropriate owning spec (`market-data-kafka-producer`, `protobuf-callback-serialization`, or `normalized-data-schema-crypto`).
  - Avoid silently “fixing” cross-spec behavior in this spec’s implementation without updating or creating the corresponding owning spec.
