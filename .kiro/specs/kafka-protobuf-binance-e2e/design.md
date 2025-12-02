# Kafka Protobuf Binance E2E - Technical Design

## 1. Purpose and Scope

This design describes how to validate an end-to-end pipeline from Binance public market data (REST + WebSocket) through Cryptofeed normalization into Kafka Protobuf topics, using the existing Kafka backend and normalized Protobuf schemas.

**In Scope**
- Orchestrating a `FeedHandler` with a Binance feed that:
  - Subscribes to public `TRADES` (and optionally `L2_BOOK`) channels
  - Normalizes raw messages into `cryptofeed.types` dataclasses
  - Routes events into the Kafka backend via `KafkaProtobufCallback`
- Using a local Redpanda cluster (via `docker/infra/base.yml`) as the Kafka test environment
- Consuming produced messages using `confluent_kafka.Consumer` and decoding them with generated Protobuf bindings under `cryptofeed.proto_bindings`
- Implementing pytest-based integration tests that are opt-in and robust to missing Docker/network prerequisites

**Out of Scope**
- Changes to core Binance normalization logic beyond what is required to exercise the existing code path
- Changes to the Kafka backend’s production behavior (topic strategies, partitioning, headers) outside test configuration
- Downstream consumer storage/analytics; consumers remain responsible for deserialization and persistence

This design assumes the Kafka backend and Protobuf serialization specs are already production-ready.

## 1.5 Compound Workstreams & Boundaries

- **Workstream Decomposition**:
  - Kafka backend and Protobuf serialization are owned by their existing specs and treated as upstream streams.
  - This spec owns only the Binance→FeedHandler→KafkaProtobufCallback→Redpanda→Consumer test harness and assertions.
  - Schema parity and protobuf message shape are validated by separate specs and reused here via imports and assertions.
- **Interfaces & Contracts**:
  - Exchange connector contract: Binance produces normalized `cryptofeed.types.*` events via `message_handler`, `_trade`, `_book`, and `_snapshot`.
  - Kafka backend contract: `KafkaProtobufCallback` accepts normalized events, applies topic/partition strategies, adds headers, and produces protobuf bytes.
  - Schema contract: Generated bindings under `cryptofeed.proto_bindings.*_pb2` define the protobuf message shapes used for decode/verify steps.
- **Synchronization Points**:
  - Topic naming, partitioning behavior, and header semantics are taken from the Kafka backend spec; this design only asserts that actual runtime behavior matches those contracts when exercised via Binance.
  - Schema versions and field-level expectations are taken from normalized-schema and parity specs; this design only asserts that messages produced from Binance respect those expectations.

## 2. High-Level Architecture

### 2.1 Data Flow Overview

The E2E tests will exercise the following path:

1. **Binance WebSocket / REST**
   - WebSocket: depth, trades, and other channels according to Binance’s public API
   - REST: order book snapshot for `L2_BOOK` via `_snapshot()` in `Binance`
2. **Cryptofeed Exchange Connector** (`cryptofeed/exchanges/binance.py`)
   - Parses raw JSON messages
   - Normalizes into `cryptofeed.types.Trade`, `OrderBook`, etc.
   - Calls `self.callback(channel, dataclass, timestamp)` / `book_callback` for books
3. **FeedHandler and Callback Wiring** (`cryptofeed/feedhandler.py`)
   - `FeedHandler` is initialized with its own `Config`
   - A `Binance` feed is added with a callbacks mapping that includes `KafkaProtobufCallback` for relevant channels
4. **Kafka Backend** (`cryptofeed.backends.kafka`)
   - `KafkaProtobufCallback` extends `KafkaCallback` and:
     - Forces `serialization_format="protobuf"`
     - Uses `serialize_to_protobuf(obj)` to encode dataclasses
     - Enriches messages with routing headers via `HeaderEnricher`
     - Uses topic/partition strategies already defined in the Kafka backend
   - Messages are produced to Redpanda over the configured bootstrap servers
5. **Kafka Consumer Test Harness**
   - A test helper uses `confluent_kafka.Consumer` to:
     - Subscribe to a target topic
     - Consume one or more messages within a timeout
     - Collect headers, key, topic, and value bytes
6. **Protobuf Decoding and Assertions**
   - Tests decode payloads using generated bindings (e.g. `trade_pb2.Trade`)
   - Tests assert on header values (`exchange`, `symbol`, `data_type`, `schema_version`, `content-type`, `cf.serialization_format`)
   - Tests assert on selected payload fields that must match normalized dataclasses

### 2.2 Component Diagram

```
Binance WS / REST
      |
      v
cryptofeed.exchanges.Binance
  (message_handler, _trade, _book, _snapshot)
      |
      v
FeedHandler
  (callbacks={TRADES: [KafkaProtobufCallback], ...})
      |
      v
KafkaProtobufCallback (cryptofeed.backends.kafka.protobuf_callback)
  - serialize_to_protobuf(obj)
  - HeaderEnricher (exchange, symbol, data_type, schema_version, content-type)
  - topic / partition strategy (per_symbol or consolidated)
      |
      v
Redpanda (docker/infra/base.yml)
      |
      v
confluent_kafka.Consumer (test harness)
  - subscribe(topic)
  - poll() → value + headers
      |
      v
cryptofeed.proto_bindings.*_pb2
  - ParseFromString
  - field assertions
```

## 3. Test Harness Design

### 3.1 Redpanda Cluster Fixture

The design reuses the Redpanda setup pattern from `tests/integration/kafka/test_kafka_protobuf_e2e.py`:

- Compose file: `docker/infra/base.yml`
- Helper `_docker_compose_available()` determines whether `docker compose` is installed
- Session-scoped `redpanda` fixture:
  - Runs `docker compose -f docker/infra/base.yml up -d`
  - Waits for `localhost:19092` to be reachable
  - Yields the bootstrap address (e.g. `"localhost:19092"`)
  - Tears down the environment with `docker compose down`

For this spec, we will:
- Either reuse the existing `redpanda` fixture by importing from `test_kafka_protobuf_e2e.py`, or
- Extract a shared fixture into a Kafka integration conftest module if reuse justifies it.

The fixture remains **guarded** by Docker availability checks; if Docker or the compose file are unavailable, tests skip.

### 3.2 Binance Feed + FeedHandler Setup

Tests require a harness that:

1. Constructs a `FeedHandler` with a minimal configuration (default `Config` or a trimmed YAML if needed).
2. Creates a `KafkaProtobufCallback` instance configured for the Redpanda cluster:
   - `bootstrap_servers=[redpanda_bootstrap]`
   - `metrics_exporter=None` and `metrics_enabled=False` (to reduce noise in tests)
   - `producer_factory=None` (real producer path)
   - Optionally sets internal attributes already used in tests, such as:
     - `_topic_strategy = "per_symbol"`
     - `_partitioner` via `PartitionerFactory.create("round_robin")` for keyless tests
3. Adds a Binance feed to the handler with callbacks wired to the Kafka backend.

Example wiring pattern (conceptual):

```python
fh = FeedHandler(config=None)

kafka_cb = KafkaProtobufCallback(
    bootstrap_servers=[redpanda_bootstrap],
    metrics_exporter=None,
    metrics_enabled=False,
)
# Route trades to KafkaProtobufCallback
from cryptofeed.defines import TRADES

binance_feed = Binance(
    channels=[TRADES],
    symbols=["BTC-USDT"],
    callbacks={TRADES: [kafka_cb]},
)

fh.add_feed(binance_feed)
```

The tests will:
- Start the feed via `feed.start(loop)` on a test event loop rather than calling `fh.run()` indefinitely
- Allow the feed to run for a bounded period while monitoring Kafka for messages
- Shut down the feed and handler using the existing `feed.shutdown()` / `FeedHandler.stop()` mechanisms

### 3.3 Kafka Consumer Helper

A helper function similar to `_consume_one` in `test_kafka_protobuf_e2e.py` will be used:

- Input: `bootstrap` (Redpanda address), `topic`, `timeout_s`
- Steps:
  - Create a `Consumer` with:
    - `bootstrap.servers=bootstrap`
    - `group.id` unique to the test suite (e.g. "cf-e2e-binance-proto")
    - `auto.offset.reset="earliest"`
  - Subscribe to `[topic]`
  - Poll until a message arrives or the timeout expires
  - Close the consumer
  - Convert headers into a `dict[bytes, bytes]` mapping for assertions
- Output: a simple dataclass `_ConsumedRecord` with `value`, `headers`, `topic`, and `key`.

This helper is **test-only** and scoped to Kafka integration tests under `tests/integration/kafka`.

### 3.4 Proxy Configuration Layering (HTTP + WebSocket)

- Tests SHALL allow proxy-enabled runs by loading `ProxySettings` (env prefix `CRYPTOFEED_PROXY_`, nested `__`) before FeedHandler startup; precedence remains env > YAML > programmatic.
- Binance WS and REST transports SHALL use the existing proxy injector; when proxies are configured, tests MAY assert resolution via `get_proxy_injector().get_http_proxy_url('binance')` / WS equivalent to confirm routing without requiring live proxy endpoints.
- Proxy pools (e.g., `...__POOL__PROXIES__0__URL`) SHALL be accepted; selection strategy (round_robin/default) must not crash, and a proxy entry MUST be returned for Binance when a pool is configured.
- SOCKS WS paths REQUIRE `python-socks`; if missing and a SOCKS proxy is configured, tests SHALL skip with a clear reason instead of failing.
- Direct path MUST remain the default when no proxy config is provided; proxy assertions MUST NOT break existing direct-mode runs.

## 4. Test Cases


### 4.1 Binance Trade Roundtrip (Live WS)

**Objective**: Validate that a live Binance trade flows through the full pipeline into a Kafka Protobuf topic with correct headers and payload.

- **Preconditions**:
  - Docker and `docker compose` available (Redpanda ready)
  - Network access to Binance public WebSocket endpoint
- **Steps**:
  1. Start Redpanda using the `redpanda` fixture.
  2. Create a `KafkaProtobufCallback` configured for `per_symbol` topics.
  3. Create a Binance feed subscribed to `TRADES` for `BTC-USDT` (or another liquid symbol).
  4. Attach `KafkaProtobufCallback` as the trade callback.
  5. Start the feed via `feed.start(loop)` and run the loop for a bounded window (e.g., up to 30 seconds) while polling Kafka.
  6. Consume a single message from the expected topic (e.g., `cryptofeed.trade.binance.btc-usdt`).
  7. Decode the payload with `trade_pb2.Trade` and assert core fields (exchange, symbol, price, amount) are populated and consistent with expectations.
  8. Assert Kafka headers include the required metadata and that `schema_version` matches `SCHEMA_VERSION` from `cryptofeed.backends.protobuf.bindings`.

- **Markers**:
  - `@pytest.mark.integration`
  - `@pytest.mark.kafka`
  - `@pytest.mark.binance_live`

### 4.2 Binance Trade Roundtrip with Round-Robin Partitioner

**Objective**: Validate that round-robin partitioning works with `KafkaProtobufCallback` in the Binance pipeline and produces `None` keys.

- **Differences from 4.1**:
  - After constructing `KafkaProtobufCallback`, set `cb._partitioner = PartitionerFactory.create("round_robin")`.
  - The test checks that the consumed record’s key is `None`.

### 4.3 Binance L2 Order Book Snapshot (Optional)

**Objective**: Exercise the order book path that involves a REST snapshot and WS deltas, verifying that at least one Protobuf-encoded order book snapshot reaches Kafka.

- **Preconditions**:
  - Same as trade tests, plus stable REST access to Binance
- **Steps (high level)**:
  1. Configure Binance feed for `L2_BOOK` with a small symbol set.
  2. Attach `KafkaProtobufCallback` for `L2_BOOK` events.
  3. Wait for `_snapshot()` to be called and the first L2 snapshot to be produced.
  4. Consume a message from the expected order book topic and decode with the order book Protobuf binding.
  5. Assert core fields (exchange, symbol) and presence of at least one bid/ask.
  6. If timeouts or REST failures are detected, mark the test as skipped with a clear reason.

This scenario may be implemented as a second phase once trade roundtrip tests are stable.

## 5. Test Execution and Skipping Strategy

### 5.1 Environment Flags and Markers

- Tests will be marked with `@pytest.mark.binance_live` to distinguish them from existing Kafka integration tests.
- Execution will be controlled via an environment variable, for example:
  - `CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true`
- If the env var is not set, tests must call `pytest.skip()` during setup with an explanatory message.

### 5.2 Docker and Network Checks

- Before starting Redpanda, the tests will reuse `_docker_compose_available()` to confirm `docker compose` is available.
- If Docker or the compose file is missing, tests skip early.
- Network connectivity issues to Binance are treated as reasons to skip, not as hard failures, to maintain CI stability.

## 6. File and Module Layout

New test code added as part of this spec will follow existing Kafka integration patterns:

- **New integration test module**:
  - `tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py`
- **Contents**:
  - Imports for `FeedHandler`, `Binance`, `KafkaProtobufCallback`, `KafkaQueuedMessage` if needed, `PartitionerFactory`, and Protobuf bindings.
  - Reuse or import of the `redpanda` fixture from `test_kafka_protobuf_e2e.py`, or a shared fixture if refactored.
  - Test functions for:
    - `test_binance_trade_roundtrip_live`
    - `test_binance_trade_roundtrip_round_robin_keyless`
    - Optional `test_binance_orderbook_snapshot_roundtrip`

The design explicitly avoids creating new production modules; all changes are confined to test files and spec documentation.

## 7. Alignment with Existing Specs

- **Kafka backend behavior**: All topic naming, partitioning, and header expectations are derived from `market-data-kafka-producer` and `kafka-backend-maintenance` specifications.
- **Protobuf serialization**: All Protobuf payloads are produced via the helpers defined by `protobuf-callback-serialization`, and any serialization errors use those exception types.
- **Schema parity**: Assertions on payload fields remain consistent with `schema-parity-hardening`; this spec does not redefine precision or field-level requirements.

## 7.5 AI Agent Design Guidance

- AI agents implementing this design MUST:
  - Use FeedHandler, Binance, and KafkaProtobufCallback exactly as described here and in their owning specs, without introducing test-only branches or alternate code paths in production modules.
  - Reuse the existing Redpanda Docker compose stack and `redpanda` fixture instead of adding new Kafka infra for this spec.
  - Keep all new code changes confined to tests and spec-aligned documentation; any changes to shared backends, schemas, or exchange connectors MUST be coordinated with their respective specs.
- When cross-spec or cross-stream behavior needs to change (e.g., header semantics, topic naming, schema fields), agents SHALL:
  - Update or create the relevant owning spec first (e.g., `market-data-kafka-producer`, `protobuf-callback-serialization`, `normalized-data-schema-crypto`, `schema-parity-hardening`).
  - Reference that spec by name and ID in commit messages and documentation to maintain traceability.

## 8. Success Criteria

The design is considered successfully implemented when:

1. At least one Binance trade E2E test passes end-to-end (Binance WS → normalized `Trade` → Kafka Protobuf → Protobuf decode) against a local Redpanda instance.
2. Kafka headers and payload fields observed in Binance E2E tests match the expectations already validated by `test_kafka_protobuf_e2e.py` and Kafka backend specs.
3. Binance E2E tests are fully opt-in, clearly marked, and skip cleanly when prerequisites (Docker, network, env var) are not met.
4. No changes to core production paths (Binance connector, Kafka backend) are required beyond configuration and test wiring.
