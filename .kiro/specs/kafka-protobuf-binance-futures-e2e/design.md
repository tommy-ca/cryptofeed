## kafka-protobuf-binance-futures-e2e – Design

This design extends the existing `kafka-protobuf-binance-e2e` (spot) design to cover Binance USDⓈ-M futures. It reuses the same Kafka backend, normalized schemas, and Redpanda harness, and adds futures-specific channels, semantics, and ergonomics.

The intent is to keep futures E2E behavior as close as possible to spot, so operators can interpret results consistently.

---

### 1. Data Flow Overview

The futures E2E tests exercise this path:

1. Binance USDⓈ-M futures WS/REST → `BINANCE_FUTURES` connector → normalized dataclasses.
2. `FeedHandler` with a `BINANCE_FUTURES` feed and callbacks wired to `KafkaProtobufCallback`.
3. `KafkaProtobufCallback` serializes to Protobuf, applies headers, resolves topic and partition, and produces to Redpanda.
4. A Kafka consumer helper subscribes to the expected topic and returns a record (value, headers, topic, key).
5. Tests decode the value using futures Protobuf bindings and assert headers + payload.

The Kafka backend, schema, and partitioning rules are owned by upstream specs; this design only verifies that the futures connector and harness exercise those rules correctly.

---

### 2. Topic Strategy and Naming

- Futures tests reuse the Kafka backend’s topic strategy system:
  - `KAFKA_E2E_TOPIC_STRATEGY=per_symbol` (default) or `consolidated`.
- A helper `_topic_strategy()` normalizes the env var and `_topic_name(channel, strategy)` maps channel/strategy to topic names, mirroring the spot E2E tests but with futures-specific suffixes (e.g., `btc-usdt-perp`).
- Tests use these helpers to pre-create topics and subscribe consumers.

---

### 3. Partitioning Semantics

- Default partitioner: composite strategy – non-`None` keys based on `exchange-symbol`.
- Round-robin partitioner: returns `None` keys and defers partition selection to the broker.
- Futures E2E tests validate both:
  - Standard futures tests use the default partitioner and assert keys are present.
  - A dedicated round-robin futures trade test configures `PartitionerFactory.create("round_robin")` with partition-key cache disabled and asserts `record.key is None`.

---

### 4. Offset Strategy

- To avoid cross-test contamination on shared topics, futures tests follow the spot E2E offset semantics:
  - `offset_reset="latest"` for TRADES (default and round-robin), TICKER, FUNDING, OPEN_INTEREST, LIQUIDATIONS, and multi-channel tests.
  - `offset_reset="earliest"` for the L2_BOOK snapshot roundtrip, where the earliest message is the snapshot.

---

### 5. Proxy-Aware Execution

- Futures tests are proxy-aware and respect `ProxySettings` and the proxy injector for `BINANCE_FUTURES`.
- Proxy sanity tests validate pool configuration and injector behavior without hitting Binance or Kafka.
- REST/WS connectivity tests use a preflight helper to call a Binance futures REST endpoint via proxy and to receive at least one WS message via proxy; failures result in clear skips.
- When no proxies are configured, futures tests run in direct mode without changing behavior.

---

### 6. Makefile Integration

- Futures E2E tests are exposed via Makefile targets (mirroring spot):
  - `test-kafka-binance-futures` – runs the futures E2E pytest file with `CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E` and `KAFKA_BOOTSTRAP_SERVERS` wired.
  - `test-kafka-binance-futures-mullvad` – runs the same suite via Mullvad HTTP/WS pools for `BINANCE_FUTURES`, defaulting `KAFKA_E2E_TOPIC_STRATEGY=consolidated`.
  - `test-kafka-all` – includes `test-kafka-binance-futures` alongside existing Kafka suites.

These targets reuse the Redpanda lifecycle helpers from the upstream Kafka specs.

---

### 7. Alignment with Existing Specs

- This spec treats the following as upstream contracts:
  - `market-data-kafka-producer` – topic naming, partitioning, headers, producer behavior.
  - `protobuf-callback-serialization` – serialization behavior and schema headers.
  - `normalized-data-schema-crypto` – futures Protobuf message shapes.
  - `kafka-protobuf-binance-e2e` – overall Kafka/Redpanda E2E harness for Binance spot.
- The futures design only adds coverage for the `BINANCE_FUTURES` connector and futures-specific channels while preserving those contracts.
