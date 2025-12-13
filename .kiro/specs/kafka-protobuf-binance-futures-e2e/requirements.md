## kafka-protobuf-binance-futures-e2e – Requirements

This spec extends the existing `kafka-protobuf-binance-e2e` (spot) work to cover Binance USDⓈ-M futures. It reuses the same Kafka/Protobuf contracts and Redpanda harness, and adds futures-specific coverage, semantics, and ergonomics.

### R1 — Futures E2E Coverage

- Provide end-to-end tests for Binance USDⓈ-M futures public market data channels, from live exchange to Kafka Protobuf topics and back to decoded Protobuf messages.
- Channels in scope:
  - High-frequency: `TRADES`, `L2_BOOK`, `TICKER`.
  - Derivatives-specific: `FUNDING` (mark price stream), `OPEN_INTEREST` (REST poll), `LIQUIDATIONS` (force orders).
- Symbols in scope:
  - Primary: `BTC-USDT-PERP`.
  - Secondary: `ETH-USDT-PERP` when using consolidated topics, to validate symbol routing.

### R2 — Header and Schema Correctness

- Futures E2E tests MUST assert Kafka headers on consumed records:
  - `content-type == application/x-protobuf`.
  - `schema_version == SCHEMA_VERSION` from normalized-data-schema.
  - `cf.serialization_format == protobuf`.
  - `exchange == binance_futures`.
  - `symbol` is one of `BTC-USDT-PERP`, `ETH-USDT-PERP`.
  - `data_type` matches the logical channel (`trade`, `l2_book`, `ticker`, `funding`, `open_interest`, `liquidation`).
- Futures payloads MUST be decoded using the generated bindings and validated for basic, non-trivial content:
  - trades: non-zero amount and price.
  - books: at least one bid or ask.
  - ticker: at least one of bid/ask populated.
  - funding: mark price and/or rate present.
  - open interest: open_interest not all-zero/empty.
  - liquidations: non-zero quantity and price.

### R3 — Topics and Partition Semantics

- Futures tests MUST support and exercise both topic strategies already defined by the Kafka backend:
  - `per_symbol` (default) and `consolidated`.
- For each strategy, tests MUST verify that:
  - Topic naming matches `TopicManager` semantics (no futures-specific divergence in producers).
  - The appropriate topic(s) receive messages for futures channels under test.
- Partitioning semantics MUST be validated for:
  - Composite partitioner (default): non-`None` key derived from `exchange-symbol`.
  - Round-robin partitioner: keyless messages (`record.key is None`), relying on broker assignment.

### R4 — Proxy-Aware Execution

- The futures E2E suite MUST honor `ProxySettings` and the proxy injector for `BINANCE_FUTURES`:
  - Support HTTP and SOCKS proxies and pools (HTTP/WS) configured via env (`CRYPTOFEED_PROXY_*`).
  - Provide proxy sanity tests that validate pool configuration and injector behavior without hitting Binance or Kafka.
- REST preflight for futures (e.g., `exchangeInfo`, open interest endpoints) MUST:
  - Execute through the configured proxy when present.
  - Skip with a clear, actionable message when proxies/geoblocks prevent reachability.
- Tests MUST default to direct connections when no proxy configuration is present; proxy logic MUST NOT break direct-mode runs.

### R5 — Opt-In and CI Safety

- All futures Kafka Protobuf E2E tests MUST be explicitly gated by an environment variable:
  - `CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E`.
- When the gate is not enabled, tests MUST cleanly `pytest.skip` with a short explanation.
- Tests MUST also skip (not fail) when any of the following are missing or unhealthy:
  - `docker compose` CLI or the Redpanda compose file.
  - Redpanda broker not reachable on the expected host/port.
  - Binance endpoints consistently unreachable within configured timeouts.
- Skips MUST be logged with enough information to distinguish proxy/configuration issues from exchange outages.

### R6 — Makefile Ergonomics

- The spec MUST provide convenient Makefile entrypoints that align with the existing Kafka E2E workflow:
  - `test-kafka-binance-futures` to run the futures Kafka Protobuf E2E suite with `CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E` and `KAFKA_BOOTSTRAP_SERVERS` wired.
  - `test-kafka-binance-futures-mullvad` to run the same suite via Mullvad HTTP/WS pools for `BINANCE_FUTURES`, defaulting `KAFKA_E2E_TOPIC_STRATEGY` to `consolidated`.
  - Inclusion of `test-kafka-binance-futures` in the `test-kafka-all` aggregate target.
- These targets MUST be safe to run repeatedly in local development and CI, and MUST propagate non-zero exit codes when tests fail.

### R7 — Alignment with Existing Specs

- This spec MUST treat the following as upstream contracts, not redefine them:
  - `market-data-kafka-producer`: topic naming, partitioning, header semantics.
  - `protobuf-callback-serialization`: serialization behavior, schema headers.
  - `normalized-data-schema-crypto`: message shapes and field semantics.
  - `kafka-protobuf-binance-e2e`: architectural patterns and spot E2E harness.
- Futures E2E documentation MUST explicitly reference these specs so changes in futures behavior remain traceable to upstream decisions.
