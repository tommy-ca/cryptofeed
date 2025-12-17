# Binance → Kafka (Protobuf) E2E Pipeline

This document describes the **end-to-end validation path**:

`Binance REST/WS` → `SOCKS/HTTP proxies (optional)` → `Cryptofeed normalization` → `KafkaProtobufCallback` → `Redpanda (Kafka)` → `consume + protobuf decode`

The implementation is tracked by:
- `.kiro/specs/kafka-protobuf-binance-e2e/` (spot)
- `.kiro/specs/kafka-protobuf-binance-futures-e2e/` (USDⓈ-M futures)

These suites are intentionally **opt-in** (they hit live Binance endpoints and require a local broker).

---

## What Gets Validated

- **Connectivity**: Binance REST bootstrap + WebSocket streams, optionally through proxies
- **Normalization**: events are emitted as `cryptofeed.types.*` dataclasses
- **Kafka production**: `KafkaProtobufCallback` publishes protobuf bytes to Redpanda
- **Routing metadata**: headers include `exchange`, `symbol`, `data_type`, `schema_version`, `content-type`, `cf.serialization_format`
- **Protobuf roundtrip**: payload decodes using generated bindings under `cryptofeed.backends.protobuf.bindings`

Spot suite focus:
- `TRADES`, `TICKER`, `CANDLES`, `L2_BOOK` (order book snapshot + deltas)
- Derivatives-only channels are represented as placeholders and will skip

Futures suite focus:
- High-frequency: `TRADES`, `TICKER`, `L2_BOOK`
- Derivatives: `FUNDING`, `OPEN_INTEREST` (REST poll), `LIQUIDATIONS`

---

## Test Suites

- **Spot E2E**: `tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py`
  - Gate: `CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true`
  - Marker: `@pytest.mark.live_binance`
- **Futures E2E (USDⓈ-M)**: `tests/integration/kafka/test_binance_futures_kafka_protobuf_pipeline.py`
  - Gate: `CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E=true`
  - Marker: `@pytest.mark.live_binance`
- **Shared infra**:
  - `tests/integration/kafka/conftest.py` (Redpanda fixture + docker compose wiring)
  - `tests/integration/kafka/topic_provision.py` (topic auto-provision via `aiokafka`)
  - `tests/integration/kafka/helpers.py` (consume helper via `confluent_kafka`)

---

## Prerequisites

### Infrastructure

- **Docker + docker compose** (Redpanda broker)
- A free local port (default `19092`)

**Redpanda lifecycle note**:
- If Redpanda is **not** running, the `redpanda` pytest fixture will start it via `docker compose up -d` and stop it after the test session.
- If Redpanda **is already running** on `REDPANDA_HOST_BOOTSTRAP` (default `localhost:19092`), the tests will reuse it and will **not** tear it down.

### Python Dependencies

These tests require Kafka client libs that are not part of Cryptofeed’s minimal install:

```bash
# From the repo root (recommended for development)
pip install -e .

# Kafka integration test dependencies
pip install confluent-kafka aiokafka
```

If you want to run through SOCKS proxies (Mullvad relays), install proxy extras too:

```bash
pip install -e ".[proxy]"
```

Notes:
- `python-socks` is required for **SOCKS WebSocket** proxying.
- `aiohttp-socks` is required for **SOCKS HTTP/REST** proxying.

---

## Quick Start (Recommended)

### Option A: Mullvad SOCKS5 proxy pool (EU/AP relays)

This matches the “Binance geofenced from US IPs” reality and is the easiest way to run reliably.

```bash
make redpanda-up

# Spot
make test-kafka-binance-mullvad

# Futures
make test-kafka-binance-futures-mullvad

make redpanda-down
```

### Option B: Direct mode (no proxy)

```bash
make redpanda-up

export CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true
make test-kafka-binance

export CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E=true
make test-kafka-binance-futures

make redpanda-down
```

---

## Configuration Reference

### Topic Strategy

Controlled by `KAFKA_E2E_TOPIC_STRATEGY`:
- `consolidated` (recommended): topics like `cryptofeed.trade`, routing via headers
- `per_symbol`: topics like `cryptofeed.trade.binance.btc-usdt`

Example:

```bash
export KAFKA_E2E_TOPIC_STRATEGY=consolidated
```

### Redpanda / Kafka Bootstrap

```bash
export KAFKA_BOOTSTRAP_SERVERS=localhost:19092
```

Note: the pytest `redpanda` fixture connects using `REDPANDA_HOST_BOOTSTRAP` (default `localhost:19092`). Keep these aligned if you override ports or run against a non-default broker.

To avoid port conflicts:

```bash
export REDPANDA_HOST_PORT=29092
export REDPANDA_HOST_BOOTSTRAP=localhost:29092
export KAFKA_BOOTSTRAP_SERVERS=localhost:29092
```

### Proxy Configuration

For the full proxy configuration matrix (single proxy vs pool, per-exchange overrides, timeouts), use:
- `docs/e2e/PROXY_TESTING.md`
- `docs/proxy/README.md`

Important runtime note:
- Binance **symbol mapping** is fetched over REST during startup; if you proxy WS but not REST, the feed can still fail early. The E2E tests try to keep REST and WS aligned when proxying is enabled.

---

## Troubleshooting Pointers

- Skip reasons and remediation: `docs/e2e/SKIP_CONDITIONS.md`
- Kafka backend behavior/spec: `docs/kafka/user-guide.md`
- Redpanda management targets: `Makefile` (`redpanda-up`, `redpanda-health`, `redpanda-down`)
