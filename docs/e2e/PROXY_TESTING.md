# Binance Kafka E2E Tests with Proxy Support

**Spec Reference**: `kafka-protobuf-binance-e2e` (FR7: Proxy-Aware Execution)

## Overview

The Binance Kafka Protobuf E2E test suites support both **spot** and **USDⓈ-M futures** pipelines and can run in three modes:
- **Direct mode** (no proxy) - default
- **Single proxy mode** (HTTP or SOCKS5)
- **Proxy pool mode** (multiple proxies with round-robin selection)

Both HTTP (REST) and WebSocket transports respect proxy configuration, including:
- Symbol metadata bootstrap (`exchangeInfo`)
- User-data listen-key acquisition/refresh (authenticated endpoints)
- Market data WebSocket streams (trades, order books, tickers, funding, etc.)

For an end-to-end pipeline overview (Binance → proxy → Cryptofeed → Redpanda → protobuf decode), see `docs/e2e/BINANCE_KAFKA_PROTOBUF_E2E.md`.

## Field-tested quick recipe (Mullvad SOCKS5, public channels)

Use a single fast relay to avoid pool-induced latency, plus a longer REST timeout:

```bash
export CRYPTOFEED_PROXY_ENABLED=true
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__POOL='{"proxies":[{"url":"socks5://de-fra-wg-socks5-101.relays.mullvad.net:1080"}],"strategy":"round_robin"}'
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__POOL='{"proxies":[{"url":"socks5://de-fra-wg-socks5-101.relays.mullvad.net:1080"}],"strategy":"round_robin"}'
export CF_SYMBOL_FETCH_TIMEOUT=30
export CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true
export KAFKA_BOOTSTRAP_SERVERS=localhost:19092

make redpanda-up            # default 19092; if busy set REDPANDA_HOST_PORT=29092 and REDPANDA_HOST_BOOTSTRAP=localhost:29092
python -m pytest tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py -k "roundtrip and not placeholder" -vv -s --maxfail=1
make redpanda-down
```

Observed results (2025-12-13): trades, round-robin trades, ticker, candle, multi-channel, and orderbook snapshot passed; candle can still be slow and may skip if no candle arrives within 180s.

## Prerequisites

### Required Dependencies

**Core dependencies** (always required):
```bash
# From this repo (recommended)
pip install -e .

# Kafka E2E test dependencies
pip install confluent-kafka aiokafka
```

**SOCKS proxy support** (required only when using SOCKS proxies):
```bash
# Proxy extras (includes aiohttp-socks + python-socks)
pip install -e ".[proxy]"
```

**Note**: If you configure a SOCKS WebSocket proxy but `python-socks` is not installed, tests will skip with a clear message.

### Infrastructure Requirements

1. **Docker + docker compose** - for Redpanda (Kafka-compatible) test cluster
2. **Network access** to Binance public endpoints (or proxy that can reach them)
3. **Environment gating** - tests are opt-in and require explicit activation

## Quick Start

### 1. Direct Mode (No Proxy)

This is the **default mode** and requires no proxy configuration:

```bash
# Start Redpanda
make redpanda-up

# Run Binance spot E2E tests
CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true \
KAFKA_BOOTSTRAP_SERVERS=localhost:19092 \
python -m pytest tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py -v

# Run Binance futures E2E tests
CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E=true \
KAFKA_BOOTSTRAP_SERVERS=localhost:19092 \
python -m pytest tests/integration/kafka/test_binance_futures_kafka_protobuf_pipeline.py -v

# Stop Redpanda
make redpanda-down
```

### 2. Single HTTP Proxy

Configure a single HTTP proxy for both REST and WebSocket:

```bash
# Set proxy environment variables
export CRYPTOFEED_PROXY_ENABLED=true
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL=http://proxy.example.com:8080
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__URL=http://proxy.example.com:8080

# Start Redpanda
make redpanda-up

# Run spot tests with HTTP proxy
CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true \
KAFKA_BOOTSTRAP_SERVERS=localhost:19092 \
python -m pytest tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py -v

# For futures, use BINANCE_FUTURES instead of BINANCE
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__HTTP__URL=http://proxy.example.com:8080
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__WEBSOCKET__URL=http://proxy.example.com:8080

CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E=true \
KAFKA_BOOTSTRAP_SERVERS=localhost:19092 \
python -m pytest tests/integration/kafka/test_binance_futures_kafka_protobuf_pipeline.py -v

# Stop Redpanda
make redpanda-down
```

### 3. Single SOCKS5 Proxy

**Important**: SOCKS5 WebSocket proxies require `python-socks`:

```bash
# Install SOCKS WebSocket support
pip install python-socks

# Set SOCKS proxy environment variables
export CRYPTOFEED_PROXY_ENABLED=true
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL=socks5://user:pass@proxy.example.com:1080
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__URL=socks5://user:pass@proxy.example.com:1080

# Start Redpanda
make redpanda-up

# Run spot tests with SOCKS proxy
CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true \
KAFKA_BOOTSTRAP_SERVERS=localhost:19092 \
python -m pytest tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py -v

# For futures
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__HTTP__URL=socks5://user:pass@proxy.example.com:1080
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__WEBSOCKET__URL=socks5://user:pass@proxy.example.com:1080

CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E=true \
KAFKA_BOOTSTRAP_SERVERS=localhost:19092 \
python -m pytest tests/integration/kafka/test_binance_futures_kafka_protobuf_pipeline.py -v

# Stop Redpanda
make redpanda-down
```

### 4. Proxy Pool (Multiple Proxies with Round-Robin)

Configure multiple proxies that will be selected in round-robin fashion:

```bash
# Set proxy pool for Binance spot (JSON format)
export CRYPTOFEED_PROXY_ENABLED=true
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__POOL='{
  "proxies": [
    {"url": "socks5://proxy1.example.com:1080", "weight": 1},
    {"url": "socks5://proxy2.example.com:1080", "weight": 1},
    {"url": "socks5://proxy3.example.com:1080", "weight": 1}
  ],
  "strategy": "round_robin"
}'

export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__POOL='{
  "proxies": [
    {"url": "socks5://proxy1.example.com:1080", "weight": 1},
    {"url": "socks5://proxy2.example.com:1080", "weight": 1},
    {"url": "socks5://proxy3.example.com:1080", "weight": 1}
  ],
  "strategy": "round_robin"
}'

# Start Redpanda
make redpanda-up

# Run spot tests with proxy pool
CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true \
KAFKA_BOOTSTRAP_SERVERS=localhost:19092 \
python -m pytest tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py -v

# For futures, use BINANCE_FUTURES
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__HTTP__POOL='{...}'
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__WEBSOCKET__POOL='{...}'

CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E=true \
KAFKA_BOOTSTRAP_SERVERS=localhost:19092 \
python -m pytest tests/integration/kafka/test_binance_futures_kafka_protobuf_pipeline.py -v

# Stop Redpanda
make redpanda-down
```

## Live E2E: Binance → Mullvad SOCKS5 → Cryptofeed (Docker Compose) → Kafka (Protobuf)

**Goal**: Validate full path with real Binance REST/WS traffic routed through Mullvad relays, normalized by Cryptofeed, and published as Protobuf to Kafka.

### Prerequisites
- Valid Binance API key/secret.
- Mullvad SOCKS5 endpoints (e.g., `socks5://user:pass@at-vie-wg-socks5-001.relays.mullvad.net:1080`).
- Docker + docker compose; host ports 8080/9090/9092 available.

### Configuration (env-first, 12-Factor)
```bash
cp .env.example .env
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...

export CRYPTOFEED_PROXY_ENABLED=true
export CRYPTOFEED_PROXY_DEFAULT__HTTP__URL=socks5://user:pass@at-vie-wg-socks5-001.relays.mullvad.net:1080
export CRYPTOFEED_PROXY_DEFAULT__WEBSOCKET__URL=$CRYPTOFEED_PROXY_DEFAULT__HTTP__URL
# (Optional per-exchange overrides)

export KAFKA_BOOTSTRAP_SERVERS=kafka:29092
```

In `config/config.yaml` (or via env with `CRYPTOFEED_EXCHANGES__BINANCE__CHANNELS` / `__SYMBOLS`):
```yaml
binance:
  channels: [trades, l2_book, ticker]
  symbols: [BTC-USDT, ETH-USDT]
kafka:
  bootstrap_servers: [kafka:29092]
  topic_strategy: consolidated
  partition_strategy: composite
```

### Run stack
```bash
docker compose up -d
```
- Wait for health: `curl http://localhost:8080/health` should be 200; metrics at `http://localhost:9090/metrics`.

### Validate Kafka output
List topics:
```bash
docker compose exec kafka kafka-topics --bootstrap-server localhost:9092 --list
```
Consume a few trade messages (headers only for quick check):
```bash
docker compose exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic cryptofeed.trades \
  --from-beginning --max-messages 5 --property print.headers=true
```
Expect headers with `exchange=binance`, `data_type=trade`; payload is Protobuf.

### Verify proxy usage
```bash
docker compose logs cryptofeed | grep \"proxy:\"
docker compose exec cryptofeed python - <<'PY'
from cryptofeed.proxy import load_proxy_settings
s = load_proxy_settings()
print("enabled", s.enabled, "http", s.get_proxy("binance","http"))
print("ws", s.get_proxy("binance","websocket"))
PY
```

### Teardown
```bash
docker compose down
```

### Notes
- If Mullvad relay is unreachable, health will degrade and `/health` returns 503; fix endpoint or disable proxy to recover.
- For Kafka inspection of Protobuf payloads, use the protobuf-callback test decoder or dedicated consumer in tests/integration/kafka/*.

## Makefile Targets

Convenient Makefile targets are available for common scenarios:

### Binance Spot E2E Tests

```bash
# Direct mode (no proxy)
make test-kafka-binance

# With Mullvad proxy pool (requires Mullvad proxy configuration)
make test-kafka-binance-mullvad
```

### Binance Futures E2E Tests

```bash
# Direct mode (no proxy)
make test-kafka-binance-futures

# With Mullvad proxy pool (requires Mullvad proxy configuration)
make test-kafka-binance-futures-mullvad
```

### Combined Tests

```bash
# Run all Kafka E2E tests (spot + futures + unit + perf)
make test-kafka-all
```

### Redpanda Management

```bash
# Start Redpanda cluster
make redpanda-up

# Check Redpanda health
make redpanda-health

# Stop Redpanda cluster
make redpanda-down

# List containers using port 19092
make docker-ps-19092

# Stop conflicting containers (use with caution)
make docker-stop-19092
```

## Environment Variables Reference

### Test Gating (Required)

| Variable | Purpose | Values | Default |
|----------|---------|--------|---------|
| `CRYPTODATA_RUN_BINANCE_KAFKA_E2E` | Enable Binance spot E2E tests | `true`, `1`, `yes`, `on` | unset (tests skip) |
| `CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E` | Enable Binance futures E2E tests | `true`, `1`, `yes`, `on` | unset (tests skip) |

### Kafka Configuration

| Variable | Purpose | Default |
|----------|---------|---------|
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka/Redpanda bootstrap address | `localhost:19092` |
| `KAFKA_E2E_TOPIC_STRATEGY` | Topic naming strategy | `per_symbol` |

Topic strategies:
- **`per_symbol`** (default): Topics like `cryptofeed.trade.binance.btc-usdt`
- **`consolidated`**: Topics like `cryptofeed.trade` (uses message headers for routing)

### Proxy Configuration

All proxy configuration uses the `CRYPTOFEED_PROXY_*` prefix with nested keys separated by `__`:

#### Enable Proxy System

```bash
export CRYPTOFEED_PROXY_ENABLED=true
```

#### Single Proxy Configuration

**Binance Spot:**
```bash
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL=http://proxy:8080
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__URL=socks5://proxy:1080
```

**Binance Futures:**
```bash
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__HTTP__URL=http://proxy:8080
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__WEBSOCKET__URL=socks5://proxy:1080
```

#### Proxy Pool Configuration (JSON)

**Binance Spot HTTP Pool:**
```bash
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__POOL='{
  "proxies": [
    {"url": "http://p1:8080", "weight": 1},
    {"url": "http://p2:8080", "weight": 1}
  ],
  "strategy": "round_robin"
}'
```

**Binance Spot WebSocket Pool:**
```bash
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__POOL='{
  "proxies": [
    {"url": "socks5://ws1:1080", "weight": 1},
    {"url": "socks5://ws2:1080", "weight": 1}
  ],
  "strategy": "round_robin"
}'
```

**Binance Futures:** Replace `BINANCE` with `BINANCE_FUTURES` in the above examples.

### Timeout Configuration

Configure timeouts for HTTP operations (see `docs/proxy/timeout-configuration.md` for details):

| Variable | Purpose | Default |
|----------|---------|---------|
| `CF_SYMBOL_FETCH_TIMEOUT` | Symbol metadata bootstrap timeout (seconds) | 10 |
| `CF_LISTEN_KEY_TIMEOUT` | Listen-key acquire/refresh timeout (seconds) | 10 |
| `CF_SCHEMA_REGISTRY_TIMEOUT` | Schema registry HTTP timeout (seconds) | 10 |

Example:
```bash
export CF_SYMBOL_FETCH_TIMEOUT=20
export CF_LISTEN_KEY_TIMEOUT=15
```

## Proxy Support Details

### What Gets Proxied

**HTTP (REST) Operations:**
- Symbol metadata bootstrap: `GET /api/v3/exchangeInfo` (spot) or `/fapi/v1/exchangeInfo` (futures)
- Listen-key generation: `POST /api/v3/userDataStream` (spot) or `/fapi/v1/listenKey` (futures)
- Listen-key refresh: `PUT /api/v3/userDataStream` (spot) or `/fapi/v1/listenKey` (futures)

**WebSocket Operations:**
- Public market data streams (trades, order books, tickers)
- Derivatives-specific streams (funding, open interest, liquidations)
- User data streams (for authenticated channels)

### Proxy Precedence

Configuration is loaded with the following precedence:
1. **Environment variables** (`CRYPTOFEED_PROXY_*`)
2. **YAML configuration file** (`config.yaml`)
3. **Programmatic configuration** (Python `ProxySettings` object)

Environment variables always override other sources.

### Implementation Notes

- **Symbol bootstrap**: Uses `aiohttp` with `ProxyInjector` integration (migrated from sync `requests`)
- **Listen-key flows**: Uses `aiohttp` with `ProxyInjector` integration (migrated from sync `requests`)
- **WebSocket proxies**: Require `python-socks` for SOCKS4/SOCKS5 support
- **HTTP proxies**: Require `aiohttp-socks` for SOCKS4/SOCKS5 REST support
- **Timeouts**: All proxy HTTP calls are timeout-bound with configurable defaults (10s)

## Skip Conditions

Tests will skip with clear messages when:

1. **Test gating env var not set**:
   ```
   Binance Kafka Protobuf E2E disabled. Set CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true to enable.
   ```

2. **Docker/docker compose unavailable**:
   ```
   Docker compose not available. Install Docker to run Redpanda-based tests.
   ```

3. **Redpanda not reachable**:
   ```
   Redpanda not reachable at localhost:19092. Ensure Redpanda is running.
   ```

4. **SOCKS WebSocket proxy configured but python-socks missing**:
   ```
   SOCKS WebSocket proxy configured but python-socks not installed. Install: pip install python-socks
   ```

5. **REST preflight fails** (proxy/geoblock issues):
   ```
   Binance REST preflight failed: HTTP 451 (geo-restriction) or connection error.
   ```

6. **Proxy configured but no proxy URL resolved**:
   ```
   Proxy configured for binance but no HTTP proxy URL resolved.
   ```

## Troubleshooting

### Issue: Tests skip due to missing python-socks

**Symptom:**
```
test_binance_trade_roundtrip_live SKIPPED [  33%] SOCKS WebSocket proxy configured but python-socks not installed.
```

**Solution:**
```bash
pip install python-socks
```

### Issue: Connection timeout to Binance

**Symptom:**
```
asyncio.TimeoutError: Symbol fetch timed out after 10.0s
```

**Solution:**
1. Check proxy connectivity: `curl --socks5 proxy.example.com:1080 https://api.binance.com/api/v3/time`
2. Increase timeout: `export CF_SYMBOL_FETCH_TIMEOUT=30`
3. Verify proxy authentication credentials in URL

### Issue: HTTP 451 Unavailable For Legal Reasons

**Symptom:**
```
Binance REST preflight failed: HTTP 451 (geo-restriction)
```

**Solution:**
1. Configure a proxy in a supported region
2. Use a SOCKS5 proxy with proper country selection
3. Tests will skip gracefully - this is expected behavior in restricted regions

### Issue: Redpanda port conflict (19092)

**Symptom:**
```
Error starting Redpanda: port 19092 already in use
```

**Solution:**
```bash
# List containers using port 19092
make docker-ps-19092

# Stop conflicting containers (review first)
make docker-stop-19092

# Or manually stop specific container
docker stop <container-id>
```

### Issue: Proxy pool selection returns None

**Symptom:**
```
Proxy configured for binance but no HTTP proxy URL resolved.
```

**Solution:**
1. Verify JSON syntax in pool configuration (use single quotes around JSON, double quotes inside)
2. Check proxy URLs are valid and reachable
3. Ensure at least one proxy in the pool has weight > 0
4. Validate pool strategy is `round_robin` (only supported strategy currently)

### Issue: Listen-key timeout

**Symptom:**
```
asyncio.TimeoutError: Listen key generation timed out after 10.0s
```

**Solution:**
1. Increase timeout: `export CF_LISTEN_KEY_TIMEOUT=30`
2. Check proxy can reach Binance authenticated endpoints
3. Verify API credentials are valid (if using authenticated channels)

## Test Coverage

### Binance Spot E2E (`test_binance_kafka_protobuf_pipeline.py`)

**Market Data Channels:**
- Trades (default composite partitioner)
- Trades (round-robin partitioner)
- L2 Order Book (snapshot + deltas)

**Proxy Tests:**
- Proxy resolution when configured
- Proxy pool selection without live Binance

### Binance Futures E2E (`test_binance_futures_kafka_protobuf_pipeline.py`)

**Market Data Channels:**
- Trades (default composite partitioner)
- Trades (round-robin partitioner)
- L2 Order Book (snapshot + deltas)
- Ticker (24hr stats)
- Funding (mark price)
- Open Interest (REST poll)
- Liquidations (force orders)
- Multi-channel (combined)

**Proxy Tests:**
- Proxy resolution when configured
- Proxy pool selection without live Binance

## Example Test Run Output

### Successful Direct Mode Run

```bash
$ CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true KAFKA_BOOTSTRAP_SERVERS=localhost:19092 \
  python -m pytest tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py -v

tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py::test_binance_trade_roundtrip_live PASSED [ 33%]
tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py::test_binance_trade_roundtrip_round_robin_keyless PASSED [ 66%]
tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py::test_binance_orderbook_snapshot_roundtrip PASSED [100%]

============================== 3 passed in 45.23s ==============================
```

### Successful Proxy Mode Run

```bash
$ export CRYPTOFEED_PROXY_ENABLED=true
$ export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL=socks5://proxy:1080
$ export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__URL=socks5://proxy:1080
$ CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true KAFKA_BOOTSTRAP_SERVERS=localhost:19092 \
  python -m pytest tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py -v

tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py::test_binance_trade_roundtrip_live PASSED [ 20%]
tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py::test_binance_trade_roundtrip_round_robin_keyless PASSED [ 40%]
tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py::test_binance_orderbook_snapshot_roundtrip PASSED [ 60%]
tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py::test_binance_proxy_resolution_when_configured PASSED [ 80%]
tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py::test_binance_proxy_pool_selection_without_live PASSED [100%]

============================== 5 passed in 48.17s ==============================
```

### Skipped Due to Missing Environment Variable

```bash
$ python -m pytest tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py -v

tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py::test_binance_trade_roundtrip_live SKIPPED [ 33%]
tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py::test_binance_trade_roundtrip_round_robin_keyless SKIPPED [ 66%]
tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py::test_binance_orderbook_snapshot_roundtrip SKIPPED [100%]

============================== 3 skipped in 0.12s ==============================
```

## Related Documentation

- **Proxy System**: `docs/proxy/README.md` - General proxy system documentation
- **Proxy User Guide**: `docs/proxy/user-guide.md` - Detailed proxy configuration examples
- **Timeout Configuration**: `docs/proxy/timeout-configuration.md` - HTTP timeout configuration
- **E2E Test Plan**: `docs/e2e/TEST_PLAN.md` - Comprehensive E2E test strategy
- **Spec Requirements**: `.kiro/specs/kafka-protobuf-binance-e2e/requirements.md` - FR7 proxy requirements

## Validation History

- **Task 6** (2025-12-11): Proxy-configured E2E runs enabled - 9 unit tests validating proxy loading
- **Task 6.1** (2025-12-11): Proxy/pool resolution validated - 13 unit tests covering HTTP/SOCKS/pools/direct mode
- **Task 6.3-6.5** (2025-12-07): Symbol bootstrap, listen-key, and preflight proxy integration completed
- **Task 6.6-6.8** (2025-12-08): Requests migration to aiohttp + ProxyInjector completed (Wave 1 + Wave 2)

## Production Usage

For production deployments using proxy-enabled Kafka ingestion:

1. **Use environment variables** for proxy configuration (preferred over YAML in containerized environments)
2. **Configure timeouts** appropriate for your proxy latency (especially for geographically distant proxies)
3. **Use proxy pools** for high-throughput deployments to distribute load across proxies
4. **Monitor proxy health** - Cryptofeed will log proxy selection and connection errors
5. **Set up retry logic** - Cryptofeed includes built-in retry with exponential backoff for proxy failures

Example production environment variables:
```bash
# Enable proxy system
CRYPTOFEED_PROXY_ENABLED=true

# Configure Binance spot proxy pool
CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__POOL='{"proxies":[...],"strategy":"round_robin"}'
CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__POOL='{"proxies":[...],"strategy":"round_robin"}'

# Configure Binance futures proxy pool
CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__HTTP__POOL='{"proxies":[...],"strategy":"round_robin"}'
CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__WEBSOCKET__POOL='{"proxies":[...],"strategy":"round_robin"}'

# Adjust timeouts for proxy latency
CF_SYMBOL_FETCH_TIMEOUT=20
CF_LISTEN_KEY_TIMEOUT=15

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS=kafka-broker1:9092,kafka-broker2:9092,kafka-broker3:9092
KAFKA_E2E_TOPIC_STRATEGY=consolidated
```

## Skip avoidance checklist (fast triage)

- **Port 19092 already bound**: either stop the conflicting container (`make docker-ps-19092` / `make docker-stop-19092`) or run Redpanda on a different host port via `REDPANDA_HOST_PORT` + `REDPANDA_HOST_BOOTSTRAP` (e.g., 29092).
- **Pool too slow/blocked**: start with a single known-good relay (e.g., `socks5://de-fra-wg-socks5-101.relays.mullvad.net:1080`) before enabling round-robin pools.
- **REST preflight timeouts**: set `CF_SYMBOL_FETCH_TIMEOUT=30` and ensure HTTP proxy matches the WebSocket proxy.
- **SOCKS dependencies**: install `python-socks` and `aiohttp-socks`; otherwise SOCKS configs will skip.
- **Candle flakiness**: candle roundtrip waits 180s; if it skips, retry with more symbols or extend the timeout locally.

---

**Maintained by**: kafka-protobuf-binance-e2e spec (FR7: Proxy-Aware Execution)
**Last Updated**: 2025-12-13
**Spec Status**: Task 6.2 (Document proxy-enabled runs) - COMPLETE
