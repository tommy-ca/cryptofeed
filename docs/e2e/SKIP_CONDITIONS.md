# E2E Test Skip Conditions Reference

This document describes all skip conditions in the Binance Kafka Protobuf E2E test suites (spot and futures) and provides clear guidance for operators when tests skip.

## Skip Condition Categories

### 1. Environment Opt-In

**Condition**: E2E environment variable not set

**Spot Tests Skip Message**:
```
Binance Kafka Protobuf E2E tests disabled. Set CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true to enable.
```

**Futures Tests Skip Message**:
```
Binance Futures Kafka Protobuf E2E disabled. Set CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E=true to enable.
```

**What to Fix**:
```bash
# For spot tests
export CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true

# For futures tests
export CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E=true

# Then rerun tests
make test-kafka-binance  # spot
make test-kafka-binance-futures  # futures
```

**Documentation**: See test file docstrings and `docs/e2e/README.md`

---

### 2. Docker Unavailable

**Condition**: Docker or `docker compose` command not found

**Skip Message**: Tests skip at Redpanda fixture level

**What to Fix**:
```bash
# Install Docker Desktop or Docker Engine
# Verify installation:
docker --version
docker compose version

# Start Docker daemon if needed
sudo systemctl start docker  # Linux
# Or start Docker Desktop application (macOS/Windows)
```

**Documentation**: `docs/e2e/README.md` (Prerequisites section)

---

### 3. Redpanda Unreachable

**Condition**: Kafka producer fails to connect to Redpanda bootstrap servers

**Skip Message**:
```
Kafka producer failed to connect to Redpanda
```

**What to Fix**:
```bash
# Start Redpanda cluster
make redpanda-up

# Verify Redpanda is healthy
make redpanda-health

# Check Redpanda logs if issues persist
docker compose -f docker/infra/base.yml logs redpanda

# Ensure port 19092 is not in use
make docker-ps-19092  # shows any conflicts
```

**Documentation**: `docs/e2e/README.md`, `Makefile` targets

---

### 4. Binance REST Endpoint Unreachable

**Condition**: Binance `exchangeInfo` API times out or returns non-200 status

**Skip Messages**:
```
# Timeout
Binance REST exchangeInfo via proxy failed: TimeoutError(...)

# Geoblocked (403/451 status)
Binance REST exchangeInfo via proxy failed (status 451); REST geoblocked or proxy blocked.

# Connection error
Binance REST exchangeInfo via proxy failed: ConnectionError(...)
```

**What to Fix**:

1. **Direct Mode (no proxy)**:
   ```bash
   # Check network connectivity
   curl -I https://api.binance.com/api/v3/exchangeInfo

   # If geoblocked, configure proxy (see below)
   ```

2. **Proxy Mode**:
   ```bash
   # Configure HTTP proxy for REST
   export CRYPTOFEED_PROXY_ENABLED=true
   export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL=http://proxy:8080

   # For SOCKS proxy
   export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL=socks5://user:pass@proxy:1080
   pip install python-socks  # required for SOCKS

   # Verify proxy is reachable
   curl -I --proxy $CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL https://api.binance.com/api/v3/exchangeInfo
   ```

3. **Timeout Configuration**:
   ```bash
   # Increase timeout if needed (default 10s)
   export CF_SYMBOL_FETCH_TIMEOUT=30
   ```

**Documentation**:
- `docs/e2e/PROXY_TESTING.md`
- `docs/proxy/timeout-configuration.md`
- `docs/proxy/README.md`

---

### 5. Binance WebSocket Timeout

**Condition**: No messages received from Binance WebSocket within timeout

**Skip Messages**:
```
# Trade timeout
Binance Kafka Protobuf E2E: no message consumed within timeout: AssertionError(...)

# Order book timeout
Binance Kafka Protobuf E2E (orderbook): no message within timeout; possible REST snapshot or WS connectivity issue
```

**What to Fix**:

1. **Check Binance WebSocket connectivity**:
   ```bash
   # Test WebSocket endpoint directly
   wscat -c wss://stream.binance.com:9443/ws/btcusdt@trade
   # Should see live trade messages
   ```

2. **Configure WebSocket proxy**:
   ```bash
   export CRYPTOFEED_PROXY_ENABLED=true
   export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__URL=http://proxy:8080

   # For SOCKS WebSocket proxy
   export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__URL=socks5://user:pass@proxy:1080
   pip install python-socks  # REQUIRED for SOCKS WebSocket
   ```

3. **Increase timeout** (for slow networks/proxies):
   - Timeouts are hardcoded in test files (60s-150s depending on channel)
   - Consider slower proxy routes or low-volume pairs may need longer waits

**Documentation**:
- `docs/e2e/PROXY_TESTING.md` (WebSocket proxy configuration)
- `tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py` docstring

---

### 6. python-socks Missing

**Condition**: SOCKS proxy configured but `python-socks` library not installed

**Skip Message**:
```
Binance Kafka Protobuf E2E: SOCKS websocket proxy configured but python-socks is not installed
```

**What to Fix**:
```bash
# Install python-socks dependency
pip install python-socks

# Or install with Cryptofeed extras
pip install -e ".[proxy]"  # includes python-socks + aiohttp-socks

# Verify installation
python -c "import python_socks; print('OK')"
```

**Documentation**:
- Test file docstrings (Quick Start sections)
- `requirements.txt` / `setup.py`

---

### 7. Missing Message Headers

**Condition**: Consumed Kafka message missing required protobuf headers

**Skip Message**:
```
Binance Kafka Protobuf E2E: missing headers [b'content-type', ...]; observed={...}
```

**What to Fix**:
- This indicates a backend bug or serialization issue
- Check `KafkaProtobufCallback` configuration
- Ensure protobuf serialization is enabled correctly
- File a bug report with observed headers

**Documentation**:
- `.kiro/specs/kafka-protobuf-binance-e2e/requirements.md` (FR1, FR4)
- `docs/kafka/technical-specification.md` (headers spec)

---

### 8. aiohttp Missing

**Condition**: `aiohttp` library not available for REST preflight

**Skip Message**:
```
aiohttp not available for REST preflight
```

**What to Fix**:
```bash
# Install aiohttp (should already be installed)
pip install aiohttp

# Or reinstall Cryptofeed dependencies
pip install -e .
```

---

### 9. Proxy Configuration Issues

**Condition**: Various proxy configuration problems

**Skip Messages**:
```
# No proxy configured when required
No proxy configuration provided for Binance

# Pool configured but no proxies available
Binance Kafka Protobuf E2E: websocket proxy configured but no proxy was selected

# Pool lease returned no release handle
Binance Kafka Protobuf E2E: websocket proxy configured but injector returned no release handle

# Proxy settings not enabled
Proxy settings not enabled; pool lease not applicable
```

**What to Fix**:

1. **Enable proxy system**:
   ```bash
   export CRYPTOFEED_PROXY_ENABLED=true
   ```

2. **Configure proxy URLs**:
   ```bash
   # Single proxy
   export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL=http://proxy:8080
   export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__URL=http://proxy:8080

   # Proxy pool
   export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__POOL='{"proxies":[{"url":"http://p1:8080","weight":1}],"strategy":"round_robin"}'
   ```

3. **Verify proxy resolution**:
   ```python
   from cryptofeed.proxy import get_proxy_injector, init_proxy_system, load_proxy_settings

   settings = load_proxy_settings()
   init_proxy_system(settings)
   injector = get_proxy_injector()

   # Test HTTP proxy
   http_url = injector.get_http_proxy_url("binance")
   print(f"HTTP proxy: {http_url}")

   # Test WebSocket proxy
   ws_url, release = injector.lease_proxy("binance", "websocket")
   print(f"WS proxy: {ws_url}")
   release()
   ```

**Documentation**:
- `docs/e2e/PROXY_TESTING.md` (comprehensive proxy guide)
- `docs/proxy/README.md`
- `.kiro/specs/kafka-protobuf-binance-e2e/requirements.md` (FR7)

---

## Common Resolution Patterns

### Full E2E Setup Checklist

```bash
# 1. Start infrastructure
make redpanda-up
make redpanda-health  # verify healthy

# 2. Configure environment (optional - for proxy mode)
export CRYPTOFEED_PROXY_ENABLED=true
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL=http://proxy:8080
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__URL=http://proxy:8080

# 3. Install dependencies (if using SOCKS)
pip install python-socks

# 4. Enable E2E tests
export CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true         # spot
export CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E=true # futures

# 5. Run tests
make test-kafka-binance           # spot, direct mode
make test-kafka-binance-futures   # futures, direct mode

# Or with explicit proxy
make test-kafka-binance-mullvad          # spot via Mullvad proxies
make test-kafka-binance-futures-mullvad  # futures via Mullvad proxies

# 6. Cleanup
make redpanda-down
```

### Troubleshooting Workflow

1. **Test skips at start** → Check env var (`CRYPTODATA_RUN_BINANCE_*_KAFKA_E2E`)
2. **Docker-related skip** → Verify Docker installed and daemon running
3. **Redpanda connection failure** → Run `make redpanda-up` and `make redpanda-health`
4. **Binance REST timeout** → Configure HTTP proxy or check network
5. **Binance WS timeout** → Configure WebSocket proxy or check connectivity
6. **SOCKS error** → Install `python-socks` library
7. **Missing headers** → Backend/serialization bug, file issue
8. **Proxy config error** → Review proxy env vars and injector initialization

---

## References

- **E2E Test Files**:
  - `tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py` (spot)
  - `tests/integration/kafka/test_binance_futures_kafka_protobuf_pipeline.py` (futures)

- **Documentation**:
  - `docs/e2e/README.md` - E2E test overview and quick start
  - `docs/e2e/PROXY_TESTING.md` - Proxy configuration guide
  - `docs/proxy/README.md` - Proxy system documentation
  - `docs/proxy/timeout-configuration.md` - Timeout settings

- **Specifications**:
  - `.kiro/specs/kafka-protobuf-binance-e2e/requirements.md` - Functional requirements
  - `.kiro/specs/kafka-protobuf-binance-e2e/design.md` - Technical design
  - `.kiro/specs/kafka-protobuf-binance-e2e/tasks.md` - Implementation tasks

- **Makefile Targets**:
  - `make redpanda-up` / `make redpanda-down` - Manage Redpanda cluster
  - `make redpanda-health` - Check Redpanda status
  - `make test-kafka-binance` - Run spot E2E tests (direct)
  - `make test-kafka-binance-futures` - Run futures E2E tests (direct)
  - `make test-kafka-binance-mullvad` - Run spot E2E via Mullvad proxy
  - `make test-kafka-binance-futures-mullvad` - Run futures E2E via Mullvad proxy
  - `make docker-ps-19092` - Check port 19092 conflicts
