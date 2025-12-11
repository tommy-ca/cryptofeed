# Timeout Configuration Guide

## Overview

Cryptofeed provides configurable timeout settings for HTTP operations to prevent indefinite hangs when accessing exchange APIs. This guide covers timeout configuration for symbol metadata bootstrap and Binance listen-key operations.

## Timeout Environment Variables

### CF_SYMBOL_FETCH_TIMEOUT

Controls the timeout for symbol metadata fetch operations (e.g., `exchangeInfo` endpoints).

**Default:** 10.0 seconds

**Environment Variable:**
```bash
export CF_SYMBOL_FETCH_TIMEOUT=15.0  # Custom timeout in seconds
# or
export CRYPTOFEED_SYMBOL_FETCH_TIMEOUT=15.0  # Alternative name
```

**Applies to:**
- Exchange symbol mapping bootstrap (all exchanges)
- Binance `exchangeInfo` endpoint
- Binance Futures `fapi/v1/exchangeInfo` endpoint
- Any exchange-specific symbol metadata endpoints

**Use Cases:**
- **Geoblocked regions**: Increase timeout when routing through slow or distant proxies
- **Fast local networks**: Decrease timeout to fail fast on connectivity issues
- **CI/testing**: Set low timeout to prevent hanging tests

### CF_LISTEN_KEY_TIMEOUT

Controls the timeout for Binance user-data stream listen-key operations.

**Default:** 10.0 seconds

**Environment Variable:**
```bash
export CF_LISTEN_KEY_TIMEOUT=8.0  # Custom timeout in seconds
# or
export CRYPTOFEED_LISTEN_KEY_TIMEOUT=8.0  # Alternative name
```

**Applies to:**
- Binance spot listen-key generation (`POST /api/v3/userDataStream`)
- Binance spot listen-key refresh (`PUT /api/v3/userDataStream`)
- Binance Futures listen-key generation (`POST /fapi/v1/listenKey`)
- Binance Futures listen-key refresh (`PUT /fapi/v1/listenKey`)

**Use Cases:**
- **Authenticated streams**: Ensure timely listen-key refresh to maintain user-data streams
- **Proxy routing**: Adjust for latency when routing through HTTP/SOCKS proxies
- **Regional restrictions**: Increase timeout for cross-region proxy pools

## Configuration Examples

### Basic Usage (Default Timeouts)

No configuration needed - defaults work for most scenarios:

```python
from cryptofeed.exchanges.binance import Binance
from cryptofeed.defines import TRADES

# Uses default timeouts (10s for both symbol fetch and listen-key)
feed = Binance(symbols=['BTC-USDT'], channels=[TRADES])
feed.start()
```

### Custom Timeouts via Environment Variables

```bash
# Set custom timeouts before starting your application
export CF_SYMBOL_FETCH_TIMEOUT=5.0   # Fast fail on symbol fetch
export CF_LISTEN_KEY_TIMEOUT=15.0    # Longer timeout for listen-key

python your_app.py
```

### Docker Compose Example

```yaml
# docker-compose.yml
services:
  cryptofeed:
    image: cryptofeed:latest
    environment:
      - CF_SYMBOL_FETCH_TIMEOUT=20.0
      - CF_LISTEN_KEY_TIMEOUT=20.0
      - CRYPTOFEED_PROXY_ENABLED=true
      - CRYPTOFEED_PROXY_DEFAULT__HTTP__URL=socks5://proxy:1080
```

### Kubernetes ConfigMap Example

```yaml
# timeout-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cryptofeed-timeout-config
data:
  CF_SYMBOL_FETCH_TIMEOUT: "15.0"
  CF_LISTEN_KEY_TIMEOUT: "12.0"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cryptofeed
spec:
  template:
    spec:
      containers:
      - name: cryptofeed
        envFrom:
        - configMapRef:
            name: cryptofeed-timeout-config
```

## Timeout Behavior with Proxies

### Direct Mode (No Proxy)

Timeouts apply to direct HTTP requests to exchange APIs:

```python
from cryptofeed.exchanges.binance import Binance

# No proxy configured - direct connection
# Timeout still enforced (default 10s)
feed = Binance(symbols=['BTC-USDT'], channels=[TRADES])
```

### Proxy Mode

Timeouts apply end-to-end including proxy negotiation and exchange response:

```bash
# Enable proxy with custom timeouts
export CRYPTOFEED_PROXY_ENABLED=true
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL=socks5://proxy:1080
export CF_SYMBOL_FETCH_TIMEOUT=30.0  # Longer timeout for slow proxy
export CF_LISTEN_KEY_TIMEOUT=30.0
```

**Important:** Timeout includes:
1. Proxy connection establishment
2. SOCKS/HTTP handshake (if applicable)
3. Exchange API request
4. Exchange API response

### Proxy Pool with Custom Timeouts

```bash
# Configure proxy pool with appropriate timeouts
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__POOL__PROXIES__0__URL=socks5://proxy1:1080
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__POOL__PROXIES__1__URL=socks5://proxy2:1080
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__POOL__STRATEGY=round_robin

# Increase timeout for potentially slower pool members
export CF_SYMBOL_FETCH_TIMEOUT=25.0
export CF_LISTEN_KEY_TIMEOUT=25.0
```

## Troubleshooting

### Timeout Errors

**Symptom:** `asyncio.TimeoutError` or `ClientTimeout` exceptions

**Possible Causes:**
1. Network latency too high for configured timeout
2. Proxy unreachable or slow
3. Exchange API experiencing high latency
4. Geoblocking causing connection delays

**Solutions:**
```bash
# Increase timeout temporarily to diagnose
export CF_SYMBOL_FETCH_TIMEOUT=60.0
export CF_LISTEN_KEY_TIMEOUT=60.0

# Check proxy connectivity
curl -x socks5://proxy:1080 https://api.binance.com/api/v3/time

# Test direct connection (no proxy)
export CRYPTOFEED_PROXY_ENABLED=false
```

### Indefinite Hangs

**Symptom:** Application hangs during startup or listen-key refresh

**Diagnosis:**
```bash
# Set aggressive timeout to identify hang location
export CF_SYMBOL_FETCH_TIMEOUT=5.0
export CF_LISTEN_KEY_TIMEOUT=5.0

# Enable debug logging
export CRYPTOFEED_LOG_LEVEL=DEBUG
```

**Common Issues:**
- Synchronous `requests` calls bypassing timeout configuration (legacy code)
- Missing timeout enforcement in custom HTTP helpers
- Event loop blocking due to sync operations

### Geoblocking and Regional Restrictions

**Symptom:** Timeouts specifically for certain exchanges (e.g., Binance)

**Solution:**
```bash
# Use SOCKS/HTTP proxy for restricted exchanges
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__URL=socks5://regional-proxy:1080
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__URL=socks5://regional-proxy:1081

# Increase timeout for cross-region latency
export CF_SYMBOL_FETCH_TIMEOUT=30.0
export CF_LISTEN_KEY_TIMEOUT=30.0
```

## Testing Timeout Configuration

### Unit Test Example

```python
import pytest
from cryptofeed.exchange import ExchangeRuntimeSettings

def test_custom_symbol_fetch_timeout(monkeypatch):
    """Verify custom timeout is honored"""
    monkeypatch.setenv("CF_SYMBOL_FETCH_TIMEOUT", "5.0")

    settings = ExchangeRuntimeSettings()
    assert settings.symbol_fetch_timeout == 5.0

def test_custom_listen_key_timeout(monkeypatch):
    """Verify custom listen-key timeout is honored"""
    monkeypatch.setenv("CF_LISTEN_KEY_TIMEOUT", "7.0")

    settings = ExchangeRuntimeSettings()
    assert settings.listen_key_timeout == 7.0

def test_default_timeouts():
    """Verify default timeouts when not configured"""
    settings = ExchangeRuntimeSettings()
    assert settings.symbol_fetch_timeout == 10.0
    assert settings.listen_key_timeout == 10.0
```

### Integration Test Example

```python
import pytest
import asyncio
from cryptofeed.exchanges.binance import Binance

@pytest.mark.asyncio
async def test_symbol_fetch_timeout_enforcement(monkeypatch):
    """Verify timeout prevents indefinite hangs"""
    monkeypatch.setenv("CF_SYMBOL_FETCH_TIMEOUT", "0.1")

    # Should timeout quickly instead of hanging
    with pytest.raises(asyncio.TimeoutError):
        Binance.symbol_mapping(refresh=True)
```

## Best Practices

### Production Recommendations

1. **Set explicit timeouts**: Don't rely on defaults in production
   ```bash
   export CF_SYMBOL_FETCH_TIMEOUT=20.0
   export CF_LISTEN_KEY_TIMEOUT=15.0
   ```

2. **Consider proxy latency**: Add 10-15s for SOCKS proxies
   ```bash
   # Direct: 10s, Proxy: 20-25s
   export CF_SYMBOL_FETCH_TIMEOUT=25.0
   ```

3. **Monitor timeout errors**: Track `asyncio.TimeoutError` frequency
   - High frequency → increase timeout or fix network path
   - Zero occurrences → consider decreasing for faster failure detection

4. **Test in staging**: Validate timeouts under load before production
   ```bash
   # Staging with aggressive timeouts
   export CF_SYMBOL_FETCH_TIMEOUT=5.0
   export CF_LISTEN_KEY_TIMEOUT=5.0
   ```

### Development Recommendations

1. **Use default timeouts**: 10s default works for local development
2. **Lower timeouts in CI**: Prevent hanging tests
   ```yaml
   # .github/workflows/test.yml
   env:
     CF_SYMBOL_FETCH_TIMEOUT: 3.0
     CF_LISTEN_KEY_TIMEOUT: 3.0
   ```

3. **Mock slow endpoints**: Test timeout enforcement
   ```python
   async def slow_fetch(*args, **kwargs):
       await asyncio.sleep(100)  # Simulate slow response
       return {}

   monkeypatch.setattr("module._fetch_json_via_proxy", slow_fetch)
   ```

## Implementation Details

### Symbol Fetch Timeout

Symbol metadata fetch uses `ExchangeRuntimeSettings.symbol_fetch_timeout`:

```python
# cryptofeed/exchange.py
class ExchangeRuntimeSettings(BaseSettings):
    symbol_fetch_timeout: float = Field(
        default=10.0,
        validation_alias=AliasChoices(
            "CRYPTOFEED_SYMBOL_FETCH_TIMEOUT",
            "CF_SYMBOL_FETCH_TIMEOUT"
        ),
    )

def _symbol_timeout_seconds() -> float:
    return float(ExchangeRuntimeSettings().symbol_fetch_timeout)
```

Applied in `_fetch_json_via_proxy()`:

```python
async with ClientSession(
    connector=connector,
    timeout=ClientTimeout(total=timeout_seconds)
) as session:
    # ... HTTP request with enforced timeout
```

### Listen-Key Timeout

Binance listen-key operations use `ExchangeRuntimeSettings.listen_key_timeout`:

```python
# cryptofeed/exchange.py
class ExchangeRuntimeSettings(BaseSettings):
    listen_key_timeout: float = Field(
        default=10.0,
        validation_alias=AliasChoices(
            "CRYPTOFEED_LISTEN_KEY_TIMEOUT",
            "CF_LISTEN_KEY_TIMEOUT"
        ),
    )

# cryptofeed/exchanges/binance.py
def _listen_key_timeout_seconds() -> float:
    return float(ExchangeRuntimeSettings().listen_key_timeout)
```

Applied in `_generate_token()` and `_refresh_token()`:

```python
await _http_request_with_proxy(
    method="POST",
    url=url,
    headers=headers,
    timeout=_listen_key_timeout_seconds(),
    exchange_id=self.id.lower()
)
```

## Related Documentation

- [Proxy System User Guide](./user-guide.md) - Full proxy configuration guide
- [Proxy Testing Guide](./testing.md) - Testing proxy-enabled applications
- [Configuration Guide](../core/configuration.md) - General Cryptofeed configuration

## Version History

- **2025-12-10**: Initial documentation (Task 6.7, kafka-protobuf-binance-e2e spec)
  - Documented `CF_SYMBOL_FETCH_TIMEOUT` and `CF_LISTEN_KEY_TIMEOUT`
  - Added examples for direct mode, proxy mode, and proxy pools
  - Included troubleshooting guide and best practices
