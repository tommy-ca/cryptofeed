# CCXT Legacy to Typed Configuration Migration

This guide helps you migrate from legacy dict-based CCXT configuration to modern Pydantic-based types.

## Overview

The CCXT integration now supports **both** legacy dict-based and modern typed configuration for backward compatibility. However, typed configuration is preferred for better IDE support, validation, and documentation.

### Legacy Pattern (Still Supported)
```python
from cryptofeed.exchanges.ccxt import CcxtFeed

feed = CcxtFeed(
    exchange_id="backpack",
    proxies={
        "rest": "http://proxy:7000",
        "websocket": "socks5://proxy:7001",
    },
    ccxt_options={
        "apiKey": "...",
        "secret": "...",
    },
    symbols=["BTC-USDT"],
    channels=["trades", "l2_book"],
)
```

### Modern Pattern (Recommended)
```python
from cryptofeed.exchanges.ccxt.config import CcxtConfig

config = CcxtConfig(
    exchange_id="backpack",
    api_key="...",
    secret="...",
    proxies={
        "rest": "http://proxy:7000",
        "websocket": "socks5://proxy:7001",
    },
)

feed = CcxtFeed(
    config=config.to_exchange_config(),
    symbols=["BTC-USDT"],
    channels=["trades", "l2_book"],
)
```

## Migration Steps

### Step 1: Audit Your Code

Find all `CcxtFeed` instantiations in your codebase:
```bash
grep -r "CcxtFeed(" --include="*.py"
```

### Step 2: Convert Dictionary Arguments

**Before:**
```python
feed = CcxtFeed(
    exchange_id="binance",
    proxies={"rest": "http://proxy:7000"},
    ccxt_options={"apiKey": "KEY", "secret": "SECRET"},
)
```

**After:**
```python
from cryptofeed.exchanges.ccxt.config import CcxtConfig

config = CcxtConfig(
    exchange_id="binance",
    api_key="KEY",
    secret="SECRET",
    proxies={"rest": "http://proxy:7000"},
)

feed = CcxtFeed(config=config.to_exchange_config())
```

### Step 3: Handle Optional Fields

**Before:**
```python
ccxt_options={
    "apiKey": "...",
    "secret": "...",
    "password": "...",  # Passphrase
    "timeout": 30000,
    "enableRateLimit": True,
}
```

**After:**
```python
CcxtConfig(
    exchange_id="...",
    api_key="...",
    secret="...",
    passphrase="...",
    timeout=30,  # Seconds, not milliseconds
    enable_rate_limit=True,
)
```

### Step 4: Transport Configuration

**Before:**
```python
CcxtFeed(
    exchange_id="...",
    snapshot_interval=5,
    websocket_enabled=True,
    rest_only=False,
)
```

**After:**
```python
from cryptofeed.exchanges.ccxt.config import CcxtExchangeConfig

config = CcxtExchangeConfig(
    exchange_id="...",
    transport=TransportConfig(
        snapshot_interval=5,
        websocket_enabled=True,
        rest_only=False,
    ),
)
```

### Step 5: Proxy Configuration

**Before:**
```python
proxies={
    "rest": "http://proxy:7000",
    "websocket": "socks5://proxy:7001",
}
```

**After:**
```python
proxies={
    "rest": "http://proxy:7000",
    "websocket": "socks5://proxy:7001",
}
# Same format! Applies to both legacy and typed configs
```

## Feature Comparison

| Aspect | Legacy Dict | Typed Config |
|--------|-------------|--------------|
| IDE autocomplete | ❌ | ✅ |
| Type validation | ⚠️ Runtime | ✅ Pydantic |
| Documentation | 📖 In code | ✅ Embedded |
| Example | `{"apiKey": "..."}` | `api_key="..."` |
| Default handling | Manual | Automatic |
| Backward compat | ✅ Fully | ✅ Full |

## Common Patterns

### Pattern 1: Environment Variables
```python
from cryptofeed.exchanges.ccxt.config import CcxtConfig
import os

config = CcxtConfig(
    exchange_id=os.getenv("CCXT_EXCHANGE", "backpack"),
    api_key=os.getenv("CCXT_API_KEY"),
    secret=os.getenv("CCXT_SECRET"),
    proxies={
        "rest": os.getenv("HTTP_PROXY", ""),
        "websocket": os.getenv("WS_PROXY", ""),
    } if os.getenv("HTTP_PROXY") else None,
)

feed = CcxtFeed(config=config.to_exchange_config())
```

### Pattern 2: Configuration File (YAML)
```python
import yaml
from cryptofeed.exchanges.ccxt.config import CcxtConfig, CcxtExchangeConfig

with open("config.yaml") as f:
    data = yaml.safe_load(f)

config = CcxtConfig(**data["ccxt"])
feed = CcxtFeed(config=config.to_exchange_config())
```

### Pattern 3: Multiple Exchanges
```python
from cryptofeed.exchanges.ccxt.config import CcxtConfig

configs = {
    "backpack": CcxtConfig(exchange_id="backpack", ...),
    "binance": CcxtConfig(exchange_id="binance", ...),
}

feeds = {
    exchange: CcxtFeed(config=cfg.to_exchange_config())
    for exchange, cfg in configs.items()
}
```

## Deprecation Timeline

| Phase | Timeline | Status |
|-------|----------|--------|
| **Current** | Now | Both patterns work |
| **Phase 1** | v2.1+ | Warnings added (optional) |
| **Phase 2** | v2.2+ | Warnings enabled by default |
| **Phase 3** | v3.0+ | Legacy pattern removed |

To opt into warnings early, set environment variable:
```bash
CRYPTOFEED_WARN_LEGACY_CCXT=1
```

## Troubleshooting

### Issue: `ValidationError: api_key must be a string`
**Solution:** Use `api_key="..."` not `apiKey="..."`:
```python
# ❌ Wrong
CcxtConfig(apiKey="...")

# ✅ Correct
CcxtConfig(api_key="...")
```

### Issue: `TypeError: exchange_id is required`
**Solution:** If using legacy pattern, explicitly pass `exchange_id`:
```python
# ✅ Both work
CcxtFeed(exchange_id="backpack", ...)
CcxtFeed(config=CcxtConfig(exchange_id="backpack").to_exchange_config())
```

### Issue: Proxy not being applied
**Solution:** Ensure proxy URL format is correct:
```python
# ✅ Correct formats
"http://proxy:7000"
"https://proxy:7000"
"socks4://proxy:7001"
"socks5://proxy:7001"

# ❌ Incorrect
"proxy:7000"  # Missing scheme
```

## Getting Help

- **API Documentation**: See `cryptofeed.exchanges.ccxt.config`
- **Examples**: Check `examples/ccxt_*.py`
- **Issues**: Report in GitHub with `[CCXT]` tag

## Next Steps

1. Run your existing code (no changes needed!)
2. Gradually adopt typed config in new code
3. Update legacy usage opportunistically
4. Reach out if you hit any issues

