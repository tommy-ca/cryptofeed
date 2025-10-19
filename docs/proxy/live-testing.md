# Live Proxy Testing Guide

This guide captures the process for running opt-in end-to-end proxy tests against
Binance using SOCKS5 relays sourced from the public Mullvad relay artifact. Tests
are designed to verify HTTP proxy routing and highlight regional geofencing
behavior.

## Prerequisites

- `pytest`
- `aiohttp-socks` (already declared in project requirements)
- Access to the Mullvad relay artifact (see below)
- Optional: region-specific SOCKS5 endpoints if you operate your own relay fleet

## Retrieve Proxy Endpoints

Download the latest relay artifact:

```bash
gh run download 18632839930 \
  --repo tommy-ca/mulvad-relay-list \
  -D /tmp/proxy_artifact

head /tmp/proxy_artifact/mullvad-relay-artifacts/mullvad_relays.csv
```

Key columns:

| Column            | Description                              |
|-------------------|------------------------------------------|
| `socks5_endpoint` | Host:port of the SOCKS5 relay             |
| `city`            | Relay city (useful for region selection)  |
| `country`         | Country code                              |

## Environment Variables

Set the proxy and optional stream configuration before invoking pytest:

```bash
export CRYPTOFEED_TEST_SOCKS_PROXY="socks5://<host>:<port>"
export CRYPTOFEED_TEST_BINANCE_SYMBOL="BTCUSDT"  # optional (defaults to BTCUSDT)
export CRYPTOFEED_TEST_BINANCE_WS_STREAM="btcusdt@trade"  # optional (defaults to btcusdt@trade)
export CRYPTOFEED_TEST_BINANCE_WS_TIMEOUT="10"            # optional receive timeout in seconds
export CRYPTOFEED_TEST_CCXT_SYMBOL="BTC/USDC:USDC"        # optional Hyperliquid symbol override
export CRYPTOFEED_TEST_CCXT_WS_TIMEOUT="20"               # optional Hyperliquid ws timeout in seconds
export CRYPTOFEED_TEST_CCXT_REST_TIMEOUT="15"             # optional Hyperliquid REST timeout (seconds)
export CRYPTOFEED_TEST_BACKPACK_CCXT_SYMBOL="BTC/USDC"    # optional Backpack ccxt symbol override
export CRYPTOFEED_TEST_BACKPACK_REST_TIMEOUT="10"         # optional Backpack REST timeout (seconds)
```

## Running the Live Test

```bash
pytest tests/integration/test_live_binance.py -v -m "live_proxy and live_binance"
```

The suite issues a REST ticker request and, when enabled, opens a Binance trade
WebSocket stream through the configured SOCKS proxy. REST requests skip on
**HTTP 451** responses; the WebSocket test skips if the handshake fails or no
message arrives within the configured timeout.

## Regional Observations (2025-10-19)

| Region | Proxy Endpoint                                                | Binance REST | Binance WS (btcusdt@trade)         | Backpack REST | Backpack WS (native) | Backpack REST (ccxt) | Backpack WS (ccxt.pro) | Hyperliquid REST | Hyperliquid WS | WS Echo |
|--------|----------------------------------------------------------------|--------------|-------------------------------------|--------------|---------------------|----------------------|------------------------|------------------|---------------|--------|
| US     | `socks5://us-nyc-wg-socks5-301.relays.mullvad.net:1080`        | ⚠️ 451 skip | ⚠️ 451 skip                        | ✅ markets    | ⚠️ parse error       | ✅ order book        | ✅ trades              | ✅ order book      | ✅ trades      | ✅      |
| Europe | `socks5://de-fra-wg-socks5-101.relays.mullvad.net:1080`        | ✅ pass      | ✅ pass                             | ✅ markets    | ⚠️ parse error       | ✅ order book        | ✅ trades              | ✅ order book      | ✅ trades      | ✅      |
| Asia   | `socks5://sg-sin-wg-socks5-001.relays.mullvad.net:1080`        | ✅ pass      | ✅ pass                             | ✅ markets    | ⚠️ parse error       | ✅ order book        | ✅ trades              | ✅ order book      | ✅ trades      | ✅      |

### Notes

- Hyperliquid and Backpack ccxt tests require `pip install ccxt ccxtpro`.
- Backpack native WebSocket currently responds with `code=4002` parse errors;
  use ccxt.pro Backpack WS for live trade verification.

Your results may vary as Binance continuously updates regional restrictions.

## Troubleshooting

- **HTTP 451**: Binance blocked the relay location. Switch to a different
  endpoint or region.
- **Connection timeout**: Verify DNS reachability for the relay hostname and
  your network’s outbound policy.
- **ImportError (aiohttp-socks)**: Install dependency via `pip install aiohttp-socks`.

## Extending Coverage

- Add additional exchanges or endpoints as needed by copying the integration
  pattern from `tests/integration/test_live_binance.py`.
- Record new regional outcomes in this guide to aid future debugging.
