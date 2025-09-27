# Backpack Native Feed Runbook

## Overview
This runbook guides operators through enabling and supervising the native Backpack feed. It complements the migration notes by focusing on day-to-day actions, observability, and rapid mitigation.

## Prerequisites
- Release deployed with the native Backpack modules and feature flag toggle.
- ED25519 credentials available for private channels (validated with `python -m tools.backpack_auth_check`).
- Proxy configuration tested against the infrastructure injector (if proxies are in use).

## Enable Procedure
1. **Bootstrap metadata** — warm the symbol cache by invoking `BackpackFeed.symbols(["BTC-USDT", ...])` in staging or running `pytest tests/integration/test_backpack_native.py`.
2. **Turn on feature flag** — set `CRYPTOFEED_BACKPACK_NATIVE=true` (or equivalent config key) and restart the FeedHandler instance.
3. **Validate subscriptions** — confirm `feed.metrics_snapshot()` reports non-zero `ws_messages` and `orderbook_resyncs` remains at 0 after initial bootstrap.
4. **Exercise private channels (optional)** — if private flows are enabled, subscribe to `order.*` topics and verify callbacks are received.

## Monitoring Checklist
- **Health endpoint** — poll `feed.health(max_snapshot_age=60)`; it should return `healthy=True`. Investigate if `parser errors encountered` or `order book snapshot stale` appear in the reasons list.
- **Metrics guardrails**:
  - `ws_errors` should remain `0` for steady-state operation.
  - `parser_errors` indicates payload schema drift; escalate to engineering if >5/min.
  - `orderbook_resyncs` will increment on genuine sequence gaps. Occasional increments are acceptable; sustained growth suggests websocket loss.
- **Logs** — watch for `Backpack detected order book gap` or `snapshot fetch failed` warnings. These correlate with REST resyncs and should be rare.

## Troubleshooting
- **Repeated resyncs** — check proxy stability and websocket latency. Enable debug logging for the router to capture raw payload sequences.
- **Authentication failures** — ensure timestamp skew <5 seconds and keys were normalised via `_normalize_key` (hex or base64 accepted).
- **Parser errors** — capture offending payloads (logged at warning level) and compare against recorded fixtures under `tests/fixtures/backpack/`.

## Rollback
1. Toggle `CRYPTOFEED_BACKPACK_NATIVE=false` and redeploy.
2. Confirm `BackpackFeed` is no longer instantiated by checking `feed.metrics_snapshot()` (should be absent) or the registry mapping.
3. Document the failure in `docs/migrations/backpack_ccxt.md` for follow-up.
