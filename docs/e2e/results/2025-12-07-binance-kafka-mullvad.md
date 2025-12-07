# Binance → Kafka Protobuf E2E (Mullvad relays)

**Date**: 2025-12-07  
**Scope**: Live Binance REST/WS through Mullvad SOCKS5 relays into Kafka Protobuf backend (Redpanda) using consolidated topics.

## Environment
- Python 3.12.11 (system venv already present)
- Docker: 29.1.2
- Required deps present: `python-socks`, `aiohttp-socks`, `websockets`, `ccxt`, `ccxtpro`, `cryptofeed`
- Redpanda via `docker/infra/base.yml` (auto-managed by pytest fixture)

## Proxy Selection
Probe command:
```bash
python tools/binance_proxy_probe.py \
  --list-url https://raw.githubusercontent.com/tommy-ca/mulvad-relay-list/refs/heads/proxy-artifacts/relays.txt \
  --list-sha256 c0975acd3fe2d28a8f8e1c8fd0cf20a74feef63b1864d438b3ae7a60151e51c8 \
  --regions eu ap --limit 3 --per-country
```
Selected OK relays (REST + WS):
- `socks5://at-vie-wg-socks5-001.relays.mullvad.net:1080`
- `socks5://be-bru-wg-socks5-101.relays.mullvad.net:1080`
- `socks5://hk-hkg-wg-socks5-201.relays.mullvad.net:1080`

## Execution
Env:
```bash
export CRYPTOFEED_PROXY_ENABLED=true
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__POOL='{"proxies":[{"url":"socks5://at-vie-wg-socks5-001.relays.mullvad.net:1080"},{"url":"socks5://be-bru-wg-socks5-101.relays.mullvad.net:1080"},{"url":"socks5://hk-hkg-wg-socks5-201.relays.mullvad.net:1080"}],"strategy":"round_robin"}'
export CRYPTOFEED_PROXY_EXCHANGES__BINANCE__WEBSOCKET__POOL="$CRYPTOFEED_PROXY_EXCHANGES__BINANCE__HTTP__POOL"
export CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true
export KAFKA_E2E_TOPIC_STRATEGY=consolidated
python -m pytest tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py -v -s
```

## Results
- ✅ 5/5 tests passed in 56.57s
  - Proxy resolution + pool selection
  - Trade roundtrip (Kafka Protobuf)
  - Trade roundtrip (round-robin partitioner)
  - Orderbook snapshot roundtrip
- ✅ Re-validated via `make test-kafka-binance-mullvad` (53.32s) using same pool and consolidated topics.
- Protobuf headers and schema_version validated; payloads parsed (BTC/USDT, ETH/USDT).

## Notes / Follow-ups
- Per-symbol topic strategy not re-checked in this run (set `KAFKA_E2E_TOPIC_STRATEGY=per_symbol` to validate legacy path).
- If repeating, reuse the SHA256 above; relay list source previously 404s on Mullvad main repo—use `tommy-ca/mulvad-relay-list`.
