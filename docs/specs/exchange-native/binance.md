## Binance Native Schemas (v1)

Scope:
- Channels: trades, bookTicker, depth, funding (UM/CM)
- Segments: SPOT, FUTURES_UM, FUTURES_CM, OPTIONS

Messages:
- `cryptofeed.exchanges.binance.v1.Trade`
- `cryptofeed.exchanges.binance.v1.BookTicker`
- `cryptofeed.exchanges.binance.v1.DepthUpdate`
- `cryptofeed.exchanges.binance.v1.Funding`

Notes:
- `symbol` uses Binance native (e.g., BTCUSDT). Mapping to `cryptofeed.v1.Symbol` happens in mappers.
- `is_buyer_maker` retained to infer side when needed.
- Depth updates store price/qty pairs; deletions represented by size=0.

TDD:
- Fixtures: sample payloads for each channel/segment.
- Tests: parse JSON -> native proto -> common proto, assert price/size/symbol/timestamps.

- Exchange fallback: if native `exchange` is unspecified, mappers derive it from `segment` (SPOT → BINANCE, UM → BINANCE_FUTURES, CM → BINANCE_DELIVERY).
- Raw payloads: `raw_data` bytes are preserved and propagated into common messages.
- aggTrade support: aggregate id `a` mapped to `trade_id`; price `p`, qty `q`, time `T`, taker side via `m`.
- BookDelta mapping: depth updates can be mapped to `BookDelta` where context requires incremental updates (size=0 indicates delete).
