## Bybit Native Schemas (v1)

Scope:
- Channels: publicTrade, orderbook (delta/snapshot), ticker
- Segments: SPOT, LINEAR, INVERSE, OPTIONS

Messages:
- `cryptofeed.exchanges.bybit.v1.Trade`
- `cryptofeed.exchanges.bybit.v1.OrderBook`
- `cryptofeed.exchanges.bybit.v1.Ticker`

Notes:
- `symbol` is retained as Bybit (e.g., BTCUSDT, BTCUSD). Side as text.

TDD:
- Fixtures: Bybit WS v5 samples for trades/books/tickers.
- Tests: Constructor and mapping to common.
- Options: map options instruments to `InstrumentType.OPTION`; detailed parsing deferred; `raw_data` preserved.
- Raw payloads: `raw_data` is preserved and propagated into common messages.
