## Bitget Native Schemas (v1)

Scope:
- Channels: trades, ticker, books
- Segments: SPOT, USDT_PERP, COIN_PERP, OPTIONS

Messages:
- `cryptofeed.exchanges.bitget.v1.Trade`
- `cryptofeed.exchanges.bitget.v1.OrderBook`
- `cryptofeed.exchanges.bitget.v1.Ticker`

Notes:
- `inst_id` retained; mapper derives common symbol.

TDD:
- Fixtures and round-trip tests for core channels.
- Options: map options instruments to `InstrumentType.OPTION`; detailed parsing deferred; `raw_data` preserved.
- Raw payloads: `raw_data` is preserved and propagated into common messages.
