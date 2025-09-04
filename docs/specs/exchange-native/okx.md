## OKX Native Schemas (v1)

Scope:
- Channels: trades, tickers, books, funding (swap)
- Segments: SPOT, FUTURES, SWAP, OPTIONS

Messages:
- `cryptofeed.exchanges.okx.v1.Trade`
- `cryptofeed.exchanges.okx.v1.Ticker`
- `cryptofeed.exchanges.okx.v1.OrderBook`
- `cryptofeed.exchanges.okx.v1.Funding`

Notes:
- `inst_id` is kept (e.g., BTC-USDT, BTC-USD-SWAP). Mapper derives common `Symbol` and `InstrumentType`.
- Depth entries carry price/size pairs.

TDD:
- Fixtures: OKX public trades/tickers/books examples.
- Tests: JSON -> native -> common mapping invariants.

- Options nuance: `instId` like BTC-USD-YYYYMMDD-STRIKE-C/P should map to `InstrumentType.OPTION`. Strike/expiry parsing is left to higher layers for now (KISS), with `raw_data` preserved.
- Sequence alignment: `seq_id` feeds `L2Book.sequence_number` for ordering checks.

Tasks:
- [x] Add options ticker/trade fixtures (from OKX V5 docs).
- [x] Tests to assert `InstrumentType.OPTION` and symbol mapping (`BTC`/`USD`).
- [ ] Document any side field nuance in options trades.
- BookDelta mapping: OrderBook deltas map to BookDelta with `seq_id` carried as `sequence_number`; raw_data preserved.