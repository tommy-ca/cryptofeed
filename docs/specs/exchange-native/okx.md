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

Mapping rules (to common):
- Exchange: always `EXCHANGE_OKX`.
- Symbol: split hyphen symbol (e.g., BTC-USDT) → `Symbol(base, quote, type from segment)`.
- Trade: `side` text to common enum; price/size mapped; optional `event_timestamp` to timestamp.
- Ticker: best bid/ask mapped to `Ticker.bid/ask`.
- L2: map `OrderBook` price levels; `seq_id` (or `seq`) → `sequence_number`.
- BookDelta: map bids/asks to changes; carry `seq`/`seq_id` to `sequence_number`.
- Funding: map `mark_price`, `rate`, `next_funding_time` when present.
- raw_data: propagate unchanged.

TDD:
- Fixtures: OKX public trades/tickers/books examples.
- Tests: JSON -> native -> common mapping invariants.

- Options nuance: `instId` like BTC-USD-YYYYMMDD-STRIKE-C/P should map to `InstrumentType.OPTION`. Strike/expiry parsing is left to higher layers for now (KISS), with `raw_data` preserved.
- Sequence alignment: `seq_id` feeds `L2Book.sequence_number` for ordering checks.

Additional Tasks:
- [x] Confirm BUY/SELL semantics for options trades; assert in tests.
- [ ] Add incremental OrderBook delta fixture using public docs with `seq` progression.

Invariants (acceptance):
- `InstrumentType` determined from segment; OPTION set for `...-C/P` forms; base/quote extracted as expected.
- Sequence numbers populated for L2 and BookDelta using `seq`/`seq_id`.
- `raw_data` equality asserted across all channels.
- Options side semantics: `side` of "buy" → `SIDE_BUY`, "sell" → `SIDE_SELL` (asserted by unit tests).

Tasks:
- [x] Add options ticker/trade fixtures (from OKX V5 docs).
- [x] Tests to assert `InstrumentType.OPTION` and symbol mapping (`BTC`/`USD`).
- [ ] Document any side field nuance in options trades.
- BookDelta mapping: OrderBook deltas map to BookDelta with `seq_id` carried as `sequence_number`; raw_data preserved.
