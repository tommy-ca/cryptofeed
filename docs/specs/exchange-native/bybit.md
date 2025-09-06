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
 - LINEAR/INVERSE segments both map to PERPETUAL instrument type in common.

Mapping rules (to common):
- Exchange: `EXCHANGE_BYBIT`.
- Symbol: split concat (e.g., BTCUSDT/BTCUSD) → `Symbol(base, quote, type from segment)`.
- Trade: `side` text → enum; price→`price`, quantity→`amount`; `event_timestamp` if present.
- Ticker: best bid/ask mapped; timestamp optional.
- L2: `OrderBook` bid/ask levels; `seq` → `sequence_number`.
- BookDelta: map levels to changes; zero-size indicates deletion; carry `seq`.
- Funding: map `rate`, `mark_price`, `next_funding_time` when present.
- raw_data: propagate unchanged.

TDD:
- Fixtures: Bybit WS v5 samples for trades/books/tickers.
- Tests: Constructor and mapping to common.
- Options: map options instruments to `InstrumentType.OPTION`; detailed parsing deferred; `raw_data` preserved.
- Raw payloads: `raw_data` is preserved and propagated into common messages.

Additional Tasks:
- [ ] Add inverse funding fixture + test (if available from docs).
- [ ] Add snapshot + delta BookDelta fixture; assert size=0 deletion.

Invariants (acceptance):
- Sequence numbers populated via `seq` for L2/BookDelta.
- `raw_data` equality asserted across channels.
