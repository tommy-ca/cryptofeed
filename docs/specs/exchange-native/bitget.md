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
 - USDT/COIN perpetual segments both map to PERPETUAL instrument type in common.

Mapping rules (to common):
- Exchange: `EXCHANGE_BITGET`.
- Symbol: split concat `inst_id` (e.g., BTCUSDT) → `Symbol(base, quote, type from segment)`.
- Trade: `side` text → enum; price→`price`, size→`amount`.
- Ticker: best bid/ask mapped; timestamp optional.
- L2: `OrderBook` levels; `seq` → `sequence_number`.
- BookDelta: map changes; `seq`/`seq_id` to `sequence_number`.
- Funding: map `rate`, `mark_price`, `next_funding_time` when present.
- raw_data: propagate unchanged.

TDD:
- Fixtures and round-trip tests for core channels.
- Options: map options instruments to `InstrumentType.OPTION`; detailed parsing deferred; `raw_data` preserved.
- Raw payloads: `raw_data` is preserved and propagated into common messages.

Additional Tasks:
- [ ] Add COIN perp funding fixture + test; verify symbol mapping and rate.
- [ ] Add sequencing fixture for L2/BookDelta with `seq` assertion.

Invariants (acceptance):
- Sequence numbers populated via `seq`/`seq_id` where provided.
- `raw_data` equality asserted across channels.
