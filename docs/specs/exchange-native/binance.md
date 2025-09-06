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
 - OPTIONS: Use underlying concat symbol (e.g., BTCUSDT) with `segment=OPTIONS`; we do not parse strike/expiry; instrument type derives from segment.

Mapping rules (to common):
- Exchange: if native `exchange` is unspecified, derive via `segment`:
  - SPOT → `EXCHANGE_BINANCE`
  - FUTURES_UM → `EXCHANGE_BINANCE_FUTURES`
  - FUTURES_CM → `EXCHANGE_BINANCE_DELIVERY`
- Symbol: split native concat symbol (e.g., BTCUSDT) → `Symbol(base='BTC', quote='USDT', type from segment)`.
- Trade: side from `is_buyer_maker` (false=BUY, true=SELL); map qty→`amount`, price→`price`, id→`trade_id`; timestamp from `event_timestamp` if present.
- Ticker: support both `BookTicker` (bid_price/bid_qty/ask_price/ask_qty) and `Ticker` (best_bid/best_ask); map to `Ticker.bid/ask`.
- L2: map `DepthUpdate` price/qty arrays to `L2Book` bids/asks; `final_update_id` → `sequence_number`.
- BookDelta: same levels as L2; size=0 represents deletion.
- Funding: map `mark_price`, `rate`, `next_funding_time` when present; segment determines instrument type.
- raw_data: if present, must propagate unchanged to common.

TDD:
- Fixtures: sample payloads for each channel/segment.
- Tests: parse JSON -> native proto -> common proto, assert price/size/symbol/timestamps.
 - aggTrade: UM/CM fixtures cover `a`→id, `p`→price, `q`→amount, `m`→side, `T`→timestamp.
 - L2/BookDelta: assert sequence mapping and deletion semantics; assert `raw_data` propagation.
 - Funding: assert rate mapping and timestamp presence when available.
 - Options (basic): construct BookTicker/Trade using underlying symbol with `segment=OPTIONS`; assert `InstrumentType.OPTION` and correct side mapping.

Invariants (acceptance):
- Correct exchange enum per segment; symbol base/quote inferred correctly.
- Sequence numbers populated: `final_update_id` → `L2Book.sequence_number` and `BookDelta.sequence_number`.
- `raw_data` equality asserted across trade/ticker/L2/BookDelta/funding tests.
