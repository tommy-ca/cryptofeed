# Exchange-Native Protobuf Schemas

Goals:
- Capture native exchange message shapes for Binance, OKX, Bybit, Bitget.
- Preserve exchange-specific fields while enabling mapping to common `cryptofeed.v1` messages.
- Cover Spot, UM/CM futures (or equivalent), and Options segments.
- Drive development via specs and TDD with fixtures and round-trip tests.

Design:
- New packages per exchange: `cryptofeed.exchanges.<name>.v1`.
- Minimal, faithful fields for key channels: trades, order book, ticker, funding (where applicable).
- Include `exchange` (from common enum), `segment` (exchange-specific enum), native symbol/id, price/size as `Decimal`, event timestamps, and `raw_data` for full payload retention.
- No RPC services are introduced yet; focus on data shapes.

Mapping to common:
- Provide mappers from native messages to `cryptofeed.v1.DataFeedEvent` and related types.
- Lossless where possible; otherwise attach original JSON to `raw_data` in both native and common envelopes.

Global invariants:
- `raw_data` present on native → must equal `raw_data` on common.
- Sequence number fields must map consistently per exchange:
  - Binance: `final_update_id` → `L2Book`/`BookDelta.sequence_number`
  - OKX: `seq`/`seq_id` → `sequence_number`
  - Bybit: `seq` → `sequence_number`
  - Bitget: `seq`/`seq_id` → `sequence_number`
- Instrument type derived from exchange `segment` across all exchanges.

Options notes:
- Instrument type is set to OPTION across exchanges.
- Binance options symbols are native (e.g., BTC-240927-60000-C) and do not encode a quote; we intentionally leave `base`/`quote` empty and preserve `symbol` as-is (no over-normalization).
- OKX options `instId` includes base-quote (e.g., BTC-USD-…); we extract base/quote via hyphen splitting.

Segments:
- Binance: SPOT, FUTURES_UM, FUTURES_CM, OPTIONS
- OKX: SPOT, FUTURES, SWAP, OPTIONS
- Bybit: SPOT, LINEAR, INVERSE, OPTIONS
- Bitget: SPOT, USDT_PERP, COIN_PERP, OPTIONS

Testing strategy (TDD):
- Fixtures: small native samples per exchange/channel/segment.
- Deserialize -> Native proto -> Mapper -> Common proto; assert key-field equivalence.
- Serialize/deserialize round-trips for stability.

Cross-exchange param tests:
- Ticker + L2 basic mapping with raw_data assertion across Binance/OKX/Bybit/Bitget.
- BookDelta mapping deltas and deletions across OKX/Bybit/Bitget, plus Binance DepthUpdate variant.
- Funding mapping (where applicable): OKX swap, Bybit linear/inverse, Bitget USDT/COIN perp, Binance UM/CM.
