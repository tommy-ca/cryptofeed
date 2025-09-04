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

Segments:
- Binance: SPOT, FUTURES_UM, FUTURES_CM, OPTIONS
- OKX: SPOT, FUTURES, SWAP, OPTIONS
- Bybit: SPOT, LINEAR, INVERSE, OPTIONS
- Bitget: SPOT, USDT_PERP, COIN_PERP, OPTIONS

Testing strategy (TDD):
- Fixtures: small native samples per exchange/channel/segment.
- Deserialize -> Native proto -> Mapper -> Common proto; assert key-field equivalence.
- Serialize/deserialize round-trips for stability.

