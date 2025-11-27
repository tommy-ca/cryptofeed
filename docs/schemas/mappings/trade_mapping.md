# Trade Schema Field Mapping

This table reconciles fields across the canonical Buf Protobuf schema,
Cryptofeed dataclasses, tardis-node JSON exports, and DBN fixed layouts.

| Canonical Field (Protobuf) | Cryptofeed Dataclass | tardis-node JSON | DBN Layout | Notes |
| --- | --- | --- | --- | --- |
| `exchange` | `Trade.exchange` | `exchange` | `exchange` (ASCII) | Normalized exchange identifier (e.g., `BITFINEX`). |
| `symbol` | `Trade.symbol` | `instrument` | `symbol` | Use normalized symbol/contract naming. |
| `side` | `Trade.side` | `side` | `side_flag` (0=buy,1=sell) | tardis-node/DBN values mapped to Protobuf enum `TradeSide`. |
| `trade_id` | `Trade.id` | `trade_id` | `trade_id` (optional) | Optional unique trade identifier. |
| `price` | `Trade.price` (`Decimal`) | `price` (`string`) | `price` (scaled int64) | Canonical scale 1e-8; DBN stores scaled integer with metadata. |
| `amount` | `Trade.amount` (`Decimal`) | `amount` (`string`) | `quantity` (scaled int64) | Use same scale as price for base asset units. |
| `timestamp` | `Trade.timestamp` (float s) | `ts_event` (µs) | `timestamp` (µs) | Convert dataclass seconds to microseconds before encoding. |
| `raw_id` | `Trade.raw`→`id` if present | `meta.raw_id` | `reserved` | Optional field for traceability; drop if unavailable. |
| `maker` | `Trade.maker` | `m` (bool) | `reserved` | true when venue reports taker/maker flag. |
| `event_time` | `Trade.event_time` | `E` | `reserved` | Exchange event time distinct from match time; microseconds. |
| `match_id` | `Trade.match_id` | `t` | `reserved` | Venue match ID when separate from trade_id. |
| `liquidity_flag` | `Trade.liquidity_flag` | venue-specific (e.g., `liquidity`) | `reserved` | Raw venue liquidity code (e.g., M/T). |

Additional documents:
- `docs/schemas/examples/tardis/trade.json` – tardis-node schema snippet for trades.
- `docs/schemas/examples/dbn/trade.yaml` – representative DBN layout for trades.
- `docs/schemas/examples/events/trades.jsonl` – sample unified events for regression pipeline.
