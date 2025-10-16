# Ticker Schema Field Mapping

| Canonical Field (Protobuf) | Cryptofeed Dataclass | tardis-node JSON | DBN Layout | Notes |
| --- | --- | --- | --- | --- |
| `exchange` | `Ticker.exchange` | `exchange` | `exchange` | |
| `symbol` | `Ticker.symbol` | `instrument` | `symbol` | |
| `bid` | `Ticker.bid` (`Decimal`) | `bid` (`string`) | `best_bid_price` (int64, scale 1e-8) | |
| `ask` | `Ticker.ask` (`Decimal`) | `ask` (`string`) | `best_ask_price` (int64, scale 1e-8) | |
| `timestamp` | `Ticker.timestamp` (float s) | `ts_event` (µs) | `timestamp` (µs) | |
