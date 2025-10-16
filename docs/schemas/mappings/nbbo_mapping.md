# NBBO Schema Field Mapping

| Canonical Field (Protobuf) | Cryptofeed Dataclass | tardis-node JSON | DBN Layout | Notes |
| --- | --- | --- | --- | --- |
| `symbol` | `NBBO.symbol` | `instrument` | `symbol` | |
| `best_bid_exchange` | `NBBO.best_bid_exchange` | `bestBid.exchange` | reserved | Exchange identifier providing best bid |
| `best_bid_price` | `NBBO.best_bid_price` (`Decimal`) | `bestBid.price` | `nbbo_bid_price` (int64, scale 1e-8) | |
| `best_bid_size` | `NBBO.best_bid_size` (`Decimal`) | `bestBid.size` | `nbbo_bid_size` (int64, scale 1e-8) | |
| `best_ask_exchange` | `NBBO.best_ask_exchange` | `bestAsk.exchange` | reserved | |
| `best_ask_price` | `NBBO.best_ask_price` (`Decimal`) | `bestAsk.price` | `nbbo_ask_price` (int64, scale 1e-8) | |
| `best_ask_size` | `NBBO.best_ask_size` (`Decimal`) | `bestAsk.size` | `nbbo_ask_size` (int64, scale 1e-8) | |
| `timestamp` | `NBBO.timestamp` (float s) | `ts_event` (µs) | `timestamp` (µs) | |
