# Order Book Schema Field Mapping

| Canonical Field (Protobuf) | Cryptofeed Dataclass | tardis-node JSON | DBN Layout | Notes |
| --- | --- | --- | --- | --- |
| `exchange` | `OrderBook.exchange` | `exchange` | `exchange` | Market data focus (L2/L3) |
| `symbol` | `OrderBook.symbol` | `instrument` | `symbol` | Normalized trading pair |
| `bids.price` | `OrderBook.book.bids` (tuple price) | `bids[][0]` | `bid_price` (int64, scale 1e-8) | Cryptofeed tuple value is canonical |
| `bids.quantity` | `OrderBook.book.bids` (tuple size) | `bids[][1]` | `bid_quantity` (int64, scale 1e-8) | |
| `asks.price` | `OrderBook.book.asks` (tuple price) | `asks[][0]` | `ask_price` (int64, scale 1e-8) | |
| `asks.quantity` | `OrderBook.book.asks` (tuple size) | `asks[][1]` | `ask_quantity` (int64, scale 1e-8) | |
| `timestamp` | `OrderBook.timestamp` | `ts_event` | `timestamp` (µs) | Cryptofeed seconds → µs |
| `sequence` | `OrderBook.sequence_number` | `sequence` | `sequence` | Optional, exchange provided |
| `checksum` | `OrderBook.checksum` | `checksum` | `reserved` | Optional integrity indicator |
| `event_time` | `OrderBook.event_time` | `E` | `reserved` | Exchange event time in microseconds when provided |
| `last_update_id` | `OrderBook.last_update_id` | `lastUpdateId` | `reserved` | Exchange book update identifier |
